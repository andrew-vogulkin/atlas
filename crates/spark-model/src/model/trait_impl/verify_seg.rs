// SPDX-License-Identifier: AGPL-3.0-only

//! Segmented CUDA-graph capture of the single-sequence K=2 verify step
//! (R6 item 3). The whole-step verify graph (`verify_b.rs`) is vetoed on any
//! model whose attention layers carry a QSA indexer; this captures the runs
//! of GDN layers between the attention layers (plus the norm/lm_head/argmax
//! tail) as separate graphs and leaves the attention layers eager.
//!
//! Cache entries are keyed by pool slot and fingerprinted by every
//! per-sequence device address the segments bake (SSM pool pointers, the PLE
//! carry). They are dropped in `invalidate_slot_graphs` (free), in
//! compaction, and on LoRA drain; the fingerprint is a second line of
//! defence so a stale entry is re-captured instead of replayed.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GraphHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::SequenceState;

/// One slot's segment graphs. `graphs[i]`: `None` = not captured yet,
/// `Some(GraphHandle(0))` = empty capture (run that segment eagerly).
pub(crate) struct SegGraphs {
    pub(crate) fingerprint: u64,
    pub(crate) graphs: Vec<Option<GraphHandle>>,
}

impl SegGraphs {
    pub(crate) fn new(fingerprint: u64, n: usize) -> Self {
        Self {
            fingerprint,
            graphs: vec![None; n],
        }
    }

    /// Every real (non-empty) exec graph, for destruction.
    pub(crate) fn into_handles(self) -> Vec<GraphHandle> {
        self.graphs
            .into_iter()
            .flatten()
            .filter(|g| g.0 != 0)
            .collect()
    }
}

/// `ATLAS_VERIFY_SEG_GRAPHS=0` restores the eager K=2 verify byte-for-byte.
pub(crate) fn verify_seg_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_VERIFY_SEG_GRAPHS").ok().as_deref() != Some("0");
        tracing::info!("segmented K=2 verify graphs: {}", if on { "ON" } else { "OFF (ATLAS_VERIFY_SEG_GRAPHS=0)" });
        on
    })
}

/// FNV-1a over every per-sequence device address a verify segment bakes.
pub(crate) fn seq_baked_fingerprint(seq: &SequenceState) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(seq.slot_idx as u64);
    for st in seq.layer_states.iter() {
        if let Some(ssm) = st.as_any().downcast_ref::<SsmLayerState>() {
            mix(ssm.h_state.0);
            mix(ssm.conv_state.0);
            mix(ssm.h_state_checkpoint.map_or(0, |p| p.0));
            mix(ssm.conv_state_checkpoint.map_or(0, |p| p.0));
            mix(ssm.h_state_intermediates.len() as u64);
            for p in &ssm.h_state_intermediates {
                mix(p.0);
            }
            mix(ssm.conv_state_intermediates.len() as u64);
            for p in &ssm.conv_state_intermediates {
                mix(p.0);
            }
            mix(ssm.h_is_f16 as u64);
            if let Some(ple) = ssm.ple.as_ref() {
                let [a, b] = ple.baked_ptrs();
                mix(a);
                mix(b);
            }
        } else {
            mix(1);
        }
    }
    h
}

impl TransformerModel {
    /// Drop (and destroy) the segment graphs of the given slots.
    pub(in crate::model) fn drop_segment_graphs(&self, slots: &[usize]) {
        let mut dead = Vec::new();
        {
            let mut cache = self.verify_segment_graphs.lock();
            for s in slots {
                if let Some(e) = cache.remove(s) {
                    dead.extend(e.into_handles());
                }
            }
        }
        for g in dead {
            if let Err(e) = self.gpu.destroy_graph(g) {
                tracing::warn!("drop_segment_graphs: destroy graph: {e:#}");
            }
        }
    }
}

/// Set when a segment capture failed: the process falls back to the eager
/// verify for good (logged once). Never set on a healthy serve.
static SEG_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn seg_disabled() -> bool {
    SEG_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// `ATLAS_VERIFY_SEG_TAIL_ONLY=1`: capture only the norm/lm_head/argmax tail
/// (R6 step 3 de-risking mode); every GDN run stays eager.
fn seg_tail_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VERIFY_SEG_TAIL_ONLY").ok().as_deref() == Some("1"))
}

pub(crate) enum SegCall<'r, 'c> {
    /// Run the segment body with this context (eager or under capture).
    Run(&'r ForwardContext<'c>),
    /// Re-arm consumed prestage state before an eager re-run.
    Rearm,
    /// The segment's kernels were replayed: do the host-side bookkeeping.
    ReplayHost,
    /// Eager stream work that must follow the segment graph's launch.
    PostLaunch,
}

impl TransformerModel {
    /// True when every layer that would land inside a captured segment
    /// (i.e. every non-attention layer) allows verify capture.
    pub(in crate::model) fn seg_layers_ok(&self) -> bool {
        self.layers.iter().enumerate().all(|(i, l)| {
            self.config.layer_type(i) == LayerType::FullAttention || !l.verify_graph_unsupported()
        })
    }

    /// Replay, capture-and-launch, or eagerly run one segment.
    fn seg_exec(
        &self,
        slot: &mut Option<GraphHandle>,
        cap_ok: bool,
        ctx: &ForwardContext<'_>,
        ctx_cap: &ForwardContext<'_>,
        seg_idx: usize,
        stream: u64,
        body: &mut dyn FnMut(SegCall<'_, '_>) -> Result<()>,
    ) -> Result<()> {
        match *slot {
            Some(g) if g.0 != 0 => {
                self.gpu.launch_graph(g, stream)?;
                body(SegCall::ReplayHost)?;
                body(SegCall::PostLaunch)
            }
            Some(_) => body(SegCall::Run(ctx)),
            None if !cap_ok || seg_disabled() => body(SegCall::Run(ctx)),
            None => {
                self.gpu.begin_capture(stream)?;
                let r = body(SegCall::Run(ctx_cap));
                let g = self.gpu.end_capture(stream);
                match (r, g) {
                    (Ok(()), Ok(g)) if g.0 != 0 => {
                        *slot = Some(g);
                        self.gpu.launch_graph(g, stream)?;
                        body(SegCall::PostLaunch)
                    }
                    (Ok(()), Ok(_)) => {
                        // Empty capture: nothing was recorded, nothing ran.
                        *slot = Some(GraphHandle(0));
                        body(SegCall::Rearm)?;
                        body(SegCall::Run(ctx))
                    }
                    (r, g) => {
                        if let Ok(g) = g
                            && g.0 != 0
                        {
                            let _ = self.gpu.destroy_graph(g);
                        }
                        let why = match (&r, &g) {
                            (Err(e), _) => format!("{e:#}"),
                            (_, Err(e)) => format!("{e:#}"),
                            _ => String::new(),
                        };
                        if !SEG_DISABLED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            tracing::warn!(
                                "segmented K=2 verify: capture of segment {seg_idx} failed ({why}); \
                                 falling back to eager verify for this process"
                            );
                        }
                        body(SegCall::Rearm)?;
                        body(SegCall::Run(ctx))
                    }
                }
            }
        }
    }

    /// Single-sequence K=2 verify forward with the GDN runs and the tail as
    /// separate CUDA graphs and the attention layers eager. Leaves the two
    /// argmax ids at `scratch()[0..8)` exactly like the whole-step path.
    pub(in crate::model) fn verify_k2_segmented(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext<'_>,
        ctx_cap: &ForwardContext<'_>,
        k: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let n = self.layers.len();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < n {
            if self.config.layer_type(i) == LayerType::FullAttention {
                i += 1;
                continue;
            }
            let s = i;
            while i < n && self.config.layer_type(i) != LayerType::FullAttention {
                i += 1;
            }
            runs.push((s, i));
        }
        let nseg = runs.len() + 1;
        let tail_only = seg_tail_only();

        let fp = seq_baked_fingerprint(seq);
        let mut cache = self.verify_segment_graphs.lock();
        let stale = cache
            .get(&seq.slot_idx)
            .is_some_and(|e| e.fingerprint != fp || e.graphs.len() != nseg);
        if stale && let Some(e) = cache.remove(&seq.slot_idx) {
            tracing::info!(
                "segmented K=2 verify: slot {} fingerprint changed, re-capturing",
                seq.slot_idx
            );
            for g in e.into_handles() {
                if let Err(e) = self.gpu.destroy_graph(g) {
                    tracing::warn!("segmented verify: destroy stale graph: {e:#}");
                }
            }
        }
        let entry = cache
            .entry(seq.slot_idx)
            .or_insert_with(|| SegGraphs::new(fp, nseg));
        let fresh = entry.graphs.iter().all(|g| g.is_none());

        let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
        let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

        let mut run_idx = 0usize;
        let mut li = 0usize;
        while li < n {
            if self.config.layer_type(li) == LayerType::FullAttention {
                let layer = &self.layers[li];
                let mut seq_state_arr: [&mut (dyn LayerState + 'static); 1] =
                    [seq.layer_states[li].as_mut()];
                let row_owner = vec![0usize; k];
                layer.decode_multi_seq_rows(
                    hidden,
                    residual,
                    k,
                    &mut seq_state_arr,
                    &row_owner,
                    kv_cache,
                    &seq_lens_vec,
                    &block_tables_vec,
                    ctx,
                    stream,
                )?;
                self.try_dflash_capture(li, k - 1, stream)?;
                li += 1;
                continue;
            }
            let (s, e) = runs[run_idx];
            debug_assert_eq!(s, li);
            let mut body = |call: SegCall<'_, '_>| -> Result<()> {
                match call {
                    SegCall::Run(c) => {
                        for j in s..e {
                            self.layers[j].decode_batched(
                                hidden,
                                residual,
                                k,
                                seq.layer_states[j].as_mut(),
                                kv_cache,
                                seq.seq_len,
                                &mut seq.block_table,
                                &mut seq.disk_block_ids,
                                &mut seq.disk_last_offloaded_per_layer,
                                c,
                                stream,
                            )?;
                            self.try_dflash_capture(j, k - 1, stream)?;
                        }
                        Ok(())
                    }
                    SegCall::Rearm => {
                        for j in s..e {
                            self.layers[j].decode_prestage_rearm(seq.layer_states[j].as_mut());
                        }
                        Ok(())
                    }
                    SegCall::ReplayHost => {
                        for j in s..e {
                            self.layers[j].verify_replay_host(seq.layer_states[j].as_mut(), k)?;
                        }
                        Ok(())
                    }
                    SegCall::PostLaunch => {
                        for j in s..e {
                            self.layers[j].verify_graph_post_launch(self.gpu.as_ref(), stream)?;
                        }
                        Ok(())
                    }
                }
            };
            self.seg_exec(
                &mut entry.graphs[run_idx],
                !tail_only,
                ctx,
                ctx_cap,
                run_idx,
                stream,
                &mut body,
            )?;
            run_idx += 1;
            li = e;
        }

        // Tail: final norm [K, H] -> lm_head -> K argmax into scratch[0..K*4).
        let mut tail = |call: SegCall<'_, '_>| -> Result<()> {
            if let SegCall::Run(_) = call {
                let normed = self.buffers.norm_output();
                self.final_norm_rows(hidden, normed, k as u32, stream)?;
                self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;
                let vocab = self.config.vocab_size;
                let argmax_out = self.buffers.scratch();
                for t in 0..k {
                    let logits_t = self.buffers.logits().offset(t * vocab * 2);
                    ops::argmax_bf16(
                        self.gpu.as_ref(),
                        self.argmax_kernel,
                        logits_t,
                        argmax_out.offset(t * 4),
                        vocab as u32,
                        stream,
                    )?;
                }
            }
            Ok(())
        };
        self.seg_exec(
            &mut entry.graphs[runs.len()],
            true,
            ctx,
            ctx_cap,
            runs.len(),
            stream,
            &mut tail,
        )?;
        if fresh && entry.graphs.iter().all(|g| g.is_some()) {
            tracing::info!(
                "Captured segmented K=2 verify graphs (slot={}, {} GDN runs + tail{})",
                seq.slot_idx,
                runs.len(),
                if tail_only { ", tail only" } else { "" }
            );
        }
        Ok(())
    }
}
