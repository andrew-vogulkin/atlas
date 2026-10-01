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

use spark_runtime::gpu::GraphHandle;

use super::super::types::TransformerModel;
use crate::layer::SsmLayerState;
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
