// SPDX-License-Identifier: AGPL-3.0-only

//! mHC multi-seq decode, Phase 7 tail: FFN + `hc_post` over the n rows.
//! Split out of `mod.rs` for the 500-LoC cap.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_hc_ffn_post(
        &self,
        c: &MultiSeqCtx<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        // K=2 verify (num_drafts=1): one batched MoE + one n=2 `hc_post`
        // instead of two single-token FFNs and two n=1 `hc_post`s. On this
        // model that replaces 12 attention layers x 2 tokens = 24 single-token
        // MoE dispatches per verify step with 12 batched ones -- which the 36
        // GDN layers have always done (`trait_decode_batched_hc.rs:305`).
        //
        // Routing is per-row identical to the per-token path: the batched
        // softmax top-k carries the same lower-index-wins tie-break as
        // `moe_topk_softmax` (kernels/gb10/common/moe_topk.cu).
        //
        // `hc_post_site` is per-row elementwise with grid.x = num_tokens, so
        // one n=2 launch is bitwise the two n=1 launches it replaces.
        //
        // Kill switch: `ATLAS_MSHC_FFN_K2=0` restores the per-token loop.
        let batch_ffn = n == 2
            && !self.ffn.is_none()
            && std::env::var("ATLAS_MSHC_FFN_K2").as_deref() != Ok("0");
        if batch_ffn {
            self.ffn.forward_k2(c.normed, ctx, stream)?;
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                ctx.buffers.moe_output(),
                hc_streams,
                post,
                comb,
                hc_streams,
                n as u32,
                h as u32,
                stream,
            )?;
        } else {
            // Per-token sequential FFN (MLA models always take this path).
            for i in 0..n {
                let normed2_i = c.normed.offset(i * c.h * c.bf16);
                let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
                // hc_streams is the FP32 mHC highway (4 bytes/elem), not BF16.
                let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
                let post_i = post.offset(i * hc.hc_mult * 4);
                let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
                ops::hc_post_site(
                    ctx.gpu,
                    self.hc_post_k,
                    hc,
                    moe_out,
                    hc_streams_i,
                    post_i,
                    comb_i,
                    hc_streams_i,
                    1,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
