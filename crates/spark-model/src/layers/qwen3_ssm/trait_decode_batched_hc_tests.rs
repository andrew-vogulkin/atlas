// SPDX-License-Identifier: AGPL-3.0-only

//! MoE arm selection for the mHC batched decode when K>3.
//!
//! Two concurrent MTP requests (num_drafts=1) verify as one K=4 batch. Before
//! the pairwise arm existed, K=4 on the 512-expert MoE fell through to a bail
//! ("no batched MoE arm for K=4") and both requests failed after 2-3 tokens.
//! These tests pin that every K the multi-sequence verify can produce has an
//! arm, and that the pairwise arm cannot clobber its own output.

use super::{MoeFallbackArm, k2_pair_order, moe_fallback_arm};

#[test]
fn k4_two_mtp_sequences_have_a_moe_arm() {
    // 2 seqs x (num_drafts=1 + 1) = K=4: the width that used to bail.
    assert_eq!(moe_fallback_arm(4, false), Some(MoeFallbackArm::PairwiseK2));
}

#[test]
fn every_multi_sequence_width_has_a_moe_arm() {
    for k in 4..=64usize {
        let want = if k.is_multiple_of(2) {
            MoeFallbackArm::PairwiseK2
        } else {
            MoeFallbackArm::PerToken
        };
        assert_eq!(moe_fallback_arm(k, false), Some(want), "K={k}");
    }
}

#[test]
fn widths_with_dedicated_arms_are_not_rerouted() {
    // K=1..3 have forward / forward_k2 / forward_k3 ahead of the fallback; the
    // fallback still refuses them so a skipped arm is an error, not a silent
    // re-route.
    for k in 1..=3 {
        assert_eq!(moe_fallback_arm(k, false), None, "K={k}");
    }
    for k in 1..=16 {
        assert_eq!(moe_fallback_arm(k, true), Some(MoeFallbackArm::Dense));
    }
}

#[test]
fn pairwise_runs_every_pair_once_and_pair_zero_last() {
    for k in (4..=64).step_by(2) {
        let order: Vec<usize> = k2_pair_order(k).collect();
        assert_eq!(order.len(), k / 2, "K={k}");
        assert_eq!(order.last(), Some(&0), "K={k}: pair 0 must run last");
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..k / 2).collect::<Vec<_>>(), "K={k}");
    }
}
