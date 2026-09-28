// SPDX-License-Identifier: AGPL-3.0-only

//! Name fallback and single-trellis selection. These decide which row cache
//! `load` opens; they do not touch the GPU.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{ngram_table_open, ple_i64_name};

fn uploaded(name: &str) -> WeightStore {
    let map = [(
        name.to_string(),
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: vec![1],
            dtype: WeightDtype::Int64,
        },
    )]
    .into_iter()
    .collect();
    WeightStore::from_map(map)
}

fn deferred(names: &[(&str, u64)]) -> WeightStore {
    let mut s = WeightStore::empty();
    for (name, offset) in names {
        s.defer(
            (*name).to_string(),
            DeferredTensor {
                path: "ngram.safetensors".into(),
                offset: *offset,
                shape: vec![4, 51],
                dtype: WeightDtype::Int16,
            },
        );
    }
    s
}

#[test]
fn i64_names_prefer_the_exl3_nesting() {
    let lp = "model.layers.0.ple";
    let exl3 = format!("{lp}.ple_embedding.ngram_embedding.layer_multipliers");
    let store = uploaded(&exl3);
    assert_eq!(
        ple_i64_name(&store, lp, "layer_multipliers", "layer_multipliers"),
        exl3
    );
}

#[test]
fn i64_names_fall_back_to_the_nvfp4_names() {
    let lp = "model.layers.0.ple";
    let old = format!("{lp}.ple_embedding.ngram_heads_vocab_sizes");
    let store = uploaded(&old);
    assert_eq!(
        ple_i64_name(&store, lp, "head_vocab_sizes", "ngram_heads_vocab_sizes"),
        old
    );
    assert_eq!(
        ple_i64_name(&store, lp, "head_offsets", "ngram_heads_offsets"),
        format!("{lp}.ple_embedding.ngram_embedding.head_offsets")
    );
}

#[test]
fn a_single_trellis_opens_at_its_own_offset() {
    let lp = "model.layers.0.ple";
    let name = format!("{lp}.ple_embedding.ngram_embedding.trellis");
    let store = deferred(&[(&name, 4096)]);
    let d = ngram_table_open(&store, lp).expect("single trellis");
    assert_eq!(d.offset, 4096);
    assert_eq!(d.shape, vec![4, 51]);
}

#[test]
fn shards_keep_the_segmented_open() {
    let lp = "model.layers.0.ple";
    let shard = format!("{lp}.ple_embedding.ngram_embedding.shard_0.weight");
    let single = format!("{lp}.ple_embedding.ngram_embedding.trellis");
    let store = deferred(&[(&shard, 128), (&single, 4096)]);
    assert!(ngram_table_open(&store, lp).is_none());
}
