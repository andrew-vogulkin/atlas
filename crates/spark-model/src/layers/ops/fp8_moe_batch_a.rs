// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Speed pass 2026-09-29: the transposed NVFP4 expert GEMVs (single-token and
/// K=2 batch) use the `_v4` kernels (4 columns per thread, uchar4 loads; same
/// per-column math). `ATLAS_MOE_B2_V4=0` restores the byte-per-thread kernels.
pub fn moe_batch2_v4() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_MOE_B2_V4").as_deref() != Ok("0"))
}

/// Handles of the `_v4` expert kernels that were actually loaded. The
/// launchers size grid/smem from the handle they are given, so a caller that
/// passes a non-v4 kernel (e.g. the ARM-2 `_e8m0` variants) keeps the
/// original one-column-per-thread grid regardless of `ATLAS_MOE_B2_V4`.
static V4_KERNELS: std::sync::RwLock<Vec<u64>> = std::sync::RwLock::new(Vec::new());

/// Load `v4` from `module` when `moe_batch2_v4()` is on (and remember the
/// handle as a v4 kernel), else `base`.
pub fn load_moe_expert_kernel(
    gpu: &dyn GpuBackend,
    module: &str,
    base: &str,
    v4: &str,
) -> Result<KernelHandle> {
    if !moe_batch2_v4() {
        return gpu.kernel(module, base);
    }
    let k = gpu.kernel(module, v4)?;
    let mut set = V4_KERNELS.write().unwrap_or_else(|e| e.into_inner());
    if !set.contains(&k.0) {
        set.push(k.0);
    }
    Ok(k)
}

pub(super) fn is_v4_kernel(kernel: KernelHandle) -> bool {
    kernel.0 != 0
        && V4_KERNELS
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&kernel.0)
}

/// Grid x for a transposed expert GEMV launch of `kernel` over `n` columns.
pub(super) fn b2_grid_x(kernel: KernelHandle, n: u32) -> Result<u32> {
    if is_v4_kernel(kernel) {
        anyhow::ensure!(
            n.is_multiple_of(T_BLOCK * 4),
            "moe expert v4: N={n} not a multiple of {}",
            T_BLOCK * 4
        );
        Ok(n / (T_BLOCK * 4))
    } else {
        Ok(div_ceil(n, T_BLOCK))
    }
}

/// Dynamic smem for the gate_up launch: the `_v4` kernel stages the f32
/// activation row (`k` floats); the original kernel uses none.
pub(super) fn gate_up_smem(kernel: KernelHandle, k: u32) -> u32 {
    if is_v4_kernel(kernel) { k * 4 } else { 0 }
}

/// NVFP4 fused gate+up GEMV (transposed). K=2 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch2_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([b2_grid_x(kernel, n)?, 2 * (top_k + 1), 2])
        .block([T_BLOCK, 1, 1])
        .shared_mem(gate_up_smem(kernel, k))
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// NVFP4 fused SiLU+down GEMV (transposed). K=2 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch2_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([b2_grid_x(kernel, n)?, 2 * (top_k + 1), 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// NVFP4 fused gate+up GEMV (transposed). K=3 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch3_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 3 * (top_k + 1), 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// NVFP4 fused SiLU+down GEMV (transposed). K=3 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch3_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 3 * (top_k + 1), 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused gate+up GEMV (transposed weight). Single-token decode.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_t_ptrs: DevicePtr,
    gate_block_scale_t_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_t_ptrs: DevicePtr,
    up_block_scale_t_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up_t: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_t_ptrs)
        .arg_ptr(gate_block_scale_t_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_t_ptrs)
        .arg_ptr(up_block_scale_t_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused SiLU+down GEMV (transposed weight). Single-token decode.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight_t_ptrs: DevicePtr,
    block_scale_t_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight_t_ptrs)
        .arg_ptr(block_scale_t_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
