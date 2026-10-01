// SPDX-License-Identifier: AGPL-3.0-only
// exllamav3's int8-activation "sq" GEMV for EXL3 mul1 weights (vendored in exl3_vendor/, see NOTICE.md): one
// regular launch, block 256, dynamic shared memory and grid chosen by the host (M6b); `locks` is a zeroed
// workspace whose counters reset themselves. Entry points: bits K in {4,5,6} x rows M in {1,2}, fp32 output.
#include <cuda_bf16.h>
#include "exl3_vendor/exl3_gemv_int8_kernel.cuh"

#define EXL3_SQ_ENTRY(K, M)                                                                                     \
    extern "C" __global__ __launch_bounds__(256) void exl3_int8_sq_k##K##_m##M(                                 \
        const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C, const int size_m,    \
        const int size_k, const int size_n, int* __restrict__ locks, const half* __restrict__ suh,             \
        half* __restrict__ A_had, const half* __restrict__ svh)                                                 \
    { exl3_gemv_int8_sq_kernel<K, M, true, false>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); }

EXL3_SQ_ENTRY(4, 1) EXL3_SQ_ENTRY(4, 2) EXL3_SQ_ENTRY(5, 1) EXL3_SQ_ENTRY(5, 2) EXL3_SQ_ENTRY(6, 1) EXL3_SQ_ENTRY(6, 2)

// B3: same GEMV with the bf16 -> f16 input conversion and the f32 -> bf16 row output folded in
// (A is bf16 [m, k]; C is bf16 rows `out_stride` elements apart). Same grid / decomposition / math.
#define EXL3_SQ_ENTRY_BF16(K, M)                                                                                \
    extern "C" __global__ __launch_bounds__(256) void exl3_int8_sq_k##K##_m##M##_bf16(                          \
        const __nv_bfloat16* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,             \
        const int size_m, const int size_k, const int size_n, int* __restrict__ locks,                         \
        const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh,                  \
        const int out_stride)                                                                                   \
    { exl3_gemv_int8_sq_kernel<K, M, true, false, false, true>((const half*) A, B, C, size_m, size_k, size_n,   \
                                                               locks, suh, A_had, svh, out_stride); }

EXL3_SQ_ENTRY_BF16(4, 1) EXL3_SQ_ENTRY_BF16(4, 2) EXL3_SQ_ENTRY_BF16(5, 1) EXL3_SQ_ENTRY_BF16(5, 2)
EXL3_SQ_ENTRY_BF16(6, 1) EXL3_SQ_ENTRY_BF16(6, 2)

// R2 vocab-l2 item B: the same entries with an L2 cache-policy hint on the trellis weight loads
// (_l2f = evict_first, _l2l = evict_last). Host picks via ATLAS_SQ_L2HINT; identical grid and math.
#define EXL3_SQ_ENTRY_L2(K, M, SUF, P)                                                                          \
    extern "C" __global__ __launch_bounds__(256) void exl3_int8_sq_k##K##_m##M##SUF(                            \
        const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C, const int size_m,    \
        const int size_k, const int size_n, int* __restrict__ locks, const half* __restrict__ suh,             \
        half* __restrict__ A_had, const half* __restrict__ svh)                                                 \
    { exl3_gemv_int8_sq_kernel<K, M, true, false, false, false, P>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); }
#define EXL3_SQ_ENTRY_BF16_L2(K, M, SUF, P)                                                                     \
    extern "C" __global__ __launch_bounds__(256) void exl3_int8_sq_k##K##_m##M##_bf16##SUF(                     \
        const __nv_bfloat16* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,             \
        const int size_m, const int size_k, const int size_n, int* __restrict__ locks,                         \
        const half* __restrict__ suh, half* __restrict__ A_had, const half* __restrict__ svh,                  \
        const int out_stride)                                                                                   \
    { exl3_gemv_int8_sq_kernel<K, M, true, false, false, true, P>((const half*) A, B, C, size_m, size_k, size_n,\
                                                                  locks, suh, A_had, svh, out_stride); }
#define EXL3_SQ_L2_ALL(SUF, P) \
    EXL3_SQ_ENTRY_L2(4, 1, SUF, P) EXL3_SQ_ENTRY_L2(4, 2, SUF, P) EXL3_SQ_ENTRY_L2(5, 1, SUF, P) EXL3_SQ_ENTRY_L2(5, 2, SUF, P) \
    EXL3_SQ_ENTRY_BF16_L2(4, 1, SUF, P) EXL3_SQ_ENTRY_BF16_L2(4, 2, SUF, P) EXL3_SQ_ENTRY_BF16_L2(5, 1, SUF, P) EXL3_SQ_ENTRY_BF16_L2(5, 2, SUF, P)
EXL3_SQ_L2_ALL(_l2f, 1)
EXL3_SQ_L2_ALL(_l2l, 2)

extern "C" __global__ void exl3_f32_to_bf16(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = __float2bfloat16_rn(in[i]);
}

// f32 [rows, cols] -> bf16 rows at `out_stride` elements apart (writes one projection's rows into a wider
// [rows, out_stride] buffer).
extern "C" __global__ void exl3_f32_to_bf16_rows(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                                 unsigned rows, unsigned cols, unsigned out_stride)
{
    unsigned n = rows * cols, stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        // Padded rows (cols > out_stride, e.g. the lm_head: 248320-wide rows at a 248077 stride) would
        // overlap the next row: drop the pad columns so rows never race or overrun.
        unsigned c = i % cols;
        if (c < out_stride) out[(i / cols) * out_stride + c] = __float2bfloat16_rn(in[i]);
    }
}
