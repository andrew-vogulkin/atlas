// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE — K=2 batch variant. Same semantics as
// moe_shared_expert_fused_batch2 but reads weight in `[K/2, N]` layout
// (input-major, prefill-coalesced) instead of `[N, K/2]`. See
// moe_shared_expert_fused_t.cu for the layout rationale.
//
// blockIdx.y: 0..2*top_k-1 routed (token = y/top_k, slot = y%top_k);
//             2*top_k..2*top_k+1 shared (token = y - 2*top_k).
// For gate_up: blockIdx.z = proj (0=gate, 1=up). silu_down has no z.
// Block: (128); each thread owns one output position `n`; lanes adjacent.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH2_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; NVIDIA path is the verbatim cast.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

extern "C" __global__ void moe_expert_gate_up_shared_batch2_t(
    const __nv_bfloat16* __restrict__ A,                    // [2, K]
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,                   // [2*top_k, N]
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,                     // [2*top_k, N]
    const unsigned int* __restrict__ expert_indices,        // [2*top_k]
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,                // [2, N]
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,                  // [2, N]
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned long long c_offset = 0;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2;
            C = sh_gate_out; c_offset = (unsigned long long)token * N;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed; B_scale = sh_up_t_scale; s2 = sh_up_s2;
            C = sh_up_out; c_offset = (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        float sc = atlas_dec_e4m3(sb) * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float a_lo = __bfloat162float(A_token[k_half * 2]);
            float a_hi = __bfloat162float(A_token[k_half * 2 + 1]);
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += a_lo * w_lo + a_hi * w_hi;
        }
    }
    C[c_offset + n] = __float2bfloat16(acc);
}

extern "C" __global__ void moe_expert_silu_down_shared_batch2_t(
    const __nv_bfloat16* __restrict__ gate_out,             // [2*top_k, K]
    const __nv_bfloat16* __restrict__ up_out,               // [2*top_k, K]
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                          // [2*top_k, N]
    const unsigned int* __restrict__ expert_indices,        // [2*top_k]
    const __nv_bfloat16* __restrict__ sh_gate_in,           // [2, K]
    const __nv_bfloat16* __restrict__ sh_up_in,             // [2, K]
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,                // [2, N]
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    unsigned long long c_offset;

    if (is_shared) {
        if (sh_down_t_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        c_offset = (unsigned long long)token * N;  // sh_down_out
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        float sc = atlas_dec_e4m3(sb) * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += s_act[k_half * 2] * w_lo + s_act[k_half * 2 + 1] * w_hi;
        }
    }

    if (is_shared) {
        sh_down_out[c_offset + n] = __float2bfloat16(acc);
    } else {
        C[c_offset + n] = __float2bfloat16(acc);
    }
}

// ── VEC4 variants (speed pass 2026-09-29) ─────────────────────────────────
// Same math, same per-column accumulation order as the kernels above, but
// each thread owns FOUR adjacent output columns and reads one uchar4 per
// k-row instead of one byte: a warp request moves 128 B instead of 32 B.
// Requires N % 4 == 0 (host checks N % 128 == 0). Grid.x = N / 128.
#define V4_COLS (BLOCK_SIZE * 4)

__device__ __forceinline__ void v4_store_zero(__nv_bfloat16* C, unsigned long long off, unsigned int n0, unsigned int N) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) if (n0 + j < N) C[off + n0 + j] = __float2bfloat16(0.0f);
}

extern "C" __global__ void moe_expert_gate_up_shared_batch2_t_v4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) { token = y - total_routed; expert_slot = 0; }
    else { token = y / top_k; expert_slot = y % top_k; }
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;
    const unsigned int n0 = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned long long c_offset = 0;
    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) { v4_store_zero(sh_gate_out, (unsigned long long)token * N, n0, N); return; }
            B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2;
            C = sh_gate_out; c_offset = (unsigned long long)token * N;
        } else {
            if (sh_up_t_packed == 0) { v4_store_zero(sh_up_out, (unsigned long long)token * N, n0, N); return; }
            B_packed = sh_up_t_packed; B_scale = sh_up_t_scale; s2 = sh_up_s2;
            C = sh_up_out; c_offset = (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) { v4_store_zero(C, c_offset, n0, N); return; }
    }

    // Activation row staged once as float (exact bf16->f32), 4*K bytes of
    // dynamic shared memory (host sets it).
    extern __shared__ float s_a[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) s_a[i] = __bfloat162float(A_token[i]);
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (n0 >= N) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    // Groups processed GD at a time with all GD*8 uchar4 loads issued up
    // front (more loads in flight per warp); the per-column FMA order is
    // unchanged: group sg, then kh_off 0..7, exactly as before.
    constexpr unsigned int GD = 4;
    unsigned int sg = 0;
    for (; sg + GD <= num_groups; sg += GD) {
        uchar4 sbv[GD];
        uchar4 bv[GD * 8];
        #pragma unroll
        for (unsigned int g = 0; g < GD; ++g) {
            sbv[g] = *(const uchar4*)(B_scale + (unsigned long long)(sg + g) * N + n0);
            #pragma unroll
            for (unsigned int kh_off = 0; kh_off < 8; kh_off++)
                bv[g * 8 + kh_off] = *(const uchar4*)(B_packed + (unsigned long long)((sg + g) * 8 + kh_off) * N + n0);
        }
        #pragma unroll
        for (unsigned int g = 0; g < GD; ++g) {
            const float sc0 = atlas_dec_e4m3(sbv[g].x) * s2;
            const float sc1 = atlas_dec_e4m3(sbv[g].y) * s2;
            const float sc2 = atlas_dec_e4m3(sbv[g].z) * s2;
            const float sc3 = atlas_dec_e4m3(sbv[g].w) * s2;
            #pragma unroll
            for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
                const unsigned int k_half = (sg + g) * 8 + kh_off;
                const float a_lo = s_a[k_half * 2];
                const float a_hi = s_a[k_half * 2 + 1];
                const uchar4 b = bv[g * 8 + kh_off];
                acc0 += a_lo * (s_lut[b.x & 0xFu] * sc0) + a_hi * (s_lut[(b.x >> 4) & 0xFu] * sc0);
                acc1 += a_lo * (s_lut[b.y & 0xFu] * sc1) + a_hi * (s_lut[(b.y >> 4) & 0xFu] * sc1);
                acc2 += a_lo * (s_lut[b.z & 0xFu] * sc2) + a_hi * (s_lut[(b.z >> 4) & 0xFu] * sc2);
                acc3 += a_lo * (s_lut[b.w & 0xFu] * sc3) + a_hi * (s_lut[(b.w >> 4) & 0xFu] * sc3);
            }
        }
    }
    for (; sg < num_groups; sg++) {
        const uchar4 sb = *(const uchar4*)(B_scale + (unsigned long long)sg * N + n0);
        const float sc0 = atlas_dec_e4m3(sb.x) * s2;
        const float sc1 = atlas_dec_e4m3(sb.y) * s2;
        const float sc2 = atlas_dec_e4m3(sb.z) * s2;
        const float sc3 = atlas_dec_e4m3(sb.w) * s2;
        const unsigned int kh_base = sg * 8;
        uchar4 bv[8];
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++)
            bv[kh_off] = *(const uchar4*)(B_packed + (unsigned long long)(kh_base + kh_off) * N + n0);
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            const unsigned int k_half = kh_base + kh_off;
            const float a_lo = s_a[k_half * 2];
            const float a_hi = s_a[k_half * 2 + 1];
            const uchar4 b = bv[kh_off];
            acc0 += a_lo * (s_lut[b.x & 0xFu] * sc0) + a_hi * (s_lut[(b.x >> 4) & 0xFu] * sc0);
            acc1 += a_lo * (s_lut[b.y & 0xFu] * sc1) + a_hi * (s_lut[(b.y >> 4) & 0xFu] * sc1);
            acc2 += a_lo * (s_lut[b.z & 0xFu] * sc2) + a_hi * (s_lut[(b.z >> 4) & 0xFu] * sc2);
            acc3 += a_lo * (s_lut[b.w & 0xFu] * sc3) + a_hi * (s_lut[(b.w >> 4) & 0xFu] * sc3);
        }
    }
    C[c_offset + n0 + 0] = __float2bfloat16(acc0);
    C[c_offset + n0 + 1] = __float2bfloat16(acc1);
    C[c_offset + n0 + 2] = __float2bfloat16(acc2);
    C[c_offset + n0 + 3] = __float2bfloat16(acc3);
}

extern "C" __global__ void moe_expert_silu_down_shared_batch2_t_v4(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) { token = y - total_routed; expert_slot = 0; }
    else { token = y / top_k; expert_slot = y % top_k; }
    const unsigned int n0 = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    __nv_bfloat16* out;
    unsigned long long c_offset;
    if (is_shared) {
        if (sh_down_t_packed == 0) { v4_store_zero(sh_down_out, (unsigned long long)token * N, n0, N); return; }
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        c_offset = (unsigned long long)token * N;
        out = sh_down_out;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        c_offset = (unsigned long long)flat_slot * N;
        out = C;
        if (B_packed == 0) { v4_store_zero(C, c_offset, n0, N); return; }
    }

    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2_T[threadIdx.x];
    __syncthreads();
    if (n0 >= N) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    // Groups processed GD at a time with all GD*8 uchar4 loads issued up
    // front (more loads in flight per warp); the per-column FMA order is
    // unchanged: group sg, then kh_off 0..7, exactly as before.
    constexpr unsigned int GD = 4;
    unsigned int sg = 0;
    for (; sg + GD <= num_groups; sg += GD) {
        uchar4 sbv[GD];
        uchar4 bv[GD * 8];
        #pragma unroll
        for (unsigned int g = 0; g < GD; ++g) {
            sbv[g] = *(const uchar4*)(B_scale + (unsigned long long)(sg + g) * N + n0);
            #pragma unroll
            for (unsigned int kh_off = 0; kh_off < 8; kh_off++)
                bv[g * 8 + kh_off] = *(const uchar4*)(B_packed + (unsigned long long)((sg + g) * 8 + kh_off) * N + n0);
        }
        #pragma unroll
        for (unsigned int g = 0; g < GD; ++g) {
            const float sc0 = atlas_dec_e4m3(sbv[g].x) * s2;
            const float sc1 = atlas_dec_e4m3(sbv[g].y) * s2;
            const float sc2 = atlas_dec_e4m3(sbv[g].z) * s2;
            const float sc3 = atlas_dec_e4m3(sbv[g].w) * s2;
            #pragma unroll
            for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
                const unsigned int k_half = (sg + g) * 8 + kh_off;
                const float x_lo = s_act[k_half * 2];
                const float x_hi = s_act[k_half * 2 + 1];
                const uchar4 b = bv[g * 8 + kh_off];
                acc0 += x_lo * (s_lut[b.x & 0xFu] * sc0) + x_hi * (s_lut[(b.x >> 4) & 0xFu] * sc0);
                acc1 += x_lo * (s_lut[b.y & 0xFu] * sc1) + x_hi * (s_lut[(b.y >> 4) & 0xFu] * sc1);
                acc2 += x_lo * (s_lut[b.z & 0xFu] * sc2) + x_hi * (s_lut[(b.z >> 4) & 0xFu] * sc2);
                acc3 += x_lo * (s_lut[b.w & 0xFu] * sc3) + x_hi * (s_lut[(b.w >> 4) & 0xFu] * sc3);
            }
        }
    }
    for (; sg < num_groups; sg++) {
        const uchar4 sb = *(const uchar4*)(B_scale + (unsigned long long)sg * N + n0);
        const float sc0 = atlas_dec_e4m3(sb.x) * s2;
        const float sc1 = atlas_dec_e4m3(sb.y) * s2;
        const float sc2 = atlas_dec_e4m3(sb.z) * s2;
        const float sc3 = atlas_dec_e4m3(sb.w) * s2;
        const unsigned int kh_base = sg * 8;
        uchar4 bv[8];
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++)
            bv[kh_off] = *(const uchar4*)(B_packed + (unsigned long long)(kh_base + kh_off) * N + n0);
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            const unsigned int k_half = kh_base + kh_off;
            const float x_lo = s_act[k_half * 2];
            const float x_hi = s_act[k_half * 2 + 1];
            const uchar4 b = bv[kh_off];
            acc0 += x_lo * (s_lut[b.x & 0xFu] * sc0) + x_hi * (s_lut[(b.x >> 4) & 0xFu] * sc0);
            acc1 += x_lo * (s_lut[b.y & 0xFu] * sc1) + x_hi * (s_lut[(b.y >> 4) & 0xFu] * sc1);
            acc2 += x_lo * (s_lut[b.z & 0xFu] * sc2) + x_hi * (s_lut[(b.z >> 4) & 0xFu] * sc2);
            acc3 += x_lo * (s_lut[b.w & 0xFu] * sc3) + x_hi * (s_lut[(b.w >> 4) & 0xFu] * sc3);
        }
    }
    out[c_offset + n0 + 0] = __float2bfloat16(acc0);
    out[c_offset + n0 + 1] = __float2bfloat16(acc1);
    out[c_offset + n0 + 2] = __float2bfloat16(acc2);
    out[c_offset + n0 + 3] = __float2bfloat16(acc3);
}
