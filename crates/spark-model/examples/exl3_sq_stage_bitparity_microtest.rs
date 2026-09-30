// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the ITEM-2 merged-phase sq staging (round 2, 2026-09-30).
//!
//! `gemv_int8_stage_slice` used to run its five barrier phases once per
//! activation row, serially (5*M+1 barriers). ITEM-2 runs each phase for every
//! row between the SAME pair of `__syncthreads` (6 barriers), which is only
//! legitimate if it does not move a single output bit: every row's quantization
//! scale and quantized values depend on that row's activations alone, so
//! merging the *scheduling* of two independent phases must be a no-op
//! numerically.
//!
//! This is the sq analogue of `w4a16_batch_bitparity_microtest.rs`, and it
//! exists because the repo's own `exl3_int8_gpu_tests` cannot be run: the
//! `spark-model` lib TEST target does not compile on this branch (a pre-existing
//! PLE/ngram signature drift, present at HEAD and unrelated to this change), so
//! `int8_row_invariance_m1_vs_m2` is unreachable via `cargo test`. An example
//! builds against the lib proper and reaches the same kernels.
//!
//! Three legs, all on RAW FP32 OUTPUT BITS (not a cosine — a cosine is exactly
//! what hid the w8a16 fused-add defect per that test's header):
//!   A. m=2 row r vs an independent m=1 call on the same row. This is the
//!      repo's own `int8_row_invariance_m1_vs_m2` check, widened to BOTH rows.
//!   B. determinism: the same m=2 call twice must be bit-identical.
//!   C. the `_rows` (bf16-out, `out_stride`) entry at the lm_head shape, where
//!      `out_stride` (248077) < padded n (248320) — the padding guard from
//!      83964a58 must still hold with the merged staging.
//!
//! Shapes are this checkpoint's real decode classes, 5-bit included (the live
//! Bree build is 3.05bpw → k5), plus the lm_head.
//!
//! Exit: 0 every leg byte-identical, 1 any leg differs, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
//!   ATLAS_TARGET_QUANT=exl3 cargo run --release -p spark-model \
//!     --example exl3_sq_stage_bitparity_microtest

use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::ops::{
    Exl3Int8Kernels, Exl3Int8Workspace, Exl3Kernels, exl3_int8_linear_bf16,
    exl3_int8_linear_bf16_rows, sq_grid, sq_plan,
};
use spark_model::weight_map::exl3::{Exl3Shape, Exl3Weight};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// (k, n, bits) — this checkpoint's decode classes. 248320 is the lm_head's
/// 256-padded n (real out_features 248077).
const CASES: [(usize, usize, u32); 8] = [
    (2560, 16384, 5),
    (2560, 12288, 5),
    (2560, 6144, 5),
    (2560, 2560, 5),
    (2560, 512, 5),
    (6144, 2560, 5),
    (2560, 2560, 6),
    (2560, 2560, 4),
];

const LM_HEAD: (usize, usize, u32) = (2560, 248320, 5);
const LM_HEAD_STRIDE: usize = 248077;

fn mix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(seed: &mut u64) -> f32 {
    (mix64(seed) >> 40) as f32 / 16_777_216.0
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn zeros(g: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    upload(g, &vec![0u8; bytes])
}

fn dl_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

struct Case {
    shape: Exl3Shape,
    dev: Exl3Weight,
}

fn random_case(g: &dyn GpuBackend, k: usize, n: usize, bits: u32) -> Result<Case> {
    let mut seed = 0xE31A_0000_0000_u64 ^ ((k * 1_000_003 + n * 7 + bits as usize) as u64);
    let words = (k / 16) * (n / 16) * 16 * bits as usize;
    let trellis: Vec<u16> = (0..words).map(|_| mix64(&mut seed) as u16).collect();
    let mut scale = |len: usize| -> Vec<f16> {
        (0..len)
            .map(|_| {
                let mag = 0.5 + unit(&mut seed);
                let sign = if mix64(&mut seed) & 1 == 0 { 1.0 } else { -1.0 };
                f16::from_f32(sign * mag)
            })
            .collect()
    };
    let suh = scale(k);
    let svh = scale(n);
    let shape = Exl3Shape {
        in_features: k,
        out_features: n,
        bits,
    };
    let b16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let bf = |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let dev = Exl3Weight {
        trellis: upload(g, &b16(&trellis))?,
        suh: upload(g, &bf(&suh))?,
        svh: upload(g, &bf(&svh))?,
        shape,
    };
    Ok(Case { shape, dev })
}

fn activations(k: usize, rows: usize, salt: u64) -> Vec<u16> {
    let mut seed = 0x0AC7_0000_0001_u64 ^ salt ^ (k as u64);
    (0..rows * k)
        .map(|_| bf16::from_f32(unit(&mut seed) * 2.0 - 1.0).to_bits())
        .collect()
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    k8: Exl3Int8Kernels,
    k: Exl3Kernels,
    ws: Exl3Int8Workspace,
    grid: u32,
}

impl<'a> Rig<'a> {
    fn new(g: &'a dyn GpuBackend) -> Result<Self> {
        let grid = sq_grid(g.sm_count()?);
        let ws_ints = CASES
            .iter()
            .chain(std::iter::once(&LM_HEAD))
            .map(|&(k, n, bits)| sq_plan(k, n, 2, bits, grid).unwrap().ws_ints)
            .max()
            .unwrap();
        Ok(Self {
            g,
            k8: Exl3Int8Kernels::resolve(g)?,
            k: Exl3Kernels::resolve(g)?,
            ws: Exl3Int8Workspace::new(g, ws_ints)?,
            grid,
        })
    }

    /// bf16 `[m, n]` output bits of the bf16 linear (the fused path's real output).
    fn run(&self, c: &Case, x_bits: &[u16], m: u32) -> Result<Vec<u16>> {
        let (k, n) = (c.shape.in_features, c.shape.out_features);
        let g = self.g;
        let x = upload(
            g,
            &x_bits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )?;
        let x_f16 = zeros(g, m as usize * k * 2)?;
        let a_had = zeros(g, m as usize * k * 2)?;
        let c_f32 = zeros(g, m as usize * n * 4)?;
        let out = zeros(g, m as usize * n * 2)?;
        let stream = g.default_stream();
        exl3_int8_linear_bf16(
            g, &self.k8, &self.k, &self.ws, x, m, &c.dev, x_f16, a_had, c_f32, out, self.grid,
            stream,
        )?;
        g.synchronize(stream)?;
        // IMPORTANT: read out_bf16, not c_f32. Since B3 (a5b98525) folded the
        // f32 -> bf16 epilogue into the sq kernel, the fused *_bf16 entries pass
        // c_f32 as the unused A_had scratch and write the result to out_bf16.
        // The repo's own int8_row_invariance_m1_vs_m2 still reads c_f32, so on
        // this branch it compares all-zero buffers and cannot fail (see report).
        let got = dl_u16(g, out, m as usize * n)?;
        for p in [x, x_f16, a_had, c_f32, out] {
            g.free(p)?;
        }
        Ok(got)
    }

    /// bf16 `[m, out_stride]` output bits of the folded `_rows` entry.
    fn run_rows(&self, c: &Case, x_bits: &[u16], m: u32, out_stride: usize) -> Result<Vec<u16>> {
        let (k, n) = (c.shape.in_features, c.shape.out_features);
        let g = self.g;
        let x = upload(
            g,
            &x_bits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        )?;
        let x_f16 = zeros(g, m as usize * k * 2)?;
        let a_had = zeros(g, m as usize * k * 2)?;
        let c_f32 = zeros(g, m as usize * n * 4)?;
        // Pre-fill the destination with a sentinel so an overrun past out_stride
        // into the next row is visible, exactly as the 83964a58 guard intends.
        let dest_elems = m as usize * out_stride;
        let sentinel = vec![0xAAu8; dest_elems * 2];
        let out = upload(g, &sentinel)?;
        let stream = g.default_stream();
        exl3_int8_linear_bf16_rows(
            g,
            &self.k8,
            &self.k,
            &self.ws,
            x,
            m,
            &c.dev,
            x_f16,
            a_had,
            c_f32,
            out,
            out_stride,
            self.grid,
            stream,
        )?;
        g.synchronize(stream)?;
        let got = dl_u16(g, out, dest_elems)?;
        for p in [x, x_f16, a_had, c_f32, out] {
            g.free(p)?;
        }
        Ok(got)
    }
}

fn digest_u16(v: &[u16]) -> u64 {
    let mut h: u64 = 0x13198A2E_03707344 ^ (v.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for (i, w) in v.iter().enumerate() {
        let mut z = (*w as u64) ^ (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 29)).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= z ^ (z >> 32);
        h = h.rotate_left(17).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    }
    h ^ (h >> 33)
}

fn main() -> Result<()> {
    let modules = match atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "exl3") {
        Some(m) => m.modules,
        None => {
            eprintln!("SKIP: no exl3 PTX for this target");
            std::process::exit(2);
        }
    };
    let g = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &modules)?;
    let rig = Rig::new(&g)?;
    eprintln!("grid = {} (sq_grid, frozen)", rig.grid);

    let mut fails = 0usize;

    // ---- Leg A: m=2 rows vs independent m=1 calls, BOTH rows ----
    eprintln!("\n== leg A: m=2 row r == m=1 on that row (raw fp32 bits) ==");
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits)?;
        for salt in [37u64, 1009, 77_711] {
            let two = activations(k, 2, salt);
            let y2 = rig.run(&c, &two, 2)?;
            let y1a = rig.run(&c, &two[..k], 1)?;
            let y1b = rig.run(&c, &two[k..], 1)?;
            let r0 = y2[..n] == y1a[..];
            let r1 = y2[n..2 * n] == y1b[..];
            let bad0 = (0..n).filter(|&i| y2[i] != y1a[i]).count();
            let bad1 = (0..n).filter(|&i| y2[n + i] != y1b[i]).count();
            // Guard against the vacuous comparison an all-zero buffer would give.
            let live = y2.iter().any(|&w| w != 0);
            if !(r0 && r1 && live) {
                fails += 1;
            }
            if !live {
                eprintln!("    !! output is ALL ZERO - comparison is vacuous");
            }
            eprintln!(
                "  k={k} n={n} b={bits} salt={salt}: row0 {} ({bad0} words differ), row1 {} ({bad1} words differ)",
                if r0 { "OK" } else { "FAIL" },
                if r1 { "OK" } else { "FAIL" }
            );
        }
    }

    // ---- Leg B0: cross-BUILD digest. Leg A only proves m=2 is self-consistent
    // with m=1 within one build; this digest is what proves the merged staging
    // equals the SERIAL staging bit for bit. Compare across two builds.
    eprintln!("\n== leg B0: cross-build digests of the m=2 output ==");
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits)?;
        let two = activations(k, 2, 20260930);
        let y = rig.run(&c, &two, 2)?;
        let h = digest_u16(&y);
        let nz = y.iter().filter(|&&w| w != 0).count();
        let first: Vec<String> = y.iter().take(4).map(|w| format!("{:04x}", w)).collect();
        eprintln!(
            "  DIGEST k={k} n={n} b={bits} m=2: {h:016x} ({} words, {nz} nonzero, head {})",
            y.len(), first.join(",")
        );
    }
    {
        let (k, n, bits) = LM_HEAD;
        let c = random_case(&g, k, n, bits)?;
        let two = activations(k, 2, 20260930);
        let y = rig.run_rows(&c, &two, 2, LM_HEAD_STRIDE)?;
        let h = digest_u16(&y);
        eprintln!("  DIGEST lm_head rows m=2: {h:016x} ({} words)", y.len());
    }

    // ---- Leg B: determinism of the m=2 call ----
    eprintln!("\n== leg B: m=2 determinism (two identical calls) ==");
    for &(k, n, bits) in &CASES {
        let c = random_case(&g, k, n, bits)?;
        let two = activations(k, 2, 4242);
        let a = rig.run(&c, &two, 2)?;
        let b = rig.run(&c, &two, 2)?;
        let ok = a == b && a.iter().any(|&w| w != 0);
        if !ok {
            fails += 1;
        }
        eprintln!("  k={k} n={n} b={bits}: {}", if ok { "OK" } else { "FAIL" });
    }

    // ---- Leg C: lm_head `_rows` entry, out_stride < padded n ----
    eprintln!("\n== leg C: lm_head _rows (out_stride {LM_HEAD_STRIDE} < n {}) ==", LM_HEAD.1);
    {
        let (k, n, bits) = LM_HEAD;
        let c = random_case(&g, k, n, bits)?;
        for salt in [37u64, 1009] {
            let two = activations(k, 2, salt);
            let y2 = rig.run_rows(&c, &two, 2, LM_HEAD_STRIDE)?;
            let y1a = rig.run_rows(&c, &two[..k], 1, LM_HEAD_STRIDE)?;
            let y1b = rig.run_rows(&c, &two[k..], 1, LM_HEAD_STRIDE)?;
            let r0 = y2[..LM_HEAD_STRIDE] == y1a[..];
            let r1 = y2[LM_HEAD_STRIDE..] == y1b[..];
            // The guard's specific failure mode: row 0's tail columns past
            // out_stride bleeding into row 1's first columns.
            let sentinels = y2.iter().filter(|&&v| v == 0xAAAA).count();
            if !(r0 && r1) {
                fails += 1;
            }
            eprintln!(
                "  salt={salt}: row0 {}, row1 {}, untouched sentinels {sentinels}",
                if r0 { "OK" } else { "FAIL" },
                if r1 { "OK" } else { "FAIL" }
            );
        }
    }

    if fails == 0 {
        eprintln!("\nALL LEGS BYTE-IDENTICAL");
        Ok(())
    } else {
        eprintln!("\n{fails} LEG(S) DIFFER");
        std::process::exit(1);
    }
}
