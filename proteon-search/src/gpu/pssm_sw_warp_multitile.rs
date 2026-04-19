//! Multitile dispatcher for the warp-collaborative PSSM SW kernel.
//!
//! Covers queries longer than the singletile's 256-row budget. See
//! `pssm_sw_warp_multitile.cu` for the algorithm — same wavefront as
//! singletile, but wrapped in a tile loop that runs the warp over
//! TILE_SIZE=256 rows at a time, passing last-row state between tiles
//! through a ping-pong global-memory scratch buffer.
//!
//! Eligibility: `query_len <= MAX_MULTITILE_TILES * TILE_SIZE`. The cap
//! is a pragmatic limit — the singletile covers ≤ 256 residues at
//! maximum speed, multitile handles 257-through-`MAX_QUERY_LEN`
//! residues at slightly lower per-pair throughput (extra global memory
//! traffic for the borders, per-tile PSSM staging). Longer queries
//! than that fall back to the thread-per-pair kernel.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::pssm_sw_warp::{GROUPSIZE, NUM_ITEMS};
use super::sw::GappedScoreEnd;
use super::GpuContext;
use crate::pssm::Pssm;

const KERNEL_SRC: &str = include_str!("pssm_sw_warp_multitile.cu");
const KERNEL_NAME: &str = "pssm_sw_warp_multitile_batch";

pub const TILE_SIZE: usize = GROUPSIZE * NUM_ITEMS; // 256 rows per tile
pub const MAX_MULTITILE_TILES: usize = 8; // up to 2048-residue queries
pub const MAX_QUERY_LEN: usize = TILE_SIZE * MAX_MULTITILE_TILES;

const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_SIZE: u32 = WARPS_PER_BLOCK * 32;

/// Per-tile shared-memory PSSM budget: 21 × 256 × 4 bytes ≈ 21 KB for
/// protein alphabets. We still cap the PSSM's per-tile footprint at
/// the same 46 KB ceiling used elsewhere for consistency.
const SHARED_MEM_LIMIT_BYTES: usize = 46 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum PssmSwWarpMultitileDispatchError {
    /// Query longer than the multitile cap. Fall back to the thread-
    /// per-pair kernel.
    #[error(
        "query length {query_len} exceeds multitile warp kernel capacity {max}; \
         fall back to pssm_sw (thread-per-pair)"
    )]
    QueryTooLong { query_len: usize, max: usize },
    #[error(
        "per-tile PSSM size {pssm_tile_bytes} bytes exceeds shared-memory budget {limit_bytes}"
    )]
    TilePssmTooLargeForSharedMem {
        pssm_tile_bytes: usize,
        limit_bytes: usize,
    },
    #[error("driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("infrastructure: {0}")]
    Other(#[from] anyhow::Error),
}

pub(crate) struct PssmSwWarpMultitileKernel {
    func: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<PssmSwWarpMultitileKernel>> = OnceLock::new();

impl PssmSwWarpMultitileKernel {
    pub(crate) fn try_global() -> Option<&'static PssmSwWarpMultitileKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!(
                            "[proteon-search-gpu] pssm_sw_warp_multitile kernel compile failed: {e:#}"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    fn compile(ctx: &GpuContext) -> Result<Self> {
        let arch = ctx.arch_flag();
        let opts = CompileOptions {
            arch: Some(Box::leak(arch.into_boxed_str())),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .with_context(|| "NVRTC compile of pssm_sw_warp_multitile.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// Multitile warp-collaborative PSSM Smith-Waterman batch dispatch.
///
/// Same contract as the singletile variant ([`super::pssm_sw_warp::pssm_sw_warp_batch_gpu`]):
/// shared query (via [`Pssm`]) + flat list of targets → one
/// [`GappedScoreEnd`] per target in input order. Per-pair "no
/// alignment" is `GappedScoreEnd { score: 0, .. }`.
///
/// Errors:
/// - [`PssmSwWarpMultitileDispatchError::QueryTooLong`] when `pssm.query_len > MAX_QUERY_LEN`.
/// - [`PssmSwWarpMultitileDispatchError::TilePssmTooLargeForSharedMem`] when
///   a single tile's PSSM footprint exceeds the per-block shared-memory
///   budget (extremely unlikely for protein alphabets).
pub fn pssm_sw_warp_multitile_batch_gpu(
    pssm: &Pssm,
    targets: &[&[u8]],
    gap_open: i32,
    gap_extend: i32,
) -> std::result::Result<Vec<GappedScoreEnd>, PssmSwWarpMultitileDispatchError> {
    if pssm.query_len > MAX_QUERY_LEN {
        return Err(PssmSwWarpMultitileDispatchError::QueryTooLong {
            query_len: pssm.query_len,
            max: MAX_QUERY_LEN,
        });
    }

    let pssm_tile_bytes = TILE_SIZE * pssm.alphabet_size * std::mem::size_of::<i32>();
    if pssm_tile_bytes > SHARED_MEM_LIMIT_BYTES {
        return Err(
            PssmSwWarpMultitileDispatchError::TilePssmTooLargeForSharedMem {
                pssm_tile_bytes,
                limit_bytes: SHARED_MEM_LIMIT_BYTES,
            },
        );
    }

    let n_pairs = targets.len();
    if n_pairs == 0 {
        return Ok(Vec::new());
    }

    let n_tiles = pssm.query_len.div_ceil(TILE_SIZE);

    let kernel = PssmSwWarpMultitileKernel::try_global()
        .ok_or_else(|| anyhow!("GPU pssm_sw_warp_multitile kernel unavailable"))?;
    let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Flatten targets + per-pair offset/length.
    let mut target_offsets: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut target_lens: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut max_target_len: usize = 0;
    let total_bytes: usize = targets.iter().map(|t| t.len()).sum();
    let mut targets_flat: Vec<u8> = Vec::with_capacity(total_bytes.max(1));
    for t in targets {
        target_offsets.push(targets_flat.len() as i32);
        target_lens.push(t.len() as i32);
        if t.len() > max_target_len {
            max_target_len = t.len();
        }
        targets_flat.extend_from_slice(t);
    }
    if targets_flat.is_empty() {
        targets_flat.push(0);
    }
    // All-empty-targets shortcut: no border traffic to worry about,
    // but max_target_len == 0 causes a zero-sized scratch alloc. Bump
    // to 1 so the allocation succeeds; the kernel writes no borders
    // because every pair has target_len=0.
    let scratch_stride_cols = max_target_len.max(1);

    let pssm_for_gpu: &[i32] = if pssm.data.is_empty() {
        &[0i32]
    } else {
        &pssm.data
    };
    let d_pssm = stream.clone_htod(pssm_for_gpu)?;
    let d_targets = stream.clone_htod(&targets_flat)?;
    let d_offsets = stream.clone_htod(&target_offsets)?;
    let d_lens = stream.clone_htod(&target_lens)?;

    // Ping-pong border scratch: 2 × n_pairs × 3 × scratch_stride_cols i32.
    // For n_pairs=1000, scratch_stride_cols=2000: 48 MB — comfortable.
    // For n_pairs=10000, scratch_stride_cols=2000: 480 MB — use smaller
    // batches upstream if this is a concern.
    let scratch_len = 2 * n_pairs * 3 * scratch_stride_cols;
    let mut d_scratch: cudarc::driver::CudaSlice<i32> =
        stream.alloc_zeros::<i32>(scratch_len.max(1))?;

    let mut d_score = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qend = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tend = stream.alloc_zeros::<i32>(n_pairs)?;

    let n_blocks = (n_pairs as u32).div_ceil(WARPS_PER_BLOCK);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: pssm_tile_bytes as u32,
    };
    {
        let mut a = stream.launch_builder(&kernel.func);
        let q_len_i = pssm.query_len as i32;
        let alph_i = pssm.alphabet_size as i32;
        let n_pairs_i = n_pairs as i32;
        let n_tiles_i = n_tiles as i32;
        let max_target_len_i = scratch_stride_cols as i32;
        a.arg(&d_pssm);
        a.arg(&q_len_i);
        a.arg(&alph_i);
        a.arg(&d_targets);
        a.arg(&d_offsets);
        a.arg(&d_lens);
        a.arg(&gap_open);
        a.arg(&gap_extend);
        a.arg(&n_pairs_i);
        a.arg(&n_tiles_i);
        a.arg(&max_target_len_i);
        a.arg(&mut d_scratch);
        a.arg(&mut d_score);
        a.arg(&mut d_qend);
        a.arg(&mut d_tend);
        unsafe { a.launch(cfg)? };
    }
    stream.synchronize()?;

    let scores_out = stream.clone_dtoh(&d_score)?;
    let qend_out = stream.clone_dtoh(&d_qend)?;
    let tend_out = stream.clone_dtoh(&d_tend)?;

    let mut out: Vec<GappedScoreEnd> = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        out.push(GappedScoreEnd {
            score: scores_out[i],
            query_end: qend_out[i] as usize,
            target_end: tend_out[i] as usize,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gapped::smith_waterman;

    fn identity_matrix(alphabet_size: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            s[i * alphabet_size + i] = m;
        }
        s
    }

    fn prng() -> impl FnMut() -> u32 {
        let mut rng: u32 = 0xABCD_9876;
        move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        }
    }

    /// Two-tile query (query_len=300) against hand-crafted + PRNG
    /// targets; multitile must match CPU bit-for-bit on scores and
    /// endpoints. This is the "did the border hand-off work" test.
    #[test]
    fn multitile_parity_two_tiles() {
        if super::PssmSwWarpMultitileKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let mut next = prng();
        let query: Vec<u8> = (0..300)
            .map(|_| (next() % alphabet_size as u32) as u8)
            .collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let targets_owned: Vec<Vec<u8>> = (0..64)
            .map(|_| {
                let len = 50 + (next() as usize % 250);
                (0..len)
                    .map(|_| (next() % alphabet_size as u32) as u8)
                    .collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let gpu_results = pssm_sw_warp_multitile_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("multitile dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu_results[i].score, cpu_score,
                "target {i} (2-tile): GPU {} != CPU {cpu_score}",
                gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu_results[i].query_end, cpu_aln.query_end, "qend i={i}");
                assert_eq!(gpu_results[i].target_end, cpu_aln.target_end, "tend i={i}");
            }
        }
    }

    /// Four-tile query (query_len=900): exercises 3 internal border
    /// hand-offs. The multitile code-path runs this entirely in GPU
    /// since no CPU can see the intermediate buffers.
    #[test]
    fn multitile_parity_four_tiles() {
        if super::PssmSwWarpMultitileKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let mut next = prng();
        let query: Vec<u8> = (0..900)
            .map(|_| (next() % alphabet_size as u32) as u8)
            .collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let targets_owned: Vec<Vec<u8>> = (0..32)
            .map(|_| {
                let len = 100 + (next() as usize % 800);
                (0..len)
                    .map(|_| (next() % alphabet_size as u32) as u8)
                    .collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let gpu_results = pssm_sw_warp_multitile_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("multitile dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu_results[i].score, cpu_score,
                "target {i} (4-tile): GPU {} != CPU {cpu_score}",
                gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu_results[i].query_end, cpu_aln.query_end, "qend i={i}");
                assert_eq!(gpu_results[i].target_end, cpu_aln.target_end, "tend i={i}");
            }
        }
    }

    /// Singletile sanity: multitile with query_len ≤ 256 must still
    /// produce correct results (it's the n_tiles=1 degenerate case,
    /// same as the singletile kernel). Uses a protein alphabet + BLOSUM
    /// flavor of scoring to add surface area beyond the 4-letter synth.
    #[test]
    fn multitile_parity_single_tile_degenerate_case() {
        if super::PssmSwWarpMultitileKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let t_full: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let t_partial: Vec<u8> = vec![0, 1, 2, 3];
        let t_longer: Vec<u8> = vec![3, 3, 0, 1, 2, 3, 0, 1, 3, 3];
        let t_unrelated: Vec<u8> = vec![3, 3, 3, 3, 3];
        let targets_owned = vec![t_full, t_partial, t_longer, t_unrelated];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let gpu_results = pssm_sw_warp_multitile_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("multitile dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu_results[i].score, cpu_score,
                "target {i} (1-tile degenerate): GPU {} != CPU {cpu_score}",
                gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu_results[i].query_end, cpu_aln.query_end, "qend i={i}");
                assert_eq!(gpu_results[i].target_end, cpu_aln.target_end, "tend i={i}");
            }
        }
    }

    /// Partial-final-tile: query_len not a multiple of TILE_SIZE. The
    /// kernel must handle the case where the final tile only partly
    /// covers the 256-row slot. query_len=500 → 2 tiles (256 + 244).
    #[test]
    fn multitile_parity_partial_final_tile() {
        if super::PssmSwWarpMultitileKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let mut next = prng();
        let query: Vec<u8> = (0..500)
            .map(|_| (next() % alphabet_size as u32) as u8)
            .collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let targets_owned: Vec<Vec<u8>> = (0..16)
            .map(|_| {
                let len = 50 + (next() as usize % 400);
                (0..len)
                    .map(|_| (next() % alphabet_size as u32) as u8)
                    .collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let gpu_results = pssm_sw_warp_multitile_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("multitile dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu_results[i].score, cpu_score,
                "target {i} (partial final): GPU {} != CPU {cpu_score}",
                gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu_results[i].query_end, cpu_aln.query_end, "qend i={i}");
                assert_eq!(gpu_results[i].target_end, cpu_aln.target_end, "tend i={i}");
            }
        }
    }

    #[test]
    fn empty_target_list_returns_empty_vec_without_dispatch() {
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..300).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let result = pssm_sw_warp_multitile_batch_gpu(&pssm, &[], -5, -1).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn query_too_long_returns_structured_error() {
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..(MAX_QUERY_LEN + 1))
            .map(|i| (i % alphabet_size) as u8)
            .collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let t: Vec<u8> = vec![0; 10];
        let targets: Vec<&[u8]> = vec![&t];
        match pssm_sw_warp_multitile_batch_gpu(&pssm, &targets, -5, -1) {
            Err(PssmSwWarpMultitileDispatchError::QueryTooLong { query_len, max }) => {
                assert_eq!(query_len, MAX_QUERY_LEN + 1);
                assert_eq!(max, MAX_QUERY_LEN);
            }
            other => panic!("expected QueryTooLong, got {other:?}"),
        }
    }
}
