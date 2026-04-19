//! GPU dispatch for the warp-collaborative PSSM Smith-Waterman kernel.
//!
//! Reference: Kallenborn, Chacon, Hundt, Sirelkhatim, Didi, Cha, Dallago,
//! Mirdita, Schmidt, Steinegger, "GPU-accelerated homology search with
//! MMseqs2", *Nat. Methods* 22, 2024-2027 (2025). This is the
//! warp-collaborative ("libmarv-shape") variant of that work's canonical
//! GPU SW kernel.
//!
//! See `pssm_sw_warp.cu` for the algorithm. This is the phase 4.5a
//! libmarv-shape kernel: one warp per alignment pair, register-tiled
//! query DP, shuffle-based diagonal wavefront. It's the counterpart to
//! `pssm_sw.rs` (which dispatches the thread-per-pair kernel).
//!
//! Singletile configuration: query_len must fit in `GROUPSIZE *
//! NUM_ITEMS = 32 * 8 = 256` rows. Longer queries → [`PssmSwWarpDispatchError::QueryTooLong`],
//! caller falls back to [`super::pssm_sw::pssm_sw_padded_batch_gpu`] or
//! [`super::sw::smith_waterman_score_batch_gpu`].
//!
//! Interface mirrors [`super::sw::smith_waterman_score_batch_gpu`]: a
//! shared query (given as [`Pssm`]) and a flat list of targets. We
//! deliberately skip the `PaddedDb` coalesced-target layout that
//! pssm_sw.cu relies on — the warp-collab kernel reads the target
//! byte-by-byte in the DP loop, and within one warp all 32 lanes read
//! the same byte (broadcast), so coalescing across lanes is moot.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::sw::GappedScoreEnd;
use super::GpuContext;
use crate::pssm::Pssm;

const KERNEL_SRC: &str = include_str!("pssm_sw_warp.cu");
const KERNEL_NAME: &str = "pssm_sw_warp_batch";

/// Must match the `#define`s inside `pssm_sw_warp.cu`. Verified by
/// [`MAX_QUERY_LEN`] being used at dispatch time as the eligibility
/// threshold.
pub const GROUPSIZE: usize = 32;
pub const NUM_ITEMS: usize = 8;
pub const MAX_QUERY_LEN: usize = GROUPSIZE * NUM_ITEMS;

/// Pairs per block = warps per block. 4 pairs/block × 32 threads/warp
/// = 128 threads/block. Kept identical to the thread-per-pair kernel's
/// block size for consistency in scheduling.
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_SIZE: u32 = WARPS_PER_BLOCK * 32;

/// Conservative shared-memory budget per block — same as the sibling
/// pssm_sw.rs; Turing onwards guarantees ≥48 KB and we leave headroom
/// for the runtime's own allocations.
const SHARED_MEM_LIMIT_BYTES: usize = 46 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum PssmSwWarpDispatchError {
    /// Query longer than the singletile capacity. Fall back to the
    /// thread-per-pair kernel or (phase 4.5b) the multitile variant.
    #[error(
        "query length {query_len} exceeds singletile warp kernel capacity {max}; \
         fall back to pssm_sw or the (future) multitile variant"
    )]
    QueryTooLong { query_len: usize, max: usize },
    #[error("PSSM size {pssm_bytes} bytes exceeds shared-memory budget {limit_bytes}")]
    PssmTooLargeForSharedMem {
        pssm_bytes: usize,
        limit_bytes: usize,
    },
    #[error("driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("infrastructure: {0}")]
    Other(#[from] anyhow::Error),
}

pub(crate) struct PssmSwWarpKernel {
    func: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<PssmSwWarpKernel>> = OnceLock::new();

impl PssmSwWarpKernel {
    pub(crate) fn try_global() -> Option<&'static PssmSwWarpKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!("[proteon-search-gpu] pssm_sw_warp kernel compile failed: {e:#}");
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
            .with_context(|| "NVRTC compile of pssm_sw_warp.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// Warp-collaborative PSSM Smith-Waterman batch dispatch.
///
/// Returns one [`GappedScoreEnd`] per input target, in input order.
/// Per-pair "no alignment" is `GappedScoreEnd { score: 0, .. }`.
///
/// Errors:
/// - [`PssmSwWarpDispatchError::QueryTooLong`] when `pssm.query_len > MAX_QUERY_LEN`.
/// - [`PssmSwWarpDispatchError::PssmTooLargeForSharedMem`] when the PSSM
///   doesn't fit the per-block shared-memory budget.
pub fn pssm_sw_warp_batch_gpu(
    pssm: &Pssm,
    targets: &[&[u8]],
    gap_open: i32,
    gap_extend: i32,
) -> std::result::Result<Vec<GappedScoreEnd>, PssmSwWarpDispatchError> {
    if pssm.query_len > MAX_QUERY_LEN {
        return Err(PssmSwWarpDispatchError::QueryTooLong {
            query_len: pssm.query_len,
            max: MAX_QUERY_LEN,
        });
    }
    let pssm_bytes = pssm.data.len() * std::mem::size_of::<i32>();
    if pssm_bytes > SHARED_MEM_LIMIT_BYTES {
        return Err(PssmSwWarpDispatchError::PssmTooLargeForSharedMem {
            pssm_bytes,
            limit_bytes: SHARED_MEM_LIMIT_BYTES,
        });
    }

    let n_pairs = targets.len();
    if n_pairs == 0 {
        return Ok(Vec::new());
    }

    let kernel = PssmSwWarpKernel::try_global()
        .ok_or_else(|| anyhow!("GPU pssm_sw_warp kernel unavailable"))?;
    let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Flatten targets into one byte buffer with per-pair offset + length.
    let mut target_offsets: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut target_lens: Vec<i32> = Vec::with_capacity(n_pairs);
    let total_bytes: usize = targets.iter().map(|t| t.len()).sum();
    let mut targets_flat: Vec<u8> = Vec::with_capacity(total_bytes.max(1));
    for t in targets {
        target_offsets.push(targets_flat.len() as i32);
        target_lens.push(t.len() as i32);
        targets_flat.extend_from_slice(t);
    }
    if targets_flat.is_empty() {
        targets_flat.push(0); // cudarc rejects zero-byte alloc
    }

    // PSSM is never empty once we've passed the query-length guards,
    // but cudarc's zero-byte rejection still applies — guard for
    // consistency with the sibling kernels.
    let pssm_for_gpu: &[i32] = if pssm.data.is_empty() {
        &[0i32]
    } else {
        &pssm.data
    };
    let d_pssm = stream.clone_htod(pssm_for_gpu)?;
    let d_targets = stream.clone_htod(&targets_flat)?;
    let d_offsets = stream.clone_htod(&target_offsets)?;
    let d_lens = stream.clone_htod(&target_lens)?;

    let mut d_score = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qend = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tend = stream.alloc_zeros::<i32>(n_pairs)?;

    let n_blocks = (n_pairs as u32).div_ceil(WARPS_PER_BLOCK);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: pssm_bytes as u32,
    };
    {
        let mut a = stream.launch_builder(&kernel.func);
        let q_len_i = pssm.query_len as i32;
        let alph_i = pssm.alphabet_size as i32;
        let n_pairs_i = n_pairs as i32;
        a.arg(&d_pssm);
        a.arg(&q_len_i);
        a.arg(&alph_i);
        a.arg(&d_targets);
        a.arg(&d_offsets);
        a.arg(&d_lens);
        a.arg(&gap_open);
        a.arg(&gap_extend);
        a.arg(&n_pairs_i);
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
    use crate::gpu::sw::smith_waterman_score_batch_gpu;

    fn identity_matrix(alphabet_size: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            s[i * alphabet_size + i] = m;
        }
        s
    }

    /// Three-way parity: warp kernel == thread-per-pair GPU == CPU on
    /// the same hand-crafted corpus the other PSSM tests use.
    #[test]
    fn warp_matches_non_warp_gpu_matches_cpu() {
        if super::PssmSwWarpKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let t_full: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        let t_query_insert: Vec<u8> = vec![0, 1, 2, 3];
        let t_target_insert: Vec<u8> = vec![0, 1, 0, 0, 2, 3, 0, 1];
        let t_unrelated: Vec<u8> = vec![0, 0, 0, 0, 0, 0];
        let t_short: Vec<u8> = vec![1, 2];
        let t_long: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let targets_owned: Vec<Vec<u8>> = vec![
            t_full,
            t_query_insert,
            t_target_insert,
            t_unrelated,
            t_short,
            t_long,
        ];
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let warp_results = pssm_sw_warp_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("warp dispatch failed");

        let plain_gpu = smith_waterman_score_batch_gpu(
            &query,
            &targets,
            &scores,
            alphabet_size,
            gap_open,
            gap_extend,
        )
        .expect("non-warp GPU dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                warp_results[i].score, cpu_score,
                "target {i}: warp {} != CPU {cpu_score}",
                warp_results[i].score,
            );
            assert_eq!(
                warp_results[i].score, plain_gpu[i].score,
                "target {i}: warp {} != non-warp GPU {}",
                warp_results[i].score, plain_gpu[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(
                    warp_results[i].query_end, cpu_aln.query_end,
                    "target {i}: qend mismatch"
                );
                assert_eq!(
                    warp_results[i].target_end, cpu_aln.target_end,
                    "target {i}: tend mismatch"
                );
            }
        }
    }

    /// Larger PRNG batch — 200 targets, varied lengths, deterministic
    /// seed. Exercises multi-block dispatch at real launch dimensions.
    /// Query length 40 stays well under the 256-singletile cap.
    #[test]
    fn warp_parity_on_larger_batch() {
        if super::PssmSwWarpKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;
        let query: Vec<u8> = (0..40).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let mut rng_state: u32 = 0x5eed_beef;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        let targets_owned: Vec<Vec<u8>> = (0..200)
            .map(|_| {
                let len = 10 + (next() as usize % 50);
                (0..len)
                    .map(|_| (next() % alphabet_size as u32) as u8)
                    .collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let warp_results = pssm_sw_warp_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("warp dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                warp_results[i].score, cpu_score,
                "target {i}: warp {} != CPU {cpu_score}",
                warp_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(warp_results[i].query_end, cpu_aln.query_end, "i={i}");
                assert_eq!(warp_results[i].target_end, cpu_aln.target_end, "i={i}");
            }
        }
    }

    /// Longer query near the singletile boundary (query_len=250, fits
    /// 256-cap). Asserts the register-tiled algorithm produces the same
    /// answer for queries that actually fill most of the warp budget.
    #[test]
    fn warp_parity_at_near_max_query_len() {
        if super::PssmSwWarpKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let mut rng: u32 = 0xABCD_1234;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };

        let query: Vec<u8> = (0..250)
            .map(|_| (next() % alphabet_size as u32) as u8)
            .collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let targets_owned: Vec<Vec<u8>> = (0..32)
            .map(|_| {
                let len = 50 + (next() as usize % 250);
                (0..len)
                    .map(|_| (next() % alphabet_size as u32) as u8)
                    .collect()
            })
            .collect();
        let targets: Vec<&[u8]> = targets_owned.iter().map(|v| v.as_slice()).collect();

        let warp_results = pssm_sw_warp_batch_gpu(&pssm, &targets, gap_open, gap_extend)
            .expect("warp dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                warp_results[i].score, cpu_score,
                "target {i}: warp {} != CPU {cpu_score}",
                warp_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(warp_results[i].query_end, cpu_aln.query_end, "i={i}");
                assert_eq!(warp_results[i].target_end, cpu_aln.target_end, "i={i}");
            }
        }
    }

    #[test]
    fn empty_target_list_returns_empty_vec_without_dispatch() {
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let result = pssm_sw_warp_batch_gpu(&pssm, &[], -5, -1).expect("ok");
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
        match pssm_sw_warp_batch_gpu(&pssm, &targets, -5, -1) {
            Err(PssmSwWarpDispatchError::QueryTooLong { query_len, max }) => {
                assert_eq!(query_len, MAX_QUERY_LEN + 1);
                assert_eq!(max, MAX_QUERY_LEN);
            }
            other => panic!("expected QueryTooLong, got {other:?}"),
        }
    }
}
