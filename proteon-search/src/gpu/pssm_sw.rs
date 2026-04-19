//! GPU dispatch for PSSM Smith-Waterman on PaddedDb.
//!
//! Reference: Kallenborn, Chacon, Hundt, Sirelkhatim, Didi, Cha, Dallago,
//! Mirdita, Schmidt, Steinegger, "GPU-accelerated homology search with
//! MMseqs2", *Nat. Methods* 22, 2024-2027 (2025). This module implements
//! the canonical "libmarv" gapped-SW kernel shape from that work.
//!
//! Combines the perf primitives from 4.4a (PSSM shared-mem staging)
//! and 4.4b (PaddedDb transposed-coalesced target layout) into the
//! canonical libmarv gapped kernel. Same Gotoh affine-gap algorithm
//! as both [`crate::gapped::smith_waterman`] (CPU oracle) and
//! [`super::sw::smith_waterman_score_batch_gpu`] (non-PSSM GPU);
//! parity tests assert bit-equal output against both.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::sw::GappedScoreEnd;
use super::GpuContext;
use crate::padded_db::PaddedDb;
use crate::pssm::Pssm;

const KERNEL_SRC: &str = include_str!("pssm_sw.cu");
const KERNEL_NAME: &str = "pssm_sw_padded_batch";

const BLOCK_SIZE: u32 = 128;
const SHARED_MEM_LIMIT_BYTES: usize = 46 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum PssmSwDispatchError {
    #[error("PSSM size {pssm_bytes} bytes exceeds shared-memory budget {limit_bytes}")]
    PssmTooLargeForSharedMem {
        pssm_bytes: usize,
        limit_bytes: usize,
    },
    #[error(
        "BLOCK_SIZE {block} is not a multiple of PaddedDb bucket_size \
         {bucket}; change block_size to avoid warp-straddle inefficiency"
    )]
    BlockSizeBucketMismatch { block: usize, bucket: usize },
    #[error("driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("infrastructure: {0}")]
    Other(#[from] anyhow::Error),
}

pub(crate) struct PssmSwKernel {
    func: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<PssmSwKernel>> = OnceLock::new();

impl PssmSwKernel {
    pub(crate) fn try_global() -> Option<&'static PssmSwKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!("[proteon-search-gpu] pssm_sw kernel compile failed: {e:#}");
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
            .with_context(|| "NVRTC compile of pssm_sw.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// Run PSSM Smith-Waterman across every target in `padded_db` against
/// the query encoded by `pssm`. Returns `Vec<GappedScoreEnd>` in
/// **original input order** — the per-bucket/per-slot layout is an
/// implementation detail the caller doesn't see.
///
/// `gap_open` / `gap_extend` match CPU
/// [`crate::gapped::smith_waterman`] convention: gap of length k costs
/// `gap_open + gap_extend * (k - 1)`, both typically negative.
///
/// Errors:
/// - [`PssmSwDispatchError::PssmTooLargeForSharedMem`] when the PSSM
///   doesn't fit the per-block shared-memory budget. Caller should
///   fall back to [`super::sw::smith_waterman_score_batch_gpu`].
/// - [`PssmSwDispatchError::BlockSizeBucketMismatch`] if the kernel's
///   fixed `BLOCK_SIZE=128` isn't a multiple of `padded_db.bucket_size`
///   (sanity guard against warp-straddling inefficiency; caller can
///   rebuild the PaddedDb with a compatible bucket_size).
pub fn pssm_sw_padded_batch_gpu(
    pssm: &Pssm,
    padded_db: &PaddedDb,
    gap_open: i32,
    gap_extend: i32,
) -> std::result::Result<Vec<GappedScoreEnd>, PssmSwDispatchError> {
    let pssm_bytes = pssm.data.len() * std::mem::size_of::<i32>();
    if pssm_bytes > SHARED_MEM_LIMIT_BYTES {
        return Err(PssmSwDispatchError::PssmTooLargeForSharedMem {
            pssm_bytes,
            limit_bytes: SHARED_MEM_LIMIT_BYTES,
        });
    }
    if (BLOCK_SIZE as usize) % padded_db.bucket_size != 0 {
        return Err(PssmSwDispatchError::BlockSizeBucketMismatch {
            block: BLOCK_SIZE as usize,
            bucket: padded_db.bucket_size,
        });
    }

    let n_targets = padded_db.num_targets();
    if n_targets == 0 {
        return Ok(Vec::new());
    }

    let kernel =
        PssmSwKernel::try_global().ok_or_else(|| anyhow!("GPU PSSM SW kernel unavailable"))?;
    let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Per-slot real lengths flattened in (bucket * bucket_size + slot) order.
    let n_buckets = padded_db.num_buckets();
    let bucket_size = padded_db.bucket_size;
    let n_threads_total = n_buckets * bucket_size;
    let mut slot_real_lens: Vec<i32> = vec![0; n_threads_total];
    for (orig, &(b, slot)) in padded_db.original_to_padded.iter().enumerate() {
        if b == usize::MAX {
            continue;
        }
        slot_real_lens[b * bucket_size + slot] = padded_db.original_lens[orig] as i32;
    }

    let bucket_starts_i32: Vec<i32> = padded_db.bucket_starts.iter().map(|&s| s as i32).collect();
    let bucket_padded_lens_i32: Vec<i32> = padded_db
        .bucket_padded_lens
        .iter()
        .map(|&l| l as i32)
        .collect();
    let max_padded_len = *padded_db.bucket_padded_lens.iter().max().unwrap_or(&0);

    // Guard for empty PSSM even though the shared-mem check above
    // rejects only oversized PSSMs — an empty one would slip through
    // into cudarc's zero-byte-alloc rejection.
    let pssm_for_gpu: &[i32] = if pssm.data.is_empty() {
        &[0i32]
    } else {
        &pssm.data
    };
    let d_pssm = stream.clone_htod(pssm_for_gpu)?;
    // `clone_htod` rejects zero-byte allocations.
    let padded_data_slice = if padded_db.data.is_empty() {
        vec![0u8]
    } else {
        padded_db.data.clone()
    };
    let d_padded = stream.clone_htod(&padded_data_slice)?;
    let d_bucket_starts = stream.clone_htod(&bucket_starts_i32)?;
    let d_bucket_padded_lens = stream.clone_htod(&bucket_padded_lens_i32)?;
    let d_slot_real_lens = stream.clone_htod(&slot_real_lens)?;

    // Per-thread scratch: 6 × (max_padded_len + 1) i32.
    let row_stride = max_padded_len + 1;
    let scratch_total = n_threads_total * 6 * row_stride;
    let mut d_scratch = stream.alloc_zeros::<i32>(scratch_total.max(1))?;

    let mut d_score = stream.alloc_zeros::<i32>(n_threads_total)?;
    let mut d_qend = stream.alloc_zeros::<i32>(n_threads_total)?;
    let mut d_tend = stream.alloc_zeros::<i32>(n_threads_total)?;

    let n_blocks = (n_threads_total as u32).div_ceil(BLOCK_SIZE);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: pssm_bytes as u32,
    };
    {
        let mut a = stream.launch_builder(&kernel.func);
        let q_len_i = pssm.query_len as i32;
        let alph_i = pssm.alphabet_size as i32;
        let bucket_size_i = bucket_size as i32;
        let n_buckets_i = n_buckets as i32;
        let max_padded_i = max_padded_len as i32;
        let n_threads_i = n_threads_total as i32;
        a.arg(&d_pssm);
        a.arg(&q_len_i);
        a.arg(&alph_i);
        a.arg(&d_padded);
        a.arg(&d_bucket_starts);
        a.arg(&d_bucket_padded_lens);
        a.arg(&d_slot_real_lens);
        a.arg(&bucket_size_i);
        a.arg(&n_buckets_i);
        a.arg(&gap_open);
        a.arg(&gap_extend);
        a.arg(&max_padded_i);
        a.arg(&n_threads_i);
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

    // Map back from (bucket, slot) layout to original input order.
    let mut out: Vec<GappedScoreEnd> = vec![GappedScoreEnd::default(); n_targets];
    for (orig, &(b, slot)) in padded_db.original_to_padded.iter().enumerate() {
        if b == usize::MAX {
            continue;
        }
        let idx = b * bucket_size + slot;
        out[orig] = GappedScoreEnd {
            score: scores_out[idx],
            query_end: qend_out[idx] as usize,
            target_end: tend_out[idx] as usize,
        };
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

    /// Three-way parity: PSSM+padded GPU == non-PSSM GPU (4.3) == CPU.
    #[test]
    fn pssm_padded_gpu_matches_non_pssm_gpu_matches_cpu() {
        if super::PssmSwKernel::try_global().is_none() {
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

        let padded = PaddedDb::build(&targets, 32, 0);

        let pssm_padded_results = pssm_sw_padded_batch_gpu(&pssm, &padded, gap_open, gap_extend)
            .expect("PSSM padded SW dispatch failed");

        let sw_gpu_results = smith_waterman_score_batch_gpu(
            &query,
            &targets,
            &scores,
            alphabet_size,
            gap_open,
            gap_extend,
        )
        .expect("non-PSSM GPU SW dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                pssm_padded_results[i].score, cpu_score,
                "target {i}: PSSM padded GPU {} != CPU {cpu_score}",
                pssm_padded_results[i].score,
            );
            assert_eq!(
                pssm_padded_results[i].score, sw_gpu_results[i].score,
                "target {i}: PSSM padded GPU {} != non-PSSM GPU {}",
                pssm_padded_results[i].score, sw_gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(
                    pssm_padded_results[i].query_end, cpu_aln.query_end,
                    "target {i}: qend mismatch"
                );
                assert_eq!(
                    pssm_padded_results[i].target_end, cpu_aln.target_end,
                    "target {i}: tend mismatch"
                );
            }
        }
    }

    /// Larger PRNG batch — 200 targets, bucket_size 32, varied lengths.
    /// Exercises multi-bucket dispatch + inactive-slot handling in
    /// the final partial bucket.
    #[test]
    fn pssm_padded_parity_on_larger_batch() {
        if super::PssmSwKernel::try_global().is_none() {
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
        let padded = PaddedDb::build(&targets, 32, 0);

        let gpu_results = pssm_sw_padded_batch_gpu(&pssm, &padded, gap_open, gap_extend)
            .expect("PSSM padded SW dispatch failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(&query, target, &scores, alphabet_size, gap_open, gap_extend);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu_results[i].score, cpu_score,
                "target {i}: GPU {} != CPU {cpu_score}",
                gpu_results[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu_results[i].query_end, cpu_aln.query_end, "i={i}");
                assert_eq!(gpu_results[i].target_end, cpu_aln.target_end, "i={i}");
            }
        }
    }

    #[test]
    fn empty_padded_db_returns_empty_vec_without_dispatch() {
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let padded = PaddedDb::build(&[], 32, 0);
        let result = pssm_sw_padded_batch_gpu(&pssm, &padded, -5, -1).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn oversize_pssm_returns_structured_error() {
        if super::PssmSwKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }
        let alphabet_size = 21;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..12000).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let t: Vec<u8> = vec![0; 100];
        let targets: Vec<&[u8]> = vec![&t];
        let padded = PaddedDb::build(&targets, 32, 0);
        match pssm_sw_padded_batch_gpu(&pssm, &padded, -5, -1) {
            Err(PssmSwDispatchError::PssmTooLargeForSharedMem { .. }) => {}
            other => panic!("expected PssmTooLargeForSharedMem, got {other:?}"),
        }
    }

    #[test]
    fn block_bucket_mismatch_returns_structured_error() {
        if super::PssmSwKernel::try_global().is_none() {
            eprintln!("SKIP: no GPU available");
            return;
        }
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1, 2];
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let t: Vec<u8> = vec![0, 1, 2];
        let targets: Vec<&[u8]> = vec![&t];
        // bucket_size=5 isn't a divisor of BLOCK_SIZE=128, so dispatch
        // must refuse rather than silently run with warp straddling.
        let padded = PaddedDb::build(&targets, 5, 0);
        match pssm_sw_padded_batch_gpu(&pssm, &padded, -5, -1) {
            Err(PssmSwDispatchError::BlockSizeBucketMismatch { .. }) => {}
            other => panic!("expected BlockSizeBucketMismatch, got {other:?}"),
        }
    }
}
