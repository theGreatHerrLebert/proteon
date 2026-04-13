//! GPU dispatch for PSSM-based batched ungapped diagonal scoring.
//!
//! Same algorithm as [`super::diagonal::ungapped_alignment_batch_gpu`]
//! — same Kadane-with-reset semantic, same outputs — but with the
//! score lookup served from shared-memory-staged PSSM rows instead of
//! global-memory `scores[query[q] * alphabet_size + target[t]]`. This
//! is the central perf primitive in libmarv: it eliminates global-
//! memory traffic for scores during the hot loop.
//!
//! Caller hands in a precomputed [`Pssm`] (typically built once per
//! query and reused across many search calls). The kernel stages it
//! into shared memory at block entry; for typical query lengths
//! (≤500 residues at alphabet 21 → ≤42 KB) this fits comfortably
//! within the per-block shared-memory limit on Turing+. Larger
//! queries should fall back to the non-PSSM `diagonal` kernel — a
//! limit-check at the dispatcher boundary handles that.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{
    CudaFunction, CudaModule, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::GpuContext;
use crate::pssm::Pssm;
use crate::ungapped::UngappedHit;

const KERNEL_SRC: &str = include_str!("pssm_diagonal.cu");
const KERNEL_NAME: &str = "ungapped_diagonal_pssm_batch";

/// Block size for PSSM staging — chosen so 256 threads cooperatively
/// load even moderately large PSSMs in a few iterations. Same value
/// `LaunchConfig::for_num_elems` would pick by default.
const BLOCK_SIZE: u32 = 256;

/// Conservative shared-memory budget per block. Turing onwards
/// guarantees ≥48 KB; we leave headroom for the runtime's own
/// allocations.
const SHARED_MEM_LIMIT_BYTES: usize = 46 * 1024;

/// Returned when the query is too large for shared-memory staging
/// — caller should fall back to `super::diagonal`.
#[derive(Debug, thiserror::Error)]
pub enum PssmDispatchError {
    #[error(
        "PSSM size {pssm_bytes} bytes exceeds shared-memory budget {limit_bytes}; \
         query is too long ({query_len} × alphabet {alphabet_size}). \
         Fall back to the non-PSSM `diagonal` kernel for this query."
    )]
    PssmTooLargeForSharedMem {
        pssm_bytes: usize,
        limit_bytes: usize,
        query_len: usize,
        alphabet_size: usize,
    },
    #[error("driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("infrastructure: {0}")]
    Other(#[from] anyhow::Error),
}

pub(crate) struct PssmDiagonalKernel {
    func: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<PssmDiagonalKernel>> = OnceLock::new();

impl PssmDiagonalKernel {
    pub(crate) fn try_global() -> Option<&'static PssmDiagonalKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!(
                            "[ferritin-search-gpu] pssm_diagonal kernel compile failed: {e:#}"
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
            .with_context(|| "NVRTC compile of pssm_diagonal.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// PSSM-staged batched ungapped diagonal scoring.
///
/// Inputs:
/// - `pssm`: precomputed query PSSM (build once per query, reuse across
///   many target batches).
/// - `targets_and_diagonals`: `(target_bytes, diagonal)` pairs —
///   identical interface to [`super::diagonal::ungapped_alignment_batch_gpu`].
///
/// Returns one [`UngappedHit`] per pair, mirroring the non-PSSM kernel.
/// Returns `Err(PssmDispatchError::PssmTooLargeForSharedMem)` if the
/// PSSM doesn't fit the per-block shared-memory budget; callers
/// should dispatch to the regular `diagonal` kernel in that case.
pub fn ungapped_alignment_pssm_batch_gpu(
    pssm: &Pssm,
    targets_and_diagonals: &[(&[u8], i32)],
) -> std::result::Result<Vec<Option<UngappedHit>>, PssmDispatchError> {
    let pssm_bytes = pssm.data.len() * std::mem::size_of::<i32>();
    if pssm_bytes > SHARED_MEM_LIMIT_BYTES {
        return Err(PssmDispatchError::PssmTooLargeForSharedMem {
            pssm_bytes,
            limit_bytes: SHARED_MEM_LIMIT_BYTES,
            query_len: pssm.query_len,
            alphabet_size: pssm.alphabet_size,
        });
    }

    let n_pairs = targets_and_diagonals.len();
    if n_pairs == 0 {
        return Ok(Vec::new());
    }

    let kernel = PssmDiagonalKernel::try_global()
        .ok_or_else(|| anyhow!("GPU PSSM diagonal kernel unavailable"))?;
    let ctx = GpuContext::try_global()
        .ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Pack targets flat + per-pair offset/length/diagonal.
    let mut target_offsets: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut target_lens: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut diagonals: Vec<i32> = Vec::with_capacity(n_pairs);
    let total_bytes: usize = targets_and_diagonals.iter().map(|(t, _)| t.len()).sum();
    let mut targets_flat: Vec<u8> = Vec::with_capacity(total_bytes.max(1));
    for &(t, d) in targets_and_diagonals {
        target_offsets.push(targets_flat.len() as i32);
        target_lens.push(t.len() as i32);
        diagonals.push(d);
        targets_flat.extend_from_slice(t);
    }
    if targets_flat.is_empty() {
        targets_flat.push(0);
    }

    let d_pssm = stream.clone_htod(&pssm.data)?;
    let d_targets = stream.clone_htod(&targets_flat)?;
    let d_offsets = stream.clone_htod(&target_offsets)?;
    let d_lens = stream.clone_htod(&target_lens)?;
    let d_diagonals = stream.clone_htod(&diagonals)?;

    let mut d_score = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qstart = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qend = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tstart = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tend = stream.alloc_zeros::<i32>(n_pairs)?;

    // Custom launch config: explicit block size + shared memory size,
    // since the kernel uses dynamic shared memory for the PSSM stage.
    let n_blocks = (n_pairs as u32).div_ceil(BLOCK_SIZE);
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
        a.arg(&d_diagonals);
        a.arg(&n_pairs_i);
        a.arg(&mut d_score);
        a.arg(&mut d_qstart);
        a.arg(&mut d_qend);
        a.arg(&mut d_tstart);
        a.arg(&mut d_tend);
        unsafe { a.launch(cfg)? };
    }
    stream.synchronize()?;

    let scores_out = stream.clone_dtoh(&d_score)?;
    let qstart_out = stream.clone_dtoh(&d_qstart)?;
    let qend_out = stream.clone_dtoh(&d_qend)?;
    let tstart_out = stream.clone_dtoh(&d_tstart)?;
    let tend_out = stream.clone_dtoh(&d_tend)?;

    let mut out: Vec<Option<UngappedHit>> = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        if scores_out[i] > 0 {
            out.push(Some(UngappedHit {
                score: scores_out[i],
                query_start: qstart_out[i] as usize,
                query_end: qend_out[i] as usize,
                target_start: tstart_out[i] as usize,
                target_end: tend_out[i] as usize,
            }));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::diagonal::ungapped_alignment_batch_gpu;
    use crate::ungapped::ungapped_alignment;

    fn identity_matrix(alphabet_size: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            s[i * alphabet_size + i] = m;
        }
        s
    }

    /// Three-way parity: PSSM-GPU == non-PSSM-GPU == CPU on identical
    /// inputs. Same algorithm via three different lookup paths.
    #[test]
    fn pssm_gpu_equals_non_pssm_gpu_equals_cpu_on_hand_crafted_batch() {
        if super::PssmDiagonalKernel::try_global().is_none() {
            eprintln!("SKIP parity test: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let t_full: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let t_diverged: Vec<u8> = vec![3, 3, 2, 3, 0, 1, 3, 3];
        let t_no_match: Vec<u8> = vec![1, 2, 3, 0];
        let t_pos_diag: Vec<u8> = vec![3, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let t_neg_diag: Vec<u8> = vec![2, 3, 0, 1];
        let t_empty: Vec<u8> = vec![0, 1, 2];

        let pairs: Vec<(&[u8], i32)> = vec![
            (&t_full, 0),
            (&t_diverged, 0),
            (&t_no_match, 0),
            (&t_pos_diag, 2),
            (&t_neg_diag, -2),
            (&t_empty, 5),
        ];

        let pssm_results = ungapped_alignment_pssm_batch_gpu(&pssm, &pairs)
            .expect("PSSM GPU dispatch failed");
        let non_pssm_results =
            ungapped_alignment_batch_gpu(&query, &pairs, &scores, alphabet_size)
                .expect("non-PSSM GPU dispatch failed");

        for (i, (target, diag)) in pairs.iter().enumerate() {
            let cpu = ungapped_alignment(&query, target, *diag, &scores, alphabet_size);
            assert_eq!(
                pssm_results[i], cpu,
                "pair {i}: PSSM GPU vs CPU disagree"
            );
            assert_eq!(
                pssm_results[i], non_pssm_results[i],
                "pair {i}: PSSM GPU vs non-PSSM GPU disagree"
            );
        }
    }

    /// Larger PRNG batch — confirms the shared-memory PSSM staging
    /// + per-thread Kadane don't desync at real launch dimensions.
    #[test]
    fn pssm_parity_on_larger_batch() {
        if super::PssmDiagonalKernel::try_global().is_none() {
            eprintln!("SKIP larger batch: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..50).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        let mut rng_state: u32 = 0xfeed_face;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        let mut targets: Vec<Vec<u8>> = Vec::new();
        let mut diagonals: Vec<i32> = Vec::new();
        for _ in 0..200 {
            let len = 10 + (next() as usize % 60);
            let target: Vec<u8> = (0..len).map(|_| (next() % alphabet_size as u32) as u8).collect();
            let diag = (next() as i32 % 31) - 15;
            targets.push(target);
            diagonals.push(diag);
        }
        let pairs: Vec<(&[u8], i32)> = targets
            .iter()
            .zip(diagonals.iter())
            .map(|(t, d)| (t.as_slice(), *d))
            .collect();

        let pssm_results = ungapped_alignment_pssm_batch_gpu(&pssm, &pairs)
            .expect("PSSM GPU dispatch failed");

        for (i, (target, diag)) in pairs.iter().enumerate() {
            let cpu = ungapped_alignment(&query, target, *diag, &scores, alphabet_size);
            assert_eq!(
                pssm_results[i], cpu,
                "pair {i}: PSSM GPU vs CPU disagree (diag={diag})"
            );
        }
    }

    /// Oversize PSSM must reject cleanly with the structured error,
    /// not silently truncate or dispatch with garbage.
    #[test]
    fn oversize_pssm_returns_structured_error() {
        if super::PssmDiagonalKernel::try_global().is_none() {
            eprintln!("SKIP oversize test: no GPU available");
            return;
        }
        // Force a PSSM larger than the shared-memory budget: 12000 ×
        // 21 i32 = ~1 MB, well above 46 KB.
        let alphabet_size = 21;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..12000).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        let target: Vec<u8> = vec![0; 100];
        let pairs: Vec<(&[u8], i32)> = vec![(target.as_slice(), 0)];
        match ungapped_alignment_pssm_batch_gpu(&pssm, &pairs) {
            Err(PssmDispatchError::PssmTooLargeForSharedMem { .. }) => {}
            other => panic!("expected PssmTooLargeForSharedMem, got {other:?}"),
        }
    }

    #[test]
    fn empty_pair_list_returns_empty_vec() {
        let scores = identity_matrix(4, 3, -2);
        let query: Vec<u8> = vec![0, 1, 2];
        let pssm = Pssm::build(&query, &scores, 4);
        let result = ungapped_alignment_pssm_batch_gpu(&pssm, &[]).expect("ok");
        assert!(result.is_empty());
    }
}
