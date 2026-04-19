//! GPU dispatch wrapper for batched ungapped diagonal scoring.
//!
//! CPU oracle is [`crate::ungapped::ungapped_alignment`] — this kernel
//! is a line-for-line port of that function, parallelized across many
//! `(target, diagonal)` pairs against one shared query. Per-pair output
//! is the same Kadane-derived `(score, q_start, q_end, t_start, t_end)`
//! tuple, returned as `Option<UngappedHit>` (None when no positive
//! segment exists, mirroring the CPU contract).
//!
//! Lazy NVRTC compile via `OnceLock` — first call pays the ~100 ms
//! compile, subsequent calls reuse the cached `CudaFunction`.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::GpuContext;
use crate::ungapped::UngappedHit;

const KERNEL_SRC: &str = include_str!("diagonal.cu");
const KERNEL_NAME: &str = "ungapped_diagonal_batch";

/// Compiled kernel + the module that owns it. Module must outlive the
/// function; both are kept in `Arc` so the static cache can hand out
/// references.
pub(crate) struct DiagonalKernel {
    func: CudaFunction,
    // `_module` keeps the cudarc module alive; `func` is a handle into it.
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<DiagonalKernel>> = OnceLock::new();

impl DiagonalKernel {
    /// Lazily compile + cache the kernel. Returns `None` if no GPU is
    /// available or compilation failed.
    pub(crate) fn try_global() -> Option<&'static DiagonalKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!("[proteon-search-gpu] diagonal kernel compile failed: {e:#}");
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
            .with_context(|| "NVRTC compile of diagonal.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// Score `(target, diagonal)` pairs in parallel against a shared query.
///
/// Returns one `Option<UngappedHit>` per input pair, in the same order
/// as `targets_and_diagonals`. Semantics per pair are identical to
/// [`crate::ungapped::ungapped_alignment`]: `Some` for a positive-
/// scoring local segment on the diagonal, `None` otherwise.
///
/// Returns `Err` only on hard infrastructure failures (no GPU, kernel
/// compile failed, allocation error). Pair-level "no hit" is `Ok(None)`,
/// not `Err`.
///
/// `scores` is a flat `alphabet_size * alphabet_size` row-major i32
/// matrix, same layout the CPU side already uses.
pub fn ungapped_alignment_batch_gpu(
    query: &[u8],
    targets_and_diagonals: &[(&[u8], i32)],
    scores: &[i32],
    alphabet_size: usize,
) -> Result<Vec<Option<UngappedHit>>> {
    assert_eq!(
        scores.len(),
        alphabet_size * alphabet_size,
        "scores length must equal alphabet_size^2"
    );

    let n_pairs = targets_and_diagonals.len();
    if n_pairs == 0 {
        return Ok(Vec::new());
    }

    let kernel =
        DiagonalKernel::try_global().ok_or_else(|| anyhow!("GPU diagonal kernel unavailable"))?;
    let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Pack variable-length targets into one flat byte buffer with
    // per-pair offset + length arrays. i32 indexing limits any single
    // launch to ~2 GB of concatenated targets — far beyond useful
    // batch sizes; revisit when we hit it.
    let mut target_offsets: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut target_lens: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut diagonals: Vec<i32> = Vec::with_capacity(n_pairs);
    let total_target_bytes: usize = targets_and_diagonals.iter().map(|(t, _)| t.len()).sum();
    let mut targets_flat: Vec<u8> = Vec::with_capacity(total_target_bytes.max(1));
    for &(t, d) in targets_and_diagonals {
        target_offsets.push(targets_flat.len() as i32);
        target_lens.push(t.len() as i32);
        diagonals.push(d);
        targets_flat.extend_from_slice(t);
    }
    if targets_flat.is_empty() {
        targets_flat.push(0); // cudarc rejects zero-byte allocations
    }

    // Upload inputs. Same sentinel-byte dance for the query: an empty
    // query means zero pairwise scores would be computed anyway, but
    // cudarc's `clone_htod` refuses the zero-byte alloc, so we hand it
    // a 1-byte buffer that the kernel never reads (q_len=0).
    let query_for_gpu: &[u8] = if query.is_empty() { &[0u8] } else { query };
    let d_query = stream.clone_htod(query_for_gpu)?;
    let d_targets = stream.clone_htod(&targets_flat)?;
    let d_offsets = stream.clone_htod(&target_offsets)?;
    let d_lens = stream.clone_htod(&target_lens)?;
    let d_diagonals = stream.clone_htod(&diagonals)?;
    let d_scores = stream.clone_htod(scores)?;

    // Output buffers, zero-initialised so the kernel's "no hit" branch
    // (which writes 0s) is consistent with our absent-pair default.
    let mut d_score: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qstart: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qend: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tstart: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tend: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;

    let cfg = LaunchConfig::for_num_elems(n_pairs as u32);
    {
        let mut a = stream.launch_builder(&kernel.func);
        let q_len_i = query.len() as i32;
        let alph_i = alphabet_size as i32;
        let n_pairs_i = n_pairs as i32;
        a.arg(&d_query);
        a.arg(&q_len_i);
        a.arg(&d_targets);
        a.arg(&d_offsets);
        a.arg(&d_lens);
        a.arg(&d_diagonals);
        a.arg(&d_scores);
        a.arg(&alph_i);
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
    use crate::ungapped::ungapped_alignment;

    fn identity_matrix(alphabet_size: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            s[i * alphabet_size + i] = m;
        }
        s
    }

    /// GPU vs CPU parity on a hand-crafted batch covering the same
    /// edge cases unit-tested on the CPU side: full match, diverged
    /// tails, zero overlap, positive/negative diagonals, no positive
    /// segment, empty input.
    #[test]
    fn parity_against_cpu_ungapped_alignment_on_hand_crafted_batch() {
        // Skip if no GPU on this host. The scaffold's `try_global` test
        // already exercises the no-GPU path; here we just need the
        // kernel to run if hardware is present.
        if super::DiagonalKernel::try_global().is_none() {
            eprintln!("SKIP parity test: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];

        // (target, diagonal) pairs covering each ungapped.rs unit-test
        // scenario.
        let t_full: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3]; // full match @ diag 0
        let t_diverged_tails: Vec<u8> = vec![3, 3, 2, 3, 0, 1, 3, 3]; // center matches
        let t_no_match: Vec<u8> = vec![1, 2, 3, 0]; // pairwise mismatch on diag 0
        let t_positive_diag: Vec<u8> = vec![3, 3, 0, 1, 2, 3, 0, 1, 2, 3]; // diag +2
        let t_negative_diag: Vec<u8> = vec![2, 3, 0, 1]; // diag -2
        let t_empty_overlap: Vec<u8> = vec![0, 1, 2]; // diag = 5 → empty

        let pairs: Vec<(&[u8], i32)> = vec![
            (&t_full, 0),
            (&t_diverged_tails, 0),
            (&t_no_match, 0),
            (&t_positive_diag, 2),
            (&t_negative_diag, -2),
            (&t_empty_overlap, 5),
        ];

        let gpu_results = ungapped_alignment_batch_gpu(&query, &pairs, &scores, alphabet_size)
            .expect("GPU batch ungapped failed");

        for (i, (target, diagonal)) in pairs.iter().enumerate() {
            let cpu = ungapped_alignment(&query, target, *diagonal, &scores, alphabet_size);
            let gpu = &gpu_results[i];
            assert_eq!(
                cpu, *gpu,
                "pair {i}: CPU vs GPU disagree for diagonal {diagonal}",
            );
        }
    }

    /// Empty query with non-empty targets must not explode on cudarc's
    /// zero-byte-alloc rejection. Kernel should run with q_len=0 and
    /// report no hit for every pair.
    #[test]
    fn empty_query_with_nonempty_targets_returns_nones() {
        if super::DiagonalKernel::try_global().is_none() {
            eprintln!("SKIP empty-query guard: no GPU available");
            return;
        }
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let t1: Vec<u8> = vec![0, 1, 2, 3];
        let t2: Vec<u8> = vec![3, 2, 1, 0];
        let pairs: Vec<(&[u8], i32)> = vec![(&t1, 0), (&t2, 1)];
        let result = ungapped_alignment_batch_gpu(&[], &pairs, &scores, alphabet_size)
            .expect("empty query must not be an infrastructure error");
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|r| r.is_none()));
    }

    #[test]
    fn empty_pair_list_returns_empty_vec_without_gpu_dispatch() {
        // Should never even probe the kernel for a zero-pair call.
        let scores = identity_matrix(4, 3, -2);
        let result = ungapped_alignment_batch_gpu(&[0, 1, 2], &[], &scores, 4).expect("ok");
        assert!(result.is_empty());
    }

    /// Larger batch (100 random pairs) — exercises real launch
    /// dimensions and confirms the per-pair index math survives at
    /// scale. Still GPU-vs-CPU oracled.
    #[test]
    fn parity_on_larger_batch() {
        if super::DiagonalKernel::try_global().is_none() {
            eprintln!("SKIP larger batch: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = (0..50).map(|i| (i % alphabet_size) as u8).collect();

        // Build 100 deterministic targets at varying lengths and
        // diagonals (positive, zero, negative) using a simple PRNG.
        let mut rng_state: u32 = 0xdead_beef;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        let mut targets: Vec<Vec<u8>> = Vec::new();
        let mut diagonals: Vec<i32> = Vec::new();
        for _ in 0..100 {
            let len = 10 + (next() as usize % 60);
            let target: Vec<u8> = (0..len)
                .map(|_| (next() % alphabet_size as u32) as u8)
                .collect();
            let diag = (next() as i32 % 31) - 15;
            targets.push(target);
            diagonals.push(diag);
        }
        let pairs: Vec<(&[u8], i32)> = targets
            .iter()
            .zip(diagonals.iter())
            .map(|(t, d)| (t.as_slice(), *d))
            .collect();

        let gpu_results = ungapped_alignment_batch_gpu(&query, &pairs, &scores, alphabet_size)
            .expect("GPU batch failed");

        for (i, (target, diagonal)) in pairs.iter().enumerate() {
            let cpu = ungapped_alignment(&query, target, *diagonal, &scores, alphabet_size);
            assert_eq!(
                cpu, gpu_results[i],
                "pair {i}: CPU vs GPU disagree for diagonal {diagonal}",
            );
        }
    }
}
