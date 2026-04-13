//! GPU dispatch wrapper for batched Smith-Waterman score+endpoint.
//!
//! CPU oracle is [`crate::gapped::smith_waterman`] — this kernel
//! returns just the `(score, query_end, target_end)` triple,
//! deliberately skipping CIGAR traceback. Callers that need full
//! alignment for the top-K hits run CPU `smith_waterman` on those
//! few pairs; the GPU's job is fast batch ranking, matching how
//! upstream's GPU prefilter hands off to CPU for output.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::GpuContext;

const KERNEL_SRC: &str = include_str!("sw.cu");
const KERNEL_NAME: &str = "smith_waterman_score_batch";

/// One entry per (query, target) pair. `score == 0` means no positive-
/// scoring local alignment exists, mirroring CPU
/// [`crate::gapped::smith_waterman`]'s `None`.
///
/// `query_end` and `target_end` are exclusive — slice indexing via
/// `query[..query_end]` gives the prefix that contains the alignment
/// endpoint. The corresponding `*_start` requires traceback and is
/// not provided by this kernel; callers that need it run CPU SW.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GappedScoreEnd {
    pub score: i32,
    pub query_end: usize,
    pub target_end: usize,
}

pub(crate) struct SwKernel {
    func: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNEL: OnceLock<Option<SwKernel>> = OnceLock::new();

impl SwKernel {
    pub(crate) fn try_global() -> Option<&'static SwKernel> {
        KERNEL
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!(
                            "[ferritin-search-gpu] sw kernel compile failed: {e:#}"
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
            .with_context(|| "NVRTC compile of sw.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            func,
            _module: module,
        })
    }
}

/// Score N (query, target) pairs in parallel. Returns one
/// [`GappedScoreEnd`] per input target, in input order.
///
/// `query` is shared across all pairs; `targets` are the per-pair
/// target sequences (any lengths). `scores` is a flat
/// `alphabet_size × alphabet_size` row-major i32 substitution matrix
/// (same layout the CPU side uses). `gap_open` and `gap_extend` are
/// the affine penalties (typically negative); convention matches CPU
/// `smith_waterman`: a gap of length k costs `gap_open + gap_extend *
/// (k - 1)`.
///
/// Returns `Err` only on infrastructure failures (no GPU, kernel
/// compile failed, allocation error). Per-pair "no alignment" is
/// `Ok(GappedScoreEnd { score: 0, .. })`.
pub fn smith_waterman_score_batch_gpu(
    query: &[u8],
    targets: &[&[u8]],
    scores: &[i32],
    alphabet_size: usize,
    gap_open: i32,
    gap_extend: i32,
) -> Result<Vec<GappedScoreEnd>> {
    assert_eq!(
        scores.len(),
        alphabet_size * alphabet_size,
        "scores length must equal alphabet_size^2"
    );

    let n_pairs = targets.len();
    if n_pairs == 0 {
        return Ok(Vec::new());
    }

    let kernel = SwKernel::try_global()
        .ok_or_else(|| anyhow!("GPU SW kernel unavailable"))?;
    let ctx = GpuContext::try_global()
        .ok_or_else(|| anyhow!("GPU context unavailable"))?;
    let stream = ctx.cuda_context().new_stream()?;

    // Pack targets flat + per-pair offset/length.
    let mut target_offsets: Vec<i32> = Vec::with_capacity(n_pairs);
    let mut target_lens: Vec<i32> = Vec::with_capacity(n_pairs);
    let total_target_bytes: usize = targets.iter().map(|t| t.len()).sum();
    let mut targets_flat: Vec<u8> = Vec::with_capacity(total_target_bytes.max(1));
    let mut max_target_len: usize = 0;
    for t in targets {
        target_offsets.push(targets_flat.len() as i32);
        target_lens.push(t.len() as i32);
        max_target_len = max_target_len.max(t.len());
        targets_flat.extend_from_slice(t);
    }
    if targets_flat.is_empty() {
        targets_flat.push(0);
    }

    let d_query = stream.clone_htod(query)?;
    let d_targets = stream.clone_htod(&targets_flat)?;
    let d_offsets = stream.clone_htod(&target_offsets)?;
    let d_lens = stream.clone_htod(&target_lens)?;
    let d_scores = stream.clone_htod(scores)?;

    // Per-pair scratch: 6 × (max_target_len + 1) i32.
    let row_stride = max_target_len + 1;
    let scratch_per_pair = 6 * row_stride;
    let scratch_total = n_pairs * scratch_per_pair;
    let mut d_scratch: cudarc::driver::CudaSlice<i32> =
        stream.alloc_zeros::<i32>(scratch_total.max(1))?;

    let mut d_score: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_qend: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;
    let mut d_tend: cudarc::driver::CudaSlice<i32> = stream.alloc_zeros::<i32>(n_pairs)?;

    let cfg = LaunchConfig::for_num_elems(n_pairs as u32);
    {
        let mut a = stream.launch_builder(&kernel.func);
        let q_len_i = query.len() as i32;
        let alph_i = alphabet_size as i32;
        let n_pairs_i = n_pairs as i32;
        let max_t_i = max_target_len as i32;
        a.arg(&d_query);
        a.arg(&q_len_i);
        a.arg(&d_targets);
        a.arg(&d_offsets);
        a.arg(&d_lens);
        a.arg(&d_scores);
        a.arg(&alph_i);
        a.arg(&gap_open);
        a.arg(&gap_extend);
        a.arg(&n_pairs_i);
        a.arg(&max_t_i);
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

    /// GPU vs CPU parity on a hand-crafted batch covering each
    /// gapped::tests scenario: full match, gap in query, gap in
    /// target, affine coalescing, local trimming, no positive
    /// alignment, empty input.
    #[test]
    fn parity_against_cpu_smith_waterman_on_hand_crafted_batch() {
        if super::SwKernel::try_global().is_none() {
            eprintln!("SKIP parity test: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1];

        // Identical to CPU gapped::tests fixtures.
        let t_full: Vec<u8> = vec![0, 1, 2, 3, 0, 1]; // full match
        let t_query_insert: Vec<u8> = vec![0, 1, 2, 3]; // q has extra 0,1 vs t
        let t_target_insert: Vec<u8> = vec![0, 1, 0, 0, 2, 3, 0, 1]; // t has extra 0,0
        let t_unrelated: Vec<u8> = vec![0, 0, 0, 0, 0, 0]; // pairwise mismatch (with mismatch=-2 no positive)
        let t_local_trim: Vec<u8> = vec![1, 1, 0, 1, 2, 3, 1, 1]; // requires q+t match design
        let q_local: Vec<u8> = vec![1, 1, 0, 1, 2, 3, 1, 1]; // ditto

        // For the local-trim case use sequences with high-mismatch flanks
        // so the parity test exercises trimming.
        let scores_strict = identity_matrix(alphabet_size, 3, -10);
        let q_strict: Vec<u8> = vec![1, 1, 0, 1, 2, 3, 1, 1];
        let t_strict: Vec<u8> = vec![2, 2, 0, 1, 2, 3, 2, 2];

        // Each test pair: (query_used, target, scores_used, gap_open_used, gap_extend_used).
        let pairs: Vec<(Vec<u8>, Vec<u8>, &[i32], i32, i32)> = vec![
            (query.clone(), t_full, scores.as_slice(), gap_open, gap_extend),
            (query.clone(), t_query_insert, scores.as_slice(), gap_open, gap_extend),
            (query.clone(), t_target_insert, scores.as_slice(), gap_open, gap_extend),
            (query.clone(), t_unrelated, scores.as_slice(), gap_open, gap_extend),
            (q_strict, t_strict, scores_strict.as_slice(), gap_open, gap_extend),
            (q_local, t_local_trim, scores.as_slice(), gap_open, gap_extend),
        ];

        for (i, (q, t, s, go, ge)) in pairs.iter().enumerate() {
            // Single-pair GPU launch (the kernel is batched but per-pair
            // gap penalties + scores can vary, so we call it once per
            // pair for these mixed-config tests).
            let gpu = smith_waterman_score_batch_gpu(
                q,
                &[t.as_slice()],
                s,
                alphabet_size,
                *go,
                *ge,
            )
            .expect("GPU SW failed");
            assert_eq!(gpu.len(), 1);

            let cpu = smith_waterman(q, t, s, alphabet_size, *go, *ge);
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu[0].score, cpu_score,
                "pair {i}: CPU score {} != GPU score {}",
                cpu_score, gpu[0].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(
                    gpu[0].query_end, cpu_aln.query_end,
                    "pair {i}: query_end mismatch"
                );
                assert_eq!(
                    gpu[0].target_end, cpu_aln.target_end,
                    "pair {i}: target_end mismatch"
                );
            }
        }
    }

    #[test]
    fn empty_target_list_returns_empty_vec_without_dispatch() {
        let scores = identity_matrix(4, 3, -2);
        let result = smith_waterman_score_batch_gpu(
            &[0, 1, 2],
            &[],
            &scores,
            4,
            -5,
            -1,
        )
        .expect("ok");
        assert!(result.is_empty());
    }

    /// Larger PRNG-driven batch — same query, 50 random targets at
    /// varying lengths. Confirms per-thread scratch slicing works at
    /// real launch dimensions.
    #[test]
    fn parity_on_larger_batch() {
        if super::SwKernel::try_global().is_none() {
            eprintln!("SKIP larger batch: no GPU available");
            return;
        }

        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let gap_open = -5;
        let gap_extend = -1;

        let query: Vec<u8> =
            (0..40).map(|i| (i % alphabet_size) as u8).collect();

        let mut rng_state: u32 = 0x1234_5678;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state
        };

        let mut targets: Vec<Vec<u8>> = Vec::new();
        for _ in 0..50 {
            let len = 10 + (next() as usize % 50);
            let target: Vec<u8> = (0..len)
                .map(|_| (next() % alphabet_size as u32) as u8)
                .collect();
            targets.push(target);
        }
        let target_refs: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();

        let gpu = smith_waterman_score_batch_gpu(
            &query,
            &target_refs,
            &scores,
            alphabet_size,
            gap_open,
            gap_extend,
        )
        .expect("GPU SW failed");

        for (i, target) in targets.iter().enumerate() {
            let cpu = smith_waterman(
                &query,
                target,
                &scores,
                alphabet_size,
                gap_open,
                gap_extend,
            );
            let cpu_score = cpu.as_ref().map(|a| a.score).unwrap_or(0);
            assert_eq!(
                gpu[i].score, cpu_score,
                "pair {i}: CPU score {} != GPU score {}",
                cpu_score, gpu[i].score,
            );
            if let Some(cpu_aln) = cpu {
                assert_eq!(gpu[i].query_end, cpu_aln.query_end, "pair {i} qend");
                assert_eq!(gpu[i].target_end, cpu_aln.target_end, "pair {i} tend");
            }
        }
    }
}
