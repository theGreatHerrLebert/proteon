//! GPU dispatch for the k-mer diagonal-voting prefilter.
//!
//! CPU oracle: [`crate::prefilter::diagonal_prefilter`]. This produces the
//! **bit-exact** same `PrefilterHit` set (the computation is integer-only).
//!
//! Phase 1: single query, exact k-mers, in-memory [`KmerIndex`], uploaded per
//! call. The index-resident handle (upload once, reuse across a query batch —
//! the real throughput win) and similar-k-mer expansion are later phases.
//!
//! Voting avoids a from-scratch GPU sort: a GPU open-addressing hash table
//! counts `(seq_id, diagonal)` co-occurrences, then an `atomicMax`-into-
//! `best[seq_id]` pass picks each target's best diagonal with the CPU
//! tie-break (max count, then smallest diagonal). See `prefilter.cu` +
//! `devdocs/GPU_PREFILTER_PLAN.md`.

use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::GpuContext;
use crate::kmer::{KmerEncoder, KmerIndex, KmerLookup};
use crate::prefilter::{PrefilterHit, PrefilterOptions};

const KERNEL_SRC: &str = include_str!("prefilter.cu");
const DIAG_BITS: u32 = 20;

/// Compiled vote + reduce kernels, cached in a process-global `OnceLock`.
pub(crate) struct PrefilterKernels {
    vote: CudaFunction,
    reduce: CudaFunction,
    _module: Arc<CudaModule>,
}

static KERNELS: OnceLock<Option<PrefilterKernels>> = OnceLock::new();

impl PrefilterKernels {
    pub(crate) fn try_global() -> Option<&'static PrefilterKernels> {
        KERNELS
            .get_or_init(|| {
                let ctx = GpuContext::try_global()?;
                match Self::compile(ctx) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        eprintln!("[proteon-search-gpu] prefilter kernel compile failed: {e:#}");
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
            .with_context(|| "NVRTC compile of prefilter.cu failed")?;
        let module = ctx.cuda_context().load_module(ptx)?;
        let vote = module.load_function("prefilter_vote")?;
        let reduce = module.load_function("prefilter_reduce")?;
        Ok(Self {
            vote,
            reduce,
            _module: module,
        })
    }
}

/// Device-resident k-mer index for the GPU prefilter. Upload the (large) index
/// ONCE, then run many queries against it without re-sending it — the index
/// dwarfs each query, so reuse is the batching throughput win.
///
/// Holds only READ-ONLY device buffers + owned host metadata, so `&self`
/// methods are safe to call concurrently (each query allocates its own
/// per-query scratch — there is no shared mutable device state).
pub struct GpuPrefilterIndex {
    d_offsets: CudaSlice<u64>,
    d_seq_id: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    encoder: KmerEncoder,
    /// Host copy of `offsets` — sizes each query's hash table from posting-list
    /// lengths without a device round-trip.
    offsets_host: Vec<u64>,
    table_size: u64,
    /// `max_seq_id + 1` (guarded so it fits u32 for the kernel arg).
    best_len: usize,
}

impl GpuPrefilterIndex {
    /// Upload an in-memory index to the device (one-time HtoD of offsets +
    /// entries). The handle is then reusable across queries.
    pub fn upload(index: &KmerIndex) -> Result<Self> {
        // best[] is indexed by seq_id (NOT dense). seq_id == u32::MAX would make
        // best_len 2^32, which truncates to 0 in the kernel arg and would
        // silently reject every target (codex review) — reject it.
        let max_seq_id = index.entries.iter().map(|h| h.seq_id).max().unwrap_or(0);
        if max_seq_id == u32::MAX {
            return Err(anyhow!(
                "GPU prefilter: seq_id u32::MAX is unsupported (best[] length would overflow u32)"
            ));
        }

        let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
        let stream = ctx.cuda_context().new_stream()?;

        let entries_seq_id: Vec<u32> = index.entries.iter().map(|h| h.seq_id).collect();
        let entries_pos: Vec<u32> = index.entries.iter().map(|h| h.pos as u32).collect();
        // cudarc rejects zero-byte allocations; an empty index has no entries.
        // A 1-element stub the kernels never read (every posting list is empty,
        // so no query hash produces a hit) keeps the handle valid.
        let seq_for_gpu = if entries_seq_id.is_empty() {
            vec![0u32]
        } else {
            entries_seq_id
        };
        let pos_for_gpu = if entries_pos.is_empty() {
            vec![0u32]
        } else {
            entries_pos
        };

        let d_offsets = stream.clone_htod(&index.offsets)?;
        let d_seq_id = stream.clone_htod(&seq_for_gpu)?;
        let d_pos = stream.clone_htod(&pos_for_gpu)?;
        stream.synchronize()?;

        Ok(Self {
            d_offsets,
            d_seq_id,
            d_pos,
            encoder: index.encoder().clone(),
            offsets_host: index.offsets.clone(),
            table_size: index.encoder().table_size(),
            best_len: max_seq_id as usize + 1,
        })
    }

    /// Single-query prefilter against the resident index. Bit-exact equal to
    /// [`crate::prefilter::diagonal_prefilter`].
    pub fn prefilter(
        &self,
        query: &[u8],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        self.run_query(query, skip_idx, opts)
    }

    /// Many queries against the resident index, reusing the upload + one stream.
    /// `out[i]` is `queries[i]`'s hit list — identical to calling [`prefilter`]
    /// per query, but the index is sent once.
    ///
    /// [`prefilter`]: GpuPrefilterIndex::prefilter
    pub fn prefilter_batch(
        &self,
        queries: &[&[u8]],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<Vec<PrefilterHit>>> {
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            out.push(self.run_query(q, skip_idx, opts)?);
        }
        Ok(out)
    }

    /// Shared per-query body — the single source of truth so `prefilter` and
    /// `prefilter_batch` can't diverge from each other (or from Phase-1
    /// semantics). Allocates its OWN per-query scratch (hash table + best[] +
    /// error flag, all `alloc_zeros`), so there is no cross-query state; each
    /// query gets its own stream (the resident index is what's reused).
    fn run_query(
        &self,
        query: &[u8],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        let qlen = query.len();
        // diag in [-(qlen-1), 65535] (target pos is u16). Bias by qlen ⇒
        // diag_biased in [1, 65535+qlen]: never 0 (EMPTY sentinel), fits DIAG_BITS.
        let diag_max: u64 = (1u64 << DIAG_BITS) - 1;
        if (65535usize + qlen) as u64 > diag_max {
            return Err(anyhow!(
                "query length {qlen} too large for GPU prefilter (diagonal bias would overflow \
                 {DIAG_BITS}-bit packing); use the CPU prefilter"
            ));
        }
        let diag_bias = qlen as i32;

        // Out-of-range hashes have no posting list (mirrors lookup_hash) — drop
        // them so the kernel never indexes `offsets` out of bounds (codex).
        let kmers: Vec<(usize, u64)> = self
            .encoder
            .iter_kmers(query, skip_idx)
            .filter(|&(_, h)| h < self.table_size)
            .collect();
        if kmers.is_empty() {
            return Ok(Vec::new());
        }

        let mut total_hits: u64 = 0;
        for &(_, h) in &kmers {
            let h = h as usize;
            total_hits += self.offsets_host[h + 1] - self.offsets_host[h];
        }
        if total_hits == 0 {
            return Ok(Vec::new());
        }

        // Hash table capacity: next pow2 >= 2*total_hits (distinct keys <=
        // total_hits ⇒ never fills). Guard the arithmetic.
        let cap_u64 = total_hits
            .checked_mul(2)
            .and_then(|v| v.checked_next_power_of_two())
            .ok_or_else(|| {
                anyhow!("GPU prefilter: hash table size overflow ({total_hits} hits)")
            })?;
        if cap_u64 > u32::MAX as u64 {
            return Err(anyhow!(
                "GPU prefilter: {total_hits} hits exceed the in-memory table cap"
            ));
        }
        let table_cap = cap_u64 as usize;
        let table_mask = (table_cap - 1) as u32;

        let kmer_qpos: Vec<i32> = kmers.iter().map(|&(p, _)| p as i32).collect();
        let kmer_hash: Vec<u64> = kmers.iter().map(|&(_, h)| h).collect();
        let n_kmers = kmers.len();

        // Acquire kernels + a stream only now (a no-hit query returned above
        // without touching the GPU). Each query uses its own stream; the
        // resident index buffers are read-only and shared.
        let kernels = PrefilterKernels::try_global()
            .ok_or_else(|| anyhow!("GPU prefilter kernels unavailable"))?;
        let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
        let stream = ctx.cuda_context().new_stream()?;

        let d_qpos = stream.clone_htod(&kmer_qpos)?;
        let d_hash = stream.clone_htod(&kmer_hash)?;
        // Fresh per-query scratch (zeroed: EMPTY=0, zero counts, best=0=no-vote).
        let mut d_keys = stream.alloc_zeros::<u64>(table_cap)?;
        let mut d_counts = stream.alloc_zeros::<u32>(table_cap)?;
        let mut d_best = stream.alloc_zeros::<u64>(self.best_len)?;
        let mut d_err = stream.alloc_zeros::<u32>(1)?;

        // Kernel A: one block per k-mer; threads stride its posting list.
        {
            let cfg = LaunchConfig {
                grid_dim: (n_kmers as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut a = stream.launch_builder(&kernels.vote);
            let n_kmers_i = n_kmers as i32;
            a.arg(&self.d_offsets);
            a.arg(&self.d_seq_id);
            a.arg(&self.d_pos);
            a.arg(&d_qpos);
            a.arg(&d_hash);
            a.arg(&n_kmers_i);
            a.arg(&diag_bias);
            a.arg(&mut d_keys);
            a.arg(&mut d_counts);
            a.arg(&table_mask);
            a.arg(&mut d_err);
            unsafe { a.launch(cfg)? };
        }

        // Kernel B: one thread per hash slot → atomicMax into best[seq].
        {
            let cfg = LaunchConfig {
                grid_dim: ((table_cap as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut a = stream.launch_builder(&kernels.reduce);
            let table_cap_u = table_cap as u32;
            let best_len_u = self.best_len as u32;
            a.arg(&d_keys);
            a.arg(&d_counts);
            a.arg(&table_cap_u);
            a.arg(&diag_max);
            a.arg(&mut d_best);
            a.arg(&best_len_u);
            unsafe { a.launch(cfg)? };
        }
        stream.synchronize()?;

        let err = stream.clone_dtoh(&d_err)?;
        if err[0] != 0 {
            return Err(anyhow!(
                "GPU prefilter: hash table probe exhausted (capacity bug)"
            ));
        }
        let best = stream.clone_dtoh(&d_best)?;

        // Decode best[] → PrefilterHit; same filter/sort/truncate as the CPU path.
        let mut hits: Vec<PrefilterHit> = Vec::new();
        for (seq, &packed) in best.iter().enumerate() {
            if packed == 0 {
                continue; // no vote
            }
            let count = (packed >> DIAG_BITS) as u32;
            let diag_b = diag_max - (packed & diag_max);
            let diagonal = diag_b as i64 - diag_bias as i64;
            let seq_id = seq as u32;
            if count >= opts.score_threshold && opts.exclude_self != Some(seq_id) {
                hits.push(PrefilterHit {
                    seq_id,
                    diagonal_score: count,
                    best_diagonal: diagonal as i32,
                });
            }
        }
        hits.sort_by(|a, b| {
            b.diagonal_score
                .cmp(&a.diagonal_score)
                .then_with(|| a.seq_id.cmp(&b.seq_id))
        });
        if let Some(limit) = opts.max_hits {
            hits.truncate(limit);
        }
        Ok(hits)
    }
}

/// Convenience: upload + single query. For repeated queries build a
/// [`GpuPrefilterIndex`] once and call [`GpuPrefilterIndex::prefilter`] /
/// [`GpuPrefilterIndex::prefilter_batch`] to skip re-uploading the index.
/// Bit-exact equal to [`crate::prefilter::diagonal_prefilter`].
pub fn diagonal_prefilter_gpu(
    index: &KmerIndex,
    query: &[u8],
    skip_idx: u8,
    opts: &PrefilterOptions,
) -> Result<Vec<PrefilterHit>> {
    GpuPrefilterIndex::upload(index)?.prefilter(query, skip_idx, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmer::{KmerEncoder, KmerIndex};
    use crate::prefilter::diagonal_prefilter;

    fn skip() -> bool {
        if PrefilterKernels::try_global().is_none() {
            eprintln!("SKIP GPU prefilter parity: no GPU available");
            true
        } else {
            false
        }
    }

    /// GPU output must equal the CPU oracle bit-for-bit.
    fn assert_parity(index: &KmerIndex, query: &[u8], skip_idx: u8, opts: &PrefilterOptions) {
        let cpu = diagonal_prefilter(index, query, skip_idx, opts);
        let gpu = diagonal_prefilter_gpu(index, query, skip_idx, opts).expect("GPU prefilter");
        assert_eq!(cpu, gpu, "CPU vs GPU prefilter disagree (query {query:?})");
    }

    fn small_index() -> KmerIndex {
        // Sparse seq_ids (10, 20, 30) — exercises the max_seq_id sizing.
        let enc = KmerEncoder::new(4, 2);
        let a = vec![0u8, 1, 2, 3]; // ACGT
        let b = vec![1u8, 2, 3, 0]; // CGTA
        let c = vec![0u8, 0, 0, 0]; // AAAA
        KmerIndex::build(
            enc,
            [
                (10u32, a.as_slice()),
                (20u32, b.as_slice()),
                (30u32, c.as_slice()),
            ],
            99,
        )
        .unwrap()
    }

    #[test]
    fn parity_basic_sparse_ids() {
        if skip() {
            return;
        }
        let idx = small_index();
        assert_parity(&idx, &[0, 1, 2, 3], 99, &PrefilterOptions::default());
    }

    #[test]
    fn parity_best_diagonal_and_count_ties() {
        if skip() {
            return;
        }
        // ACGTAC: AC occurs at 0 and 4 ⇒ multiple diagonals, incl. a count tie
        // structure that stresses the smallest-diagonal tie-break.
        let enc = KmerEncoder::new(4, 2);
        let seq = vec![0u8, 1, 2, 3, 0, 1];
        let idx = KmerIndex::build(enc, [(1u32, seq.as_slice())], 99).unwrap();
        assert_parity(&idx, &seq, 99, &PrefilterOptions::default());
    }

    #[test]
    fn parity_with_options() {
        if skip() {
            return;
        }
        let idx = small_index();
        let q = vec![0u8, 1, 2, 3];
        assert_parity(
            &idx,
            &q,
            99,
            &PrefilterOptions {
                score_threshold: 3,
                ..Default::default()
            },
        );
        assert_parity(
            &idx,
            &q,
            99,
            &PrefilterOptions {
                exclude_self: Some(10),
                ..Default::default()
            },
        );
        assert_parity(
            &idx,
            &q,
            99,
            &PrefilterOptions {
                max_hits: Some(1),
                ..Default::default()
            },
        );
    }

    #[test]
    fn parity_empty_result() {
        if skip() {
            return;
        }
        // A query whose k-mers (TA, AT) are absent from the AAAA-heavy index
        // built below ⇒ no hits.
        let enc = KmerEncoder::new(4, 2);
        let only_a = vec![0u8, 0, 0, 0];
        let idx = KmerIndex::build(enc, [(0u32, only_a.as_slice())], 99).unwrap();
        assert_parity(&idx, &[3u8, 0, 3, 0], 99, &PrefilterOptions::default());
    }

    /// Larger pseudo-random index + dense seq_ids — exercises real launch
    /// dimensions and the hash table at scale, still CPU-oracled.
    #[test]
    fn parity_larger_random_index() {
        if skip() {
            return;
        }
        let alphabet = 6usize;
        let enc = KmerEncoder::new(alphabet as u32, 3);
        let mut rng: u32 = 0x1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let seqs: Vec<Vec<u8>> = (0..40)
            .map(|_| {
                let len = 12 + (next() as usize % 40);
                (0..len)
                    .map(|_| (next() as usize % alphabet) as u8)
                    .collect()
            })
            .collect();
        let corpus: Vec<(u32, &[u8])> = seqs
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_slice()))
            .collect();
        let idx = KmerIndex::build(enc, corpus, 99).unwrap();

        for _ in 0..8 {
            let qlen = 12 + (next() as usize % 40);
            let query: Vec<u8> = (0..qlen)
                .map(|_| (next() as usize % alphabet) as u8)
                .collect();
            assert_parity(&idx, &query, 99, &PrefilterOptions::default());
            assert_parity(
                &idx,
                &query,
                99,
                &PrefilterOptions {
                    score_threshold: 2,
                    max_hits: Some(5),
                    ..Default::default()
                },
            );
        }
    }

    // --- Phase 2: resident handle + batching ---

    /// Batch result[i] must equal the CPU oracle for queries[i].
    fn assert_batch_parity(
        index: &KmerIndex,
        queries: &[&[u8]],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) {
        let handle = GpuPrefilterIndex::upload(index).expect("upload");
        let gpu = handle
            .prefilter_batch(queries, skip_idx, opts)
            .expect("batch");
        assert_eq!(gpu.len(), queries.len());
        for (i, q) in queries.iter().enumerate() {
            let cpu = diagonal_prefilter(index, q, skip_idx, opts);
            assert_eq!(cpu, gpu[i], "batch query {i} ({q:?}) mismatch");
        }
    }

    #[test]
    fn batch_mixed_hit_empty_hit_lower() {
        if skip() {
            return;
        }
        let idx = small_index();
        // ACGT hits 10 (score 3) & 20 (score 2); TTTT is absent (empty); AC hits
        // ONLY 10 with score 1 — a query reusing the same target at a LOWER score
        // right after a high-score one (the stale-state catch). Then ACGT again.
        let q_hit: &[u8] = &[0, 1, 2, 3];
        let q_empty: &[u8] = &[3, 3, 3, 3];
        let q_lower: &[u8] = &[0, 1];
        assert_batch_parity(
            &idx,
            &[q_hit, q_empty, q_lower, q_hit],
            99,
            &PrefilterOptions::default(),
        );
    }

    #[test]
    fn batch_empty_then_hit() {
        if skip() {
            return;
        }
        let idx = small_index();
        // empty first ⇒ a clears-skipped-after-early-return bug would surface.
        assert_batch_parity(
            &idx,
            &[&[3, 3, 3, 3], &[0, 1, 2, 3]],
            99,
            &PrefilterOptions::default(),
        );
    }

    #[test]
    fn batch_with_options() {
        if skip() {
            return;
        }
        let idx = small_index();
        assert_batch_parity(
            &idx,
            &[&[0u8, 1, 2, 3], &[0u8, 1]],
            99,
            &PrefilterOptions {
                score_threshold: 2,
                max_hits: Some(1),
                ..Default::default()
            },
        );
    }

    #[test]
    fn resident_handle_survives_source_drop() {
        if skip() {
            return;
        }
        // Upload, then drop the source KmerIndex — the handle owns its device
        // buffers + cloned encoder/offsets, so it must keep working.
        let handle = {
            let idx = small_index();
            GpuPrefilterIndex::upload(&idx).expect("upload")
        };
        let idx2 = small_index();
        let q = vec![0u8, 1, 2, 3];
        let cpu = diagonal_prefilter(&idx2, &q, 99, &PrefilterOptions::default());
        let gpu = handle
            .prefilter(&q, 99, &PrefilterOptions::default())
            .expect("prefilter");
        assert_eq!(cpu, gpu);
    }
}
