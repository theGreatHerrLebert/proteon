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
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::GpuContext;
use crate::kmer::{KmerEncoder, KmerIndex, KmerLookup};
use crate::kmer_generator::for_each_similar_kmer;
use crate::prefilter::{PrefilterHit, PrefilterOptions, SimilarityConfig};

const KERNEL_SRC: &str = include_str!("prefilter.cu");
const DIAG_BITS: u32 = 20;

/// Compiled vote + reduce kernels, cached in a process-global `OnceLock`.
pub(crate) struct PrefilterKernels {
    vote: CudaFunction,
    reduce_seqhash: CudaFunction,
    compact: CudaFunction,
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
        let reduce_seqhash = module.load_function("prefilter_reduce_seqhash")?;
        let compact = module.load_function("prefilter_compact")?;
        Ok(Self {
            vote,
            reduce_seqhash,
            compact,
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
    /// `dense_to_orig[d]` is the original `seq_id` for dense target index `d`.
    /// The uploaded `entries_seq_id` use dense `[0, n_targets)` indices (so the
    /// seq-keyed reduction never depends on the external id range, which may be
    /// sparse — codex); decode maps the compacted output back through this.
    dense_to_orig: Vec<u32>,
}

/// Reusable per-call device scratch for the voting pipeline: a stream plus all
/// the growable device buffers, so a batch of queries reuses ONE allocation set
/// (and one stream) instead of `cudaMalloc`-ing — and creating a stream — per
/// query. That per-query alloc/stream churn was the fixed-overhead wall the
/// benchmark exposed (P2).
///
/// Grow-not-shrink: each buffer is reallocated only when a query needs more than
/// its current capacity (mirrors `DPWorkspace`). The scratch is CALLER-owned, so
/// the resident [`GpuPrefilterIndex`] stays `&self` and concurrency-safe — there
/// is no shared mutable device state on the handle (claudex: a `Mutex` on
/// `&self` would silently serialize concurrent callers).
///
/// Hold one across many single-query [`GpuPrefilterIndex::prefilter_with`] calls
/// (e.g. an engine reusing it across `search()` calls) to get the same
/// amortization the batch methods get internally — a fresh scratch per query
/// pays the alloc/stream cost and pushes the GPU-vs-CPU crossover much higher.
pub struct PrefilterScratch {
    stream: Arc<CudaStream>,
    d_qpos: CudaSlice<i32>,
    d_hash: CudaSlice<u64>,
    d_keys: CudaSlice<u64>,
    d_counts: CudaSlice<u32>,
    d_best_keys: CudaSlice<u64>,
    d_best_vals: CudaSlice<u64>,
    d_out_seq: CudaSlice<u32>,
    d_out_packed: CudaSlice<u64>,
    /// Compaction counter, len 1. Zeroed per query.
    d_out_count: CudaSlice<u32>,
    /// `[0]` vote-probe, `[1]` seq-probe, `[2]` output-overflow. Zeroed per query.
    d_err: CudaSlice<u32>,
}

impl PrefilterScratch {
    /// Allocate the scratch (one stream + stub buffers grown on first use).
    pub fn new() -> Result<Self> {
        let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;
        let stream = ctx.cuda_context().new_stream()?;
        // cudarc rejects zero-byte allocations, so start the growable buffers at
        // a 1-element stub; `ensure` grows them to the first query's needs.
        Ok(Self {
            d_qpos: unsafe { stream.alloc::<i32>(1) }?,
            d_hash: unsafe { stream.alloc::<u64>(1) }?,
            d_keys: unsafe { stream.alloc::<u64>(1) }?,
            d_counts: unsafe { stream.alloc::<u32>(1) }?,
            d_best_keys: unsafe { stream.alloc::<u64>(1) }?,
            d_best_vals: unsafe { stream.alloc::<u64>(1) }?,
            d_out_seq: unsafe { stream.alloc::<u32>(1) }?,
            d_out_packed: unsafe { stream.alloc::<u64>(1) }?,
            d_out_count: stream.alloc_zeros::<u32>(1)?,
            d_err: stream.alloc_zeros::<u32>(3)?,
            stream,
        })
    }
}

/// Grow `buf` to at least `needed` elements (never shrinks). The realloc drops
/// the old buffer's contents — callers `memset`/`memcpy` the used range before
/// the kernels read it, so uninitialised growth is safe.
fn ensure_cap<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<T>,
    needed: usize,
) -> Result<()> {
    if buf.len() < needed {
        *buf = unsafe { stream.alloc::<T>(needed) }?;
    }
    Ok(())
}

impl GpuPrefilterIndex {
    /// Upload an in-memory index to the device (one-time HtoD of offsets +
    /// entries). The handle is then reusable across queries.
    pub fn upload(index: &KmerIndex) -> Result<Self> {
        let ctx = GpuContext::try_global().ok_or_else(|| anyhow!("GPU context unavailable"))?;

        // Verify the kernels actually compile (NVRTC) BEFORE the (multi-GB)
        // upload. On an unsupported arch the compile fails permanently; uploading
        // first would leave a large resident index cached but unusable, with every
        // query retrying + logging and the wasted device memory starving downstream
        // GPU alignment (codex). `None` ⇒ Err ⇒ the caller's CPU fallback.
        if PrefilterKernels::try_global().is_none() {
            return Err(anyhow!(
                "GPU prefilter kernels unavailable (NVRTC compile failed); using CPU prefilter"
            ));
        }

        // Resident-only capacity pre-check FIRST — reject an archive-scale index
        // (offsets 8 B/slot + entries 8 B) BEFORE scanning every posting to build
        // the dense map, so a too-big index falls back to CPU cheaply (codex).
        let total_mem = ctx.cuda_context().total_mem().unwrap_or(0) as u64;
        let resident_bytes = (index.offsets.len() as u64)
            .saturating_mul(8)
            .saturating_add((index.entries.len() as u64).saturating_mul(8));
        if total_mem > 0 && resident_bytes > total_mem / 2 {
            return Err(anyhow!(
                "GPU prefilter: resident index ~{resident_bytes} B exceeds half of {total_mem} B \
                 device memory; using CPU prefilter"
            ));
        }

        // Dense-remap seq_ids → [0, n_targets) so the kernels vote in a compact
        // index space (the seq-keyed reduction + decode map back via
        // `dense_to_orig`); a sparse huge external id costs nothing. Every host
        // allocation (the map AND its key vec) is fallible so a pathological index
        // falls back to CPU instead of OOM-aborting on infallible growth.
        let mut dense_to_orig: Vec<u32> = Vec::new();
        let mut orig_to_dense: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        for h in &index.entries {
            if orig_to_dense.contains_key(&h.seq_id) {
                continue;
            }
            let d = dense_to_orig.len();
            if d >= u32::MAX as usize {
                return Err(anyhow!(
                    "GPU prefilter: > u32::MAX distinct targets; using CPU prefilter"
                ));
            }
            // Reserve fallibly BEFORE inserting so neither the vec nor the map
            // grows via an infallible (abort-on-OOM) allocation.
            dense_to_orig
                .try_reserve(1)
                .and_then(|()| orig_to_dense.try_reserve(1))
                .map_err(|_| anyhow!("GPU prefilter: host OOM building target map; CPU"))?;
            dense_to_orig.push(h.seq_id);
            orig_to_dense.insert(h.seq_id, d as u32);
        }
        // No second capacity guard: the resident pre-check above already bounds
        // the only upload-time buffers (offsets + entries). Per-query scratch
        // (vote + seq-hash tables + output) is sized by QUERY work, allocated and
        // freed per query — a too-large query's alloc fails → Err → CPU fallback,
        // never a fixed target-count tax at upload (codex: P1 removed the dense
        // best[] this guard used to reserve for).

        // Host SoA via FALLIBLE allocation (entries use DENSE seq ids).
        let n = index.entries.len();
        let mut entries_seq_id: Vec<u32> = Vec::new();
        let mut entries_pos: Vec<u32> = Vec::new();
        let mut offsets_host: Vec<u64> = Vec::new();
        entries_seq_id
            .try_reserve(n.max(1))
            .and_then(|()| entries_pos.try_reserve(n.max(1)))
            .and_then(|()| offsets_host.try_reserve(index.offsets.len()))
            .map_err(|_| anyhow!("GPU prefilter: host OOM allocating upload buffers; CPU"))?;
        for h in &index.entries {
            entries_seq_id.push(orig_to_dense[&h.seq_id]);
            entries_pos.push(h.pos as u32);
        }
        offsets_host.extend_from_slice(&index.offsets);
        // cudarc rejects zero-byte allocations; an empty index has no entries.
        // A 1-element stub the kernels never read (every posting list is empty)
        // keeps the handle valid.
        if entries_seq_id.is_empty() {
            entries_seq_id.push(0);
            entries_pos.push(0);
        }

        let stream = ctx.cuda_context().new_stream()?;
        let d_offsets = stream.clone_htod(&offsets_host)?;
        let d_seq_id = stream.clone_htod(&entries_seq_id)?;
        let d_pos = stream.clone_htod(&entries_pos)?;
        stream.synchronize()?;

        Ok(Self {
            d_offsets,
            d_seq_id,
            d_pos,
            encoder: index.encoder().clone(),
            offsets_host,
            table_size: index.encoder().table_size(),
            dense_to_orig,
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
        let mut scratch = PrefilterScratch::new()?;
        self.run_query(&mut scratch, query, skip_idx, opts)
    }

    /// Single-query prefilter reusing a CALLER-owned [`PrefilterScratch`] — hold
    /// one across many calls (e.g. per `search()`) to amortize the device-buffer
    /// allocation + stream creation a fresh-scratch [`prefilter`] would repay
    /// every call. Bit-exact equal to [`crate::prefilter::diagonal_prefilter`].
    ///
    /// [`prefilter`]: GpuPrefilterIndex::prefilter
    pub fn prefilter_with(
        &self,
        scratch: &mut PrefilterScratch,
        query: &[u8],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        self.run_query(scratch, query, skip_idx, opts)
    }

    /// Many queries against the resident index, reusing the upload + ONE scratch
    /// allocation set + stream across the whole batch (the per-query
    /// malloc/stream churn was the fixed-overhead wall — P2). `out[i]` is
    /// `queries[i]`'s hit list — identical to calling [`prefilter`] per query.
    ///
    /// [`prefilter`]: GpuPrefilterIndex::prefilter
    pub fn prefilter_batch(
        &self,
        queries: &[&[u8]],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<Vec<PrefilterHit>>> {
        let mut scratch = PrefilterScratch::new()?;
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            out.push(self.run_query(&mut scratch, q, skip_idx, opts)?);
        }
        Ok(out)
    }

    /// Sensitive single-query prefilter: each query k-mer is expanded into every
    /// similar k-mer scoring `>= similarity.threshold` before voting. Bit-exact
    /// equal to [`crate::prefilter::diagonal_prefilter_sensitive`].
    pub fn prefilter_sensitive(
        &self,
        query: &[u8],
        skip_idx: u8,
        similarity: &SimilarityConfig<'_>,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        let mut scratch = PrefilterScratch::new()?;
        self.prefilter_sensitive_with(&mut scratch, query, skip_idx, similarity, opts)
    }

    /// Sensitive single-query prefilter reusing a CALLER-owned
    /// [`PrefilterScratch`] (see [`prefilter_with`]). Bit-exact equal to
    /// [`crate::prefilter::diagonal_prefilter_sensitive`].
    ///
    /// [`prefilter_with`]: GpuPrefilterIndex::prefilter_with
    pub fn prefilter_sensitive_with(
        &self,
        scratch: &mut PrefilterScratch,
        query: &[u8],
        skip_idx: u8,
        similarity: &SimilarityConfig<'_>,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        let diag_bias = self.validate_diag_bias(query.len())?;
        let kmers = self.build_sensitive_kmers(query, skip_idx, similarity)?;
        self.vote_and_reduce(scratch, &kmers, diag_bias, opts)
    }

    /// Many sensitive queries against the resident index (one scratch + stream
    /// reused across the batch, like [`prefilter_batch`]).
    ///
    /// [`prefilter_batch`]: GpuPrefilterIndex::prefilter_batch
    pub fn prefilter_sensitive_batch(
        &self,
        queries: &[&[u8]],
        skip_idx: u8,
        similarity: &SimilarityConfig<'_>,
        opts: &PrefilterOptions,
    ) -> Result<Vec<Vec<PrefilterHit>>> {
        let mut scratch = PrefilterScratch::new()?;
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            let diag_bias = self.validate_diag_bias(q.len())?;
            let kmers = self.build_sensitive_kmers(q, skip_idx, similarity)?;
            out.push(self.vote_and_reduce(&mut scratch, &kmers, diag_bias, opts)?);
        }
        Ok(out)
    }

    /// Expand a query into its similar-k-mer `(q_pos, hash)` list — the only
    /// thing the sensitive path does differently from exact (the voting is
    /// identical). Mirrors `diagonal_prefilter_sensitive`'s window loop: skip
    /// X-windows; for each window, every neighbor scoring `>= threshold`.
    ///
    /// Keeps ONLY in-range hashes whose posting list is **non-empty**: an empty
    /// posting contributes no votes (the CPU `for_each_hit` is a no-op on it), so
    /// dropping it is parity-preserving AND bounds the RETAINED list to neighbours
    /// actually present in the index.
    ///
    /// Uses the STREAMING generator (`for_each_similar_kmer`) so neighbours are
    /// filtered one at a time and the full `alphabet^k` per-window set is NEVER
    /// materialised — even a pathological threshold only ever holds the retained
    /// (in-index) hashes (codex). A hard cap on the retained list is the backstop.
    fn build_sensitive_kmers(
        &self,
        query: &[u8],
        skip_idx: u8,
        similarity: &SimilarityConfig<'_>,
    ) -> Result<Vec<(usize, u64)>> {
        let k = self.encoder.kmer_size();
        let mut kmers: Vec<(usize, u64)> = Vec::new();
        let mut overflow = false;
        for q_pos in 0..query.len().saturating_sub(k - 1) {
            let window = &query[q_pos..q_pos + k];
            if window.contains(&skip_idx) {
                continue;
            }
            for_each_similar_kmer(
                &self.encoder,
                window,
                similarity.scores,
                similarity.threshold,
                |h, _| {
                    if h >= self.table_size || overflow {
                        return;
                    }
                    let hu = h as usize;
                    if self.offsets_host[hu + 1] > self.offsets_host[hu] {
                        kmers.push((q_pos, h));
                        if kmers.len() > i32::MAX as usize {
                            overflow = true;
                        }
                    }
                },
            );
            if overflow {
                return Err(anyhow!(
                    "GPU sensitive prefilter: expanded k-mer list exceeds the launch-grid width"
                ));
            }
        }
        Ok(kmers)
    }

    /// Validate the query length against the `DIAG_BITS` diagonal packing and
    /// return the diagonal bias. Called BEFORE any k-mer build/expansion so an
    /// oversized query fails fast (claudex) — sensitive expansion is expensive.
    ///
    /// diag in [-(qlen-1), 65535] (target pos is u16). Bias by qlen ⇒
    /// diag_biased in [1, 65535+qlen]: never 0 (EMPTY sentinel), fits DIAG_BITS.
    fn validate_diag_bias(&self, qlen: usize) -> Result<i32> {
        let diag_max: u64 = (1u64 << DIAG_BITS) - 1;
        if (65535usize + qlen) as u64 > diag_max {
            return Err(anyhow!(
                "query length {qlen} too large for GPU prefilter (diagonal bias would overflow \
                 {DIAG_BITS}-bit packing); use the CPU prefilter"
            ));
        }
        Ok(qlen as i32)
    }

    /// Exact-match per-query body. Builds the exact k-mer list, then votes via
    /// the shared [`vote_and_reduce`](Self::vote_and_reduce) core.
    fn run_query(
        &self,
        scratch: &mut PrefilterScratch,
        query: &[u8],
        skip_idx: u8,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        let diag_bias = self.validate_diag_bias(query.len())?;
        // Out-of-range hashes have no posting list (mirrors lookup_hash) — drop
        // them so the kernel never indexes `offsets` out of bounds (codex).
        let kmers: Vec<(usize, u64)> = self
            .encoder
            .iter_kmers(query, skip_idx)
            .filter(|&(_, h)| h < self.table_size)
            .collect();
        self.vote_and_reduce(scratch, &kmers, diag_bias, opts)
    }

    /// Shared GPU voting core for a prebuilt `(q_pos, hash)` list — exact OR
    /// similar-k-mer-expanded; the kernels are identical for both. Reuses the
    /// caller-provided [`PrefilterScratch`] (grown as needed, zeroed per query),
    /// so a batch reuses one allocation set + stream. The resident index buffers
    /// are read-only `&self`.
    fn vote_and_reduce(
        &self,
        scratch: &mut PrefilterScratch,
        kmers: &[(usize, u64)],
        diag_bias: i32,
        opts: &PrefilterOptions,
    ) -> Result<Vec<PrefilterHit>> {
        let diag_max: u64 = (1u64 << DIAG_BITS) - 1;
        if kmers.is_empty() {
            return Ok(Vec::new());
        }
        // The expanded list length is the vote grid dim + an i32 kernel arg, and
        // is NOT bounded by `total_hits` (most neighbor hashes may miss) — guard
        // it independently (covers the i32 arg AND the CUDA grid-x limit). claudex.
        let n_kmers = kmers.len();
        if n_kmers as u64 > i32::MAX as u64 {
            return Err(anyhow!(
                "GPU prefilter: {n_kmers} (expanded) k-mers exceed the launch-grid width"
            ));
        }

        let mut total_hits: u64 = 0;
        for &(_, h) in kmers {
            let h = h as usize;
            let posting = self.offsets_host[h + 1] - self.offsets_host[h];
            total_hits = total_hits
                .checked_add(posting)
                .ok_or_else(|| anyhow!("GPU prefilter: total hit count overflow"))?;
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
        // u32 arithmetic / kernel-width guard (not a memory cap — a table this
        // large simply fails to allocate, which is correct, not silent).
        if cap_u64 > u32::MAX as u64 {
            return Err(anyhow!(
                "GPU prefilter: {total_hits} hits exceed the addressable table width (u32)"
            ));
        }
        let table_cap = cap_u64 as usize;
        let table_mask = (table_cap - 1) as u32;

        let kmer_qpos: Vec<i32> = kmers.iter().map(|&(p, _)| p as i32).collect();
        let kmer_hash: Vec<u64> = kmers.iter().map(|&(_, h)| h).collect();

        let kernels = PrefilterKernels::try_global()
            .ok_or_else(|| anyhow!("GPU prefilter kernels unavailable"))?;

        // Output capacity: at most one hit per DISTINCT target, and distinct
        // targets <= total_hits (every posting could be a different seq). The
        // table_cap guard already proved 2*total_hits <= u32::MAX ⇒ total_hits
        // fits u32 and is a safe, target-count-INDEPENDENT bound (claudex §2b).
        let out_cap = total_hits as usize;

        // Reuse the caller's scratch: grow buffers to this query's needs, then
        // zero only the ranges the kernels touch (the malloc is amortized across
        // the batch; P2). Clone the stream Arc so `scratch`'s buffer fields can
        // be borrowed disjointly while the stream drives the launches.
        let stream = scratch.stream.clone();
        ensure_cap(&stream, &mut scratch.d_qpos, n_kmers)?;
        ensure_cap(&stream, &mut scratch.d_hash, n_kmers)?;
        ensure_cap(&stream, &mut scratch.d_keys, table_cap)?;
        ensure_cap(&stream, &mut scratch.d_counts, table_cap)?;
        ensure_cap(&stream, &mut scratch.d_best_keys, table_cap)?;
        ensure_cap(&stream, &mut scratch.d_best_vals, table_cap)?;
        ensure_cap(&stream, &mut scratch.d_out_seq, out_cap)?;
        ensure_cap(&stream, &mut scratch.d_out_packed, out_cap)?;

        // Zero only the USED ranges. Vote table: EMPTY=0 keys, 0 counts.
        // Seq-hash table: EMPTY=0 keys, 0 (= "no vote") packed vals. Counters +
        // error flags reset. out_seq/out_packed need NO zeroing — compaction
        // writes by index and the host reads only [0, n_out).
        stream.memset_zeros(&mut scratch.d_keys.slice_mut(0..table_cap))?;
        stream.memset_zeros(&mut scratch.d_counts.slice_mut(0..table_cap))?;
        stream.memset_zeros(&mut scratch.d_best_keys.slice_mut(0..table_cap))?;
        stream.memset_zeros(&mut scratch.d_best_vals.slice_mut(0..table_cap))?;
        stream.memset_zeros(&mut scratch.d_out_count)?;
        stream.memset_zeros(&mut scratch.d_err)?;

        // Upload the query's k-mer list into the (reused) input buffers.
        stream.memcpy_htod(&kmer_qpos, &mut scratch.d_qpos)?;
        stream.memcpy_htod(&kmer_hash, &mut scratch.d_hash)?;

        // Kernel A: one block per k-mer; threads stride its posting list,
        // insert-or-increment into the (seq, diag) vote table.
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
            a.arg(&scratch.d_qpos);
            a.arg(&scratch.d_hash);
            a.arg(&n_kmers_i);
            a.arg(&diag_bias);
            a.arg(&mut scratch.d_keys);
            a.arg(&mut scratch.d_counts);
            a.arg(&table_mask);
            a.arg(&mut scratch.d_err); // slot [0]
            unsafe { a.launch(cfg)? };
        }

        // Kernel B: one thread per VOTE slot → insert-or-max into the seq-keyed
        // hash table (replaces the old dense best[#targets]).
        {
            let cfg = LaunchConfig {
                grid_dim: ((table_cap as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut a = stream.launch_builder(&kernels.reduce_seqhash);
            let table_cap_u = table_cap as u32;
            a.arg(&scratch.d_keys);
            a.arg(&scratch.d_counts);
            a.arg(&table_cap_u);
            a.arg(&diag_max);
            a.arg(&mut scratch.d_best_keys);
            a.arg(&mut scratch.d_best_vals);
            a.arg(&table_mask); // best cap == vote cap ⇒ same mask
            let d_err_best = scratch.d_err.slice(1..2);
            a.arg(&d_err_best); // slot [1]
            unsafe { a.launch(cfg)? };
        }

        // Kernel C: one thread per SEQ-HASH slot → stream-compact occupied
        // (seq, packed) into the dense output list. Host copies back O(hits).
        {
            let cfg = LaunchConfig {
                grid_dim: ((table_cap as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut a = stream.launch_builder(&kernels.compact);
            let best_size_u = table_cap as u32;
            let out_cap_u = out_cap as u32;
            a.arg(&scratch.d_best_keys);
            a.arg(&scratch.d_best_vals);
            a.arg(&best_size_u);
            a.arg(&mut scratch.d_out_seq);
            a.arg(&mut scratch.d_out_packed);
            a.arg(&out_cap_u);
            a.arg(&mut scratch.d_out_count);
            let d_err_out = scratch.d_err.slice(2..3);
            a.arg(&d_err_out); // slot [2]
            unsafe { a.launch(cfg)? };
        }
        stream.synchronize()?;

        let err = stream.clone_dtoh(&scratch.d_err)?;
        if err[0] != 0 {
            return Err(anyhow!(
                "GPU prefilter: vote hash table probe exhausted (capacity bug)"
            ));
        }
        if err[1] != 0 {
            return Err(anyhow!(
                "GPU prefilter: seq hash table probe exhausted (capacity bug)"
            ));
        }
        if err[2] != 0 {
            return Err(anyhow!(
                "GPU prefilter: compaction output overflow (capacity bug)"
            ));
        }

        let n_out = (stream.clone_dtoh(&scratch.d_out_count)?[0] as usize).min(out_cap);
        // Copy back only the compacted prefix (O(hits)); the buffers may be
        // larger than `n_out` after grow-not-shrink, so slice before DTOH.
        let out_seq = stream.clone_dtoh(&scratch.d_out_seq.slice(0..n_out))?;
        let out_packed = stream.clone_dtoh(&scratch.d_out_packed.slice(0..n_out))?;

        // Decode the compacted output → PrefilterHit; same filter/sort/truncate
        // as the CPU path. Compaction order is nondeterministic, but each seq
        // appears once and the sort below is a total order (score desc, seq_id
        // asc), so the result is deterministic and bit-exact vs CPU.
        let mut hits: Vec<PrefilterHit> = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let packed = out_packed[i];
            let count = (packed >> DIAG_BITS) as u32;
            let diag_b = diag_max - (packed & diag_max);
            let diagonal = diag_b as i64 - diag_bias as i64;
            // The kernel voted in DENSE space; map back to the external seq_id.
            let seq_id = self.dense_to_orig[out_seq[i] as usize];
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
    fn sparse_huge_seq_id_handled_via_dense_remap() {
        if skip() {
            return;
        }
        // A single target with a near-u32::MAX seq_id used to size best[] at
        // ~34 GB (OOM risk). Dense remapping sizes best[] by the target COUNT
        // (1 here), so it uploads fine — and the original seq_id is preserved in
        // the result via dense_to_orig (codex).
        let enc = KmerEncoder::new(4, 2);
        let t = vec![0u8, 1]; // k-mer "01"
        let big_id = u32::MAX - 1;
        let idx = KmerIndex::build(enc, [(big_id, t.as_slice())], 99).unwrap();
        let handle =
            GpuPrefilterIndex::upload(&idx).expect("dense remap should accept a sparse huge id");
        let q = vec![0u8, 1];
        let gpu = handle
            .prefilter(&q, 99, &PrefilterOptions::default())
            .unwrap();
        let cpu = diagonal_prefilter(&idx, &q, 99, &PrefilterOptions::default());
        assert_eq!(gpu, cpu, "dense-remapped result must match CPU");
        assert!(
            gpu.iter().any(|h| h.seq_id == big_id),
            "original seq_id must be preserved through the dense remap"
        );
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

    // --- Phase 3a: sensitive (similar-k-mer expanded) ---

    use crate::prefilter::{diagonal_prefilter_sensitive, SimilarityConfig};

    fn id_matrix(alphabet: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet * alphabet];
        for i in 0..alphabet {
            s[i * alphabet + i] = m;
        }
        s
    }

    fn assert_sensitive_parity(
        index: &KmerIndex,
        query: &[u8],
        skip_idx: u8,
        sim: &SimilarityConfig,
        opts: &PrefilterOptions,
    ) {
        let handle = GpuPrefilterIndex::upload(index).expect("upload");
        let cpu = diagonal_prefilter_sensitive(index, query, skip_idx, sim, opts);
        let gpu = handle
            .prefilter_sensitive(query, skip_idx, sim, opts)
            .expect("sensitive");
        assert_eq!(cpu, gpu, "sensitive CPU vs GPU disagree (query {query:?})");
    }

    #[test]
    fn sensitive_widens_beyond_exact_and_matches_cpu() {
        if skip() {
            return;
        }
        // Index: seq 0 = "00" (k-mer 00), seq 1 = "01" (k-mer 01). Query "00".
        // identity +2/-1, threshold 1: window "00" expands to neighbours incl
        // "01" (score 2-1=1) ⇒ catches seq 1 that the EXACT path misses.
        let enc = KmerEncoder::new(4, 2);
        let t0 = vec![0u8, 0];
        let t1 = vec![0u8, 1];
        let idx =
            KmerIndex::build(enc, [(0u32, t0.as_slice()), (1u32, t1.as_slice())], 99).unwrap();
        let scores = id_matrix(4, 2, -1);
        let sim = SimilarityConfig {
            scores: &scores,
            threshold: 1,
        };
        let query = vec![0u8, 0];

        let handle = GpuPrefilterIndex::upload(&idx).expect("upload");
        let exact = handle
            .prefilter(&query, 99, &PrefilterOptions::default())
            .unwrap();
        let sens = handle
            .prefilter_sensitive(&query, 99, &sim, &PrefilterOptions::default())
            .unwrap();
        assert!(exact.iter().all(|h| h.seq_id != 1), "exact must miss seq 1");
        assert!(
            sens.iter().any(|h| h.seq_id == 1),
            "sensitive must catch seq 1"
        );
        assert_sensitive_parity(&idx, &query, 99, &sim, &PrefilterOptions::default());
    }

    #[test]
    fn sensitive_self_score_threshold_equals_exact() {
        if skip() {
            return;
        }
        // threshold = k*m (the self-score) ⇒ only the exact k-mer of each window
        // expands ⇒ sensitive == exact.
        let idx = small_index();
        let scores = id_matrix(4, 2, -1);
        let sim = SimilarityConfig {
            scores: &scores,
            threshold: 4, // k=2 * m=2
        };
        let q = vec![0u8, 1, 2, 3];
        let handle = GpuPrefilterIndex::upload(&idx).expect("upload");
        let exact = handle
            .prefilter(&q, 99, &PrefilterOptions::default())
            .unwrap();
        let sens = handle
            .prefilter_sensitive(&q, 99, &sim, &PrefilterOptions::default())
            .unwrap();
        assert_eq!(
            exact, sens,
            "self-score-threshold sensitive must equal exact"
        );
        assert_sensitive_parity(&idx, &q, 99, &sim, &PrefilterOptions::default());
    }

    #[test]
    fn sensitive_parity_with_options_and_batch() {
        if skip() {
            return;
        }
        let idx = small_index();
        let scores = id_matrix(4, 2, -1);
        let sim = SimilarityConfig {
            scores: &scores,
            threshold: 1,
        };
        assert_sensitive_parity(
            &idx,
            &[0u8, 1, 2, 3],
            99,
            &sim,
            &PrefilterOptions {
                score_threshold: 2,
                exclude_self: Some(20),
                max_hits: Some(2),
                ..Default::default()
            },
        );
        // Batch: hit, empty, hit — vs per-query CPU sensitive.
        let handle = GpuPrefilterIndex::upload(&idx).expect("upload");
        let queries: Vec<&[u8]> = vec![&[0, 1, 2, 3], &[3, 3, 3, 3], &[0, 1]];
        let gpu = handle
            .prefilter_sensitive_batch(&queries, 99, &sim, &PrefilterOptions::default())
            .expect("batch");
        for (i, q) in queries.iter().enumerate() {
            let cpu = diagonal_prefilter_sensitive(&idx, q, 99, &sim, &PrefilterOptions::default());
            assert_eq!(cpu, gpu[i], "sensitive batch query {i} mismatch");
        }
    }

    #[test]
    fn sensitive_parity_larger_random_index() {
        if skip() {
            return;
        }
        let alphabet = 6usize;
        let enc = KmerEncoder::new(alphabet as u32, 3);
        let mut rng: u32 = 0x9e37_79b9;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        let seqs: Vec<Vec<u8>> = (0..30)
            .map(|_| {
                let len = 10 + (next() as usize % 30);
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
        let scores = id_matrix(alphabet, 2, -1);
        let sim = SimilarityConfig {
            scores: &scores,
            threshold: 5, // expands a few neighbours per 3-mer (self-score 6)
        };
        for _ in 0..6 {
            let qlen = 10 + (next() as usize % 30);
            let query: Vec<u8> = (0..qlen)
                .map(|_| (next() as usize % alphabet) as u8)
                .collect();
            assert_sensitive_parity(&idx, &query, 99, &sim, &PrefilterOptions::default());
        }
    }
}
