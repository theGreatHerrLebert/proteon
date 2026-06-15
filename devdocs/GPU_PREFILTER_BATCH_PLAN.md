# GPU_PREFILTER_BATCH_PLAN — resident index + query batching (Phase 2)

Status: DRAFT. Builds on Phase 1 (#157, `gpu/prefilter.{cu,rs}`). The kernels are unchanged and
already CPU-parity-validated; this phase removes Phase 1's **per-call index re-upload** — the
dominant cost — by holding the uploaded index resident and reusing it across many queries.

## 1. Why
Phase 1 `diagonal_prefilter_gpu` uploads `offsets` (u64×(table_size+1)) + `entries_seq_id`
(u32×N) + `entries_pos` (u32×N) on EVERY call. For a real search that's the same large index
re-sent per query — the index dwarfs the query. The throughput win is: **upload once, reuse**.

## 2. Design — `GpuPrefilterIndex` resident handle
A struct owning the device-resident index + metadata:
```
pub struct GpuPrefilterIndex {
    d_offsets: CudaSlice<u64>,
    d_seq_id:  CudaSlice<u32>,
    d_pos:     CudaSlice<u32>,
    table_size: u64,
    max_seq_id: u32,
    best_len:   usize,          // max_seq_id + 1 (guarded != u32::MAX)
    // a clone of the encoder so the host can extract query k-mers without the
    // original KmerIndex; OR keep a reference — decide (lifetime vs owned).
    encoder: KmerEncoder,
    offsets_host: Arc<Vec<u64>>, // for the host total_hits sum (sizing the table)
}
```
- `GpuPrefilterIndex::upload(index: &KmerIndex) -> Result<Self>`: does the one-time HtoD of the
  three arrays + captures `table_size`, `max_seq_id` (reject `u32::MAX`), `encoder`, and an
  `Arc<Vec<u64>>` of `offsets` (host needs `offsets[h+1]-offsets[h]` to size each query's hash
  table without a DtoH).
- `fn prefilter(&self, query, skip_idx, opts) -> Result<Vec<PrefilterHit>>`: the Phase-1 per-query
  body MINUS the index upload — extract k-mers (filter out-of-range hashes), size + alloc the
  per-query hash table, alloc/zero `best[best_len]` + `error_flag`, upload `kmer_qpos`/`kmer_hash`,
  launch vote+reduce against the RESIDENT `d_offsets/d_seq_id/d_pos`, decode + filter/sort/truncate.
- `fn prefilter_batch(&self, queries: &[&[u8]], skip_idx, opts) -> Result<Vec<Vec<PrefilterHit>>>`:
  reuse ONE stream across the batch; reuse the `best[best_len]` + `error_flag` buffers (their size
  is index-determined, not query-determined) **zeroing them between queries** (`memset`); alloc the
  hash table per query (size is query-determined). Returns one hit list per query, in order.

## 3. Back-compat
`diagonal_prefilter_gpu(index, query, skip_idx, opts)` stays as a thin wrapper:
`GpuPrefilterIndex::upload(index)?.prefilter(query, skip_idx, opts)`. The Phase-1 parity tests keep
passing unchanged; add batch parity tests.

## 4. Correctness invariants
- **best[]/error_flag reuse across the batch requires zeroing between queries** — a leftover vote
  from query i would corrupt query i+1. Use `stream.memset_zeros` (or re-alloc if memset is
  unavailable in cudarc — check). Hash table is fresh per query (alloc_zeros), so no reuse hazard
  there.
- The resident `d_*` buffers are READ-ONLY in the kernels ⇒ safe to share across queries/streams.
- Per-query `best_len`, `table_size`, `max_seq_id`, the guards (qlen bound, out-of-range hash
  filter, u32::MAX seq_id) are unchanged from Phase 1 — keep them.
- Bit-exact parity per query is inherited from Phase 1 (same kernels, same host finish).

## 5. Tests
- `prefilter_batch` over several queries (incl. an empty-result query and a high-hit query) ==
  per-query `diagonal_prefilter` (CPU) — proves the resident-index reuse + best[]/flag zeroing.
- A single `prefilter` via the resident handle == `diagonal_prefilter_gpu` (the wrapper) ==
  CPU — proves the refactor preserved Phase-1 behavior.
- Reuse-without-leak: run query A (hits), then query B (empty) through the SAME handle/batch and
  assert B is empty (catches a best[]-not-zeroed bug).

## 6. Non-goals (Phase 3+)
- TRUE fused multi-query kernels (one launch, all queries, per-query hash-table regions) — the
  bigger parallelism win; per-query sequential launches here already amortize the upload.
- Concurrent multi-stream overlap across queries.
- Wiring into `search.rs` / the connector batch search path (default stays CPU until benchmarked).
- on-disk/memmap index residency at UniRef scale; similar-k-mer expansion.

## 7. Review log (claudex) — adopted
1. **Reuse `best[]`+`error_flag`, clear immediately before each GPU-reaching query** (not "after
   the previous one"): early-return queries (empty k-mers / `total_hits==0`) never touch the
   scratch, so clearing at the start of the GPU path makes early returns/errors harmless. Same-
   stream issue order serializes `reduce(i)→DtoH(i)→memset(i+1)→vote(i+1)` — no overlap hazard.
2. **Per-query re-alloc is NOT cheaper** (`alloc_zeros` = alloc + `memset_zeros`; plus allocator /
   drop-sync overhead). So reuse + `stream.memset_zeros` before each query. Hash table stays
   per-query (`alloc_zeros`, size is query-determined).
3. **Scratch is LOCAL to each call, not on the handle** — `prefilter(&self)`/`prefilter_batch(&self)`
   own their `best`/`err`/stream, so `&self` calls are concurrency-safe. The handle holds ONLY the
   read-only resident index buffers (Send+Sync, safe to share across streams) + owned `encoder` +
   owned `offsets_host: Vec<u64>` (for host table sizing; `Arc` wouldn't avoid the one-time clone —
   plain `Vec`) + `table_size`/`max_seq_id`/`best_len`.
4. **Extract ONE shared per-query body** (`run_query`) used by `prefilter` and `prefilter_batch` so
   the refactor can't drift from Phase-1 semantics (empty-index handling, guard order, sync-before-
   decode, filter/sort/truncate). Parity is preserved by construction, not "inherited."
5. **Tests:** `hit→empty→hit`, `empty→hit`, **two queries on the SAME `seq_id` where B scores LOWER
   than A** (the critical stale-`atomicMax` catch), same-score-different-diagonal (stale tie-break),
   `total_hits==0` between active queries, varying options per call, handle survives source
   `KmerIndex` drop.
