# GPU_PREFILTER_SENSITIVE_PLAN — sensitive (similar-k-mer) GPU prefilter (Phase 3a)

Status: DRAFT. Builds on #157/#158 (`GpuPrefilterIndex`). Adds the SENSITIVITY half — each query
k-mer expands into every similar k-mer scoring `>= threshold` (substitution matrix), matching the
CPU oracle `diagonal_prefilter_sensitive`. This is what makes the prefilter find remote homologs;
the exact-k-mer path (#157/#158) only catches identical k-mers.

## 1. Key insight — reuse the kernels unchanged
The CPU sensitive path (`prefilter.rs:152`) differs from exact ONLY in how the `(q_pos, hash)`
list is built: per query window it calls `generate_similar_kmers(encoder, window, scores,
threshold)` and votes over EVERY similar hash; the diagonal voting + reduction + sort are
IDENTICAL. The GPU vote/reduce kernels already take a flat `kmer_qpos[]` / `kmer_hash[]` list and
don't care whether the hashes are exact or expanded. So:

**GPU sensitive = host-side expansion (the same `generate_similar_kmers` call) → the existing
resident vote+reduce kernels.** No kernel change. Bit-exact vs the CPU sensitive oracle.

**No double-counting risk:** a target position holds exactly ONE k-mer, so a given `(seq_id, pos)`
appears in exactly one posting list ⇒ at most one similar-hash hits it per `q_pos` (same as exact;
the hash-table count per `(seq_id, diagonal)` stays correct).

## 2. Refactor
Split `run_query` (currently: build exact k-mers → GPU vote/reduce → decode) into:
- `vote_and_reduce(&self, kmers: &[(usize, u64)], qlen: usize, opts) -> Result<Vec<PrefilterHit>>`
  — everything from the qlen/diag-bias guard + `total_hits` sizing through the GPU launches and
  decode/filter/sort/truncate. The single GPU core, shared by exact and sensitive.
- exact `run_query`: build `kmers` via `iter_kmers(query, skip_idx).filter(h < table_size)`, call
  `vote_and_reduce(&kmers, query.len(), opts)`. (Behaviour unchanged — Phase-1/2 tests still pass.)

## 3. New API (mirrors the exact methods)
- `GpuPrefilterIndex::prefilter_sensitive(&self, query, skip_idx, similarity: &SimilarityConfig, opts)`
- `GpuPrefilterIndex::prefilter_sensitive_batch(&self, queries, skip_idx, similarity, opts)`
Both build the expanded list and call `vote_and_reduce`:
```
let k = self.encoder.kmer_size();
let mut kmers = Vec::new();
for q_pos in 0..query.len().saturating_sub(k - 1) {
    let window = &query[q_pos..q_pos + k];
    if window.contains(&skip_idx) { continue; }   // X-window skip, same as CPU
    for (h, _) in generate_similar_kmers(&self.encoder, window, similarity.scores, similarity.threshold) {
        if h < self.table_size { kmers.push((q_pos, h)); }  // OOB-hash guard (mirrors lookup_hash)
    }
}
vote_and_reduce(&kmers, query.len(), opts)
```
`SimilarityConfig` + `generate_similar_kmers` come from `crate::prefilter` / `crate::kmer_generator`.

## 4. Correctness / parity
Bit-exact vs `diagonal_prefilter_sensitive` (same expansion, same voting). Same guards as exact:
qlen bound, OOB-hash filter, `total_hits==0`/empty ⇒ `Ok(vec![])`, u32::MAX seq_id rejected at
upload. The expanded list can be much larger than exact (many neighbors/window) ⇒ `total_hits`
larger ⇒ bigger hash table; the existing `2*total_hits` sizing + overflow guards already cover it.

## 5. Tests (vs CPU `diagonal_prefilter_sensitive`)
- Parity on a small protein-ish alphabet with an identity-ish score matrix + a threshold that
  expands SOME neighbors (not just the exact k-mer) — proves expansion actually widens the hit set
  AND matches CPU. Include: a near-match that the EXACT path misses but sensitive catches; count
  ties; `score_threshold`/`exclude_self`/`max_hits`; an empty result; a batch mixing sensitive
  queries.
- A high threshold that expands to only the exact k-mer ⇒ sensitive == exact == CPU (degenerate
  check).
- Larger random index + several queries, sensitive GPU == CPU.

## 6. Non-goals (Phase 3b+)
- GPU-side k-mer expansion (branch-and-bound DFS on device) — host expansion is the tractable
  step; the GPU still wins on the (larger) voting.
- A STREAMING `generate_similar_kmers` (callback/iterator yielding neighbours one at a time, so a
  pathological threshold can't materialise `alphabet^k` per window) — a shared follow-up for the
  CPU + GPU sensitive paths (both call the current Vec-returning generator); realistic thresholds
  bound it today. Phase-1 filter already bounds the RETAINED list to in-index neighbours.
- Fused multi-query kernels (one launch, all queries).
- on-disk/memmap index residency; `search.rs` default wiring.

## 7. Review log (claudex) — adopted
No-double-counting CONFIRMED airtight: the generator DFS visits each candidate once + injective
encoding ⇒ unique hashes; even if it emitted duplicates, CPU (`for sim_hash … for_each_hit`) and
GPU (one posting-list vote per hash) double-count IDENTICALLY ⇒ parity holds regardless. Adopted:
1. **[HIGH] Guard the expanded `n_kmers` separately** — it's the grid dim + an `i32` kernel arg and
   is NOT bounded by `total_hits` (most neighbor hashes may have empty posting lists). Add a checked
   `n_kmers <= i32::MAX` (covers the i32 arg AND the CUDA grid-x limit) guard in `vote_and_reduce`.
2. **Validate qlen BEFORE expansion** — `validate_diag_bias(qlen) -> Result<i32>` at the top of BOTH
   public entry points (exact + sensitive), before building/expanding k-mers, so an oversized
   sensitive query errors immediately instead of doing enormous expansion first. `vote_and_reduce`
   then takes the validated `diag_bias`, not raw `qlen`.
3. **`total_hits` via `checked_add`** (sensitive duplication makes overflow more plausible).
4. Reword the `u32` table cap as an "arithmetic / kernel-width guard," not an "in-memory cap"
   (allocation failure past it is correct, not silent).
5. **Tests:** the deterministic widening case (k=2, query `[0,0]`, targets `[0,0]`+`[0,1]`, identity
   `+2/−1`, threshold `1` ⇒ exact returns only target 0; CPU-sensitive == GPU-sensitive return both,
   full `PrefilterHit` equality); threshold-above-max ⇒ empty expansion ⇒ sensitive == exact;
   X-windows mixed with valid; ties; options; empty; batch sequence; larger random index.
6. Shared `build_sensitive_kmers(query, skip_idx, similarity)` helper backs both `prefilter_sensitive`
   and `_batch` (single source of truth, like `run_query`); filter `h < table_size` defensively.
