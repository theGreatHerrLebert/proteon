# GPU_PREFILTER_PLAN — GPU k-mer diagonal-voting prefilter (Phase 1)

Status: DRAFT (pre-implementation). The last tracked Tier-2 item. The MMseqs2-style k-mer
prefilter (`proteon-search/src/prefilter.rs`) runs on CPU; the GPU search path (ungapped
diagonal + SW kernels) already exists and is fed by the CPU prefilter. This adds a GPU path for
the prefilter's hot inner loop — k-mer index lookup + per-`(seq_id, diagonal)` vote + best-
diagonal-per-target — producing **parity-exact** `PrefilterHit`s.

## 0. Why this shape (and why staged)
The genuine win is doing the **voting on-device** so only the small result list crosses PCIe — a
GPU-gather/CPU-vote hybrid would lose to transferring every hit back. But counting per
`(seq_id, diagonal)` over a large corpus rules out a dense `[n_targets][n_diag]` atomic
histogram (too big). So Phase 1 votes via a **GPU open-addressing hash table** (avoids a
from-scratch GPU radix sort) + an `atomicMax`-into-`best[n_targets]` reduction whose packed key
reproduces the CPU tie-break exactly. This is a multi-PR effort; Phase 1 is the single-query,
exact-k-mer, **in-memory-index** core. Later: query batching, similar-k-mer expansion,
on-disk/memmap indexes, perf tuning.

## 1. CPU reference (must match exactly) — `diagonal_prefilter` (prefilter.rs:73)
1. k-mers `(q_pos, hash)` from `index.encoder().iter_kmers(query, skip_idx)`.
2. For each, `for_each_hit(hash)` → `(seq_id, target_pos)`; `diagonal = target_pos − q_pos`;
   `counts[(seq_id, diagonal)] += 1`.
3. Best per `seq_id`: max count; **tie → smaller diagonal**.
4. Filter (`score_threshold`, `exclude_self`), sort (**score desc, seq_id asc**), truncate `max_hits`.
Index layout (`KmerIndex`): `offsets: Vec<u64>` (len `table_size+1`, CSR), `entries: Vec<KmerHit{seq_id:u32, pos:u16}>` sorted by hash. Lookup = `entries[offsets[h]..offsets[h+1]]`.

## 2. GPU design (single query, in-memory index)
**Upload once (resident, reused across queries):** `offsets` (u64×(table_size+1)), `entries_seq_id`
(u32×N), `entries_pos` (u16×N). On a `GpuPrefilterIndex` handle cached alongside the index.

**Per query (host):** compute `(q_pos, hash)` k-mers (CPU `iter_kmers`, cheap); upload two arrays
`kmer_qpos: i32[m]`, `kmer_hash: u64[m]`. Compute total hit count `H = Σ (offsets[h+1]−offsets[h])`
on host (cheap, just offset reads) to size the hash table; if `H == 0`, return empty.

**Kernel A — vote (insert-or-increment):** grid over the *flattened* (k-mer, hit) space. To get a
flat thread→(k-mer,hit) map without atomics, host computes a prefix sum `kmer_hit_base[m]` over
per-k-mer hit counts and the kernel binary-searches it (or: one block per k-mer, threads stride
its hit list — simpler, no prefix sum). Each thread:
- `seq = entries_seq_id[e]`, `diag = (i32)entries_pos[e] − kmer_qpos[k]`;
- `key = pack(seq, diag)` (u64: `seq` high 32, `diag_biased` low 32, `diag_biased = diag + DIAG_BIAS`);
- open-addressing insert-or-increment into `table_keys[u64]` / `table_counts[u32]`
  (size `2·H` round to pow2): linear probe; claim empty slot with `atomicCAS(&table_keys[s], EMPTY, key)`;
  on success or key-match, `atomicAdd(&table_counts[s], 1)`.

**Kernel B — best per target:** one thread per table slot. If occupied, decode `(seq, diag_biased, count)`;
`packed = ((u64)count << DIAG_BITS) | (DIAG_MAX − diag_biased)`; `atomicMax(&best[seq], packed)`.
`best[n_targets]` u64, init 0. The packed order = max count, then (on tie) **max `DIAG_MAX−diag`** =
**min diagonal** ⇒ exactly the CPU tie-break. `DIAG_BITS` covers the biased-diagonal range
(`target_pos ≤ 65535`, so `diag_biased < 65535 + qlen`; 20 bits ⇒ ≤ ~1M, count in the upper bits).

**Host finish:** copy `best[]` back; for each `seq` with `count>0` decode `(count, diagonal)`; apply
`exclude_self`/`score_threshold`; sort (score desc, seq_id asc); truncate. Identical to CPU steps 3–4.

## 3. Parity (exact)
Diagonal scoring is integer/deterministic ⇒ **bit-exact** vs `diagonal_prefilter`. Test:
`diagonal_prefilter_gpu(index, query, …) == diagonal_prefilter(index, query, …)` on hand-crafted
small indexes (incl. ties on count → smaller diagonal; negative diagonals; `exclude_self`;
`score_threshold`; `max_hits`; empty result; a k-mer whose hit list is large). Gate `#[cfg(feature="cuda")]`,
skip when `try_global().is_none()` (existing pattern).

## 4. Correctness risks (for claudex)
1. **Hash insert-or-increment race:** the CAS-claim then atomicAdd pattern — is the "claim empty
   OR matches my key" loop correct under concurrent inserts of the SAME key (two threads racing to
   claim the same empty slot: loser must detect the now-present matching key and add to it, not
   probe past it)? Need the canonical insert-or-increment loop.
2. **Table sizing / full table:** `2·H` pow2; guard against a pathological full table (load factor)
   — if a probe wraps fully, that's a bug (over-count or lost vote). Bound H; assert.
3. **Tie-break packing:** does `(count<<DIAG_BITS)|(DIAG_MAX−diag_biased)` + `atomicMax` truly
   reproduce "max count, then min diagonal"? Bit widths: count can be up to #k-mers (≤ qlen ≤
   ~65k) → needs ≤ 44 bits above DIAG_BITS=20; u64 fits. Verify no overflow.
4. **diag bias range:** `target_pos` is `u16` (≤65535); `q_pos < qlen`. `diag ∈ [−(qlen−1), 65535]`.
   `DIAG_BIAS = qlen` (or a safe constant); ensure `diag_biased ≥ 0` and `< 2^DIAG_BITS`.
5. **best[] size = n_targets:** need `n_targets` (max seq_id + 1). The index knows it (max over
   entries, or stored). Confirm.
6. **u16 pos vs i32 diagonal:** match the CPU cast (`hit.pos as i32 - q_pos as i32`).

## 5. Files
- `proteon-search/src/gpu/prefilter.cu` (kernels A + B) + `gpu/prefilter.rs` (upload handle, launch,
  host finish, parity test) — mirror `gpu/diagonal.rs` (NVRTC lazy compile, `GpuContext::try_global`,
  `OnceLock` kernel cache).
- `gpu/mod.rs` — register the module.
- Wire into `search.rs` behind `opts.use_gpu` as an opt-in alternative to `diagonal_prefilter`
  (default stays CPU until benchmarked) — OR leave wiring to Phase 2; Phase 1 ships the kernel +
  parity test + a public `diagonal_prefilter_gpu` entry. Decide in review.

## 6. Non-goals (later phases)
Query batching (the real throughput multiplier — index resident, thousands of queries); similar-
k-mer expansion (`KmerGenerator`); on-disk/memmap index upload at UniRef scale; replacing the CPU
prefilter by default; perf tuning (warp-cooperative probing, shared-mem staging).

## 7. Open questions
1. One block per k-mer (threads stride its hit list, no prefix sum) vs flat thread space (prefix
   sum + binary search)? Lean **block-per-k-mer** for Phase 1 simplicity; revisit if hit-list
   length is very skewed.
2. Hash-table-count + atomicMax vs sort-based (radix) counting? Hash table avoids a from-scratch
   GPU sort and is simpler to get parity-exact; sort-based is the upstream approach and scales
   better — defer to a perf phase.
3. Wire into `search.rs` now or Phase 2?

## 8. Review log (claudex) — adopted
Design confirmed parity-exact and right for Phase 1. Resolutions:
- **Insert-or-increment loop** (canonical): `old = atomicCAS(&keys[s], EMPTY, key); if (old==EMPTY || old==key) { atomicAdd(&counts[s],1); return; } s = (s+1)&mask;`. Correct under same-key races (winner gets EMPTY, loser gets key, both add). `table_counts` zeroed first; kernel B after A (same stream).
- **EMPTY=0 is safe** with `DIAG_BIAS = qlen`: min `diag_biased = −(qlen−1)+qlen = 1`, so `key = (seq<<32)|diag_biased ≥ 1` always (even seq=0). No sentinel collision. Apply a 64-bit **mix** before masking to avoid linear-probe clustering.
- **Tie-break packing:** `packed = ((u64)count << 20) | ((1<<20)−1 − diag_biased)`; `atomicMax(u64)` ⇒ max count, then min diagonal. EXACT. Bit budget: count(≤~65k, 32b) + 20b = 52b < 64b. **Enforce `65535 + qlen < 2^20`** (i.e. qlen ≤ ~65000) and `q_pos ≤ i32::MAX` — guard, don't silently narrow.
- **Sizing:** distinct keys ≤ H ⇒ capacity `next_pow2(2·H)` cannot fill, but check every host arithmetic step for overflow (`H` usize, `2H`, pow2, `cap*size`) and handle `H==0` before pow2. Probe loop bounded by `table_size`; on exhaustion set a **device error flag**, copy it back, return `Err` (never silently drop a vote or spin).
- **best[] sizing:** seq_ids are NOT guaranteed dense (tests use 10/20/30). Size `best[max_seq_id+1]` (host computes `max_seq_id` over `entries_seq_id`, or the index supplies it). **Bounds-check `seq < best_len`** in kernel B (guard malformed metadata).
- **count overflow:** `atomicAdd(u32)` wraps; CPU u32 add wraps in release / panics in debug — bounded by #k-mers (≤ qlen, small) given the qlen guard, so safe; note it.
- **Parity:** `diag = (i32)pos − (i32)q_pos` (match the cast); threshold is `>=` (incl. 0); sort score-desc / id-asc; `best==0` ⇒ no vote. **Adversarial tests:** count ties → smaller diagonal; negative diagonals; sparse seq_ids; `exclude_self`; `score_threshold`; `max_hits`; empty result; a long hit list.
- **Wiring (Q3):** Phase 1 ships the kernel + a public `diagonal_prefilter_gpu` + parity tests; do NOT change the default `search.rs` path (stays CPU until a benchmark justifies it). Batching/sensitive/on-disk are later phases.
