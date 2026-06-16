# GPU_PREFILTER_FUSED_PLAN — fused multi-query prefilter (throughput unlock)

Status: REVIEWED (claudex round folded in — see §8). Ready to implement from P1.
Builds on the resident `GpuPrefilterIndex`
(#157–#162) and the measured benchmark (`GPU_PREFILTER_BENCHMARK.md`): the GPU
prefilter is ~0.36× CPU and **degrades with corpus size** (0.49×@5k →
0.32×@100k targets). This plan removes the two reasons.

## 1. Why the current path loses (measured)

`prefilter_batch` is a host loop over `run_query`; each query in
`vote_and_reduce`:
1. `alloc_zeros` two `table_cap`-wide buffers (`d_keys`, `d_counts`) **+ one
   `d_best` sized to `best_len` = #distinct targets**,
2. two kernel launches (vote, reduce) + a stream sync,
3. `clone_dtoh(d_best)` of `best_len` u64 **and a host scan over all
   `best_len` entries** to decode hits.

Two costs that never amortize and scale with the WRONG quantity:
- **Per-query allocation + zeroing** of device scratch (launch-overhead-bound;
  throughput is flat ~4 400 q/s regardless of query count → pure fixed
  overhead).
- **`best[]` is O(#targets)**, but a single query only touches ≤ `total_hits`
  (its posting-list sum, a few thousand) distinct targets. The full-width
  dtoh + host scan is what makes throughput *degrade as the corpus grows*.

## 2. Core idea — make every buffer O(query-work), then batch + persist

### 2a. Kill the dense `best[#targets]` → seq-keyed hash reduction
Replace `d_best[seq]` (dense, #targets-wide) with a **second open-addressing
hash table keyed by `seq` only** (`best_keys`/`best_vals`), sized like the vote
table at `next_pow2(2·total_hits)` (distinct seqs ≤ total_hits ⇒ never fills).
Reduction becomes: for each occupied vote-table slot, `atomicMax` the packed
`(count, diag_max−diag_b)` into `best_vals[probe(seq)]` (open-addressing on
`seq`). `atomicMax` on the packed integer is commutative/associative ⇒ the
per-seq max is order-independent, identical tie-break math to today's
`atomicMax(&best[seq], …)`, just into a hashed slot. **Every buffer is now
O(total_hits); nothing depends on #targets.** That alone should erase the
degradation-with-N.

Correctness details that make it bit-exact (claudex — these are the real
hazards, not "seq vs (seq,diag) collisions" which open addressing handles):
- **Insert convergence:** claim a slot with `atomicCAS(best_keys[slot], EMPTY,
  key)`; treat BOTH `EMPTY→key` (we won) and already-`key` (someone else claimed
  it for the same seq) as success, then `atomicMax(best_vals[slot], packed)`.
- **Sentinel:** dense seq ids start at 0 and `EMPTY=0`, so **store `seq+1` as the
  key** (or any non-zero encoding) — a real seq must never equal `EMPTY`.
- **Init ordering:** bulk-zero `best_vals` up front (0 = neutral packed = "no
  vote"), so a slot is never observed owned-but-uninitialised.
- **Compaction keys off `best_keys`** (occupied?), not `best_vals`, in case a
  valid packed value is ever 0 (it isn't — `count≥1 ⇒ packed>0` — but key off
  the key anyway for safety).
- **Probe exhaustion poisons the query** via `error_flag` → host Err → CPU
  fallback; never trust partial results.

### 2b. On-device compaction → O(hits) copyback
A compaction kernel scans `best_keys` (O(total_hits), small) and appends each
occupied `(seq, packed)` to a per-query output segment via an atomic counter.
Host receives only `out_count` + the compacted `(seq, packed)` entries — the
host scan over #targets is gone. Sort/filter/truncate stay on the host (tiny:
≤ #distinct-targets-hit entries), preserving the exact CPU
order/threshold/exclude_self/max_hits semantics.

**Output must hold ALL distinct seqs the query touched — NEVER cap compaction at
`max_hits`** (claudex, critical): host sort+threshold+exclude_self+truncate runs
*after* compaction, so the top-`max_hits` are unknown on-device; capping there
would drop a hit that survives the host sort and break exactness. The output
segment is sized to the same `total_hits` upper bound as the tables, with its
own per-query **output-overflow flag** (→ Err → CPU fallback) alongside the
probe-exhaustion flag.

### 2c. Batch the launches
One vote launch + one reduce launch + one compact launch for the WHOLE tile of
queries (not per query). Grid is keyed by a flattened `(query, k-mer)` work
list for vote, and by `(query, slot)` for reduce/compact. Per-query parameters
(hash-list offset, diag_bias, table base offset, output base) ride in
device arrays indexed by `query_in_tile`. This amortizes launch latency across
the tile — the fix for the flat-4 400-q/s overhead wall.

### 2d. Persistent, grow-not-shrink scratch
`GpuPrefilterIndex` gains a lazily-allocated, reused scratch struct (mirrors
`DPWorkspace`): vote tables, best tables, compaction output + counters, sized
to the current tile and grown (never shrunk) across `prefilter_batch` calls.
Removes per-query alloc+zero. Zeroing between tiles is bounded by tile capacity
(one `memset`/tile), not per query.

## 3. Tiling (memory budget)
**Bucket queries by `total_hits` into power-of-two cap classes, then tile within
a bucket** (claudex) — do NOT let `max_total_hits` set one cap for the whole
batch, or a single high-repeat query inflates `table_cap` for 999 small ones
(catastrophic waste + memory blowup). Within a bucket every query shares a
uniform `table_cap = next_pow2(2·bucket_max_hits)` (coalesced layout). Tile
width `T` per bucket chosen so `T · table_cap · (8+4 vote + 8+8 best) bytes +
output` fits a fraction (≤ ¼?) of free VRAM (`ctx…total_mem()`), same guard
style as `upload`. A query whose `total_hits` exceeds the largest practical
bucket cap falls back to the existing single-query path. **Log any per-query
fallback** (no silent cap). The output segment is sized by the same per-bucket
`total_hits` bound (§2b), never `max_hits`.

## 4. Parity — non-negotiable
Output must stay **bit-exact vs `diagonal_prefilter`** (the CPU oracle), same as
every prior phase. The reduction math (packed key, tie-break: max count then
smallest diagonal) is unchanged; only the *container* for the per-seq max
changes (dense index → seq-hashed slot). The seq-keyed table must use the SAME
`atomicMax`-of-packed semantics so ties resolve identically. Open-addressing on
`seq` needs its own probe-exhaustion `error_flag` (host turns it into an Err →
CPU fallback, never a silent miscount). Exact AND sensitive both route through
the batched core (the sensitive path already shares `vote_and_reduce`).

## 5. Implementation phases
Each phase has an explicit acceptance gate; later phases are gated on the
measured result of the earlier one (claudex — don't commit to all four up
front).
- **P0 (optional diagnostic, ~1h).** Keep dense `best[#targets]`, but add
  on-device compaction (scan dense best, append non-zero) + remove the host
  scan/copyback. If degradation-with-N persists, it proves the dense GPU-side
  scan/zero itself is the cost ⇒ go straight to the seq-hash. A cheap way to
  confirm the cost model before building the second table.
- **P1 — seq-keyed reduction (single query first). DONE.** Swapped dense
  `best[]` for the seq-hash table (`prefilter_reduce_seqhash`) + on-device
  compaction (`prefilter_compact`) in `vote_and_reduce`, one query per call.
  **Gate MET:** bit-exact vs CPU (14 parity tests) AND N-degradation eliminated —
  *reversed*, in fact: speedup rose 0.40×→0.87× across 5k→100k targets (was
  0.49×→0.32×). See `GPU_PREFILTER_BENCHMARK.md`. Does not yet beat CPU (still
  per-query alloc + 3 launches) — that's P2/P3.
- **P2 — persistent scratch. DONE.** Hoisted per-query device buffers + stream
  into a caller-owned, grow-not-shrink `PrefilterScratch` reused across the batch
  (NOT `Mutex`-on-`&self` — the handle stays `&self`; the batch methods own one
  scratch and thread it through `vote_and_reduce`). **Gate MET:** GPU q/s +~60%
  at small N; **GPU crosses 1.0 — beats CPU at 50k+ targets (1.23× @100k)**.
  Bit-exact (14 parity tests). The P3 throughput goal is already reached at scale
  by P2 alone; P3 now only targets the small-N (<50k) regime.
- **P3 — batched launches + tiling.** Flatten the tile work lists; one
  launch-set per tile; bucket-by-`total_hits` tiling + per-query fallback.
  **Gate:** beats CPU steady-state at realistic batch sizes — *required if the
  goal is a throughput unlock, optional if the goal is only fixing
  N-degradation.*
- **P4 — wire `search()` to the crossover. DONE (ahead of P3).** `search()` is
  single-query, so it wires the single-query path. A fresh scratch per query only
  crosses 1.0 near ~200k targets, so the engine now CACHES one `PrefilterScratch`
  (`Mutex<Option<…>>` on `&self`) and reuses it across `search()` calls via the
  new public `prefilter_with` / `prefilter_sensitive_with`, recovering the ~55k
  crossover. `gpu_handle()` gates on `target_count() >= GPU_PREFILTER_MIN_TARGETS`
  (default 75k, overridable via `SearchOptions::gpu_prefilter_min_targets`); below
  it the index is never uploaded. Bit-exact wiring tests + a below-threshold
  skip test. P3 (small-N batched launches) remains the only open item, and only
  matters for batched callers below ~55k targets.

## 6. Tests
- Per-phase **bit-exact vs CPU** on the existing parity corpora (exact +
  sensitive, with reducer + `reduce_to: None`).
- Seq-hash **probe-exhaustion** path → Err → CPU fallback (force a tiny table).
- **Tiling correctness:** batch split across ≥2 tiles == one-shot == per-query.
- **Outlier-query fallback** triggers + logs.
- Benchmark re-run committed as the acceptance gate: P1 flat-vs-N; P3 > CPU
  steady-state (state the target, e.g. ≥2× at 50k targets, or record the
  honest number and keep CPU default).

## 7. Open questions (for claudex)
1. **Three kernels vs fusion.** Is vote→reduce→compact as three launches per
   tile fine (the tile already amortizes), or worth fusing reduce+compact (one
   scan of `best_keys`)? Cooperative-groups single-kernel fusion seems
   over-engineered here — confirm.
2. **Seq-keyed table sizing.** `2·total_hits` bounds distinct seqs safely but
   over-allocates when postings are dense on few targets. Worth a tighter bound
   (distinct seqs ≤ min(total_hits, #targets))? Probably not — keep simple.
3. **Tile cap set by max_total_hits** wastes slots when query sizes are skewed.
   Bucket queries by `total_hits` into same-cap tiles, or accept the waste?
4. **Is P1 alone enough?** If removing the #targets dependence gets GPU
   competitive (the degradation was the main loss), P3 batching may be optional
   polish. Should we gate P2/P3 on P1's measured result rather than commit to
   all four up front?
5. **Persistent scratch + `&self` concurrency.** RESOLVED (claudex): give the
   batched optimized path a `&mut self` signature with persistent scratch; leave
   the existing `&self` single-query path on temporary scratch (unchanged). Do
   NOT hide shared mutable scratch behind a `Mutex` on `&self` — it silently
   serializes concurrent callers and can alias device buffers. (A stream-keyed
   scratch pool preserving `&self` is possible but over-complex; only build it
   if a concurrent batched API is actually required.)
6. **Bit-exactness of the hashed per-seq max.** RESOLVED (claudex): safe in
   principle — `atomicMax` is commutative, so the hashed slot computes the same
   max as the dense index. The only risks are implementation-level and are now
   pinned in §2a (CAS insert convergence, `seq+1` sentinel, value-init ordering,
   compaction keyed off `best_keys`, probe-exhaustion poisoning).

## 8. Review log (claudex) — adopted
1. Seq-keyed reduction is bit-exact iff each seq maps to one initialized slot;
   real hazards are atomic insert race, EMPTY sentinel (use `seq+1`), value-init
   ordering, probe-exhaustion poisoning, compaction keyed off `best_keys`. Folded
   into §2a.
2. **Output sized to all distinct-seqs-touched, NEVER `max_hits`** — host
   sort/filter/truncate runs after compaction, so on-device capping breaks
   exactness. Added per-query output-overflow flag. Folded into §2b.
3. **Bucket queries by `total_hits` (pow2 cap classes) before tiling** — don't
   let `max_total_hits` set one cap for the whole batch. Folded into §3.
4. **Phasing is gated, not all-up-front:** P1 likely fixes N-degradation but
   won't beat CPU alone; P3 batching is required only if the goal is a true
   throughput unlock. Added P0 diagnostic + per-phase gates. Folded into §5.
5. **Scratch: `&mut self` batched path, not `Mutex`-on-`&self`.** Folded into
   §7.5.
6. Three kernels (vote→reduce→compact) per tile is fine — the tile amortizes
   launches; cooperative-groups single-kernel fusion is over-engineering here
   (confirmed). No change needed.
