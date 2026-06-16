# GPU_PREFILTER_BENCHMARK — measured CPU vs GPU prefilter throughput

Status: MEASURED (RTX 2070). Benchmark: `proteon-search/examples/prefilter_bench.rs`.

## What this measures

The diagonal-voting k-mer prefilter **in isolation** — CPU
[`diagonal_prefilter`] vs the resident GPU
[`GpuPrefilterIndex::prefilter_batch`] — on a synthetic 13-letter reduced
alphabet indexed at k=6 (the config `SearchEngine` uses by default). It is
*not* `cpu_vs_gpu_bench`, which times the whole `search()` path and conflates
prefilter + ungapped + SW. Each run verifies the GPU result is **bit-exact** vs
CPU.

```bash
cargo run --release -p proteon-search --features cuda \
    --example prefilter_bench -- <n_targets> <n_queries> <len_min> <len_max>
```

## Result: GPU loses — keep the CPU prefilter as the default

| corpus (targets) | CPU q/s | GPU q/s | steady-state speedup |
|------------------|---------|---------|----------------------|
| 5 000            | 22 800  | 11 200  | 0.49×                |
| 20 000           | 17 300  | 6 300   | 0.37×                |
| 50 000           | 11 100  | 4 300   | 0.39×                |
| 100 000          | 7 700   | 2 500   | 0.32×                |

- GPU throughput is **flat at ~4 400 q/s regardless of query count** (200 →
  4 000 queries at 50k targets all ~0.36×) ⇒ pure per-query launch overhead,
  **no batch amortization**.
- It **degrades as the corpus grows** (0.49× → 0.32×).
- Folding in the one-time upload makes it worse still: **0.1–0.2× amortized**.
- Bit-exact vs CPU on every query — the correctness work (Phases 1–3d) holds.

## Why — the bottleneck

In `vote_and_reduce`, per query:
1. `alloc_zeros` two `table_cap` hash buffers **+ one `best[]` sized to the
   whole target set**,
2. two kernel launches (vote, then reduce),
3. **copy `best[]` back and host-scan all `#distinct_targets` entries** to
   decode hits.

`prefilter_batch` is a host loop over `run_query`, so none of this amortizes
across the batch — and the target-set-sized `best[]` copyback + scan is exactly
what makes throughput **degrade with N**. The actual voting work per query
(~200 k-mers, each hitting ~1–10 postings) is dwarfed by this fixed overhead.

## Update — P1 (seq-keyed reduction) landed

`GPU_PREFILTER_FUSED_PLAN.md` P1 replaced the dense `best[#targets]` with a
seq-keyed hash table + on-device compaction (copyback is now O(hits)). Same
RTX 2070 sweep, 1 000 queries, bit-exact verified:

| corpus (targets) | CPU q/s | GPU q/s | steady-state | (was) |
|------------------|---------|---------|--------------|-------|
| 5 000            | 22 000  | 8 800   | 0.40×        | 0.49× |
| 20 000           | 16 700  | 9 500   | 0.57×        | 0.37× |
| 50 000           | 12 600  | 8 300   | 0.66×        | 0.39× |
| 100 000          | 7 800   | 6 700   | **0.87×**    | 0.32× |

The **degradation-with-N is eliminated and reversed**: GPU q/s now holds roughly
flat while CPU falls off as postings densify, so the speedup *rises* with corpus
size (0.40× → 0.87×) instead of sinking (0.49× → 0.32×). GPU still doesn't beat
CPU at these sizes — the remaining gap is per-query alloc churn + three
un-batched launches, which **P2 (persistent scratch)** and **P3 (batched
launches)** target. The slight regression at 5k (0.49× → 0.40×) is the extra
compaction kernel + best-hash table fixed overhead, which P2/P3 also amortize.

## What would make GPU win

A **fused multi-query kernel**:
- one launch covering *all* queries in the batch (amortize launch latency),
- persistent, reusable device scratch (no per-query alloc+zero),
- **compacted output** — copy back O(hits), not O(#targets), so cost stops
  scaling with corpus size.

Until that exists, `use_gpu` is a no-win for the prefilter; the
`SearchEngine::search` GPU path (Phase 3c/3d) is correctness-complete but not a
throughput win, and the silent CPU fallback is the fast path in practice.
