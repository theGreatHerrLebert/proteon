# GPU_PREFILTER_WIRING_PLAN — wire the GPU prefilter into SearchEngine (Phase 3c)

Status: DRAFT. The GPU prefilter (#157/#158/#159/#160 — exact/resident/batched/sensitive/streaming)
is currently reachable ONLY via the direct `GpuPrefilterIndex` API. `SearchEngine::search` still
runs the CPU `diagonal_prefilter`. This wires the GPU exact prefilter into `search()` so the engine
actually uses it — opt-in, bit-exact, silent CPU fallback.

## 1. Scope
- `search()` uses the **exact** prefilter (`diagonal_prefilter`) today, so wire the **exact** GPU
  prefilter (`GpuPrefilterIndex::prefilter`). Sensitive-in-search is a follow-up (search doesn't use
  it yet).
- **In-memory index only.** `GpuPrefilterIndex::upload` needs a `KmerIndex`; the engine's
  `KmerIndexStorage` is `InMemory(KmerIndex)` or `OnDisk(KmerIndexFile)`. OnDisk falls back to CPU
  (GPU memmap residency is a separate phase).
- **Default unchanged in spirit:** the prefilter SOURCE (GPU vs CPU) follows the existing
  `opts.use_gpu` + `gpu::is_available()` gate — the same gate that already routes ungapped/SW to
  GPU. No new knob; bit-exact ⇒ results identical either way.

## 2. Design
- Add a cuda-gated cached handle to `SearchEngine`:
  `#[cfg(feature = "cuda")] gpu_prefilter: OnceLock<Option<GpuPrefilterIndex>>`.
  Lazily built on first GPU prefilter from the InMemory index (`get_or_init(|| match &self.index {
  InMemory(idx) => GpuPrefilterIndex::upload(idx).ok(), OnDisk(_) => None })`). Cached ⇒ the index
  is uploaded ONCE and reused across queries (Phase 2's whole point). `&self` + `OnceLock` keeps
  `search(&self)` immutable + thread-safe. Init `OnceLock::new()` in the 3 `Ok(Self { … })` sites.
- Private `run_prefilter(&self, q_for_prefilter: &[u8], opts: &PrefilterOptions) -> Vec<PrefilterHit>`:
  - `#[cfg(feature="cuda")]`: if `opts/self.opts.use_gpu && gpu::is_available()`, get the cached
    handle; if `Some(h)`, `match h.prefilter(q, self.skip_idx, opts) { Ok(hits) => return hits,
    Err(e) => { eprintln!(...CPU fallback...); } }`. Falls through on Err / None / OnDisk.
  - CPU fallback: `diagonal_prefilter(&self.index, q, self.skip_idx, opts)`.
- `search()` calls `run_prefilter` instead of the inline `diagonal_prefilter`.

## 3. Correctness
- `GpuPrefilterIndex::prefilter` is **bit-exact** vs `diagonal_prefilter` (Phase 1, exhaustively
  tested) ⇒ the wired GPU prefilter produces the SAME `PrefilterHit` list ⇒ `search_gpu`/`search_cpu`
  see identical input ⇒ identical `SearchHit`s. The wiring changes only the prefilter *source*.
- Fallback paths (no GPU / OnDisk / a runtime upload-or-prefilter Err) silently use the CPU
  prefilter — never panic, never wrong results.
- The cached handle is built from the SAME InMemory `KmerIndex` the CPU path queries, with the SAME
  `skip_idx` + `PrefilterOptions` (`exclude_self: None`, the engine's `diagonal_score_threshold` /
  `max_prefilter_hits`). So GPU and CPU prefilter are fed identical inputs.

## 4. Tests
- **Wired parity:** build an engine (use_gpu), call `run_prefilter(q)` and `diagonal_prefilter(&self
  .index, q, skip_idx, opts)` on the same reduced query; assert equal (catches a wiring bug — wrong
  opts/skip_idx/query, or a stale cached handle). Skip if no GPU.
- **Full-search parity:** two engines, `use_gpu=true` vs `use_gpu=false`, same corpus + query ⇒
  identical `SearchHit`s (end-to-end; relies on the already-tested downstream GPU/CPU parity too).
- **OnDisk fallback:** an OnDisk-backed engine with `use_gpu=true` still returns correct hits (CPU
  fallback, no panic). If an OnDisk fixture is awkward, assert the handle init yields `None` for
  OnDisk and the result matches CPU.
- **Cache reuse:** two `search()` calls on one engine reuse the cached handle (the index isn't
  re-uploaded) — assert via a second query returning correctly (functional proof; the OnceLock is
  init-once by construction).

## 5. Non-goals
- Sensitive prefilter in `search()` (search doesn't use it yet).
- GPU residency for OnDisk/memmap indexes (separate phase).
- Batched multi-query `search()` (one query at a time today).
- Making GPU the default when `use_gpu` is unset (it already defaults true in `SearchOptions`).

## 6. Review log (claudex) — adopted
Design CONFIRMED sound (`OnceLock<Option<>>`+`get_or_init` = correct single-flight thread-safe
primitive for `search(&self)`; `GpuPrefilterIndex` safe to share — read-only buffers, per-query
stream/scratch; failure-caching reasonable; cuda-gated field on 3 constructors low-risk, compiler-
checked). Refinements:
1. **Isolate the parity test on the prefilter SOURCE:** primary test = `run_prefilter(gpu)` ==
   `diagonal_prefilter(cpu)` on the same reduced query (varied ties / `max_hits` / threshold). A
   `use_gpu` true-vs-false FULL-search comparison also compares GPU-vs-CPU *alignment* downstream
   (GPU ranks/truncates before CPU traceback), so it relies on the separate downstream parity — keep
   it only as an end-to-end smoke, not the isolating check.
2. **Log the upload error once INSIDE the `get_or_init` initializer** (`match upload { Ok=>Some,
   Err(e)=>{ eprintln!once; None } }`), not `.ok()` which discards it.
3. **Reword:** it's a "CPU **prefilter** fallback" — downstream alignment may still run on GPU. And
   "never panic" → "introduces no panic path; a CUDA alloc Err surfaces as `Err` → CPU prefilter".
4. **Failure-caching = engine-lifetime circuit breaker** (upload-Err caches `None`, never retried —
   avoids repeated large-alloc attempts + log spam). A per-query `prefilter()` Err keeps the valid
   handle cached, so later queries still try GPU. Document both.
5. **Tests:** exact `PrefilterHit` equality incl. ties + `max_hits` boundary; full `SearchHit`
   equality smoke; OnceLock populated after a GPU search (cache proof); OnDisk ⇒ `None` ⇒ CPU; CUDA-
   disabled build. (Concurrent-one-upload + forced-upload/prefilter-Err want mock infra — noted as
   gaps, not built here.)

## 7. Open questions
1. `index` is set once at construction; the cached handle stays valid for the engine's life. No
   invalidation needed.
