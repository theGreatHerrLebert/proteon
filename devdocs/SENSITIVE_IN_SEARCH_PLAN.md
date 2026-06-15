# SENSITIVE_IN_SEARCH_PLAN — sensitive prefilter in SearchEngine::search (Phase 3d)

Status: DRAFT. Builds on #161 (`run_prefilter` wiring) + #159 (GPU `prefilter_sensitive`) +
`diagonal_prefilter_sensitive`. `search()` runs the EXACT prefilter only, so it misses the remote
homologs the sensitive (similar-k-mer-expanded) prefilter catches — the whole point of MMseqs2's
sensitivity. `SearchOptions` already documents "optionally with similar-k-mer expansion (2.3b)";
this wires it.

## 1. Design
- **`SearchOptions.similar_kmer_threshold: Option<i32>`** — `None` (default) = exact prefilter
  (unchanged). `Some(t)` = sensitive: each query k-mer expands to every similar k-mer scoring `>= t`.
- **Engine precomputes a prefilter-alphabet i32 score matrix** (the generator scores k-mers in the
  PREFILTER alphabet, which is the *reduced* alphabet when a reducer is set):
  - reducer `Some`: `reducer.reduce_matrix(matrix)` (f32 averages, reduced_size²) → scale to i32 by
    the same `bit_factor` the full matrix uses: `(v * bit_factor).round() as i32`.
  - reducer `None` (full-alphabet index): reuse the existing full `matrix_int` (already i32).
  Stored once at construction (`prefilter_score_matrix: Vec<i32>`, reduced_size² or full²).
- **`run_prefilter` dispatch:** when `similar_kmer_threshold == Some(t)`, build
  `SimilarityConfig { scores: &self.prefilter_score_matrix, threshold: t }` and call the SENSITIVE
  path — GPU `handle.prefilter_sensitive(q, skip_idx, &sim, opts)` (cached handle) with CPU
  `diagonal_prefilter_sensitive(&self.index, q, skip_idx, &sim, opts)` fallback; `None` keeps the
  exact path. (Same GPU-availability + in-memory + Err-fallback structure as #161.)

## 2. Scaling — the crux (for claudex)
The full `matrix_int = matrix.to_integer_matrix(bit_factor, 0.0)` = `clamp_round_i8(v*bit_factor)`
widened to i32. For the prefilter's reduced matrix I propose `(reduce_matrix(matrix)[c] *
bit_factor).round() as i32` — i.e. average raw scores into reduced space, THEN apply the same
`bit_factor`. Open questions:
- Is averaging-then-scaling the right order vs scaling-then-averaging? (Linear, so
  `avg(v)*bf == avg(v*bf)` up to rounding — rounding last is fine.)
- Should I i8-clamp per cell to match the full matrix exactly, or keep full i32 precision? The
  prefilter scoring space is independent of the ungapped/SW (which use `matrix_int`); the reduced
  matrix + `threshold` only need internal consistency. Lean: full i32 (no clamp) — a k-mer sums k
  cells; clamping individual reduced averages buys nothing and can distort. **The `threshold` is in
  this `bit_factor`-scaled reduced space; document that.**
- No-reducer case reuses `matrix_int` (full i8-clamped) — the threshold is then in the full-matrix
  integer scale. Acceptable (a different but consistent space per index type); document.

## 3. Correctness / parity
- The GPU `prefilter_sensitive` is **bit-exact** vs `diagonal_prefilter_sensitive` (Phase 3a) ⇒
  identical hits ⇒ identical downstream `SearchHit`s, regardless of GPU vs CPU.
- Default (`None`) path is byte-for-byte the current exact behaviour.
- The same memory/fallback hardening from #161 applies (the sensitive path reuses the cached
  `GpuPrefilterIndex` + its guards; `build_sensitive_kmers` already streams + filters in-index).

## 4. Tests
- **Sensitive-in-search parity:** engine with `similar_kmer_threshold: Some(t)`; assert
  `run_prefilter(q)` (GPU sensitive) == `diagonal_prefilter_sensitive(&index, q, skip_idx, &sim,
  opts)` (CPU) through the engine, isolating the prefilter source (varied options). Skip without GPU.
- **Widening smoke:** a moderate `t` yields >= the exact-prefilter candidate set on a small corpus
  (sensitive never drops a hit the exact path found; document the monotonicity expectation).
- **Default unchanged:** `None` ⇒ identical to the current exact search (a couple of existing tests
  already cover this; add an explicit `None == exact` assertion).
- **No-reducer:** a full-alphabet engine (`reduce_to: None`) with sensitivity also matches CPU.

## 5. Non-goals
- A *separate* sensitive threshold auto-tuned per matrix (upstream derives it) — user sets `t`.
- Sensitive in the on-disk-index path beyond what #161's CPU fallback already gives.
- Changing the default (stays exact / `None`).

## 6. Review log (claudex) — adopted
1. Reduced matrix = `round(avg(raw)·bit_factor)` (average THEN scale THEN round-nearest) — confirmed
   correct (NOT integerize-then-reduce). `clamp_round_i8` rounds nearest, so the scales match in
   spirit. **Keep i32, no i8 clamp** (a k-mer sums k cells; clamping distorts averages). Add a helper
   that rejects non-finite reduced scores while rounding to i32.
2. **Threshold is a footgun** — document in `SearchOptions` that it lives in the ACTIVE prefilter
   alphabet's `bit_factor`-scaled matrix space, is NOT a final alignment score, and is NOT portable
   across reducer / matrix / k / `bit_factor`.
3. **Drop the unconditional monotonicity claim.** Sensitive ⊇ exact ONLY when `threshold ≤` the
   min exact self-k-mer score (else exact windows whose self-score `< t` vanish) AND
   `max_prefilter_hits == None` (truncation after sort can drop an exact candidate). Test
   monotonicity only under those conditions; otherwise just test GPU==CPU parity. Add an explicit
   non-monotonicity-with-`max_hits` test that DOCUMENTS the truncation behaviour.
4. **Dispatch (critical):** when `Some(t)`, EVERY branch — GPU success, GPU-Err fallback, no-GPU,
   on-disk — must call the SENSITIVE path (`diagonal_prefilter_sensitive`), never exact. Factor the
   cached-handle lookup into a `gpu_handle()` helper so the exact and sensitive arms share it without
   duplicating the get_or_init. `debug_assert` the score-matrix shape (`== alphabet_size²`).
5. **Tests:** reduced-matrix unit (merged classes + rounding-last); `prefilter_score_matrix.len()
   == encoder.alphabet_size()²`; CPU sensitive engine parity WITH a reducer AND with `reduce_to:
   None`; GPU-Err falls back to sensitive CPU (not exact); the `max_hits` non-monotonicity doc test.
6. Update `SimilarityConfig`'s doc (it currently says scores come from `to_integer_matrix` only).
