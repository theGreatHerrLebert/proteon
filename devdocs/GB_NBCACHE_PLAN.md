# GB_NBCACHE_PLAN — cache the cutoff-GB neighbor list in NbCache

Status: DRAFT (pre-implementation). Follow-up to #154 (opt-in CutoffNonPeriodic OBC GB).
codex code-review P2 / GB_CUTOFF_PLAN §6. Builds on #154 being merged to main.

## 1. Problem
#154 computes cutoff GB on the NBL path via `gb_obc_energy_and_forces_nbl`, but builds the
**exclusion-free GB cell list fresh on every force evaluation** inside
`compute_energy_and_forces_nbl` (energy.rs). For iterative callers (minimizer line searches,
MD steps) that's a full O(N) cell-grid build per eval — wasted work, since `NbCache` already
retains + drift-refreshes the LJ list. Correct, but pays the rebuild each call.

## 2. Fix — one cached GB list in NbCache, via a PRIVATE inner entry point
**No public `Option` footgun (claudex #3).** Keep `compute_energy{,_and_forces}_nbl`'s exact
#154 signature + internal-build semantics so existing/`*_auto`/test callers are untouched and
nobody can silently pass `None` and quietly restore the per-eval rebuild. Instead:
- Extract the body into a private `compute_energy_and_forces_nbl_inner(coords, topo, params,
  lj_nbl, gb_nbl: Option<&NeighborList>)`. The GB block uses `gb_nbl` when `Some`, else builds
  internally. Public `compute_energy_and_forces_nbl` calls `inner(.., None)` (unchanged
  behavior); `compute_energy_nbl` forwards likewise. **Only `NbCache` calls `inner` with the
  cached list** — the cache path is reachable through exactly one, tested caller.
- **NbCache** (`nb_cache.rs`): add `gb_nbl: Option<NeighborList>` and store `gb_cutoff:
  Option<f64>` captured at construction. Build `gb_nbl` in `new_with_exec` iff
  `params.gb_cutoff().is_some()` AND the LJ NBL path is active (same size gate), via
  `NeighborList::build(coords, gb_cutoff, &empty, &empty)` — exclusion-free. Refresh it in
  `refresh()` alongside the LJ list: ONE shared `needs_rebuild` check (both share coords +
  cutoff + 2 Å buffer; claudex confirmed no case needs different refresh times). Rebuild both
  atomically; `debug_assert!` LJ and GB cutoffs equal to protect the shared-refresh assumption.
- **Param consistency (claudex #4):** `NbCache::energy{,_and_forces}` `debug_assert!` that
  `params.gb_cutoff() == self.gb_cutoff` so a caller can't pass a force field whose GB method
  differs from the one the cached list was built for.
- **GPU**: nothing to do — `GpuStructState::new` already refuses the cutoff method (#154), so
  `NbCache.gpu` is `None` under cutoff GB and the GB call always takes the CPU branch.

## 3. Correctness invariants (do not regress #154)
- **Cached `gb_nbl` at coords X == a fresh `NeighborList::build` at X** byte-for-byte: the
  cell-list build is deterministic, so same coords+cutoff+exclusions ⇒ identical pairs in
  identical order ⇒ identical FP. (The `_inner(.., None)` public path is literally unchanged.)
- **Cached path vs all-pairs cutoff: TOLERANCE, not byte identity (claudex #1).** Different
  enumeration ⇒ different summation order ⇒ ~1e-9 FP drift. Assert `< 1e-9`, not `==`.
- Refresh consistency: after the GB list refreshes, Born radii + spread read the same
  refreshed `gb_nbl`. The 2 Å buffer + `r²≤cutoff²` predicate make a not-yet-rebuilt list
  correct between refreshes (drift-out → predicate-filtered; drift-in → buffer covers it until
  the Verlet trigger). claudex confirmed: while every atom stays within `buffer/2` of its
  reference, pair separation changes ≤ `buffer`, so a pair outside `cutoff+buffer` cannot enter
  the physical cutoff. Requires callers to `refresh()` before each eval at new coords — MD +
  line search already do.

## 4. Tests
- **Cached == fresh-build == all-pairs**: NbCache (`CpuNbl` exec) cutoff-GB energy+forces vs a
  freshly-built `gb_nbl` (exact) AND vs the all-pairs cutoff path (`< 1e-9`), forced through
  the NBL path.
- **Drift test that PROVES a rebuild (claudex #2)** — not just "perturb + compare":
  a fixture with a GB pair initially **beyond `cutoff + 2 Å`** (absent from `gb_nbl`); move one
  atom > 1 Å until the pair is inside the physical cutoff; `refresh()`; assert the pair is now
  present in `gb_nbl` (expose `pairs`/a len accessor) AND the GB energy gains the contribution
  that would be missing from a stale list.
- **Sub-threshold preserves the list**: a move < buffer/2 leaves `gb_nbl` unchanged — assert
  via a test-only rebuild-generation counter (incremented in `refresh` when it rebuilds), so
  "no rebuild" is proven, not inferred from matching energies.
- **Exclusion semantics (claudex):** a bonded 1-2/1-3 pair is ABSENT from the LJ list but
  PRESENT in the cached exclusion-free `gb_nbl`.
- **Method coverage:** cutoff GB under `CpuNbl` and `Auto`; NoCutoff GB (`gb_nbl` stays `None`,
  unchanged); a non-OBC force field (no GB list built). `debug_assert` trips if eval-time
  `gb_cutoff` ≠ construction-time.

## 5. Non-goals
- GPU cutoff-GB kernels (separate).
- Caching the GB list for the single-point `*_auto` path (one build is negligible there).
- Changing the GB cutoff to differ from the LJ cutoff.

## 6. Review log (claudex)
Both correctness invariants CONFIRMED (shared single refresh is sound; staleness/buffer
argument holds). 4 findings adopted, none rejected: #1 cached-vs-all-pairs is tolerance not
byte-identical (cached-vs-fresh-build stays exact); #2 a real rebuild-proving drift fixture +
a rebuild-generation counter for the sub-threshold case; #3 NO public `Option` footgun — a
private `_inner` entry point, cache reachable only through `NbCache`; #4 store `gb_cutoff` at
construction + `debug_assert` eval-time params match. Plus exclusion-semantics + method-coverage
tests. Open questions resolved: one shared refresh (with cutoff/buffer-equality assert);
private-inner over a public param or bundle struct.
