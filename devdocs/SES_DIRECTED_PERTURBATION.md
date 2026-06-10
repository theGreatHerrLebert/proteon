# Directed perturbation for the SES CDT-crossing latency

## Problem

The shipped fix (commit d7708b8) routes a CDT "boundary crossing" through the
whole-protein atom-perturbation retry (`build_with_perturbation_retry`,
`is_degeneracy_error`). It is **correct and BALL-validated** (recovered surfaces
match BALL area ≤0.32%), but slow: measured recovery depth is **5–11 perturbations**
(median 9 of 12). Each attempt re-meshes the *whole protein*, so a recovery costs
5–11 whole-protein remeshes and a residual case pays up to 12 before the grid
fallback. On the 22-protein crossing subset, 15 (the large ones) time out at 600 s.

Why so deep: `perturb_atoms` jitters **all** atoms in a **random direction** at a
magnitude that only reaches its 1e-2 Å cap at attempt 8. The crossing is a *local*
near-tangency between two contact circles on **one** atom's sphere, so an undirected
global walk wastes most attempts: early ones are too small to open the tangency, and
even at 1e-2 the random direction may not open *this* tangency. (Measured: 1ijp
cleared at 1.6e-3/attempt-5 by luck; 3lwa/8t5l/9hfa/5qs5 all needed the 1e-2 cap.)

A budget cap is therefore not viable — it loses the recoveries (verified: cap=3 →
9hfa ERRs). The cost is intrinsic to *undirected, whole-protein* perturbation.

## Proposed fix: directed perturbation

The crossing happens inside `fill_spherical_region` for a **specific** contact cap —
atom `a` (known at the call site, `assemble.rs:714/724`), whose cap is bounded by
arcs to a known neighbour set. Instead of jittering all atoms randomly:

1. **Identify the failing atom.** When the cap fill for atom `a` returns a "crosses
   an existing constraint" error, attach `a` (and optionally its cap-neighbour atom
   indices) to the error.
2. **Perturb only that local set.** On retry, jitter **only atom `a`** (and/or its
   cap neighbours) — the atoms whose mutual geometry defines the near-tangent
   contact circles — leaving the rest of the protein at its exact coordinates.
3. **Whole-protein remesh stays** (weld-safe by construction: the whole graph is
   rebuilt from the perturbed atom set). The win is *fewer attempts*: a directed
   nudge of the right atom opens its tangency in ~1–2 tries instead of 5–11.

This keeps everything that makes the current fix correct (whole-protein rebuild +
weld + AnalyticPerturbed provenance) and only changes *which* atoms move and *why*,
collapsing the attempt count.

## Design choices / open questions

1. **Which atoms to perturb?** Options: (a) atom `a` alone; (b) `a` + its cap
   neighbours; (c) the specific near-tangent neighbour *pair* (requires mapping the
   two crossing chart-edges back to their arcs → neighbours, more plumbing). (a) is
   simplest and moving `a` perturbs *all* its contact circles at once; is it enough,
   or is (b) needed because moving `a` rigidly may preserve a tangency that only
   moving a neighbour breaks?
2. **Magnitude / schedule.** Undirected needed ~1e-2. A directed nudge can likely
   start larger (skip the sub-1e-3 attempts that can't open a tangency). Proposed: a
   few directions at, say, 5e-3 and 1e-2. Keep ≤1e-2 cap (area stays within ~0.05%).
   Does directed change the area-fidelity argument (fewer atoms move ⇒ *smaller*
   area perturbation than today)?
3. **Determinism.** Seed the directed jitter by (atom index, attempt) exactly as
   `perturb_atoms` does, so it stays reproducible run-to-run.
4. **Threading the atom set** error→retry→perturb. The retry loop currently is
   generic and classifies by error string. Cleanest: a typed error (or a small
   side-channel) carrying the atom index, vs. encoding it in the message and parsing
   (consistent with the existing string-matching but hacky). Which is cleaner given
   the `anyhow`-based flow?
5. **Multiple / shifting failure sites.** Perturbing `a` may surface a crossing at a
   *different* atom `b` next attempt. Accumulate the perturbed set across attempts
   (union), or perturb only the current failing atom each time? Accumulation risks a
   growing moved-set; per-attempt risks oscillation. Which converges?
6. **Fallback.** Keep the undirected whole-protein jitter as a backstop if directed
   fails after k attempts, or go straight to the grid fallback? (The hybrid grid
   fallback already guarantees a watertight mesh.)
7. **Other degeneracy classes** (cospherical, degenerate caps, RS-face) keep the
   existing undirected retry — directed only applies to the crossing class. Confirm
   no regression to those paths.

## OUTCOME — the bottleneck was `build_graph`, not the retry

Profiling (env `SES_PROFILE`) overturned the premise. The latency is **not**
attempt-count: per build on 9hfa (1558 atoms) was `build_graph` **9.2s** + toric 1.1s
+ caps 4.3s, and `build_graph` re-runs every retry. On 1sq9 (2981 atoms) `build_graph`
**alone exceeded 120s**. Root cause: `enumerate_rs_faces` / `enumerate_toric_faces`
brute-forced **all O(N³) atom triples/pairs** with an O(N) clearance/blocker scan —
no spatial acceleration. (This is also why BALL, which uses spatial acceleration, is
seconds where proteon was minutes.)

**Fix (shipped): a uniform `NeighborGrid`** (cell = `2·(r_max+probe)`, the interaction
cutoff). An RS face's three atoms are pairwise within the cutoff, and every
clearance/blocker atom is within the cutoff of the probe/roll-centre, so the 27-cell
stencil is provably complete — the enumerated faces are **bit-identical** to brute
force (asserted by `grid_enumeration_matches_brute_force`, and all BALL-gated tests
pass). Complexity O(N³)→O(N·k²).

Measured: `build_graph` 9.2s→0.73s (12.6×) on 9hfa; 1sq9 went from "couldn't finish
`build_graph` in 120s" to a full mesh in 411s. On the 22-protein crossing subset the
analytic recoveries went 5→11 and timeouts 15→7. This also attacks the general
large-protein slowness behind the original 163/317 corpus timeouts — not just the
crossing cases. The remaining 7 timeouts are the 4000–6000-atom proteins where the
cap-loop (chart fill) × retries is now the dominant cost.

## Validation plan

- Re-run the 22-protein crossing subset; expect the 5 current recoveries to clear in
  ≤2 attempts (vs 5–11), and the large timeouts to either recover fast or reach the
  grid fallback far below the 600 s wall.
- Area parity unchanged vs BALL (≤0.32%) on the recovered set.
- All surface unit tests pass; determinism test for the directed jitter.
