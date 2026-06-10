# P6.5 — Near-singular remediation for the regular-Yukawa collocation

**Status:** implemented (opt-in; default stays Fixed — see §5). Reviewed by: codex on the
design (round 1, §§ below) and on the implementation (round 2); Duffy self-term round 3.

**Duffy self-term (round 3, landed).** The on-panel **single-layer** self term now uses a
**Duffy** graded rule (centroid fan → collapse the cusp vertex → tensor Gauss–Legendre,
GL-16), replacing the cusp-limited fixed 7-point value. It matches an independent
high-resolution quadrature to ≤1e-4 on quality panels and beats the fixed 7-point self by
100×–10⁷× (the fixed self is **5e-3 → 28%** wrong as κ grows — much larger than first
estimated). The **double-layer** self stays NESSie's fixed value on purpose: with ξ and x
coplanar, `(x−ξ)·n ≡ 0`, so its integrand is zero a.e. and `κ²/(2√3)` is a regularisation
convention to preserve, not a quadrature target. Net solve effect remains small
(energy-insensitivity gate Δ rose from ~3e-6 to ~5e-5, still ≪ 1e-4) — consistent with §5.

**Headline finding (solve level):** adaptive provably fixes the near-singular *cross-
entries* (kernel: 0.3%/3% → ~1e-7), but that error **does not propagate to the solved
energy** (~1e-6 on a two-sphere cleft, even charges-at-cleft / gap 0.1). So the default
stays Fixed/fast; adaptive is opt-in. Details in §5.

Implementation-review fixes, all landed:
- **[P2a]** `point_to_triangle_distance` now has a degenerate-triangle fallback (closest
  over the three edges) — the Voronoi divisions could divide by zero on collinear
  vertices that `Tri::with_normal` does not reject.
- **[P2b]** the matrix assembly treats the **self** term by index identity (`j == i`),
  not the `d ≤ ETOL` distance test (which kept only as a direct-call safety net) — so a
  sub-ETOL cleft can never be misclassified as self.
- **[P3]** `Quadrature::Adaptive(AdaptiveConfig)` carries its config, so a caller who
  sees `SolveStats::capped_panels > 0` can raise `max_depth`/`rtol`.

## 0. Key correction from review — the kernel is *not* smooth

The regular single-layer kernel expands as
`(e^{−κr}−1)/r = −κ + κ²r/2 − …`, which contains `r = |x−ξ|`. `r` has a **cusp**
(non-differentiable) at the source point, so although the *value* is finite (the `1/r`
singularity cancels), the integrand is **not smooth** in Cartesian coordinates. Two
consequences drive the revised design:

- Polynomial cubature (Radon) loses its nominal high-order accuracy near the cusp, so a
  pure coarse-vs-refined **difference test can falsely converge** (both rules alias the
  same missed feature). The estimator therefore needs a **deterministic resolution
  floor** first, with the difference test as a *secondary* refinement, plus
  **estimator-effectivity** validation (`|true error| / |estimate|`) against references.
- The **self / near-self** panel (cusp inside or adjacent to the panel) is better
  handled by a **centroid polar fan** (split into 3 sub-triangles at the closest point)
  than by deep generic recursion — bounded and predictable.

The sections below are revised accordingly; review notes are inlined as `[R#]`.

## 1. Problem, located precisely

The nonlocal BEM energy plateaus at a few percent off the closed-form Born energy
and **does not converge** as the mesh refines
(`tests/born_convergence.rs::nonlocal_bem_energy_matches_born_within_radon_floor`),
while the *local* energy (analytic Laplace collocation) converges monotonically to
<3% on the same meshes. The difference isolates the floor:

- **Laplace** single/double-layer collocation is the **exact** analytic Rjasanow
  integral of `1/r` over a flat triangle — accurate at any observation distance, so
  the local solve has no quadrature floor.
- **Regular Yukawa** (`Vy`/`Ky` = Yukawa − Laplace) is integrated with a **fixed
  7-point Radon cubature** (`quadrature.rs`, `yukawa.rs`). The regular kernel is
  *smooth* (the `1/r` singularity cancels analytically), but for an observation point
  `ξ` close to triangle `j` relative to its size, the integrand is sharply **peaked**
  near the closest point; 7 points under-resolve that peak. On a refined SES/sphere
  mesh, near-neighbour pairs proliferate, so the error stops shrinking.

So the remediation target is **exactly** the regular-Yukawa collocation for
near-field element pairs. Laplace is left untouched (exact already).

## 2. The remediation: adaptive panel subdivision with an error estimate

Not "adaptive quadrature" in the abstract — a concrete, bounded scheme.

### 2.1 Near-field criterion (when to subdivide)

For observation point `ξ` and triangle `T`:

```
d  = point_to_triangle_distance(ξ, T)        // exact closest-point, not centroid
h  = longest_edge(T)                          // diameter proxy — [R2] NOT sqrt(2A),
                                              //   which understates skinny triangles
near  ⟺  d < NEAR_FACTOR · h                  // NEAR_FACTOR ≈ 4 (EMPIRICAL, calibrated
                                              //   across κh and shape — not a bound)
```

`d < NEAR_FACTOR·h` ⇒ subdivide; else one 7-point Radon eval is within tolerance.
**[R2]** difficulty also depends on `κh` (the Yukawa scale across the panel) and element
shape/closest-point location (interior / edge / vertex); `NEAR_FACTOR` is an *empirically
calibrated* trigger, **not** a numerical guarantee, so far-field acceptance is justified
only by the far-field micro-corpus (§4), not by the criterion alone. Poor-quality
elements (high aspect ratio) are flagged for the mesh-acceptance gate rather than trusted
silently. The self pair (`ξ` = own centroid, `d = 0`) is the limiting near case → §2.3.

`point_to_triangle_distance`: standard closest-point-on-triangle (project to plane;
clamp to nearest edge/vertex if outside). ~30 lines, unit-tested against brute-force
sampling, covering interior/edge/vertex closest points.

### 2.2 Subdivision + estimator (how much to subdivide)

**[R1/R3]** A bare difference test is unsafe near the cusp (false convergence). The
scheme is therefore *resolution floor first, difference test second*, and it returns a
**status**, never a silent "converged":

```
adaptive(T, depth):
    // [R1] deterministic floor: refine until the panel is fine enough in BOTH the
    // geometric (d/h) and Yukawa-scale (κh) senses, regardless of the difference test.
    if depth < min_depth(d(ξ,T)/h(T), κ·h(T)):
        return Σ_k recurse(subs(T), depth+1)              // forced refinement

    coarse  = radon7_collocation(T)
    subs    = midpoint_split(T)                            // 4 congruent children
    refined = Σ_k radon7_collocation(subs[k])
    // [R5] scale tol to a non-cancelling local magnitude (handles double-layer zero
    // crossings), with a panel-scaled absolute floor; SEPARATE single/double scales.
    est  = |refined − coarse|
    tol  = RTOL · Σ_k Σ_i |w_i · f(p_i^k)|  +  ATOL_panel(kind, A, κ)
    if est ≤ tol:
        return (refined, Converged)
    if depth ≥ MAX_DEPTH:
        return (refined, Capped)                           // [R3] NOT "converged"
    return Σ_k ( near(ξ, subs[k]) ? adaptive(subs[k], depth+1)
                                  : (radon7_collocation(subs[k]), Converged) )
```

- `min_depth(d/h, κh)` is a small deterministic table/formula (e.g. deeper when `d/h`
  small or `κh` large), calibrated so estimator **effectivity** `|true err|/|est| ≈ O(1)`
  on the corpus — validated, not assumed.
- `RTOL ≈ 1e-6`; `ATOL_panel` from a kernel bound × area × physical coefficient,
  separate for single vs double. `MAX_DEPTH ≈ 6` (4⁶ leaves worst case).
- **[R3]** a `Capped` status propagates up; the accurate solver **counts capped entries
  and surfaces them** (a diagnostic, optionally an error) — it never reports a capped
  matrix as converged.
- Far children are accepted with one eval **only because** the near predicate is gated
  as accuracy-sufficient on the corpus (§4) — the gate, not the geometry alone,
  certifies it.

### 2.3 On-panel (self) vs off-panel (cleft) — **revised after experiment**

The two cases turned out to need different treatment:

- **Off-panel interior** (`ξ` projects inside the face but is at distance `d > ETOL` —
  the **cleft / opposing-surface** case): the cusp at `x = ξ` is *off* the integration
  domain, so the integrand is smooth on the panel. A **centroid fan** about the
  projected point `p*` (3 sub-triangles `(p*, vᵢ, vᵢ₊₁)`) + resolution-floor recursion
  converges cleanly. *Verified:* `cleft_opposing_panels_beats_fixed` — fixed is off by
  0.3% (single) / 3% (double) per cross-entry, adaptive by ~1e-7.
- **On-panel** (`d ≤ ETOL`, the **self / coincident** term): the cusp lies *on* the
  domain. Subdivision does **not** converge here — every fan/midpoint refinement still
  straddles the cusp, so the estimator caps (observed: the whole diagonal capped). This
  is the regime where polynomial cubature genuinely fails (review [R1/R4]). **Resolution
  (round 3): a Duffy graded rule for the single layer.** Fan the panel at the centroid
  (cusp → a vertex of each sub-triangle), then Duffy-map each sub-triangle so the
  collapsed-vertex Jacobian `u` provides the radial grading that absorbs the cusp
  (`∫ r dA` in polar is `∫ ρ·ρ dρ dθ`, smooth), with a tensor Gauss–Legendre rule
  (GL-16). The **double layer** keeps NESSie's fixed `κ²/(2√3)` regularised value: with
  ξ, x coplanar the integrand is zero a.e., so there is no quadrature to improve — only a
  convention to preserve. (Laplace self stays the exact analytic InPlane form.) In the
  matrix assembly the self term is routed by index identity (`j == i`); the `d ≤ ETOL`
  distance branch is the equivalent direct-call path.

## 3. Integration — keeping the GPU parity gates valid

The GPU matrix-free Yukawa kernel (`yukawa_matvec`) is the **fast** path; porting
recursive subdivision to it is out of scope (divergent per-thread work). Decision:

- `yukawa_matrices` (fixed 7-point) is **unchanged** — it remains the GPU-parity anchor
  (`gpu_matrix_free_yukawa_matvec_matches_cpu_dense` keeps comparing fixed-vs-fixed).
- New `yukawa_matrices_adaptive(elements, yukawa, cfg)` — the accurate CPU assembly.
- The CPU nonlocal solve gains a quadrature selector:
  `solve_nonlocal_elements_q(…, Quadrature::{Fixed,Adaptive})`.
- **[R6]** the quadrature mode is **reported in `SolveStats`** (and surfaced in the
  Python result dict + a log line), so two same-named solves with different physics can
  never be confused.
- **[R6] the size-aware dispatcher must NOT silently change accuracy.** As designed, the
  auto-dispatcher would route large nonlocal meshes to the fixed-quadrature GPU path and
  small ones to adaptive CPU — a silent accuracy switch by mesh size. Resolution:
  `solve_nonlocal_elements_auto` carries the requested `Quadrature`; when the GPU path
  cannot honour it (GPU = Fixed only), it either (a) stays on CPU adaptive, or (b)
  returns Fixed **with the mode explicitly reported and warned** — never a silent swap.
  The default-policy decision (§ open Q3) interacts with this.
- The GPU-vs-dense solve parity test is repointed at the **Fixed** CPU solve (a valid
  matrix-free-vs-dense gate *at equal quadrature*); adaptive accuracy is gated by the
  Born-floor + cleft corpus (§4).

This gives a coherent, *observable* story: **GPU = fast, fixed/floored; CPU =
accurate when Adaptive is requested**, with the mode always reported. A future P8 item
ports adaptive (or a corrected near-singular rule) to the GPU.

## 4. Gates (how we know it works — not "trust the subdivision")

1. **Near-singular micro-corpus vs high-precision quadrature.** Triangle pairs spanning
   the *actual* difficulty axes **[R7]**, not just `d/h`:
   - gap `d/h` ∈ {2, 1, 0.5, 0.1, self};
   - element shape: equilateral, **skinny**, **obtuse**;
   - closest point: interior, edge, vertex;
   - Yukawa scale: sweep `κh` (small → large);
   - **a manufactured cleft**: two nearly-parallel opposing panels at small separation
     (the real SES re-entrant failure mode the sphere test cannot exhibit), checking the
     **double-layer cancellation** (signed, near-zero) specifically.

   Reference **[R1/R7]**: the brute-force `integrate_tri` of the *physical* regular
   kernel is trustworthy only after its **own convergence study** (increasing order/
   depth) and, for self/near-self, a **polar (fan) high-precision** evaluation and/or an
   **independent formulation** — a same-branch brute force can reproduce an
   implementation error. Adaptive must hit ≈1e-6 rel where fixed 7-point misses by ≫ that.

2. **Series/closed-form + self convergence rate.** Across the `κr = 0.1` series boundary
   and the `r ≤ ETOL` self branch, test the **limit value and convergence rate** of the
   fan (not only continuity) — **[R4]** a continuous-but-inaccurate branch must fail.

3. **Operator-level cleft error (the real proof) + honest sphere observation.**
   **Revised after experiment — [R7] was right:** the convex-sphere Born energy is **not**
   a near-singular demonstrator. It has no opposing surfaces, so adaptive ≈ fixed there
   (verified, `nonlocal_adaptive_matches_fixed_on_convex_sphere`) and its few-percent
   floor is general discretisation, not the near-singular artifact. *Do not* gate on
   "adaptive tightens the sphere Born" — it doesn't, and claiming so would be false.
   The proof is the **opposing-panel cleft** collocation gate
   (`cleft_opposing_panels_beats_fixed`): per cross-entry, fixed is off 0.3%/3%, adaptive
   ~1e-7. A full **solve-level** demonstration needs a non-convex analytic mesh with a
   cleft (a dumbbell / two-sphere geometry) — **future work**, since the sphere mesher is
   the only analytic generator in-tree today.

4. **No far-field regression.** Far elements (the bulk) must be **bit-identical** to the
   fixed path (the criterion gates them out), so entrywise assembly / Cauchy-data parity
   is untouched there.

5. **Estimator effectivity + cap accounting.** Report `|true err|/|estimate|` on the
   corpus (must be O(1), not ≪1 — proving no false convergence), and assert that the
   corpus drives **zero `Capped` entries** at `MAX_DEPTH` (else the cap, or the floor, is
   mis-set).

## 5. Scope / non-goals

- **In:** adaptive regular-Yukawa collocation (resolution-floor recursion + centroid
  fan for self/near-self); point-to-triangle distance; the five gates incl. the cleft
  corpus; `Quadrature` selector reported in stats.
- **Default policy — RESOLVED by experiment: stays opt-in, do NOT flip.** The solve-level
  investigation (`tests/cleft_solve.rs`) found that although adaptive fixes the
  near-singular *cross-entries* (0.3%/3% → ~1e-7, kernel gate), that error **does not
  propagate to the integrated reaction-field energy**: on two close spheres, even with
  the charges *at* the cleft and the gap squeezed to 0.1, the solved nonlocal energy
  moves by only ~1e-6 between fixed and adaptive. The global GMRES solution averages out
  a few slightly-wrong entries, and the energy is dominated by each component's
  self-solvation (identical under both). So flipping the default would impose adaptive's
  subdivision cost on **every** nonlocal solve for ~1e-6 of energy accuracy on every
  testable geometry — not worth it. Adaptive stays **opt-in** (`Quadrature::Adaptive`),
  correct and available for callers who want it. Whether near-singular *ever* materially
  moves an energy is reopened only by a true **SES re-entrant** (toric/saddle) mesh,
  where near-singular pairs form a connected concave region rather than a few weakly-
  coupled facing caps — not generable analytically in-tree today (awaits the SES mesher).
  This refines the plan's "single highest failure risk" framing with evidence: the
  per-entry floor is real, but its **energy impact is small** for separable geometries.
- **Out (this round):** Laplace near-singular (analytic/exact); GPU adaptive; Duffy
  coordinate transforms (value is finite — the fan suffices); the full P6.5 mesh-
  acceptance gate (separate work item).

## 6. Open questions — resolved by review

1. **Estimator validity** → **resolved [R1]:** bare difference test is unsafe (cusp,
   false convergence). Adopt resolution-floor-first + difference-test-second, validate
   effectivity. Done in §2.2.
2. **`NEAR_FACTOR` / tolerances** → **[R2]:** `h` = longest edge not √(2A); criterion
   includes `κh`; `NEAR_FACTOR` empirical, calibrated on the corpus, not a bound.
3. **Default = Adaptive** → **[R7]:** opt-in until the cleft + operator-level gates pass,
   then flip. Done in §5.
4. **Double-layer tolerance** → **[R5]:** mixed `RTOL·Σ|wᵢfᵢ| + ATOL_panel`, separate
   single/double scales. Done in §2.2.
5. **Self panel** → **[R4]:** centroid polar fan (cusp at a vertex), gate the convergence
   rate; no full Duffy transform (value finite). Done in §2.3.
6. **GPU/CPU accuracy split** → **[R6]:** report quadrature mode in stats/API; dispatcher
   must never silently change accuracy by size. Done in §3.

### Remaining calibration (settled during implementation, on the corpus)

- The exact `min_depth(d/h, κh)` table and `NEAR_FACTOR` value — fit to estimator
  effectivity O(1) and zero `Capped` entries on the corpus (gate 5).
- `ATOL_panel(kind, A, κ)` constants for single vs double.
