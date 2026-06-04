# Plan: complete the EXACT analytic SES mesh (finish the BALL triangulation port)

> **DECISION (hybrid port).** Clean-room Rust for the meshing (reduced surface →
> patches → shared registry → CDT contact caps) where elegance is fine; **port
> BALL's `SESSingularityCleaner` faithfully** (its graph-rewrite choices +
> thresholds) since oracle parity on singular cases demands BALL's exact logic.
> Gated vs `ball-py ses_mesh`/`ses_area` on BOTH non-singular and singular
> fixtures. Rationale: strongest refutation of "not portable" — *every* part
> ports, including the gnarly cleaner — without transcribing the global-EPSILON
> hacks where a clean equivalent is provably equivalent. Phasing: two-atom (done)
> → triangle3 → general non-singular N → faithful singularity cleaner →
> density/large-protein gates. claudex-reviewed.

## Why (the actual point)

The originating challenge: the boss claims BALL's SES **mesh triangulation** is
"not portable." Goal = port it (clean Rust, gated against BALL) to prove him
wrong. The grid/SDF mesher (`volume.rs`) we built is a *different algorithm* — a
watertight approximation that converges to BALL's numbers — so it does **not**
settle the bet. The deliverable that does is the **exact analytic Connolly mesh**:
vertices lying exactly on the SES patches (contact spherical caps, toric reentrant
surfaces, spheric concave triangles), exact patch boundaries, watertight, gated
vs `ball-py ses_mesh`/`ses_area`.

**Clean-room, not transcription.** "Portable" is proven by an independent clean
implementation that produces an equivalent (oracle-gated) mesh — NOT by copying
BALL's `solventExcludedSurface.C` line-for-line with its global mutable
`Constants::EPSILON` and magic refinement constant. A clean port that matches the
oracle is a *stronger* refutation than a faithful transcription of the hacks.

## What is already built (reuse, don't rewrite)

- `geom` (L0), `rs` (L1, gated vs `reduced_surface_stats`), `ses` (L2, gated vs
  `ses_graph`) — the combinatorial pipeline through the SES element graph is DONE.
- `arrangement` — spherical-circle arrangement of an atom's incident contact
  circles (`exposed_arcs`, robust angular classifier) = the boundary engine for
  multi-hole contact caps.
- `patches` — `spheric_face_mesh` (geodesic triangle), `toric_face_mesh`,
  single-hole `contact_cap_mesh`, `fill_loop_on_sphere`.
- `stitch` — two-atom assembly (toric ring + 2 caps, shared rims), gated vs
  `ses_area` to 1%.
- `mesh` — index mesh + invariants (area, signed volume, watertight, Euler,
  orient_consistently, weld).

## What remains (the hard half)

### L4a — multi-hole contact cap (THE crux that caused the pivot)
An atom's contact face = its sphere minus the union of buried caps of all incident
neighbours. With ≥2 holes the exposed region is **not a disk** (sphere-with-k-holes
= genus-0 with k boundary loops) so `fill_loop_on_sphere` (single fan) cannot mesh
it. `arrangement::exposed_arcs` already gives the exact boundary loops. Need a
**sphere-region triangulator** for those loops. Options:
  - **(A) Stereographic chart + constrained Delaunay.** Project the exposed
    region (boundary loops sampled from `exposed_arcs`) from a pole inside the
    region to a plane; constrained Delaunay triangulate with the loops as
    constraints; lift vertices back onto the sphere (exact). Robust, handles any
    #holes / non-convex. Needs a 2D CDT (crate `spade`, or hand-rolled
    Bowyer–Watson + constraint insertion). **Recommended.**
  - **(B) BALL's clip-template-icosphere + advancing-front gift-wrap.** Faithful
    but gnarly (global EPSILON, snap tol). Rejected earlier.
  - **(C) Arrangement + per-loop fan.** Only works for a single loop (disk) —
    insufficient for the annulus/k-hole case.

### L4b — bounded toric + spheric faces, shared registry
- Toric faces between a pair are **bounded** (θ capped between the two spheric
  probe positions) for triple-incident edges; free (full 2π ring) otherwise.
  `toric_face_mesh` already parameterizes θ-range.
- Spheric faces = `spheric_face_mesh` on the probe sphere, bounded by 3 concave
  arcs.
- **Shared-index registry** (`TO_SES_STITCHING.md`): every shared boundary curve
  (contact circle, concave arc) sampled ONCE; adjacent patches index the same
  vertices → watertight by construction. Canonical direction + per-patch reversed
  flag for orientation. Intern SES vertices (probe-contact corners) by topological
  identity.

### L3 — singularity handling (`singular.rs`)
When the probe self-intersects (atoms close enough that the reentrant surface
folds), BALL's `SESSingularityCleaner` splits/deletes singular toric faces. This
is the fragile, robustness-critical layer. Open question: port BALL's cleaner as a
graph rewrite (gated vs `ses_graph(cleaned=true)` pre/post), OR resolve
singularities geometrically via the arrangement/clipping we already have. Likely
needs the faithful graph-rewrite logic (same thresholds) — isolate + gate hard
with metamorphic tests (permuted order, translation, scale, ε across threshold).

## Oracle gates (per layer)
- L3: `ses_graph` before/after cleaning — face/edge/vertex deltas, Euler χ of the
  SES complex, metamorphic invariance.
- L4: `ses_mesh` + `ses_area` — INVARIANTS not vertex-exact: watertight + 2-manifold,
  per-component Euler/genus vs BALL, mesh area → analytic `ses_area` as density
  rises, signed volume > 0 and ≈ BALL, **no self-intersections**, every vertex on
  its owning patch within ε, contact samples outside every buried cap (ownership).

## Review adjustments (claudex)

- **Contact cap = registry-first, CDT crate, no Steiner on the boundary.** The CDT
  consumes 2D *images of existing registry vertices*; it must NOT insert Steiner
  points on constrained boundary edges (or, if it does, split that curve in the
  global registry and force the adjacent toric/spheric patch to the same split).
  Interior Steiner points are fine — lift by inverse projection and normalize onto
  the atom sphere (still exactly on-patch). Use a real CDT crate (`spade` / `cdt`),
  not hand-rolled. **Chart selection is part of the algorithm, not a helper**:
  pick a pole/tangent-chart *outside* the exposed component and away from all
  boundary samples, check pole→boundary angular conditioning, allow multiple
  charts for multiple/near-hemispheric components.
- **"Exact" = exact boundaries + on-sphere interior.** Interior triangulation need
  not be geodesically exact — every vertex on the analytic sphere + exact boundary
  identity + convergence under refinement is what "analytic mesh" requires. The
  grid mesh can't be reused (its vertices aren't organized by patch ownership).
- **Singularities are a separate ACCEPTANCE GATE, not an enhancement** — and the
  dominant risk (specification, not engineering): `SESSingularityCleaner` is
  topological surgery encoding BALL's fold-through decisions. Detection can be
  geometric (self-intersecting toric, invalid probe arcs); **oracle parity likely
  needs reproducing BALL's graph-rewrite choices**. Not rare in real proteins
  (dense packing, omitted H, alt radii). Excluding singular cases makes the final
  "portable" claim weak — so it's a gate, phased last, but required.
- **Watertight = global edge-incidence invariant.** Registry must own ALL boundary
  samples incl. endpoints, triple-point SES vertices, reversed-direction views,
  and refinement splits. The real failure is topological cracks from
  numerically-coincident-but-different-index duplicates — assert every
  boundary-derived edge is used by exactly 2 triangles after assembly.
- **Deliverable phrasing (scoping):** "portable analytic Connolly/BALL-equivalent
  SES mesh, oracle-gated against BALL behavior." A clean-room equivalent proves
  the construction is *portable/reimplementable*; it does NOT prove BALL's exact
  tangled source transplants line-for-line. If the boss's claim is the latter,
  the clean-room route leaves room to move goalposts — **confirm what "ported"
  must mean before committing.**

## Open questions for review
1. Contact cap: stereographic+CDT (A) vs revisit gift-wrap (B)? Pull in a CDT
   crate (`spade`) or hand-roll? Is stereographic projection numerically safe for
   near-hemisphere exposed regions (pole placement)?
2. Singularities: faithful graph-rewrite port vs geometric resolution via the
   arrangement? Can we even *reach* singular configs cleanly with the clean-room
   approach, or do we inherit BALL's perturbation `correct()` (already in `rs`)?
3. Watertight stitching: is the shared-registry sampling enough, or do
   contact↔toric↔spheric boundaries need explicit vertex interning at the
   triple-point SES vertices? Where are the crack risks?
4. Phasing: smallest gateable increments from two-atom (done) → triangle3 (one
   spheric face, 3 bounded toric, 3 two-hole caps) → general N.
5. Effort/robustness: is the singularity layer the dominant risk, and is there a
   clean-room way to sidestep BALL's global-EPSILON without losing oracle parity?
