# TO_SES_NONRADIAL_INTEGRATION — wiring burial into the analytic SES

## Goal

Close the crambin self-intersection gap (watertight but +1.1% area, 5000+
self-intersecting triangle pairs) by trimming the **nonradial** (distinct-probe
collision) overlaps — the measured-dominant class (1575 vs 21 spindle). The four
foundations are built, tested, committed:

- `nonradial::probe_burial_cap` / `spheric_face_caps` — burial cap + the
  great-circle reformulation (`spheric face = exposed({3 great circles})`).
- `intersect::self_intersections` — the embedding gate.
- `arrangement::arrange_loops` — the DCEL linker that survives multivalent
  vertices (burial caps create them).
- `nonradial::canonical_burial_circle` / `sample_circle_rim` — the bit-identical
  shared seam keyed by the unordered probe pair.

This plan is the integration that combines them in `ses_mesh_analytic`.

## Current spheric-face path (assemble.rs:565–591)

Each RS face → a spheric triangle on its probe sphere, bounded by 3 toric-boundary
arcs (`arc_on_sphere` between the 3 contact points), filled by
`fill_spherical_region`. Toric faces are meshed separately and share those 3 arcs
bit-exactly.

## Proposed change

### A. Spheric faces minus burial

For each RS face `f` (probe centre `p`):
1. Contact directions `d_k = (cs_k − p)/probe`; great-circle caps
   `spheric_face_caps([d0,d1,d2])`.
2. Burial caps: every other probe `q` with `|p−q| < 2·probe` →
   `probe_burial_cap(p, q)` (a `SphereCircle` in `p`'s frame).
3. `arrange_loops([3 great circles] ++ [burial caps])` → boundary loops as
   `BoundaryArc`s, each tagged by its source circle.
4. **Sample with seam awareness**:
   - great-circle arcs → as today (`arc_on_sphere`, the bit-exact toric seam).
   - burial arcs → from `canonical_burial_circle(f, q)` + `sample_circle_rim`,
     so face `f` and face `q` lay down the *same* world points on `C`.
5. `fill_spherical_region` over the sampled loops (azimuthal chart, pole =
   buried-antipode as today).

### B. Toric faces also collide

Two fixed probes `< 2·probe` apart also bury parts of each other's **toric**
reentrant arcs (the rolling-probe sweep passes through a neighbour's exclusion).
The toric face mesh (`toric_face_mesh`) samples reentrant arcs `T_i(θ)→T_j(θ)`;
some of those points are inside a neighbour probe → must be trimmed, sharing the
same canonical `C` seam with the spheric/other-toric faces that border it.

## The hard part: seam matching at corners

A shared seam between faces `f` and `q` is the arc of `C = sphere_f ∩ sphere_q`
that is (i) inside **both** spheric triangles and (ii) globally exposed. Its
*interior* welds via `sample_circle_rim` (bit-identical). Its **corners** are the
problem:

- **Global corner** — `C` meets a *third* sphere `r` (probe or atom). That is a
  frame-independent world point (`C ∩ sphere_r`), so it can be computed
  canonically and shared by both faces. ✓ weldable.
- **Face-local corner** — `C` exits face-`f`'s triangle by crossing one of `f`'s
  *own* toric great circles. That point is **not** on `q`'s boundary (per SEAM
  SCOPE) — it bounds a *non-shared* stretch. There, face-`f`'s burial arc must
  hand off to face-`f`'s great-circle arc, and the surface on the `q` side is a
  *different* SES element. ✓ no weld needed with `q`, but the local hand-off must
  still be exact.

So the sampler must, per burial arc, classify each endpoint global-vs-local and
source global corners canonically.

## REARCHITECTURE after codex review (DECISIVE — supersedes A+B above)

Codex verdict: **do not** subtract burial caps per spheric triangle — that is a
spherical-envelope op, not BALL's cleaned topology. The per-face DCELs cannot
independently agree on the shared circle's arc decomposition, and toric burial is
*not* `sphere(f)∩sphere(q)` (toric points ride a rolling probe sphere). Port
BALL's `SESSingularityCleaner` graph-rewrite instead, staged:

- **Stage 1 — triple-sphere vertex registry.** `triple_sphere_intersections`
  (0/1/2 points equidistant `probe` from three probe centres) cached by key
  `(sorted i,j,k, branch)` — three spheres give *two* branches, so the branch bit
  is mandatory (Q2). These are the canonical SES vertices where singular edges
  meet; both incident edges/faces look them up → bit-identical.
- **Stage 2 — singular edges on probe-pair circles.** Each `C_ij`
  (`canonical_burial_circle`) is split by *all* triple events `C_ij ∩ sphere_k`
  (the registry vertices) and classified exposed (outside every other accessible
  probe ball). The exposed arcs are the **singular edges**. ONE global
  arrangement per pair — not two per-face DCELs (Q1).
- **Stage 3 — face rewrite.** Spheric faces re-bound on the singular edges
  incident to them; **toric faces are trimmed** against neighbour probes with
  their *own* seam geometry (the rolling-probe circle family, Q3/Q4); undersized
  toric faces deleted. Each retained boundary interval is assigned to its incident
  patches from the global arrangement.
- **Stage 4 — richer gate (Q6).** `self_intersections==0` is necessary not
  sufficient; also gate signed volume, Euler/genus/components, boundary-edge
  count, per-face-type area, and sampled distance to analytic patches vs BALL.

The four existing foundations feed this: `canonical_burial_circle` = the
pairwise-intersection circle the singular edges live on; `arrange_loops` = the
per-pair circle arrangement; the burial cap = the global exposure test;
`self_intersections` = part of the Stage-4 gate.

## Open questions for review

1. **Is the SEAM SCOPE decomposition correct and complete?** Is every shared seam
   really "arc of `C` inside both triangles and globally exposed," and is the
   `arrange_loops` per-face boundary guaranteed to follow exactly that arc on the
   shared stretch (so the two faces' burial arcs coincide there before sampling)?
   Or can `arrange_loops` in `f`'s frame pick a *different* sub-arc of `C` than in
   `q`'s frame even on the shared stretch (e.g. when a third burial cap clips `C`)?

2. **Corner canonicalization.** For a global corner `C ∩ sphere_r`, both faces
   must compute the *same* world point. `C` is canonical (registry), but the arc
   endpoint comes out of `arrange_loops`' `circle_intersections` in each face's
   frame → not bit-identical. Do I need a *vertex* registry too (key
   `(probe_pair, third_sphere)`), or can I snap arrange_loops' corners to
   registry-recomputed canonical points by proximity?

3. **Toric burial.** Is trimming toric reentrant arcs against neighbour probes
   actually necessary, or do the spheric-face trims plus the existing toric mesh
   already cover the colliding region? Concretely: when probes `i,j` collide, is
   the buried surface entirely on the spheric faces, or does it also remove part
   of the toric faces incident to `i` and `j`? If toric trimming is needed, does
   it share `C` with the spheric trim, or a *different* circle?

4. **Does B even share `C`?** The toric reentrant arc lies on a *rolling* probe
   sphere (centre on the roll circle), not on the fixed probe `f`. A fixed
   neighbour `q` burying a rolling-probe point is a different intersection than
   `sphere_f ∩ sphere_q`. So toric burial likely needs its own seam geometry —
   confirm, and sketch what it is.

5. **Degenerate/measure-zero collisions.** The 14 near-coincident duplicate
   probes on crambin: should the integration dedup probe placements up front
   (RS-level) rather than handle them as burial? Recommended?

6. **Gate sufficiency.** After A (+ B if needed), the acceptance is
   `self_intersections(crambin) == 0` and area within ~0.3% of BALL. Is there a
   failure mode that zeroes self-intersections but still mismatches BALL area
   (e.g. over-trimming a sliver), that the gate would miss?
