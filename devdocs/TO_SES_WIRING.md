# TO_SES_WIRING — integrating the cleaner into ses_mesh_analytic

## State

The general-N analytic mesher (`ses_mesh_analytic`) is watertight on crambin but
self-intersects (+1.1% area, 5000+ crossing pairs) because colliding probes
(`< 2·probe`) double-cover the reentrant surface — the nonradial defect. All
cleaner primitives are built + tested in isolation:

- `cleaner::SingularVertices` — canonical triple-probe vertex registry `(sorted
  i,j,k, branch)`.
- `cleaner::singular_edges` — the globally-exposed arcs of each collision circle
  `C_ij`.
- `cleaner::sample_singular_edge` — bit-identical seam sampling, symmetric in
  `(i,j)`.
- `cleaner::clip_spheric_face` — one spheric face trimmed by its colliding
  neighbours (grid-validated); meshes in `p`'s own frame.

This plan wires them into `ses_mesh_analytic` to actually drop the crambin
self-intersection count, without regressing the tri/tetra/chain *valid
embeddings* (which have no probe-probe collisions).

## The current mesh (assemble.rs)

- **toric faces** — `toric_face_mesh` over the rolling-probe reentrant arcs;
  share their θ-end great-circle arcs with spheric faces (bit-exact, via
  `arc_on_sphere` in canonical atom order).
- **contact caps** — `walk_cap_loops` + `fill_spherical_region` per atom.
- **spheric faces** — one per RS face: 3 great-circle arcs (`arc_on_sphere`)
  filled. **This is what `clip_spheric_face` replaces.**

The whole thing welds by bit-identical shared samples + `mesh.welded()`.

## The coupling problem

When probe `j` buries part of spheric face `i`:
1. it removes a chunk of face `i` (handled by `clip_spheric_face`), **and**
2. it cuts the shared great-circle edge between face `i` and an adjacent **toric**
   face at the burial point, **and**
3. it buries part of that toric face too — along a *different* seam (the rolling-
   probe family `sphere(P(θ)) ∩ sphere(j)`, **not** `C_ij`; codex Q4).

So a spheric-only trim leaves the toric overhang → still self-intersecting, and
the great-circle seam no longer matches (the spheric arc got shortened, the toric
θ-end did not). The trim must be **coordinated**.

## Proposed staging

### W1 — preserve the non-burial seams exactly

`clip_spheric_face` currently samples *all* boundary arcs in `p`'s frame
(`sample_loop`). For watertightness the **great-circle** arcs must instead be
sampled exactly as the toric θ-end is (`arc_on_sphere`, canonical atom order), and
only the **burial** arcs sampled canonically (`sample_singular_edge`). So rewrite
the sampler to be source-aware: from `arrange_loops`' tagged `BoundaryArc`s, route
great-circle arcs (cap index < 3) → `arc_on_sphere` between their (possibly
burial-cut) corners; burial arcs → `sample_singular_edge`.

### W2 — measurement-first wiring (spheric only)

Gate the clip: a spheric face with **no** colliding neighbour → the existing path
(unchanged → tri/tetra/chain regression-safe). With colliding neighbours →
`clip_spheric_face`. Run crambin and **measure** `self_intersections` before vs
after. Expectation: a large drop (the spheric double-cover removed) but **not 0**
(toric overhangs remain). This validates the spheric half and quantifies the
toric residue.

### W3 — toric trim

Trim each toric face against neighbour probes along its own seam
(`sphere(P(θ)) ∩ sphere(j)`), meeting `C_ij` at the shared great-circle corner and
the triple vertices. Sample canonically so it welds to the spheric clip.

### W4 — Stage-4 gate vs BALL on crambin

`self_intersections == 0`, plus signed volume / Euler / per-face-type area /
sampled distance-to-analytic-patches within tolerance of BALL.

## MEASURED on crambin (ses_singular_diag)

With the cleaner primitives built, a per-face diagnostic (no weld — just plain vs
clipped spheric area, which codex confirmed is meaningful even when a
spheric-only *mesh* is not):

- `clip_spheric_face` ran on **all 480 RS faces with 0 errors** → production-robust
  on real geometry (`arrange_loops` + the chart hold up).
- **Spheric clip removes 12.9 Å²** over 35 faces (plain 573.6 → 560.7).
- Crambin's analytic excess vs BALL is **≈26 Å²** (the +1.1%).

- **Toric trim removes 12.0 Å²** over 21 arcs (0 errors; includes the spindle cap).
- **Cleaner total removed = 24.9 Å²** (spheric 12.9 + toric 12.0), **0 errors** on
  every face/arc.

⇒ the cleaner now accounts for **~96 %** of the ~26 Å² excess. The **spindle**
(radial) self-overlap turned out to be *also* a per-column burial cap (axis toward
the roll-axis centre, `half = acos(R_roll/probe)`), so it trims through the same
`toric_trim_mesh` path — adding exactly the 21 spindle arcs (5.6 → 12.0 Å²). The
residual **~1 Å²** is contact-cap trimming + discretization. The cleaner geometry
is essentially complete and validated on real data; what remains is the
**coordinated canonical-weld assembler** (spheric clip + toric trim + contact caps
sharing registry seams → a *watertight* cleaned mesh) + the Stage-4 gate.

## Open questions for review

1. **W1 corner identity.** A great-circle arc cut by burial ends at a point that
   is `C_ij ∩ great_circle` — a point on `p`'s sphere, on the contact great circle
   *and* at distance `probe` from `j`. Both the spheric clip and the toric trim
   must use the *same* corner. Is that corner a registry vertex? It is **not** a
   triple-*probe* vertex (one party is an atom/contact circle, not a third probe).
   Do I need a second registry keyed `(probe_pair i,j, contact_circle/atom)` for
   these, or can the toric trim derive it from the same `C_ij` θ that the spheric
   clip used (shared by construction)?

2. **W2 honesty.** Is a spheric-only intermediate even meaningful, or will the
   broken great-circle seams (spheric shortened, toric not) make the mesh so
   non-watertight that `self_intersections` is uninterpretable? Should W1+W3 land
   together instead, never shipping a half-trimmed mesh?

3. **W3 toric seam.** The toric reentrant arc at fixed θ is a meridian arc on
   `sphere(P(θ))`. Burial by `j` removes the sub-arc inside `sphere(j)`. Is
   trimming each θ-column's reentrant arc against `sphere(j)` (a 1-D clip per
   column) sufficient and watertight, given adjacent columns clip at slightly
   different φ? Or does the toric burial boundary need to be a proper curve
   (interpolated across columns) to avoid a jagged, leaking seam?

4. **W3 vs BALL.** BALL deletes *undersized* toric faces and rewrites singular
   toric faces into singular edges. Does per-column clipping reproduce that, or
   miss the case where a toric face is *entirely* buried (should be deleted) or
   *spindle-singular* and buried at once?

5. **Activation predicate.** "No colliding neighbour" = no other probe within
   `2·probe` of `p`. Sufficient to preserve the exact old path, or can a neighbour
   within `2·probe` still not actually bury any of *this* face (so clipping is a
   no-op that nonetheless reroutes sampling and risks a weld regression)? Should
   the gate be "burial cap actually intersects the triangle" instead?

6. **Welded tolerance.** The mesh welds by bit-identical samples. The canonical
   seam guarantees bit-identical burial arcs across faces, but the great-circle
   corners (W1) are computed once per face — are they bit-identical across the two
   faces sharing them, or do they also need registry interning?
