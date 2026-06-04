# Design note: watertight stitching of SES patches (L4 final step)

> **DECISION (2026-06): build the SES mesh by iso-surfacing a distance field,
> not by analytic patch stitching.** Reading BALL's own triangulator
> (`triangulatedSES.C`) settled it: BALL meshes contact caps by clipping a
> template icosphere with each contact-circle *plane* and gift-wrapping an
> advancing front over the union of {clipped interior points} ∪ {shared
> boundary points the toric pass stored in `edge_[ses_edge]`} — i.e. clip-and-snap
> over a registry, with a global mutable `Constants::EPSILON` and a magic
> `(4·density·π·r²−12)/30` refinement sizing. Faithfully porting that is not sane.
>
> The sane port uses the **erosion identity**: the SES solid is the
> solvent-accessible solid (atoms inflated by the probe) **eroded by a probe-radius
> ball**. So a signed distance field `f(x) = dist(x, complement(A_p)) − probe`
> (with `A_p` = union of inflated atoms) has the *entire* SES — contact, toric, and
> reentrant — as its `f = 0` iso-surface. Iso-surfacing (surface nets / marching
> cubes) gives a watertight mesh by construction, handles any hole/annulus/pocket
> topology automatically, and needs no patch stitching. Grid spacing is the single
> convergence knob; gated against `ball-py ses_area` (analytic area **and** volume,
> an independent Connolly path). Implemented in `surface/volume.rs`.
>
> The grid/SDF mesher (`volume.rs`) shipped as ONE output. But it does not refute
> the boss's "BALL's triangulation isn't portable" claim — that needs the EXACT
> analytic Connolly mesh, which is now being completed (hybrid clean-room +
> faithful singularity cleaner — see `TO_SES_EXACT_COMPLETION.md`). So the
> analytic machinery below (`arrangement.rs`, `patches.rs`, this registry) is back
> to being **the mesher**, not just a cross-check oracle. The stitching strategy
> below is the design for that registry-first assembly.

## Context

The SES port (`proteon-core/src/surface/`, plan in `TO_SES_TRIANGULATION.md`) has,
gated against `ball-py`:
- L1 reduced surface — vertices (surface atoms), edges (atom pairs), faces (atom triples)
- L2 SES element graph — contact / toric / spheric faces + ownership
- L4 patch primitives, each individually area-gated:
  - `icosphere` — a contact cap as a *full* sphere (correct only for a free atom)
  - `spheric_face_mesh` — geodesic spherical triangle, inward normals
  - `toric_face_mesh` — torus patch, every vertex exactly `probe_radius` from the roll-circle

Each patch is correct in isolation. **The remaining problem: assemble them into one
*closed* (watertight, consistently oriented) multi-atom mesh** that matches BALL's
`ses_mesh` on area / volume / Euler characteristic. The whole difficulty is the
shared boundaries — adjacent patches must reference the *same* vertices along the
curve they share, or the mesh cracks.

## The boundary topology (what shares what)

Three reentrant/contact patch types meet along two kinds of curve:

1. **Contact circle** — where a toric face meets a contact cap. For RS edge
   `(i,j)`, the probe rolling over `i,j` traces a circle of contact points *on
   atom i* (and another on atom j). That circle bounds both `toric(i,j)` and
   `contact(i)`. **NOT independent holes (codex-review correction):** a contact
   cap is *not* the sphere minus each disc subtracted independently. Incident
   contact circles overlap / nest / touch / near-touch and can split the cap into
   several exposed components. The cap is the **exposed region of the spherical
   *arrangement* of all incident contact circles** (the boundary of the union of
   buried discs). A full contact circle is a valid boundary loop only on the arc
   intervals where it actually survives that arrangement. **The per-atom circle
   arrangement is the central data structure for contact patches** — the registry
   stores only the *surviving boundary intervals*, not whole circles.
2. **Concave arc (spheric edge)** — where a toric face meets a spheric face. At a
   probe position resting on `(i,j,k)` (an RS face = `spheric(i,j,k)`), the probe
   touches `i,j,k` at three points; the three arcs between them bound the spheric
   triangle, and each arc is the θ-end of a toric face (`toric(i,j)` ends where
   the probe reaches this RS-face position). For a *free* toric edge there are no
   spheric ends — the torus is a full ring closed in θ.

So the SES is a cell complex: spheric faces (triangular, 3 concave-arc edges) +
toric faces (quad-ish, 2 contact-circle edges + 2 concave-arc edges, or a full
ring for free edges) + contact faces (sphere minus circular holes). SES vertices
are the probe-contact points where three patches meet.

## Proposed stitching strategy — shared boundary registry

Watertightness by construction: **discretize every shared boundary curve exactly
once into an ordered list of vertices, and have both adjacent patches index into
that shared list** (never re-emit boundary vertices independently).

1. **Enumerate boundary curves** from the SES element graph:
   - one **contact circle** per (atom, incident RS edge);
   - one **concave arc** per (RS face, one of its 3 atom pairs) — i.e. each spheric
     triangle edge, which is also a toric θ-end.
2. **Partition each curve** into `N(curve)` segments at the target `density`
   (matching BALL's `partitionOfCircle`: points ∝ √density · arc-length). Store the
   resulting vertex indices in a registry keyed by the curve's identity (e.g.
   `(atom, neighbor)` for a contact circle, `(rs_face, pair)` for an arc). The two
   endpoints of a concave arc are **SES vertices** (probe-contact points) shared
   across all patches meeting there — interned once.
3. **Orientation contract (codex-review).** A registry shares the same *vertices*
   but not the same *walk*. Each curve has a **canonical direction** stored once;
   each consuming patch records `same`/`reversed`. Winding is derived from that
   contract, never inferred ad-hoc from sorted keys — otherwise adjacent patches
   silently disagree on which side is interior.
4. **Intern SES vertices by topological identity (codex-review)** — the
   probe-contact points where patches meet are keyed by the RS element that
   creates them, not by coordinate welding (which would merge distinct near-degenerate
   features).
5. **Mesh each patch up to its boundary** using the registry vertices as the fixed
   rim:
   - **spheric** — fill the spherical triangle whose 3 edges are the registry arcs
     (constrained geodesic triangulation to the given rim).
   - **toric** — grid in (θ, φ): the two φ-rims are the contact circles, the two
     θ-rims are the concave arcs; a **free ring** needs an explicit wrap *seam*
     (shared parameter seam, not a duplicated geometric boundary).
   - **contact** — triangulate the **exposed cells of the per-atom circle
     arrangement** (step 1), with the surviving boundary intervals as constraints.
     Prefer spherical-arrangement panels or a constrained Delaunay in a *local*
     projection chart; **avoid clip-icosphere-and-snap** (the snap inverts skinny
     triangles / creates degeneracies in dense circle clusters — codex-review).
6. **Assemble**: concatenate patch vertices + the shared registry vertices; emit
   triangles with consistent (outward-solvent) winding; assert closed.

## Gating (end-to-end, against `ball-py ses_mesh`)

Invariants are **necessary but insufficient** (codex-review: a closed, oriented
mesh can still self-intersect, mesh the wrong region, or hide compensating
errors). Two tiers:

*Topological/global invariants:*
- `is_watertight()` and `is_consistently_oriented()` — **closed manifold**;
- per-component **Euler characteristic** vs BALL's mesh;
- **surface area → analytic** `ses_area` as density rises (convergence claim);
- **signed volume** > 0 and matching BALL's enclosed volume;
- vertex/triangle counts as a *measured* band (triangulation differs), not gated.

*Pointwise analytic-residual + ownership (the gates that catch
"closed-but-wrong" — codex-review):*
- every vertex/triangle sample lies on its intended primitive within tol —
  contact → atom sphere, toric → probe torus (dist `probe_radius` from the
  roll-circle), spheric → probe sphere;
- contact samples are **outside every buried disc** for that atom (correct side
  of the arrangement); toric samples are within the RS edge's valid θ interval;
- full-mesh triangle-centroid **signed distance to the analytic SES ≈ 0, correct
  side**;
- explicit **self-intersection** check (watertight + oriented is not enough).

## Central principle (the question the first draft missed — codex-review)

**The authoritative source of truth for a contact cap is the spherical arrangement
of all incident contact circles, not the set of full circles.** Build that
arrangement first; everything else (registry curves, boundary loops, contact
meshing) derives from its surviving exposed cells. Designing around "independent
hole loops" will pass simple fixtures and crack on crowded molecules.

## Resolved by the review

- **Contact cap** → spherical-arrangement panels (or local-chart constrained
  Delaunay); NOT clip-and-snap.
- **Registry identity** → `(atom, neighbor)` / `(rs_face, sorted_pair)` suffices
  for *vertex* identity only; add an **explicit orientation contract** (canonical
  direction + per-patch reversed). Requires the graph to already have split
  singular/multiple faces.
- **Partition `N`** → geometry+density derived is fine for *sharing*; reproduce
  BALL's `partitionOfCircle` only if count *bands* must mean something.
- **Free rings** → registry handles them, but the wrap **seam** needs explicit
  handling (not "same as a bounded quad").
- **Gating** → invariants alone are too weak; add the pointwise analytic-residual,
  ownership/exposure classification, and self-intersection checks above.
- **L3** → design the registry to accept **post-L3 split faces now**; don't bake
  non-singular identity keys.
- **Strategy** → registry+arrangement is right for BALL-faithful analytic patches;
  marching-cubes/dual-contouring would be more robustly watertight but discards
  exact patch boundaries, area convergence, and Euler stability — rejected.

## Open questions still to settle

1. **Spherical arrangement implementation** — small-circle×small-circle
   intersection on a sphere + cell classification (exposed vs buried) is now the
   core new component. Build a minimal bespoke arrangement (incident counts ≤ ~tens
   per atom), or is there a simpler exposure test that avoids a full arrangement
   (e.g. sample-and-classify the cap, accepting the L1 sampling caveat)?
2. **Where do contact-circle / spheric-arc endpoints coincide** (an SES vertex on
   atom m that's also on a contact circle of m) — the interning + tolerance policy
   that keeps both patches agreeing at that point.
3. **Self-intersection check cost** at corpus scale — spatial hash / BVH, or only
   run it on the small gate corpus?
4. **Density → partition** mapping: pin BALL's `partitionOfCircle` now (for count
   bands) or defer and treat counts as diagnostics only?
