# Closing the residual open edges in the cleaned SES weld

## Context

We triangulate the solvent-excluded surface (SES) analytically: contact caps
(on atom spheres), **toric** reentrant faces (probe rolling over an atom pair),
and **spheric** reentrant faces (probe resting on an atom triple). The analytic
mesher (`ses_mesh_analytic`) is watertight by **bit-identical shared samples**:
every shared boundary is sampled once and passed verbatim to both incident
patches.

We added a *cleaner* (`ses_mesh_cleaned`) that trims away probe-probe
self-overlap singularities. It produces concatenated (open) patches:
- toric faces via `toric_trim_mesh`: each θ-column (a rolling-probe position)
  keeps the φ-intervals of its reentrant arc not buried by neighbour probes
  (`toric_kept_intervals(dir_a, dir_b, caps)`), sampled by `toric_column_curve`
  (uniform in φ: `p + (dir_a·cosφ + tangent·sinφ)·probe`).
- spheric faces via `clip_spheric_face`: the probe-sphere triangle bounded by 3
  great circles, trimmed by neighbour burial caps, meshed via an arrangement +
  chart fill.

We weld with a **tolerance merge** (`Mesh::welded_within(eps)`, eps=1e-4), since
the cleaned seams are sampled by different parameterizations that coincide
mathematically but not in f64 bits.

## What's done (brick 2, just landed)

A spheric face's 3 edges are great-circle reentrant arcs; each is shared with
the **θ-end column** of an adjacent toric face (the toric arc's probe sweep
reaches the spheric face's probe at its two ends). We made `clip_spheric_face`
sample those edges with the **same** `toric_column_curve` parameterization and
the same `dir_a` reference (lower-atom contact, matching the toric `edge=[i,j]`
i<j convention). Now `dir_a`/`dir_b`/`tangent` are bit-identical between the two
sides; only the φ-range endpoints differ (~1e-9), which the eps weld fuses.

Result on crambin (327 atoms): open edges **34290 → 8241** (76% closed), area
held at +0.04% vs BALL, signed volume corrected (the old in-frame `rim_point`
sampling gave orientation-inconsistent patches). Zero true (≥3) non-manifold
edges. The inert tetra/chain still match BALL.

## The residual (~8241 open on crambin, 30 on a tight tetra)

The toric θ-end and the spheric edge share a reentrant arc but **trim it with
different neighbour-exclusion sets**:

- The toric arc `(i,j)` computes its kept φ-intervals against `neighbours = all
  probes except its own two end-face probes {P, P'}` (the two RS faces at the
  arc ends — excluded because the θ-end columns coincide with them).
- The spheric face `P` (an RS face) clips its edge `(i,j)` against `all probes
  except itself`. So it does NOT exclude `P'`, and it DOES include third probes
  `P''`, `P'''` (siblings via its other edges).

So whenever a third probe buries the reentrant arc on one side but is excluded
on the other, the two trimmed polylines diverge → the shared edge stays open.
This is symmetric: sometimes the toric is trimmed more, sometimes the spheric.

We tried the obvious fix — exclude the spheric face's "sibling" end-face probes
to match the toric. It made things **worse** on crambin (area +0.04% → +0.38%,
i.e. *under*-trimming; open 8241 → 9417), because a sibling probe CAN genuinely
bury part of a spheric face. So a naive symmetric exclusion is wrong.

## The question

What is the cleanest architecture to make the toric θ-end and the spheric edge
trim the **same** reentrant arc identically, so their polylines coincide (then
the eps weld fuses them)?

Candidate approaches we're weighing:

**A. Emit-and-consume.** Run the toric pass first; for each bounded toric arc,
emit its two trimmed θ-end polylines keyed by `(rs_face_idx, atom_pair)`. Then
`clip_spheric_face` consumes those polylines verbatim as its 3 edge boundaries
(instead of re-deriving them), and only computes the burial arcs (collision
circles) and chart-fills the interior. Guarantees a match (literally same
verts). Cost: the spheric face's boundary is currently assembled by a generic
arrangement (`arrange_loops`) over {3 great circles ∪ burial caps}; splicing in
pre-made edge polylines while keeping the arrangement topology (where burial
caps cut the edges, and the burial arcs connect them) is fiddly.

**B. One shared cap-set per reentrant arc.** Define, for each reentrant arc
`(i,j)` at probe `P`, the canonical neighbour set that should trim it, and use it
on BOTH the toric θ-end and the spheric edge. Question: what IS the correct set?
The toric's "exclude both end faces" is a special-case hack for the θ-ends; the
spheric's "all neighbours" is the general rule. Is there a single rule (e.g.
"all probes that are not RS faces on a triple sharing this atom pair, plus the
genuinely-burying ones") that both can use and that's geometrically correct?

**C. Something else** — e.g. don't trim the toric θ-end columns at all (leave
the θ-ends full, let the spheric face own all the edge trimming), and have the
toric strip's first/last columns simply follow the spheric edge.

Which approach is soundest? Are there failure modes we're missing (e.g. an arc
buried in its *middle* but not its ends, splitting into 2 — does each candidate
handle a θ-end that splits into multiple kept intervals)? Is there a cleaner
formulation where the seam is a single shared object owned by neither face?

Concrete, specific feedback. ~700 words.

## Codex review + decision (2026-06-06)

Codex's verdict: **the residual is a geometric-visibility disagreement, not a
sampling problem. Welding hides numerical noise; it cannot repair different
Boolean boundaries.** So the *trim itself* must agree on both sides — no eps can
fix it.

- **A (emit-and-consume):** fine plumbing/diagnostic, **unsafe as the source of
  truth** — the toric "exclude both end faces" rule is not canonical, and a
  consumed polyline must enter the spheric **arrangement as constrained
  segments**, not bypass it.
- **B (shared cap-set):** right direction but "cap set" is too fragile; promote
  it to a shared **1D Boolean object** with one canonical *visibility predicate*
  based on geometric provenance (exclude the probe whose sphere owns the seam as
  its own boundary; exclude same continuous rolling-sheet adjacency; include
  every other probe whose open interior contains a seam point; one tolerance
  policy for tangency).
- **C (spheric owns):** unsound — a full toric θ-end against a partial spheric
  boundary leaves an exposed remainder; it collapses into A/B.

**Mandatory failure handling (any solution must cover):**
- a seam has **0 / 1 / several** kept intervals — an API returning "*the*
  endpoint interval" is structurally wrong;
- 0 → no toric↔spheric adjacency there, neither face emits it;
- middle-burial → two **distinct** intervals; joining their outer endpoints is a
  false chord;
- tangency / near-zero intervals → deterministic suppression **shared** by both
  consumers;
- the toric column-zipper is invalid when interval counts change across θ — the
  retained region needs a `(θ, φ)` arrangement or explicit critical-event
  processing (component birth/death/split/merge).

**Chosen architecture — `ReentrantSeam`, owned by neither face.** Keyed by
`(toric_face, rs_face, atom_pair, end_side)`. Holds: the canonical
`toric_column_curve`; oriented kept φ-intervals; the shared sampled vertices;
clip-event provenance + intersection vertices. Built **once** by arranging all
relevant burial-cap intersections on the 1D arc and classifying each open
segment with the canonical visibility predicate. Both the toric θ-end and the
spheric edge consume the **same** seam components as constrained boundaries;
each face's own 2D arrangement then decides which interior attaches.

The keystone risk is the **canonical visibility predicate** (B's provenance
rule) — that's where the toric/spheric disagreement actually lives. Build and
unit-test that predicate in isolation first (against `rs::probe_clear` and the
analytic exposure used by `singular_edges`), then wire the `ReentrantSeam`
object, then have both meshers consume it. The tolerance weld stays as the final
numerical fuse for the (now-matching) seams.
