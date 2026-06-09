# SES analytic mesher — the CDT "boundary crossing" failure

## RESOLUTION (what shipped)

**Fix = A (atom-perturbation retry), not D (refinement).** The empirical test
overturned both the Codex review's lean and the initial plan:

| Case | flavor | D-lite (escalate n_theta 48→384) | **A (perturbation)** |
|------|--------|----------------------------------|----------------------|
| 9hfa | 2.4° sliver | ✗ still crosses | ✓ watertight (158s) |
| 5qs5 | inter-loop pinch | ✗ still crosses | ✓ watertight (167s) |
| 8t5l | 76° pinch | ✗ timed out | ✓ watertight (153s) |
| 1ijp | 126° + pinch | ✗ timed out | ✓ watertight (96s) |
| 3lwa | pinch | ✓ watertight (388s) | ✓ watertight (112s) |
| 4iej | 748-atom | (untested) | ✗ residual → grid |

**Why A beats D:** the dominant crossings are *zero-clearance* near-tangencies in
the reduced-surface arrangement — the sampled boundary nearly self-touches. No
sampling density (D) removes a crossing that exists at zero clearance, and the
global n_theta escalation is 2–8× the (already slow) whole-protein remesh. A tiny
atom jitter (≤1e-2 Å) instead opens the near-tangency at the *arrangement* level, so
the boundary becomes simple again — fixing 5/6 cases at base resolution, cheaper than
D, including the sliver/pinch cases D could not touch. Codex's caveat that jitter
"can't reliably fix a pinch" was too pessimistic in practice.

**Shipped change:** add `"crosses an existing constraint"` to
`is_degeneracy_error` (assemble.rs) so the existing perturbation retry handles it
(the chart pole-retry still runs first, cheaply). The lone residual (4iej) falls to
the hybrid grid fallback — still watertight. The D escalation was implemented,
measured, and reverted.

The diagnosis below stands; only the chosen remedy changed.

---

## Status / why this doc

Analytic Connolly SES coverage on a 317-protein RCSB corpus: among proteins that
produce a verdict, **68% mesh, 63% watertight**. Of the 36 hard failures, **22
(61%) are one error class**:

```
boundary edge N->N crosses an existing constraint
```

raised in `surface/cdt.rs:78` (`can_add_constraint` returns false). This is the
single highest-leverage robustness target. This doc states the diagnosis and lays
out candidate fixes for review.

## Pipeline (where it breaks)

Contact-cap interior meshing (`surface/chart.rs::fill_spherical_region`):

1. The exposed region of an atom = sphere minus *k* buried-cap holes, bounded by
   loops of **contact-circle arcs**. These arc samples come from
   `arrangement::sample_loop(lp, &caps, n_boundary)` and are **shared, by value,
   with the neighbouring toric faces** — that bit-identical sharing is what makes
   the assembled mesh watertight.
2. Pick a chart pole inside the region; project all boundary loops to 2D via an
   **azimuthal-equidistant** chart (injective on the whole sphere minus the
   antipode).
3. Sprinkle interior Steiner points; **constrained-Delaunay-triangulate** the
   planar polygon-with-holes, passing each boundary loop as constraint edges.
4. Lift every vertex back onto the sphere **by index, using the exact 3D sample
   positions** (`world[]`) — the 2D coords are used ONLY to derive triangulation
   *connectivity*.

The failure: at step 3, two boundary **constraint edges cross in 2D**, so the CDT
refuses the constraint. Because azimuthal-equidistant is injective, a *smooth*
simple boundary can never project to a self-crossing curve — **every crossing is a
chord artifact of the sampled polygon** (two straight chords cross where the smooth
arcs they approximate do not).

## Current mitigations (and why they're insufficient)

- **Chart pole-retry** (`chart.rs`, 32 trials: 8 directions × 4 magnitudes,
  rotations of the pole in its tangent plane). Re-projects only — boundary 3D verts
  unchanged, so weld-safe. This is the *only* thing tried for a crossing today.
- **Whole-protein perturbation-retry** (`assemble.rs::build_with_perturbation_retry`,
  jitter 1e-4→1e-2 Å, deterministic). The CDT crossing is **deliberately excluded**
  from `is_degeneracy_error` (assemble.rs:830) — rationale in-code: "a tiny atom
  jitter would not reliably fix a chord crossing," and it was reverted once for cost
  (whole-protein re-mesh per attempt; the 1aaj 102s regression).

## Diagnosis (instrumented, 5 fastest-failing cases)

Per-failure: number of loops, samples/loop, region half-extent (max angle of any
boundary sample from the chart pole), and count of actual 2D edge crossings:

| Case | loops | samples | region half-extent | crossings | reading |
|------|-------|---------|--------------------|-----------|---------|
| 9hfa | 1 | 20 | **2.4°** | 1 | sliver — near-degenerate loop; ~zero projection distortion |
| 8t5l | 1 | 51 | **76.2°** | 2 | **within** a hemisphere — mild distortion (~1.37×), genuine self-pinch |
| 5qs5 | 2 | 41, 31 | 89.8° | 4 | two holes pinching against each other (inter-loop) |
| 1ijp | 1 | 288 | **126.4°** | 2 | self-pinch **+** >hemisphere distortion |

**Conclusion:** the root cause is **the sampled boundary polygon self-intersecting**
at slivers / self-pinches / near-tangencies in the reduced surface — *not* the chart
pole or projection. The 9hfa (2.4°) and 8t5l (76°) cases prove it: at those extents
projection distortion is negligible, yet the polygon still crosses. This is exactly
why the pole-retry can't fix them — no pole removes a crossing that exists in the
*3D-sampled* boundary itself. The cases span tiny→large and single→multi-loop, so
one mechanism (degenerate/under-sampled boundary geometry) with several flavours.

## The invariant that constrains any fix

Boundary vertices are **shared by value with the toric faces** for watertightness.
So a fix may **not** freely add or move a vertex that lies on a shared contact arc
(that creates a T-junction the tolerance weld `welded_within(1e-5)` cannot close).

BUT: the lift in step 4 uses the **3D** sample positions by index; the 2D coords
only determine connectivity. So **any change confined to the 2D chart (re-project,
re-embed, perturb-in-plane) is weld-safe** as long as the boundary index set and the
hole structure are unchanged.

## Candidate fixes (for review — which to pursue, what's missing?)

**A. Re-enable crossing → whole-protein perturbation-retry, but only after the
pole-retry is exhausted.** Cheap to implement (add the crossing string back to
`is_degeneracy_error`). The in-code claim that jitter "won't reliably fix a chord
crossing" is untested against the new evidence — for a 2.4° sliver (9hfa) a 1e-2 Å
atom jitter materially changes the contact geometry and may open/collapse the
sliver. Risk: cost on large proteins (re-mesh per attempt), and it may not converge
for genuine pinches (8t5l/1ijp). Worth measuring as a baseline.

**B. 2D-only re-embedding (decouple connectivity from the projection).** Since the
lift uses 3D positions, replace the distorted azimuthal coords with a *crossing-free*
planar embedding of the same boundary cyclic order + holes (e.g. lay each loop out
as a non-overlapping simple curve and Tutte-embed the interior), CDT that, lift by
index. Fully removes projection distortion and adds no verts. Risk: a "wild"
embedding can yield skinny or orientation-flipped triangles once lifted to 3D (the
near-isometric chart gave good triangle quality for free); needs an
orientation/quality guard on the lifted triangles.

**C. Local crossing resolution by constraint subdivision.** Detect the crossing
pair, insert the smooth-arc midpoint(s) as new boundary samples to separate them.
Correct geometrically but the new samples land on **shared** arcs → must be
propagated to the toric neighbour (couples patches; non-trivial plumbing). Is there
a way to keep this purely local?

**D. Adaptive arc refinement, globally consistent.** Before meshing, detect arcs
whose non-adjacent 3D samples come within < sample-spacing (a pinch) and double their
sampling everywhere they're used (both the cap and its toric neighbour), so the
shared boundary stays consistent. Correct, but uniform-ish cost and plumbing; we are
*already* timing out on large proteins, so blanket densification is unattractive.

**E. Heal sub-resolution slivers (targeted at the 9hfa flavour).** A 2.4° region is
below mesh resolution; detect a near-degenerate boundary loop (enclosed solid angle
or min feature width < weld_eps-scale) and either drop it (let the weld close the
hole) or collapse its near-coincident samples. Cheap, but only addresses the sliver
subset, and "dropping" must be proven not to open a real hole.

## Questions for the reviewer

1. Which of A–E (or a combination) is the most robust *and* cheapest given the
   shared-boundary weld invariant? Is B (2D re-embedding) sound — does decoupling
   connectivity from the chart introduce a failure mode I'm missing (triangle
   quality / orientation after lift)?
2. Is the in-code claim that atom-jitter can't fix a chord crossing actually right?
   The sliver case (A) seems jitter-tractable; the genuine-pinch cases may not be.
3. Is there a standard technique for triangulating a **simple spherical
   polygon-with-holes** that sidesteps planar projection entirely (spherical CDT,
   or a geodesic-refinement scheme) and would be more robust here?
4. For watertightness: is dropping a sub-resolution sliver region (E) ever unsafe —
   can a 2.4° contact cap be a real (non-removable) topological feature?
5. Anything in the diagnosis that points to a simpler root-cause fix upstream — e.g.
   the reduced-surface arrangement producing these degenerate loops in the first
   place (should `sample_loop` guarantee a minimum non-adjacent separation)?
