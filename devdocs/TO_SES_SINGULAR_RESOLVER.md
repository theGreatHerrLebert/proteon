# TO_SES_SINGULAR_RESOLVER — geometric resolution of probe self-intersections

## Problem (established, gated)

The general-N analytic SES assembler (`ses_mesh_analytic`) produces a
**combinatorially watertight but self-intersecting** mesh on dense inputs:

- tri / tetra / chain: watertight **and** 0 self-intersections → valid embedding.
- crambin (327 atoms): watertight, but `self_intersections(cap 5000) = 5000`
  and **+1.1 % area vs BALL**, stable under refinement (2344.470 coarse →
  2344.415 fine) → systematic, not discretization.

Root cause: **singular toric faces**. A toric face whose roll-circle radius
`R_roll < probe` is a *spindle* torus, not a ring. Its reentrant surface reaches
**past the roll axis** and overlaps itself. The raw mesher samples the full
reentrant arc, double-covering the past-axis region → the excess area + the
self-intersections.

Already landed (foundation, tested + committed):

- `intersect::self_intersections(mesh, cell, cap)` — the valid-embedding gate.
- `singular::is_singular(roll, probe)` = `R_roll < probe`.
- `singular::spindle_poles(roll, probe)` — the two axis points at distance
  exactly `probe` from the whole roll circle: `roll.center ± n·√(probe²−R_roll²)`.

## The geometry (meridian clip)

A toric face is rotationally symmetric about the atom-pair axis. In the meridian
half-plane at azimuth θ, the reentrant profile is a **circular arc** of radius
`probe` centred at the probe centre `P(θ)` (which sits at radial distance
`R_roll` from the axis). That circle crosses the axis (radial = 0) at exactly the
two **spindle poles** — and the poles are the *same two fixed points for every
θ*, because they lie on the axis.

For a ring torus (`R_roll ≥ probe`) the reentrant arc never reaches the axis:
the whole arc `T_i → T_j` is valid; mesh as today.

For a spindle (`R_roll < probe`) the arc `T_i → T_j` dips to radial
`R_roll − probe < 0` — **past the axis**. The portion with radial-coordinate on
the far side of the axis is the self-overlap (it is the region owned by the probe
on the opposite azimuth). The SES keeps only the near-axis-side portion:

```
clip each meridian arc at the two pole-crossings:
   keep  T_i → pole_a   and   pole_b → T_j
   drop  pole_a → pole_b   (the past-axis loop)
```

All θ share the two poles, so the clipped i-side arcs sweep a **cone** from the
atom-i contact circle down to the pole pair, and likewise a cone on the j-side.
The two poles become **singular SES vertices** where the reentrant surface
pinches (a vesica/lens cross-section).

## Rescope after codex review + measurement (DECISIVE)

Codex flagged that `R_roll < probe` (spindle/**radial**) is only one singularity
class; the other is **distinct-probe collision** (**nonradial**) — two different
fixed probes < 2·probe apart, one's reentrant surface buried in the other — and
`rs::probe_clear` excludes only *atom* overlap, never *probe* overlap, so the RS
enumeration does not prevent it.

`ses_singular_diag` measured the split on crambin (327 atoms, probe 1.4):

| class | count |
|-------|-------|
| **spindle** toric faces (`R_roll < probe`) | **21** / 720 |
| **probe-probe** genuine overlaps (< 2·probe) | **1575** / 480 RS faces |
| near-coincident duplicate placements | 14 |

**The crambin +1.1 % / 5000+ self-intersections are dominated by the nonradial
probe-probe class, not spindle.** A free-edge spindle clip would resolve 21 faces
— a rounding error against 1575 collisions. **The real work is the nonradial
probe-sphere arrangement rewrite**: for each spheric (reentrant) probe face,
compute its intersection circles with neighbouring probes, arrange them on the
probe sphere, classify cells by burial, and keep only the unburied cells —
shared consistently with the toric faces. This is the bulk of BALL's
`SESSingularityCleaner`.

Revised order:
- **A. Nonradial arrangement rewrite** (the gap-closer; large). Per spheric face:
  gather probes within 2·probe; for each, the sphere–sphere intersection circle;
  arrange circles on the probe sphere; drop buried cells; retriangulate the kept
  region; stitch its new boundary arcs to the neighbour faces via the registry
  (`(probe_pair, component, endpoint)` keys — bit-exact welding is *not* enough,
  codex #5).
- **B. Spindle clip** (small, secondary): the free-RS-edge meridian clip below,
  with the per-θ in-span pole test (0/1/2 poles, codex #2/#3). Only meaningful
  after A, and only for genuinely free RS edges.

The meridian-clip plan below is retained as the **Step B** spec.

## Plan (Step B — spindle clip, secondary)

### Step 1 — clip in the toric mesher *(core change)*

`toric_face_mesh` (and its general-N caller in `assemble.rs`) currently samples
`n_phi` points along each meridian arc `T_i(θ) → T_j(θ)`. Change:

1. Detect singular (`is_singular(roll, probe)`); ring case unchanged.
2. Compute the two spindle poles once (`spindle_poles`).
3. Per θ, find the two arc parameters where the meridian arc crosses the pole
   (closed form: the arc is a circle of radius `probe` about `P(θ)` in the
   meridian plane; intersect with the axis line — the solutions are exactly the
   precomputed poles, so map each pole to its arc angle).
4. Emit two strips: `contact_i_rim → pole_a` and `pole_b → contact_j_rim`,
   each pole a **single shared vertex** (bit-identical, keyed on the pole
   position) so the welder fuses across θ and across neighbouring faces.

The poles must be emitted as the *same* `Vec3` bits everywhere they appear
(toric clip on both sides, and any spheric/contact face that touches them) — the
existing exact-weld mechanism then closes the surface with no new registry.

### Step 2 — pole boundary into the cap/spheric faces

A spindle's poles sit on the atom-pair axis and may coincide with, or bound,
adjacent spheric (probe-cap) faces. Where a spheric face's boundary circle passed
through the now-removed past-axis region, its loop must instead route through the
pole vertex. Concretely: the contact-arc walk (`walk_cap_loops`) and the spheric
boundary loops already consume *exact* shared samples; feed the pole vertices
into the same shared-sample set so the loops pick them up. Verify no face still
references a clipped (past-axis) sample.

### Step 3 — gate

- `self_intersections == 0` on crambin (currently 5000+). **Hard gate.**
- Area/volume within BALL tolerance (target ≤ 0.3 %, from +1.1 %).
- Watertight + consistently oriented preserved.
- tri/tetra/chain unchanged (they have no singular faces — regression guard).
- A *constructed* singular two-atom case (wide gap, `R_roll < probe`) gated vs
  `ball-py ses_area` directly.

## Open questions for review

1. **Pole degeneracy.** Both poles are single points shared by all θ — the cone
   strips degenerate to a point there (zero-area sliver triangles). Acceptable,
   or collapse the apex to a triangle fan with a single apex vertex (preferred —
   avoids needle triangles)?
2. **Spindle vs probe-probe collision.** `R_roll < probe` is the *self*-overlap
   of one toric face. A *separate* singularity is two **different** probes
   colliding (cusp where three+ reentrant faces meet). Is that already excluded
   by the RS enumeration (the probe-placement non-intersection test), or does it
   need its own clip? The crambin excess may be purely spindle, or may include
   probe-probe — Step 3's `self_intersections` count after Step 1 will tell us.
3. **Both poles always real?** When the contact circles are very asymmetric
   (C vs O radii), can one pole fall *outside* the `T_i → T_j` arc span (so only
   one crossing is on the kept arc)? Need the per-θ arc-span check, not a blind
   two-pole clip.
4. **Validity-predicate fallback.** Instead of the analytic pole clip, a robust
   alternative: drop any reentrant sample within `probe` of *any other* probe
   centre (universal outer-envelope test), then stitch to the clip curve. Slower
   and needs curve recovery, but handles spindle + probe-probe uniformly. Worth
   it as a correctness backstop, or keep the analytic clip for speed?
