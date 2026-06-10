**Q1**

The heuristic is useful but not geometrically sound as a correctness criterion.

“Same hemisphere as the interior hint” does **not** imply that `cap_c` lies inside the selected spherical region. In a non-convex disk, or a disk with several holes, the boundary’s enclosing-cap center can lie across a narrow bay or inside a hole while still having positive dot product with the hint. Lowering the maximum vertex angle also controls only vertex positions, not:

- whether the antipode lies outside the filled region;
- clearance from the antipode;
- distortion or chord error along boundary arcs;
- projected separation between nonadjacent edges.

Thus it can still produce crossing polygonal chords. The smooth projected loop itself cannot self-cross unless it passes through the antipode or the spherical boundary was already non-simple.

A region-constrained Chebyshev center is cleaner than the hemisphere test:

\[
p^*=\arg\max_{p\in R}\min_{q\in\partial R} d_S(p,q).
\]

For a small disk it selects its interior center; for the complement of a small cap it selects roughly the cap’s antipode. It therefore identifies the correct side without an interior-hint dot-product test.

However, it is not by itself a complete chart objective. A single azimuthal chart additionally requires:

\[
-p\notin\overline R
\]

with positive clearance. A sufficiently large or winding disk-like region may contain both `p` and `-p`, including when `p` is its deepest point. No pole then gives a regular chart over the whole region.

Cheapest correct policy:

1. Generate candidate poles, including the trusted hint and region-constrained Chebyshev candidates.
2. Reject candidates whose antipode lies in or too close to the region.
3. Among valid candidates, minimize projected boundary conditioning, preferably maximum projected arc/chord error, not merely maximum boundary-vertex angle.
4. Validate projected constraints and adaptively subdivide before CDT.

Do not use the unconstrained boundary enclosing-cap center as the primary objective.

**Q2**

Given the repeatable failure near `90.6°`, and especially that a loop’s *first sampled edge* crosses another loop, **coarse projected chords are the more likely cause**. Azimuthal-equidistant distortion is finite at 90°, but radial scale and direction variation are already enough that a long projected chord can cut across a nearby loop while the projected analytic arc bends around it.

A genuine spherical crossing remains possible at singular probe configurations. `degree == 2`, exact endpoint joins, and complete consumption prove graph closure, not that separate cycles are embedded disjointly.

The right fix is a diagnostic refinement loop, not blind global densification:

1. When two planar constraints intersect, retain their source analytic arcs and intersection parameters.
2. Test the corresponding spherical small-circle/toric-rim arcs directly.
3. If the analytic arcs are disjoint, recursively subdivide the offending arcs in chart space until their projected chord sagitta is safely below their mutual clearance.
4. If the analytic arcs intersect or converge to the same spherical point within the geometric tolerance, split both arcs there and rebuild the arrangement. That is a genuine singular merge, not a triangulation issue.

A particularly cheap implementation is “refine on crossing and retry,” with a subdivision/depth limit. After each subdivision, recompute the crossing. A false chord crossing disappears rapidly; a real intersection converges to a stable spherical location. Also use an arc-error criterion involving midpoint-to-chord distance so near misses are handled before CDT rejects them.

Multi-chart splitting is not indicated by `90.6°`. It is needed only when no pole can keep the antipode outside the region with adequate clearance, not merely because polygonal sampling is too coarse.

**Q3**

Port the bounded retry perturbation. Full Simulation of Simplicity is unlikely to justify its cost here.

SoS would need to be applied consistently to all predicates and constructions: tangent-third-atom ordering, probe creation, arc incidence, singularity cleanup, and face orientation. Applying it only in `build_graph` can select combinatorics for which the floating-point construction still produces coincident vertices or zero-area faces. It also does not directly provide usable coordinates for the symbolically separated probe positions.

Use deterministic, scale-controlled jitter:

- Seed directions from stable atom/probe IDs, not a runtime RNG.
- Start around `1e-6`–`1e-5 Å` and increase geometrically.
- Cap displacement at `0.01 Å` and attempts at roughly 10.
- Retry only the local degenerate probe configuration.
- Accept only results passing incidence, orientation, and minimum-feature checks.
- Record that perturbation occurred for reproducibility.

This matches the uncertainty and finite precision of PDB-derived coordinates while preserving the validated geometry almost everywhere. SoS becomes worthwhile only if exact combinatorial reproducibility under arbitrary degeneracies is a hard requirement.
