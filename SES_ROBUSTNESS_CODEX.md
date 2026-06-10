# SES mesher robustness — codex consult

> **UPDATE (2026-06-08, later): the dominant failure was a different bug entirely —
> now FIXED.** Instrumenting the actual failing proteins (104m, 109m, 1a7j, 1a5p,
> all erroring identically at CDT `boundary edge 11->12`) showed the cause was NOT
> codex's pole/chord-distortion theory (Q1/Q2) at all. `arrange_loops`
> (`arrangement.rs`) was emitting a near-fully-buried spheric-face rim as a
> *duplicated* full-circle boundary loop: a rim buried by the **union** of
> neighbours (so it misses the single-cap "fully buried" early-out) was left with
> two ~6e-9-radian exposed slivers (round-off). Both slivers had coincident 3D
> endpoints, so the `start.distance(end) < TOL` test wrongly promoted **each** to a
> full-circle loop — duplicating a boundary, which then made the contact-cap CDT
> reject the constraint and the whole analytic SES fall back to the grid mesher.
> **Fix:** classify a coincident-endpoint exposed arc by angular *span* —
> `coincident_arc_is_full_rim(span, tol) = span >= TAU - tol` — so a genuine full
> rim (span ≈ TAU) stays a loop but a sub-circle sliver is dropped. Result on the
> repro set: **7/7 proteins now take the analytic path (was 3/7); crambin still
> watertight, area 2319.9 unchanged**; 102 surface + 12 arrangement tests green
> (new guard `negligible_exposed_sliver_is_not_a_full_rim`). The proteins now mesh
> analytically but are not yet *watertight* (open edges 22–180) — that is the
> SEPARATE cross-face weld stage (NEXT_WELD §2/§3), not a crash. Codex's Q1/Q3 were
> already-done or not-the-cause; Q2's refine-on-crossing is **not currently needed**
> for this class. Re-measure exact-analytic coverage once the validation corpus is
> restored (the `validation/pdbs_1k_sample` symlinks → `/scratch/TMAlign/ferritin`,
> a dir lost in the crash — `ses-repro/` holds re-fetched repros meanwhile).
>
> **STATUS (2026-06-08): codex review done → actionable plan below.** Full verbatim
> review in sibling `SES_ROBUSTNESS_CODEX.review.md`. Key correction to this doc:
> the "63% ERROR" figures predate the **perturbation-retry jitter**, which has
> since landed (commit `caee18a` #108, `assemble.rs::build_with_perturbation_retry`
> + `perturb_atoms`). So **class 2 (Q3) is already DONE** and codex confirms the
> approach. The hybrid `ses_mesh()` never errors — it falls back to a numerical
> grid — so the real metric is **exact-analytic coverage** (`exact`+`pert`) vs
> **grid-fallback**, measured by `ses_corpus`. Re-measure before/after each item.
>
> ## Plan (cheapest correct first)
>
> ### Item 1 — Q3 jitter retry — ✅ DONE (`caee18a`)
> `perturb_atoms(atoms, attempt)` jitters each atom by a deterministic direction
> seeded from `(atom index, attempt)` via splitmix (NOT a runtime RNG — exactly
> codex's "seed from stable IDs" point), magnitude geometric from 1e-4 Å capped at
> 1e-2 Å, ≤12 attempts, retry only on `is_degeneracy_error` (cospherical / tangent
> third / degenerate caps / does-not-close / RS-face / singular), real errors pass
> through. Test `perturbation_jitter_is_tiny_and_deterministic`. Codex's only added
> nuance vs what we have: start the ramp lower (~1e-6) so the smallest escape that
> works distorts least — *optional* tuning, not a correctness fix. **No SoS** (codex:
> not worth the cost; it would have to cover every predicate, not just build_graph).
>
> ### Item 2 — Q2 refine-on-crossing arc-space loop — ⏳ TODO (high value)
> The CDT "boundary edge crosses an existing constraint" (`cdt.rs:78-83`) is raised
> on *projected* 2D chords, after the analytic arc identity has been flattened to
> bare points in `chart.rs::fill_spherical_region` (`loops2d`). Codex: for the
> residual 90.6° 4-loop failures, **coarse projected chords are the likely cause,
> but don't guess — distinguish (a) genuine spherical crossing from (b) false chord
> crossing with one mechanism**:
> 1. Thread per-segment analytic-arc provenance (the source circle/torus-rim +
>    its parametric endpoints) through sampling into the CDT input, instead of
>    discarding it at `loops2d`.
> 2. When two planar constraints cross, test the *spherical* arcs directly.
> 3. Disjoint on the sphere → false chord crossing → subdivide those arcs in chart
>    space until chord sagitta (midpoint-to-chord) < mutual clearance, retry. A
>    false crossing vanishes fast.
> 4. Converge to the same spherical point within tolerance → **genuine singular
>    merge** (the `walk_cap_loops` per-RS-face chaining missed an inter-loop touch)
>    → split both arcs there and rebuild the arrangement.
> Depth-limited "refine-on-crossing-and-retry". Subsumes both CDT sub-causes and
> the residual 4-loop cases. **Multi-chart is NOT indicated by 90.6°** — only when
> no pole clears the antipode.
>
> ### Item 3 — Q1 Chebyshev pole + antipode-rejection — ⏳ TODO
> Current pole pick (`chart.rs:173-178`) is the unsound heuristic codex flagged:
> `cap_c.dot(hint) > 0 && cap_r < maxang(hint)`. "Same hemisphere as hint" does NOT
> imply `cap_c` is *inside* the region (non-convex bay / hole case), and minimizing
> max *vertex* angle ignores chord error along arcs. Replace with: candidate poles
> (hint + region-constrained Chebyshev center `argmax_{p∈R} min_{q∈∂R} d_S(p,q)`) →
> **reject any whose antipode lies in/near the region** (with clearance) → among
> survivors minimize projected *arc/chord conditioning*, not vertex angle → adaptive
> subdivide before CDT. Multi-chart only when no candidate clears the antipode. Folds
> into the same chart machinery Item 2 touches, so do it alongside/after Item 2.
>
> ### Validation gate (every item)
> - `ses_corpus` over the 1k validation sample: **exact-analytic coverage must rise,
>   none regress** (track `exact`/`pert`/`grid` tallies).
> - Crambin stays watertight, area within +0.04% of BALL (the items must not move it).
> - clippy clean, MSRV 1.75 (`map_or` not `is_none_or`), `main` PR-gated.
> - codex review of Item 2 before/after (it touches the working analytic weld).
>
> ---
> *Original consult below (pre-jitter; kept for history).*

## Situation

We have a clean-room analytic Connolly SES mesher (contact caps on atom spheres,
toric faces, spheric faces; singularity cleaner; tolerance weld). Validated on
crambin: watertight, area within +0.04% of BALL.

We then tested 30 diverse proteins (327–10744 atoms). Findings:
- **Area vs BALL `ses_area`** (identical sphere inputs): median **+0.05%**, max
  +0.072% across every protein that meshes. The analytic geometry is portable.
- **Robustness is the gap: ~63% of proteins ERROR in the mesher**, before the
  weld, in three classes:
  1. `boundary edge N->M crosses an existing constraint` (CDT) — 8×
  2. `toric endpoint of pair [i,j] has 2 tangent third atoms (cospherical/singular)` — 6× (build_graph)
  3. `exposed boundary does not close — degenerate caps` (arrangement) — 5×
- BALL's *own* `ses_mesh` is also not perfectly watertight on these (leaks a few
  edges), so we're in its quality class; the gap is purely *not crashing*.

This doc is about class **1** (the CDT crossing), which we've partially fixed,
plus a question on the deeper remaining cause.

## What we found for the CDT crossing

The contact-cap interior is meshed by projecting its boundary loops to a planar
**azimuthal-equidistant** chart (pole inside the region), then constrained-
Delaunay-triangulating. The error means the *projected* boundary polygon
self-intersects, which spade's CDT refuses.

Azimuthal-equidistant is a homeomorphism from sphere-minus-antipode to the open
disk of radius π, so a *simple* spherical loop projects to a simple 2D loop. We
instrumented the failures and found **two distinct sub-causes**:

### Sub-cause A — off-center pole (FIXED)
The old pole heuristic (`-deepest_buried_cap.axis`) could throw boundary points
past 90° from the pole, where azimuthal distortion (scale factor θ/sinθ) makes
the *polygonal chords* between samples bow enough to cross. Re-centering the pole
on the boundary's **minimal enclosing cap** pulls the worst boundary point as
close to the pole as possible. This unlocked e.g. 160l (CDT error → meshes).

But naively centering on the boundary's enclosing cap is **wrong for a complement
region**: "sphere minus one small cap" has a tiny boundary loop whose enclosing
cap centers *on the hole*, opposite the region, so the region then wraps to the
antipode. So we only adopt the centered pole when it is on the region's side
(same hemisphere as the caller's trusted interior hint) AND it strictly reduces
the max boundary angle. Current code below. **Q1: is this heuristic sound, or is
there a region shape where "same hemisphere as hint AND lower max-angle" still
picks a pole that makes the projection self-cross? Is there a cleaner objective —
e.g. the region's Chebyshev center (deepest interior point) — that handles
disk-like and complement regions uniformly without the side test?**

### Sub-cause B — still failing (the question)
After centering, ~4 proteins (1a5p, 104m, 109m, 1a7j) STILL fail with the SAME
crossing, now at max boundary angle ~90.6°, with a **4-loop** contact cap: a
*hole* loop's first edge crosses the *outer* loop. Since the projection is
injective, two genuinely disjoint spherical loops can't project to overlapping 2D
loops — so either (a) the loops actually touch/cross **on the sphere** (the
contact-cap boundary extraction, `walk_cap_loops`, chains toric φ-rim arcs into
loops by shared RS-face index; it validates degree-2 and bit-exact joins but does
NOT check that distinct loops are mutually non-crossing), or (b) the loops pass
very close and their coarse **polygonal chords** cross near the high-distortion
90° rim even though the smooth curves don't.

**Q2: which is more likely, and what's the right fix?** Options we see:
- Densify boundary samples adaptively where two loops are close / near the rim
  (cheap; fixes (b) only).
- Detect inter-loop proximity and, if below the chord scale, treat as a genuine
  merge (the two buried-cap-derived rims actually meet) — i.e. the arrangement
  should have merged them into one boundary, and the per-RS-face chaining missed
  it.
- Multi-chart: split the region when one azimuthal chart can't keep all loops
  simple (general but heavy).

**Q3 (class 2, cheaper to ask now):** the `cospherical/singular` build_graph
errors — BALL survives these via a probe-position perturbation retry (jitter the
probe by ≤0.01 Å, up to 10×). Is that the right thing to port, or is there a
more principled symbolic-perturbation (SoS) approach worth the complexity for a
structure-bio mesher where inputs are floating-point PDB coordinates?

## Current pole-selection code (`chart.rs`, after the fix)

```rust
fn enclosing_cap(dirs: &[Vec3], seed: Vec3) -> (Vec3, f64) {
    let mut c = seed.normalized().unwrap_or(dirs[0]);
    for step in 0..256 {
        // Farthest boundary direction from the current center (smallest dot).
        let far = dirs.iter().copied().fold((c, 2.0_f64), |(best, mind), d| {
            let dot = c.dot(d);
            if dot < mind { (d, dot) } else { (best, mind) }
        }).0;
        let rate = 1.0 / (step as f64 + 2.0);
        c = (c + (far - c) * rate).normalized().unwrap_or(c);
    }
    let r = dirs.iter().map(|&d| c.dot(d).clamp(-1.0,1.0).acos()).fold(0.0_f64, f64::max);
    (c, r)
}

// inside fill_spherical_region, choosing the chart pole:
let all_dirs: Vec<Vec3> = loops.iter().flatten().map(|&w| dir_of(w)).collect();
let hint = interior_dir.normalized().expect("interior_dir nonzero");
let maxang = |p: Vec3| all_dirs.iter()
    .map(|&d| p.dot(d).clamp(-1.0,1.0).acos()).fold(0.0_f64, f64::max);
let (cap_c, cap_r) = enclosing_cap(&all_dirs, hint);
let pole = if cap_c.dot(hint) > 0.0 && cap_r < maxang(hint) { cap_c } else { hint };
let chart = Chart::new(pole);
// ... project each boundary point via chart.forward (azimuthal-equidistant),
//     reject if any point reaches the antipode, build Steiner grid, CDT.
```

## Contact-cap boundary extraction (`assemble.rs::walk_cap_loops`, abridged)

Each `ContactArc.pts` is a toric φ-rim (sampled `ses_vertex` points on the atom
sphere); arcs are chained into closed loops by shared RS-face index. It enforces
degree-2 per RS face, bit-exact joins, closure, and full consumption — but
**nothing prevents two separate loops from being geometrically close or crossing**
once projected. (Full source in the repo if needed.)

Concrete feedback, prioritized Q1→Q3. ~800 words.
