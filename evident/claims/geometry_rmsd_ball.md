# Kabsch RMSD: proteon vs BALL on crambin

Operational case writeup for the RMSD-vs-BALL CI claim in
`claims/geometry_rmsd_ball.yaml`.

## Problem

`proteon.geometry.rmsd` is a public Python primitive over
`proteon-align/src/core/kabsch.rs` — Kabsch optimal-rotation
superposition followed by RMSD over the rotated coordinates. It's
used downstream by every alignment subsystem (TM-align, US-align,
MM-align, FlexAlign) and by users who pull `proteon.geometry.rmsd`
directly for MD trajectory analysis or post-docking pose
comparisons. Until now, none of those uses had an external oracle
checking that proteon's Kabsch implementation actually produces the
same numerical answer as another tool's Kabsch implementation.

The risk surface is narrow but real: Kabsch involves a 3×3
eigenvalue solve plus a sign-convention reconstruction of the optimal
rotation matrix. Most implementations land on the same answer; a
handful (especially older numerical-recipes ports) carry sign-
convention bugs that produce a valid rotation matrix with the wrong
chirality, so RMSDs match for non-chiral structures and silently
disagree on chiral ones. An independent-implementation oracle is the
cheapest insurance.

## Trust Strategy

Validation. BALL's `RMSDMinimizer` is an independent C++
implementation of the Coutsalis et al. eigenvalue method
(J. Comput. Chem., 25(15), 1849, 2004) — different team, different
language, different paper-of-reference from proteon's port. The
two implementations sharing a result on a non-trivial Kabsch case
is a strong falsifiability signal.

- **Oracle**: `ball.rmsd(pdb_a, pdb_b, atoms="ca", superpose=True)`
  from the `theGreatHerrLebert/ball` fork (added in ball-py 0.1.0a3,
  PR #2 on the bindings repo). The binding wraps `RMSDMinimizer`
  over an `AtomBijection` in CA mode.
- **Engine**: `proteon.geometry.rmsd(coords_a, coords_b)` — Nx3
  numpy arrays of paired CA coordinates extracted via
  `proteon.select(structure, "name CA")`.
- **Fixture**: a pair of structures sharing residue order — the
  original crambin (1crn) and a 20-step `ball.minimize_energy`
  relaxation of it. The minimisation moves CAs by ~0.3–0.5 Å on
  average — non-trivial, so neither side bottoms out at zero, but
  small enough that any visible disagreement points at a real bug
  rather than a stress-test edge case.

## Evidence

`tests/oracle/test_rmsd_ball_oracle.py` runs four assertions:

- `test_self_rmsd_proteon_is_zero` — `proteon.geometry.rmsd(coords,
  coords)` < 1e-6 Å. Sanity floor: the Kabsch path bottoms out at
  zero on identical input.
- `test_self_rmsd_ball_is_zero` — `ball.rmsd(pdb, pdb, atoms="ca")`
  < 1e-4 Å. Same floor on the BALL side; the wider tolerance
  reflects the round-trip through file I/O on BALL's path.
- `test_kabsch_rmsd_agrees` — the load-bearing assertion.
  `|proteon.rmsd(coords_orig, coords_min) - ball.rmsd(orig.pdb,
  min.pdb, superpose=True)|` < 1e-3 Å on the crambin pair. The
  test also asserts both sides report a non-zero RMSD as a guard
  against a degenerate fixture.
- `test_no_superpose_rmsd_agrees` — `proteon.geometry.rmsd_no_super`
  vs `ball.rmsd(superpose=False)` < 1e-4 Å. Tighter than Kabsch
  because the no-superpose metric has no algorithmic freedom (it
  reduces to `sqrt(mean(||x_i - y_i||²))` on identical paired
  arrays). Includes a Kabsch-is-the-minimum sanity check.

The test imports `ball` via `pytest.importorskip("ball")` and
additionally guards `hasattr(ball, "rmsd")` so the suite passes
(skipped) when `ball-py` is not installed or predates the rmsd
binding.

## Assumptions

- `ball-py >= 0.1.0a3` is installed in the test environment. The
  rmsd binding shipped in that release; earlier wheels skip cleanly.
- The fixture pair shares CA-atom residue order. Crambin is a
  single chain with contiguous numbering, and BALL's
  `AtomBijection.assignCAlphaAtoms` enumerates CA atoms in the same
  residue order proteon's `name CA` selection produces. Multi-chain
  or alt-loc-bearing inputs would need explicit pairing reconciliation.
- The minimised PDB is generated on-the-fly via `ball.minimize_energy`
  with `max_iter=20`. This couples the test to the existing
  ball.minimize_energy binding (shipped in 0.1.0a2). A regression
  there cascades into a fixture-construction failure, but
  `ball.minimize_energy` itself is exercised by the BALL smoke suite,
  so the failure mode is at most a one-hop hop.
- Both implementations target identical algorithmic semantics
  (Kabsch optimal rotation + RMSD over rotated coords; or no-super
  direct RMSD when `superpose=False`). The 1e-3 Å Kabsch band leaves
  margin for residue-ordering or float-precision edge cases without
  triggering on numerical noise; the 1e-4 Å no-super band is the
  tighter floor that a "no algorithmic freedom" comparison should
  hold to.

## Failure Modes

- **Single fixture.** Crambin (46 CAs) only. A release-tier 1000-PDB
  version paired with a structurally diverse corpus (multi-chain,
  altlocs, NMR ensembles) is the natural extension. Crambin
  exercises a well-conditioned point cloud — degenerate-geometry
  edge cases (collinear CAs, three-atom subsets) are not covered.
- **The "second structure" comes from ball.minimize_energy.** If
  that binding regresses (returns the input unchanged, or corrupts
  the structure), `test_kabsch_rmsd_agrees` fires the
  `assert p_rmsd > 0.0` guard with a clear message rather than
  silently passing on a trivial case. Adding a pre-minimised
  fixture (e.g. `test-pdbs/1crn_min.pdb`) would decouple this claim
  from the BALL minimizer entirely; deferred to v1.
- **ball-py wheel availability.** The claim's test
  `pytest.importorskip`s the binding, so the suite stays green when
  the wheel is missing. The python-smoke workflow on
  theGreatHerrLebert/ball is the upstream guard against a regression
  in the binding itself; this claim trusts it.
- **Sign-convention bugs would show up here.** The whole point of
  having an independent implementation. If proteon's port introduces
  (or BALL's port already has) a chirality-flipped rotation matrix,
  the post-superposition RMSD is still valid but the molecules end
  up mirrored. CA-RMSD-only is robust to this (it's a length-only
  metric), so this oracle does NOT catch chirality bugs in the
  *rotation matrix itself* — only in the eigenvalue's magnitude.
  A complementary "rotation matrix parity" claim would close that
  gap; out of scope for v0.

## Lessons

- The cheapest version of "independent oracle" is also the cheapest
  to ship: identical algorithm + identical paired input + assertion
  on the scalar output. Once that's in place, any algorithmic
  divergence (BALL changes its eigenvalue solver, or proteon does)
  surfaces immediately on the existing fixture without further
  scaffolding.
- A measurement claim that depends on another binding for fixture
  construction (here `ball.minimize_energy`) is a one-hop coupling.
  As long as the upstream binding has its own smoke gate, the hop
  is acceptable; without that gate, the dependent claim's
  assertions become unreliable. Worth recording the coupling in
  `assumptions:` so future maintainers can cut it via a pre-built
  fixture file when convenient.
