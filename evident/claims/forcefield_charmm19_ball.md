# CHARMM19+EEF1 vs BALL on crambin

Operational case writeup for the CHARMM19-vs-BALL CI claim in
`claims/forcefield_charmm19_ball.yaml`. Closes the gap documented in
`claims/forcefield_charmm19_internal.yaml` — that reference claim
was a budget-line item for exactly this oracle, and its
**Retire-when** condition is met by this measurement claim landing.

## Problem

CHARMM19+EEF1 is proteon's production-default force field
(`feedback_charmm_priority` memory) but had no external per-component
oracle until the BALL Python bindings (`ball-py`) shipped. The
inversion was structural:

- OpenMM ships CHARMM36 not CHARMM19, and EEF1 has no OpenMM
  equivalent.
- BALL Julia (BiochemicalAlgorithms.jl) covers AMBER but not CHARMM.
- The original BALL C++ has CHARMM19+EEF1 fully implemented but
  required a Julia or C++-standalone harness to access — out of
  scope for proteon's pytest pipeline.

The pybind11-based `ball-py` wheel removes that harness requirement.
`pip install ball-py; import ball` and proteon's CHARMM19+EEF1
implementation now have a per-component oracle reachable from a
single `pytest tests/oracle/...` invocation.

## Trust Strategy

Validation. BALL is an independent C++ implementation of CHARMM19+EEF1
— different team (Hildebrandt-lab), different language, different
parameter-file vintage from any OpenMM build. Importantly, BALL's
EEF1 module IS the canonical EEF1 implementation outside of CHARMM
itself: there is no other reachable oracle for that specific
component without standing up a CHARMM build.

- **Oracle**: `ball.charmm_energy(pdb_path, use_eef1=True,
  nonbonded_cutoff=1e6, add_hydrogens=False)` from the
  `theGreatHerrLebert/ball` fork. The binding wraps BALL's
  `CharmmFF` with the standard `FragmentDB::normalize_names`
  + `FragmentDB::build_bonds` preprocessing chain; we pass
  `add_hydrogens=False` so BALL accepts the hydrogens proteon
  itself placed upstream, removing H-placement as a divergence
  source.
- **Engine**: `proteon.compute_energy(structure, ff="charmm19_eef1",
  nonbonded_cutoff=1e6)` returning the kJ/mol component dict.
- **Fixture**: crambin (1crn), single-chain, hydrogens placed by
  proteon's standard pipeline.
- **Settings**: `nonbonded_cutoff=1e6` on both sides, matching the
  established AMBER+BALL claim's no-cutoff convention.

## Evidence

`tests/oracle/test_charmm19_ball_oracle.py` runs seven assertions on
the same crambin structure passed to both engines, with hydrogens
placed once by proteon and shared:

- `bond_stretch`: <1% relative agreement.
- `angle_bend`: <1% relative agreement.
- `proper_torsion`: <2.5% — split out from BALL's `getTorsionEnergy`
  total (which includes impropers) via the
  `ball.charmm_energy()` dict's separate `proper_torsion` key.
- `improper_torsion`: <2.5% — same key separation.
- `vdw`: <2.5% relative at NoCutoff.
- `electrostatic`: <25% relative — wide by design, mirrors the
  AMBER+BALL band, accommodates the documented BALL-vs-canonical
  partial-charge / 1-4 scaling gap.
- `solvation`: <5% relative — tightest external-only band because
  EEF1 has no other reachable oracle.

The test imports `ball` via `pytest.importorskip("ball")` so the
suite passes (skipped, not failed) when `ball-py` is not installed.

## Assumptions

- `ball-py` is on PyPI as a pip-installable wheel. First publish is
  expected on tag `v0.1.0a0` of the
  `theGreatHerrLebert/ball` repo via the `build-wheels.yml`
  cibuildwheel + Trusted Publishing flow. Until that publishes, the
  binding is built locally via `pip install /path/to/ball-zomball`
  and the test still runs.
- BALL's `use_eef1=True` engages the full EEF1 gaussian-solvation
  kernel as documented in `parm19_eef1.ini`. proteon's
  `ff="charmm19_eef1"` is the same EEF1 logic ported to Rust via
  `proteon-connector/src/forcefield/`.
- The hydrogens proteon places upstream are valid CHARMM19 polar-H
  positions. BALL's CharmmFF `setup()` accepts them without
  re-placement. If setup() rejects (atom typing fails for some
  H types), this claim widens its tolerances or moves to a smaller
  fixture.
- The `nonbonded_cutoff=1e6` convention on both sides yields
  mathematically identical no-cutoff long-range evaluation —
  same assumption as the AMBER+BALL claim.
- Crambin's coverage of standard-residue bond/angle/torsion classes
  plus amide-plane impropers is sufficient for component-level
  claims; non-standard residues fall under the (still unwritten)
  release-tier version of this claim.

## Failure Modes

- **Single fixture.** Crambin only. A release-tier follow-up over
  the 1000-PDB pool (mirroring `forcefield_amber_openmm_release`)
  is the natural extension; it would catch regressions on
  non-standard residues, terminus handling, large-loop
  conformations.
- **BALL is the only EEF1 oracle.** A bug in BALL's EEF1
  implementation propagates through this claim invisibly. The
  closest mitigation — cross-check against the original CHARMM
  program's EEF1 output — is heavyweight and out of scope here.
- **H placement divergence.** BALL's AddHydrogenProcessor and
  proteon's add_hydrogens differ in ring / chiral handling on a
  few residue types. We work around by placing hydrogens once
  (proteon side) and passing `add_hydrogens=False` to BALL. If
  BALL's setup() rejects proteon's H positions on some atom types,
  the test will surface as a `setup failed` error rather than
  silently widening the energy disagreement — at that point the
  fixture should narrow to a poly-Ala or similar to isolate the
  noise.
- **ball-py wheel availability.** The claim's test
  `pytest.importorskip`s the binding, so the suite stays green
  when the wheel is missing. That preserves CI velocity but means
  a regression in the binding itself can hide as a skip rather
  than a fail. The python-smoke workflow on the
  theGreatHerrLebert/ball repo is the upstream guard for that case;
  this claim trusts it.

## Lessons

- An external oracle that requires building a heavyweight C++
  toolchain (e.g. BALL Julia + Boost + Qt6) is structurally weaker
  than one that ships as a pip wheel. The forcefield_charmm19_ball
  claim was deferred for months as the BALL access story remained
  "use the C++ standalone harness"; the moment it became `pip
  install ball-py`, the claim went from "documented gap" to
  "ship in v1".
- Closing a `kind: reference` gap claim with a `kind: measurement`
  is a manifest-visible promotion. The retired reference
  (`forcefield_charmm19_internal`) should be deleted from the
  manifest's `include:` list once this claim's `last_verified`
  fields are populated for the first time, not before — the
  reference still documents the gap so long as the measurement
  has not actually been verified.
