# CHARMM19+EEF1 component oracle gap

Operational case writeup for the CHARMM19 internal-only reference claim
in `claims/forcefield_charmm19_internal.yaml`.

## Problem

Proteon's CHARMM19+EEF1 implementation
(`proteon-connector/src/forcefield/`) is the production-default force
field — `feedback_charmm_priority` memory: *"CHARMM is production
default; validate CHARMM19+EEF1 first; AMBER96 has the oracle but nobody
uses it."* The current oracle inventory inverts that priority:

- **AMBER96** has two external oracles —
  `forcefield_amber_openmm.yaml` (1000 PDBs vs OpenMM, <0.5% median) and
  `forcefield_amber_ball.yaml` (per-component vs BALL Julia on crambin).
- **CHARMM19+EEF1** has neither. Its component-level validation is
  internal cross-path parity (NBL vs exact); its whole-system validation
  is fold-preservation against OpenMM CHARMM36+OBC2 — different force
  field, contextual SOTA only.

This is not an oversight. It is a structural blocker:

1. **OpenMM does not ship CHARMM19.** It ships CHARMM36 (`amber14-all`,
   `charmm36`). Loading CHARMM19 in OpenMM requires `toppar_c19` files
   from the original CHARMM distribution or a CHARMM-GUI export, plumbed
   through `CharmmParameterSet`. Sourcing and pinning those files is real
   v1.1 work.
2. **EEF1 is CHARMM-only.** Even with toppar_c19 in hand, EEF1's
   gaussian-solvation kernel has no OpenMM equivalent. A defensible
   external oracle would compare proteon's bonded + non-bonded
   components with EEF1 *disabled*; EEF1 itself stays internal-validated.
3. **BALL Julia does not implement CHARMM19** — `BiochemicalAlgorithms.jl`
   covers AMBER, not CHARMM. Adding CHARMM to BALL would require
   upstream work outside proteon's control.

The gap is real, structural, and worth surfacing at the manifest level so
it is queryable rather than buried in commit history.

## Trust Strategy

Validation by reference shadowing — the same pattern as the
cross-path-parity claim. This claim is an **inventory note**: it points
at the existing internal validation and explicitly names what's missing
so coverage queries find it.

- **Cross-path parity (CI-tier, today)**:
  `tests/test_cross_path_parity.py` parametrises over both `amber96` and
  `charmm19_eef1`, four structures (1crn, 1bpi, 1ubq, 1ake), and both
  NBL and exact paths. Asserts every component (`bond_stretch`,
  `angle_bend`, `torsion`, `improper_torsion`, `vdw`, `electrostatic`,
  `solvation`) agrees to `max(1e-6 kcal/mol, 1e-9 * |E|)` between
  paths.
- **Fold preservation (release-tier, today)**:
  `validation/tm_fold_preservation_openmm.py` runs proteon CHARMM19+EEF1
  minimisation on 1000 PDBs and compares fold-TM to OpenMM CHARMM36+OBC2
  on the same corpus. Cited at `proteon TM=0.9945, gap=0.0046` — strong
  whole-system evidence, weak component attribution.
- **External per-component oracle (v1.1 gap)**: not implemented. The
  defensible shape is documented in the YAML's `assumptions` so the
  next person to pick this up has a starting point.

## Evidence

`pytest tests/test_cross_path_parity.py -k "charmm19_eef1" -v` runs the
full CHARMM19 cross-path matrix. Eight assertions per (structure, path)
pair; 32 cells total over the four-structure × two-path corpus. The
failure mode of an asymmetric implementation bug — the 2026-04-11 EEF1
1-2/1-3 exclusion regression and the NBL path silently skipping EEF1 —
is exactly what this matrix catches.

What it cannot catch:

- A formula bug present **in both** paths (e.g. a wrong vdW switching
  function applied identically in `compute_energy_impl` and
  `compute_energy_nbl`).
- A parameter-table regression that shifts every reference structure by
  the same factor (e.g. unit conversion on the par file at load time).
- An EEF1 reference-state error that is internally consistent.

The release-tier fold-preservation claim catches some of these at
whole-system level if they degrade fold geometry, but a regression that
shifts energies without shifting forces meaningfully — or that shifts
forces in a way the minimiser absorbs — would ship.

## Assumptions

- The cross-path-parity tolerance (1e-6 kcal/mol absolute, 1e-9
  relative) is tight enough that any real per-component bug visible at
  PDB-coordinate precision would surface against either path.
- The four-structure corpus (327, 454, 602, 3317 atoms) spans both
  small/medium and the NBL-heuristic boundary. A regression specific
  to e.g. >5000-atom systems with unusual topology would not surface
  here. The 50K-PDB battle test exercises that regime as a
  physics-invariant sanity check, not as a parity assertion.
- A future external CHARMM19 oracle should compare *bonded* terms
  (bond, angle, torsion, improper) and *non-bonded* terms
  (vdw, electrostatic at NoCutoff) against OpenMM with toppar_c19
  loaded via `CharmmParameterSet`. EEF1 must be disabled in proteon
  for the comparison; EEF1 stays guarded by cross-path parity and
  by fold preservation.
- `feedback_charmm_priority` memory is current: CHARMM remains
  proteon's production default. If that flips (e.g. AMBER becomes
  default), this claim's prioritisation rationale weakens but the
  documented gap remains real.

## Failure Modes

- **No asserting test exists for the gap itself.** This claim is a
  kind=reference inventory note. Closing it requires landing
  `forcefield_charmm19_openmm.yaml` as a kind=measurement claim with a
  real `validation/charmm19_oracle.py` script behind it.
- **Symmetric bug invisibility.** Cross-path parity is the highest-leverage
  guard against asymmetric regressions and the lowest-leverage guard
  against symmetric ones. The 2026-04-11 EEF1 sign-bug history shows
  the asymmetric case being caught; a future symmetric regression would
  ship under the current oracle inventory.
- **"Internal-only" understates the fold-preservation evidence.**
  `proteon-fold-preservation-charmm-vs-openmm-release` IS an external
  comparison (vs OpenMM CHARMM36+OBC2). It is just not a per-component
  one and not a same-force-field one. Treat this claim as documenting
  a *component-level external* gap, not a total absence of external
  validation.
- **Reference-claim retirement is a manual step.** When the v1.1
  measurement claim lands, this reference claim should be deleted and
  the manifest's `include:` list updated. Skipping that step would
  leave a redundant entry visible in coverage queries.

## Lessons

- Production-default subsystems deserve the strongest oracle stack,
  not the historical one. The AMBER96 oracle landed first because it
  was easiest (OpenMM ships it). That's the wrong reason to leave
  CHARMM19 thinner.
- Document gaps as claims. A `kind=reference` entry naming the missing
  oracle is queryable and budget-able; a known-but-undocumented gap
  rots into "everyone forgets it's missing." This claim is the budget
  line item for the CHARMM19 component oracle work.
- An external oracle that requires sourcing parameter files from a
  third-party tool (toppar_c19 from CHARMM-GUI) is fundamentally
  different from one where the oracle ships the parameters
  (OpenMM AMBER96). Pin the source and version of the parameter files
  alongside the oracle binary; otherwise the claim will rot the moment
  CHARMM-GUI updates its export format.
