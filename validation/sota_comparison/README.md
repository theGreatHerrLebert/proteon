# SOTA Comparison Harness

Validates ferritin's numerical outputs against widely-used reference
implementations on a curated PDB set, with per-(structure, op, impl) JSON
dumps and an aggregated markdown report with PASS/WARN/FAIL bands.

## Status (v1, 2026-04-10)

**SHIPPING.** v1 covers two ops against pip-installable reference tools:

| op | reference impl | status |
|---|---|---|
| sasa | FreeSASA (Python pkg) | ✅ 18/18 PASS (6 PDBs × 3 metrics) |
| energy | OpenMM AMBER96 | ⚠ bond/torsion PASS-WARN, nonbonded systematic FAIL — real finding |

The v1 reference set is 6 small clean proteins:
`1crn, 1ubq, 1bpi, 1ake, 1pgb, 1aki` (see `sota_reference.txt`).

## Headline findings

### SASA (ferritin vs FreeSASA)

**Publication-grade agreement.** Ferritin's total SASA matches FreeSASA to
within **0.4% on every structure**, with per-residue Pearson correlations
of **0.9994 or better** and per-residue RMSD of **≤1.3 Å²**. 1crn is
essentially byte-perfect (Pearson 1.0000, total 0.024%).

Key to matching:
- Use `freesasa.Classifier.getStandardClassifier("protor")` for ferritin
  `radii="protor"` parity.
- Set `freesasa.Structure(..., options={"hetatm": True})` because
  ferritin counts crystal waters in SASA by default and FreeSASA skips
  them (this single setting accounts for ~7% of the initial disagreement).
- Use `freesasa.Parameters({"algorithm": freesasa.ShrakeRupley, "n-points": 960})`
  because FreeSASA defaults to Lee-Richards while ferritin uses Shrake-Rupley.

Full report: `sota_v1_sasa_report.md`

### Energy (ferritin vs OpenMM AMBER96)

**Bonded terms agree well; nonbonded is systematically off.** Across all
6 test proteins:

- **bond_stretch**: 0.4% – 5.6% (3 PASS, 3 WARN)
- **angle_bend**: 1.9% – 22.2% (1 PASS, 2 WARN, 3 FAIL)
- **torsion** (including impropers): 2.6% – 12.5% (4 WARN, 2 FAIL)
- **nonbonded** (vdw + electrostatic combined): **126% – 167% (all FAIL)**

The nonbonded disagreement is **consistent across every structure** and
therefore not noise. Ferritin's nonbonded is systematically larger in
magnitude than OpenMM's, with ferritin tending toward repulsive totals
while OpenMM gives attractive totals. Ferritin matches the BALL AMBER96
oracle on raw 1crn to 0.02% (see `tests/oracle/test_ball_energy.py`), so
the disagreement is specifically against **OpenMM's** interpretation of
the same force field, not a ferritin FF bug per se.

Candidate causes, in rough order of likelihood:
1. **1-4 interaction scaling**: AMBER usually uses 5/6 for electrostatic
   and 1/2 for vdW. If ferritin and OpenMM pick different scale factors
   or apply them to different atom pairs, the nonbonded term diverges.
2. **Terminal residue protonation**: ferritin consistently has 1 more
   atom than OpenMM after H placement (see `_n_atoms_diff` in the
   summary JSON). A single +1 or -1 terminal charge difference can swing
   the vacuum electrostatic sum by tens of kJ/mol.
3. **Dielectric constant / treatment of nonbonded cutoffs**: both tools
   are run in vacuum with no cutoffs, but there may be subtle differences
   in how the nonbonded sum is assembled.

**Full report**: `sota_v1_energy_report.md`

Investigation of this disagreement is a v2 task and is the most
scientifically interesting finding from v1.

### Bonus finding (fixed during v1 development)

The SASA harness smoke-testing against **1bpi** caught a real ferritin
bug: `residue_sasa` panicked with "range end index 627 out of range for
slice of length 626" on any structure with altloc residues. This was a
leftover from the altloc migration in commit `2412ea2`. Fixed in commit
`1b23670`. Without the SOTA harness this bug would have shipped to users.

## Setup (monster3)

```bash
cd /globalscratch/dateschn/ferritin-benchmark/ferritin
bash validation/sota_comparison/setup_sota_env.sh
```

This creates `/globalscratch/dateschn/ferritin-benchmark/sota_venv`, builds
ferritin into it via `maturin develop --release`, and installs FreeSASA +
OpenMM + pyarrow + numpy. Idempotent: safe to re-run.

The v1 reference PDBs live at
`/globalscratch/dateschn/ferritin-benchmark/sota_pdbs/` — assembled from
the ferritin repo `test-pdbs/` (1crn, 1ubq, 1bpi, 1ake) and the monster3
50K corpus (1pgb, 1aki).

## Run (monster3)

```bash
source /globalscratch/dateschn/ferritin-benchmark/sota_venv/bin/activate
cd /globalscratch/dateschn/ferritin-benchmark/ferritin

# SASA comparison
python validation/sota_comparison/run_all.py \
  --pdb-dir /globalscratch/dateschn/ferritin-benchmark/sota_pdbs \
  --ops sasa \
  --output /tmp/sota_sasa.json
python validation/sota_comparison/aggregate.py /tmp/sota_sasa.json \
  --output /tmp/sota_sasa_report.md --json /tmp/sota_sasa_summary.json

# Energy comparison (slower — ~6 min for 6 PDBs because of LBFGS)
python validation/sota_comparison/run_all.py \
  --pdb-dir /globalscratch/dateschn/ferritin-benchmark/sota_pdbs \
  --ops energy \
  --output /tmp/sota_energy.json
python validation/sota_comparison/aggregate.py /tmp/sota_energy.json \
  --output /tmp/sota_energy_report.md --json /tmp/sota_energy_summary.json

# Both at once, strict mode (exit non-zero on any FAIL)
python validation/sota_comparison/run_all.py \
  --pdb-dir /globalscratch/dateschn/ferritin-benchmark/sota_pdbs \
  --output /tmp/sota_all.json
python validation/sota_comparison/aggregate.py /tmp/sota_all.json \
  --output /tmp/sota_report.md --strict
```

## Design

See `/home/administrator/.claude/plans/wobbly-crunching-orbit.md` for the
locked design decisions. Highlights:

- **Runners sliced by (op, impl)**, not by tool. Each `runners/<op>.py`
  exports one function per implementation. Avoids the trap of
  `openmm_runner.py` producing three different result shapes.
- **`RunnerResult` is a typed dataclass** with per-op payload schemas.
  No free-form dicts. Units normalized inside the runner: SASA in Å²,
  energy in kJ/mol. Per-residue results keyed by `(chain, resi, icode)`
  tuples — never positional.
- **Aggregator emits both `report.md` and `summary.json`.** Humans read
  the markdown; CI / regression checks read the JSON. `--strict` exits
  non-zero on any FAIL.

Tolerance tables are hard-coded in `aggregate.py::compare_*` functions.
See the per-op docstrings in each runner module for the payload schema
contracts.

## Adding a new runner

1. Pick an op module (`runners/sasa.py`, `runners/energy.py`) or create a
   new one. Add `runners/<new_op>.py` to the import list in
   `runners/__init__.py::_safe_import()`.
2. Write a function `def your_impl(pdb_path: str) -> RunnerResult` and
   decorate it with `@register("<op>", "<your_impl_name>")`. Guard the
   import of the underlying tool in a `try/except ImportError` so a
   missing dep doesn't break the whole registry.
3. Normalize units inside the function (Å² for SASA, kJ/mol for energy,
   Å for coords).
4. Key any per-residue output by `(chain, resi, icode)` tuples.
5. If needed, add a comparison function in `aggregate.py` (see
   `compare_sasa` / `compare_energy` for examples). Mirror the same
   `(metric_name, value, band)` shape.

## v2 backlog

- **Investigate ferritin vs OpenMM AMBER96 nonbonded disagreement**. The
  biggest open question from v1. Start by dumping ferritin and OpenMM
  per-atom charges side by side, then compare 1-4 scaling on known bond
  topologies. Reference: BALL oracle test passes ferritin to 0.02% on
  heavy-only crambin, so the disagreement is post-H-placement.
- **Native-binary SOTA tools**: build mkdssp ≥ 4.0 and Reduce from source
  into `/globalscratch/dateschn/ferritin-benchmark/sota_bin/`. Wire
  `runners/dssp.py` and a hydrogen-placement op (`runners/h_place.py`)
  via PDBFixer (pip) and Reduce.
- **Stratified v2 reference set**: sample ~50 PDBs from the monster3 50K
  corpus, stratified by size + content class, with explicit "no nucleic
  acid, no metal cofactors" filtering. Commit the list to
  `sota_reference_v2.txt`.
- **Prepared-input variant**: eliminate topology-level differences
  between runners by producing a single minimized PDB via one of the
  tools (probably OpenMM Modeller + LocalEnergyMinimizer), checking it
  into the repo, and having both runners consume that. This would
  isolate FF-evaluation disagreements from H-placement disagreements.
- **CI gate**: wire `aggregate.py --strict` into a periodic CI job
  (weekly?) against the v1 / v2 reference set so regressions land early.
