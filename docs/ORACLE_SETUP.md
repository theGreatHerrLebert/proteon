# Oracle environment setup

This doc is the single recipe for standing up the oracle tools ferritin
validates against, and for running the evaluations. If the "0.2% OpenMM
AMBER96 parity" or "0.1 Å reduce hydrogen-placement parity" claims are
ever in doubt, this recipe is how you reproduce them on your own machine.

Read `docs/WHY.md` for the rationale behind oracle validation and
`devdocs/ORACLE.md` for the tolerance philosophy. This doc is purely
mechanical — copy, paste, run.

## Scope

Eight external oracles are covered here, split by what runs where:

- **CI-gated slice** (pip-installable, small, fast — installed by
  `.github/workflows/test.yml` and exercised on every PR): pydssp,
  biopython, gemmi, FreeSASA. These are the library-parity checks
  under `tests/oracle/` — ferritin's DSSP vs pydssp, I/O vs
  biopython/gemmi, SASA vs FreeSASA.
- **Release-quality slice** (pip-installable but heavier — OpenMM alone
  is ~500 MB — or source-built): OpenMM, reduce, USAlign, BALL Julia
  (BiochemicalAlgorithms.jl). These are **not** installed in CI. They
  power the 1000-PDB parity benchmarks under `validation/` (OpenMM),
  the hydrogen-placement parity tests (reduce), the TM-align pair
  benchmark (USAlign), and the per-release crambin reference regen
  (BALL Julia). Install them on your own machine to reproduce the
  published numbers or regenerate a reference.

MMseqs2 and GROMACS are used for additional benchmark runs under
`validation/` but are not part of this setup recipe — see their
dedicated docs.

## Pinned versions

These are the versions against which ferritin's current reference
numbers (in `validation/reports/`, when a release is tagged) were
produced. If you install newer versions and see drift, that is the
oracle drifting, not ferritin — follow the regeneration workflow in the
last section.

| Tool | Version | Source |
|---|---|---|
| OpenMM | 8.5.0 | `pip install openmm` |
| biopython | 1.87 | `pip install biopython` |
| gemmi | 0.7.5 | `pip install gemmi` |
| pydssp | 0.9.1 | `pip install pydssp` |
| FreeSASA | 2.2.1 | `pip install freesasa` |
| reduce | 4.16.250520 (commit `d723303`) | source build, see below |
| USAlign | 20260329 | source build, see below |
| BALL Julia (BiochemicalAlgorithms.jl) | commit `99e4acb` | `Pkg.instantiate()`, see below |
| Julia | 1.11.5 | https://julialang.org/downloads/ |

## Prerequisites

- A Python venv with ferritin installed — `packages/ferritin` built via
  `maturin develop --release` and the repo's Python deps on the path.
  The existing `/scratch/TMAlign/ferritin/.venv` on the dev machine is
  configured this way; on a new machine, see the top-level README's
  Quick Start.
- A C++ toolchain (`gcc` / `clang`, `cmake` >= 3.12, `make`) for the
  source-build oracles.
- Julia 1.11.5 or newer for the BALL oracle.

All commands below assume the ferritin repo is checked out at
`$FERRITIN` and the venv is activated.

## Install — CI-gated slice

These four are what `.github/workflows/test.yml` installs on every PR.
Install them locally to reproduce CI exactly.

```bash
source $FERRITIN/.venv/bin/activate
pip install \
    biopython==1.87 \
    gemmi==0.7.5 \
    pydssp==0.9.1 \
    freesasa==2.2.1
```

Smoke test (prints four lines, no errors — `freesasa` has no
`__version__` attribute, so we instantiate a `Classifier` to confirm
the native library loads):

```bash
python -c "
import Bio, gemmi, pydssp, freesasa
print('biopython', Bio.__version__)
print('gemmi    ', gemmi.__version__)
print('pydssp   ', pydssp.__file__)
print('freesasa ', 'ok' if freesasa.Classifier() else 'FAIL')
"
```

At this point `pytest tests/oracle/` runs the library-parity slice —
reduce, USAlign, BALL Julia, and OpenMM tests all skip cleanly
because their binaries / env vars / modules aren't set up yet.

## Install — OpenMM (force-field reference)

OpenMM is the authoritative oracle for ferritin's AMBER96 + OBC GB
claims (`validation/amber96_oracle.py` runs the 1000-PDB benchmark).
It's pip-installable but heavy (~500 MB with its molecular-dynamics
dependencies), which is why it's not in the CI install list.

```bash
source $FERRITIN/.venv/bin/activate
pip install openmm==8.5.0
```

Smoke test:

```bash
python -c "import openmm; print('openmm   ', openmm.__version__)"
# expect: openmm    8.5
```

## Install — reduce (hydrogen placement)

```bash
cd $HOME/src   # or wherever you keep external repos
git clone https://github.com/rlabduke/reduce.git
cd reduce
git checkout d723303c0ae7a9991a4f723b74f9bfa268d33e87  # pinned
mkdir -p build && cd build
cmake ..
make -j4
```

Only the `reduce` binary target matters for ferritin's oracle; the
Python extension (`mmtbx_reduceOrig_ext`) can fail to build against
modern Boost.Python — that is not a problem for us.

Tell ferritin's tests where the binary and het dictionary live:

```bash
export REDUCE_BIN="$HOME/src/reduce/build/reduce_src/reduce"
export REDUCE_DB="$HOME/src/reduce/reduce_wwPDB_het_dict.txt"
```

Smoke test (should print `reduce.4.16.250520` and add 315 H to crambin):

```bash
$REDUCE_BIN -version
$REDUCE_BIN -NOFLIP -Quiet -NUClear -DB "$REDUCE_DB" $FERRITIN/test-pdbs/1crn.pdb \
    | grep -c "^ATOM"
# expect 642 (327 heavy + 315 H)
```

## Install — USAlign (C++ TM-score reference)

```bash
cd $HOME/src
git clone https://github.com/pylelab/USalign.git
cd USalign
make
export USALIGN_BIN="$HOME/src/USalign/USalign"
```

Smoke test (should print version 20260329 or newer):

```bash
$USALIGN_BIN | grep "US-align (Version"
```

## Install — BALL Julia (BiochemicalAlgorithms.jl)

```bash
# 1. Install Julia 1.11.5 if you don't have it
curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel=1.11.5
#   or download a tarball from https://julialang.org/downloads/

# 2. Clone BiochemicalAlgorithms.jl at the pinned commit
cd $HOME/src
git clone https://github.com/hildebrandtlab/BiochemicalAlgorithms.jl.git
cd BiochemicalAlgorithms.jl
git checkout 99e4acb35e1820b8c5557561ba24f9218d93dcc7

# 3. Instantiate the environment (downloads + compiles deps, ~5-15 minutes)
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Smoke test (should print crambin energies and a JSON blob):

```bash
export BALL_JL=$HOME/src/BiochemicalAlgorithms.jl
julia --project=$BALL_JL \
    $FERRITIN/tests/oracle/ball_energy_raw.jl \
    $FERRITIN/test-pdbs/1crn.pdb
```

The expected JSON has `total` ≈ 78911 kJ/mol and `NonBonded::Electrostatic`
≈ 77998; if those numbers have drifted, see the regeneration workflow
below.

## Running the evaluations

### Library-parity slice — `pytest tests/oracle/`

Runs everything under `tests/oracle/`. Tests whose oracle isn't
installed skip cleanly (reduce if `REDUCE_BIN` unset, USAlign if
`USALIGN_BIN` unset, OpenMM / BALL Julia tests guard similarly). The
four CI-installed oracles (biopython, gemmi, pydssp, freesasa) run.

```bash
cd $FERRITIN
source .venv/bin/activate
pytest tests/oracle/ -v
```

With everything installed, expect **108 passed, 35 skipped** (the 35
skips are MMseqs2 recall@10 slow tests + some edge-case I/O tests
without reference data). Wall time: ~15 seconds.

Scoped runs:

```bash
pytest tests/oracle/ -m oracle                # all oracles
pytest tests/oracle/test_dssp_oracle.py       # DSSP only
pytest tests/oracle/test_ball_energy.py       # BALL only
pytest tests/oracle/ -k reduce                # reduce only
```

### Release-quality measurements — `validation/`

These produce the reference numbers quoted in the README's "Evidence it
works" section. They do not run in CI. Expected to take minutes to
hours depending on dataset size.

```bash
# AMBER96 vs OpenMM on 1000 random PDBs (primary force-field oracle).
# Needs pre-downloaded PDB corpus at the path configured in the script.
python $FERRITIN/validation/amber96_oracle.py

# Fold preservation benchmark (ferritin CHARMM19+EEF1 minimizer vs
# OpenMM CHARMM36+OBC2 on 1000 PDBs).
python $FERRITIN/validation/tm_fold_preservation_amber.py

# TM-align vs USAlign on a curated pair set.
python $FERRITIN/validation/bench_alignment.py
```

Outputs are JSON / JSONL files; numerical summaries + plots live under
`validation/report/`.

## Regenerating reference values when an oracle drifts

Oracle libraries publish new versions, fix bugs, change defaults. When
that happens ferritin's reference numbers need to move with them. The
workflow (using BALL Julia as the worked example from 2026-04-19):

1. **Run the oracle fresh** on the canonical input:
   ```bash
   julia --project=$BALL_JL \
       $FERRITIN/tests/oracle/ball_energy_raw.jl \
       $FERRITIN/test-pdbs/1crn.pdb > /tmp/ball_fresh.json
   ```

2. **Diff vs the hardcoded expected values** in the relevant test file
   (e.g. `BALL_CRAMBIN_RAW` in `tests/oracle/test_ball_energy.py`) or
   against `validation/reports/`. Is the gap explained by the upstream
   oracle's changelog, by a ferritin change, or by a deeper convention
   difference (charge dictionary, 1-4 scaling, cutoff policy)?

3. **If the oracle moved for a defensible reason**, update the reference:
   - Hardcoded values in the test: update + note the regen date.
   - `validation/reports/`: regenerate the report.
   - Tolerance: tighten if the new reference is closer; loosen with a
     documented reason if it surfaces a convention gap (see the BALL
     electrostatic row for what that looks like).

4. **If ferritin moved**, investigate before touching the reference —
   the oracle is the authority here. Check against a second oracle
   (e.g. OpenMM if BALL drifted) to triangulate.

5. **Bump the pinned-versions table** in this file to match what you
   actually ran against.

An oracle failure is not a "fix ferritin and move on" signal; it's an
investigation prompt. The regeneration workflow is part of the oracle,
not an escape hatch from it.
