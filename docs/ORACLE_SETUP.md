# Oracle environment setup

This doc is the single recipe for standing up the oracle tools proteon
validates against, and for running the evaluations. If the "0.2% OpenMM
AMBER96 parity" or "0.1 Å reduce hydrogen-placement parity" claims are
ever in doubt, this recipe is how you reproduce them on your own machine.

Read `docs/WHY.md` for the rationale behind oracle validation and
`devdocs/ORACLE.md` for the tolerance philosophy. This doc is purely
mechanical — copy, paste, run.

## Scope

This setup recipe covers the external oracles behind proteon's main CI,
local, and release validation runs, split by what runs where:

- **CI-gated slice** (pip-installable, small, fast — installed by
  `.github/workflows/test.yml` and exercised on every PR): pydssp,
  biopython, gemmi, FreeSASA. These are the library-parity checks
  under `tests/oracle/` — proteon's DSSP vs pydssp, I/O vs
  biopython/gemmi, SASA vs FreeSASA.
- **Release-quality slice** (pip-installable but heavier — OpenMM alone
  is ~500 MB — or source-built): OpenMM, reduce, TMalign, USAlign,
  GROMACS, BALL C++, BALL Julia (BiochemicalAlgorithms.jl). These are
  **not** installed in CI. They power the 1000-PDB parity benchmarks
  under `validation/` (OpenMM), the hydrogen-placement parity tests
  (reduce), monomer TM-align spot-checks and port debugging (TMalign),
  the pairwise alignment benchmark harness (`validation/bench_alignment.py`,
  via USAlign's tabular output), the fold-preservation / triangulated
  AMBER96 comparison runs (GROMACS), the CHARMM-side standalone oracle
  binaries under `validation/ball_cpp/` (BALL C++), and the per-release
  crambin reference regen (BALL Julia). Install them on your own
  machine to reproduce the published numbers or regenerate a reference.

MMseqs2 and Foldseek are also used for search/retrieval validation, but
their install flow lives with the search-specific docs and workflows
instead of this file.

## Pinned versions

These are the versions against which proteon's current reference
numbers (in `validation/reports/`, when a release is tagged) were
produced. If you install newer versions and see drift, that is the
oracle drifting, not proteon — follow the regeneration workflow in the
last section.

| Tool | Version | Source |
|---|---|---|
| OpenMM | 8.5.0 | `pip install openmm` |
| biopython | 1.87 | `pip install biopython` |
| gemmi | 0.7.5 | `pip install gemmi` |
| pydssp | 0.9.1 | `pip install pydssp` |
| FreeSASA | 2.2.1 | `pip install freesasa` |
| reduce | 4.16.250520 (commit `d723303`) | source build, see below |
| TMalign | 20220412 | source build, see below |
| USAlign | 20260329 | source build, see below |
| GROMACS | 2026.1 | source build / package install, see below |
| BALL C++ (`libBALL.so`) | commit `d85d2dd` | source build, see below |
| BALL Julia (BiochemicalAlgorithms.jl) | commit `99e4acb` | `Pkg.instantiate()`, see below |
| Julia | 1.11.5 | https://julialang.org/downloads/ |

## Prerequisites

- A Python venv with proteon installed — `packages/proteon` built via
  `maturin develop --release` and the repo's Python deps on the path.
  On any machine, set `PROTEON=/path/to/your/proteon/checkout` and use
  `$PROTEON/.venv`; on a new machine, see the top-level README's Quick
  Start.
- A C++ toolchain (`gcc` / `clang`, `cmake` >= 3.12, `make`) for the
  source-build oracles.
- Julia 1.11.5 or newer for the BALL oracle.

All commands below assume the proteon repo is checked out at
`$PROTEON` and the venv is activated.

## Install — CI-gated slice

These four are what `.github/workflows/test.yml` installs on every PR.
Install them locally to reproduce CI exactly.

```bash
source $PROTEON/.venv/bin/activate
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

OpenMM is the authoritative oracle for proteon's AMBER96 + OBC GB
claims (`validation/amber96_oracle.py` runs the 1000-PDB benchmark).
It's pip-installable but heavy (~500 MB with its molecular-dynamics
dependencies), which is why it's not in the CI install list.

```bash
source $PROTEON/.venv/bin/activate
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

Only the `reduce` binary target matters for proteon's oracle; the
Python extension (`mmtbx_reduceOrig_ext`) can fail to build against
modern Boost.Python — that is not a problem for us.

Tell proteon's tests where the binary and het dictionary live:

```bash
export REDUCE_BIN="$HOME/src/reduce/build/reduce_src/reduce"
export REDUCE_DB="$HOME/src/reduce/reduce_wwPDB_het_dict.txt"
```

Smoke test (should print `reduce.4.16.250520` and add 315 H to crambin):

```bash
$REDUCE_BIN -version
$REDUCE_BIN -NOFLIP -Quiet -NUClear -DB "$REDUCE_DB" $PROTEON/test-pdbs/1crn.pdb \
    | grep -c "^ATOM"
# expect 642 (327 heavy + 315 H)
```

## Install — TMalign (canonical C++ monomer reference)

Proteon's `tmalign` port follows the original C++ TMalign codepath for
monomer alignment. We do not currently shell out to the upstream
`TMalign` binary in `tests/oracle/`, but it is still the right tool for
spot-checking monomer behavior and debugging drift in the Rust port.

```bash
cd $HOME/src
git clone https://github.com/zhanggroup/TM-align.git TMAlign
cd TMAlign
g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
export TMALIGN_BIN="$HOME/src/TMAlign/TMalign"
```

Smoke test (should print version `20220412` on the current dev-machine
checkout):

```bash
$TMALIGN_BIN -v
```

## Install — USAlign (C++ alignment oracle harness)

USAlign is the external binary that the current automated alignment
oracle uses. `tests/oracle/test_tmscore_oracle.py` and
`validation/bench_alignment.py` both parse its `-outfmt 2` tabular
output.

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

## Install — GROMACS (prep + fold-preservation comparator)

GROMACS is part of proteon's release-quality oracle story in two places:

- `validation/tm_fold_preservation_gromacs.py` runs `gmx pdb2gmx`,
  `grompp`, `mdrun`, and compares TM-score before/after minimization.
- `validation/amber96_oracle_triangulate.py` uses `gmx pdb2gmx` and a
  0-step single-point run to triangulate AMBER96 energies against
  OpenMM.

If you already have `gmx` on `$PATH`, use it. Otherwise build or
install GROMACS 2026.1 and point proteon's scripts at the binary with
`GMX`.

```bash
# package-manager install is fine if it gives you `gmx`
# apt install gromacs
# conda install -c bioconda gromacs

# or build from source
cd $HOME/src
tar xf gromacs-2026.1.tar.gz   # if using a release tarball
cd gromacs-2026.1
mkdir -p build && cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF
make -j$(nproc)

export GMX="$HOME/src/gromacs-2026.1/build/bin/gmx"
```

Smoke test:

```bash
$GMX --version
# expect: GROMACS version: 2026.1
```

## Install — BALL C++ (`libBALL.so`, CHARMM oracle path)

Proteon still uses BALL's C++ library for some CHARMM-side oracle work
because the old SIP Python bindings do not build cleanly on current
toolchains. The standalone binaries under `validation/ball_cpp/` link
against `libBALL.so` and emit JSON for comparison.

```bash
cd $HOME/src
git clone https://github.com/BALL-Project/ball.git
cd ball
git checkout d85d2dd84   # local oracle checkout
mkdir -p build && cd build
cmake ..
make -j$(nproc)

export BALL_CPP_ROOT="$HOME/src/ball/build"
export LD_LIBRARY_PATH="$BALL_CPP_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

Smoke test:

```bash
test -f "$BALL_CPP_ROOT/lib/libBALL.so" && echo "libBALL.so ok"
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
    $PROTEON/tests/oracle/ball_energy_raw.jl \
    $PROTEON/test-pdbs/1crn.pdb
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
cd $PROTEON
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
python $PROTEON/validation/amber96_oracle.py

# Fold preservation benchmark (proteon CHARMM19+EEF1 minimizer vs
# OpenMM CHARMM36+OBC2 on 1000 PDBs).
python $PROTEON/validation/tm_fold_preservation_amber.py

# Pairwise alignment benchmark (proteon TM-align vs C++ USAlign on a
# curated pair set).
python $PROTEON/validation/bench_alignment.py

# Fold-preservation comparison with GROMACS AMBER96.
python $PROTEON/validation/tm_fold_preservation_gromacs.py

# Three-way AMBER96 spot check: proteon vs OpenMM vs GROMACS.
python $PROTEON/validation/amber96_oracle_triangulate.py \
    $PROTEON/test-pdbs/1crn.pdb
```

Outputs are JSON / JSONL files; numerical summaries + plots live under
`validation/report/`.

## Regenerating reference values when an oracle drifts

Oracle libraries publish new versions, fix bugs, change defaults. When
that happens proteon's reference numbers need to move with them. The
workflow (using BALL Julia as the worked example from 2026-04-19):

1. **Run the oracle fresh** on the canonical input:
   ```bash
   julia --project=$BALL_JL \
       $PROTEON/tests/oracle/ball_energy_raw.jl \
       $PROTEON/test-pdbs/1crn.pdb > /tmp/ball_fresh.json
   ```

2. **Diff vs the hardcoded expected values** in the relevant test file
   (e.g. `BALL_CRAMBIN_RAW` in `tests/oracle/test_ball_energy.py`) or
   against `validation/reports/`. Is the gap explained by the upstream
   oracle's changelog, by a proteon change, or by a deeper convention
   difference (charge dictionary, 1-4 scaling, cutoff policy)?

3. **If the oracle moved for a defensible reason**, update the reference:
   - Hardcoded values in the test: update + note the regen date.
   - `validation/reports/`: regenerate the report.
   - Tolerance: tighten if the new reference is closer; loosen with a
     documented reason if it surfaces a convention gap (see the BALL
     electrostatic row for what that looks like).

4. **If proteon moved**, investigate before touching the reference —
   the oracle is the authority here. Check against a second oracle
   (e.g. OpenMM if BALL drifted) to triangulate.

5. **Bump the pinned-versions table** in this file to match what you
   actually ran against.

An oracle failure is not a "fix proteon and move on" signal; it's an
investigation prompt. The regeneration workflow is part of the oracle,
not an escape hatch from it.
