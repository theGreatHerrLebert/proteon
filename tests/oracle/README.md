# Oracle tests

Everything in this directory compares ferritin output against an **independent,
externally-implemented tool** and fails if the two disagree beyond a documented
tolerance. That is the single discipline that makes the validation claims in
[`devdocs/ORACLE.md`](../../devdocs/ORACLE.md) and
[`docs/WHY.md`](../../docs/WHY.md) verifiable rather than aspirational.

If a new numerical claim is going into the codebase, it lands **with an oracle
test**, not just a unit test.

## Current oracles

| File | Oracle | What it pins |
|---|---|---|
| `test_io_oracle.py` | Biopython, Gemmi | Atom/residue/chain counts, coordinates, B-factors, hetero flags on 1ubq / 1crn / 4hhb / mmCIF variants |
| `test_tmscore_oracle.py` | USAlign (C++) | TM-score agreement with the canonical C++ reference across pdbtbx's example set |
| `test_ball_energy.py` | BALL (Julia) | CHARMM19 / AMBER96 energy components on crambin, per-component tolerances pinned |
| *(CI-only)* MMseqs2 round-trip oracle | MMseqs2 (binary, pinned release) | Byte-exact DB I/O on the vendored 50-seq fixture + end-to-end recall@10 on 20k upstream targets |

Oracles tested in the main test tree but not under `tests/oracle/` (historical,
candidates for consolidation): AMBER96 vs OpenMM (`test_amber_openmm_parity.py`),
SASA vs Biopython / FreeSASA, OBC GB vs OpenMM, supervision-export Rust/Python
parity.

## The pattern

```python
import pytest

@pytest.mark.oracle("openmm")
def test_amber_matches_openmm_on_crambin():
    ours = ferritin.compute_energy(s, ff="amber96", nonbonded_cutoff=1e6, units="kJ/mol")
    theirs = openmm_amber96_energy(s, nonbonded="NoCutoff")

    for component in ENERGY_COMPONENTS:
        assert abs(ours[component] - theirs[component]) / abs(theirs[component] + 1e-9) < 0.002, (
            f"{component}: ferritin={ours[component]:+.4f} openmm={theirs[component]:+.4f}"
        )
```

Four conventions:

1. **Tag with `@pytest.mark.oracle("<tool>")`.** One word, lowercase — `openmm`,
   `ball`, `biopython`, `gemmi`, `usalign`, `mmseqs2`, `freesasa`. Lets `pytest
   -m oracle` filter and lets `pytest -m 'oracle and openmm'` scope.
2. **Assert a numerical tolerance, not equality.** Floats differ across
   platforms, `-ffast-math` reorderings, and sometimes legitimate convention
   gaps. The tolerance is *part of the claim* — pick it deliberately and
   document what it encodes (see `devdocs/ORACLE.md` §Tolerances).
3. **Make disagreements diagnostic.** The assertion message should print both
   values and the component/name so that a failure in CI tells you what
   diverged, not just "Booleans differed".
4. **If the oracle is slow to install, gate it.** Use `pytest.importorskip` or
   an env var (`FERRITIN_SEARCH_REQUIRE_ORACLE=1` for MMseqs2) so dev loops
   stay fast; CI turns the gate on explicitly.

## Running

```bash
pytest -m oracle                       # run every oracle test
pytest tests/oracle -v                 # run the directory
pytest tests/oracle/test_ball_energy.py # run one oracle
pytest -m oracle -k usalign            # filter by tool via test-id keyword
pytest --collect-only -m oracle -q     # inspect the coverage
```

The marker is `oracle(tool)` — `tool` is an argument on the single marker, not
a second marker, so `-m 'oracle and usalign'` won't filter by tool. Use
`-k <tool>` or a path instead.

CI runs the MMseqs2 oracle as its own workflow job (`MMseqs2 byte-exact
round-trip oracle` in `.github/workflows/test.yml`) with a pinned upstream
release, so PRs are gated on it. Other oracles run in the main Python matrix
when the oracle tool is importable.

## Getting the oracles

Every oracle is a real, third-party tool. To run the matching tests locally
you need the tool installed and (in a few cases) told where to find it.
Python-wrapped oracles install with pip; the heavy ones ship via conda or
upstream binaries.

| Tool | Install | Invocation in tests |
|---|---|---|
| [Biopython](https://biopython.org) | `pip install biopython` | Imported directly; no config |
| [Gemmi](https://gemmi.readthedocs.io) | `pip install gemmi` | Imported directly; no config |
| [FreeSASA](https://freesasa.github.io) | `pip install freesasa` | Imported directly; no config |
| [OpenMM](https://openmm.org) | `pip install openmm` (or `conda install -c conda-forge openmm`) | Imported via `pytest.importorskip("openmm")` |
| [USAlign](https://github.com/pylelab/USalign) | `git clone && make`, put `USalign` on `$PATH` | Set `USALIGN_BIN=/path/to/USalign` if not on `$PATH` |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | `conda install -c bioconda mmseqs2`, or the pinned binary release (see `.github/workflows/test.yml` `MMSEQS_VERSION`) | Set `FERRITIN_SEARCH_MMSEQS_BIN=/path/to/mmseqs` and `FERRITIN_SEARCH_REQUIRE_ORACLE=1` to gate |
| [BALL](https://github.com/hildebrandtlab/BiochemicalAlgorithms.jl) (Julia) | `julia --project=<path>` and `Pkg.add("BiochemicalAlgorithms")` | `ball_energy_raw.jl` / `ball_energy_oracle.jl` shell out; see the scripts in this directory |

Tests that `pytest.importorskip` a missing oracle will **skip** rather than fail,
so you can run `pytest -m oracle` locally and just see results for the tools
you have — CI runs the full set.

## Adding a new oracle

1. Install the oracle locally and get a known-good reference value for a
   small, reproducible input (crambin is the usual start).
2. Write the assertion in `tests/oracle/test_<thing>_oracle.py`, tagged
   `@pytest.mark.oracle("<tool>")`, with a tolerance documented in the
   docstring.
3. If install is heavy (OpenMM, BALL, CUDA): add `pytest.importorskip` so
   the test skips silently on machines without it.
4. Update the table above.
5. Update `devdocs/ORACLE.md` if the new oracle extends the principle
   (new tolerance class, new convention gap, new slowness tier).
