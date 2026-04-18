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
| [BALL C++](https://github.com/BALL-Project/ball) (`libBALL.so`) | Build from source (`cmake + make`); no pip/conda distribution. Used for the CHARMM19 oracle because the SIP Python bindings don't build cleanly on current toolchains. | Standalone C++ binaries under `validation/ball_cpp/` link `libBALL.so` and emit JSON; tests read those |
| [GROMACS](https://www.gromacs.org) | `conda install -c bioconda gromacs`, `apt install gromacs`, or [build from source](https://manual.gromacs.org/current/install-guide/) | `gmx` / `gmx_mpi` binary on `$PATH`; referenced from `validation/tm_fold_preservation_gromacs.py` as a fold-preservation comparator against ferritin's CHARMM19+EEF1 minimizer |

Tests that `pytest.importorskip` a missing oracle will **skip** rather than fail,
so you can run `pytest -m oracle` locally and just see results for the tools
you have — CI runs the full set.

## Candidate oracles (not yet wired up)

These are tools that would extend oracle coverage into components currently
validated only internally (or cross-checked against a single other tool).
Listed here so the gap is visible; they're not in the install table above
because no test actually uses them yet. Contributions welcome.

| Tool | Would cover | Why it's the right oracle |
|---|---|---|
| [reduce](https://github.com/rlabduke/reduce) (Richardson Lab, Duke) | `add_hydrogens` / protonation placement | Canonical asymmetric hydrogen placer; mature, widely trusted, independent of any MD engine. Current hydrogen placement is only cross-checked against BALL + GROMACS (both also MD-linked), so reduce would give a third voice that doesn't share force-field assumptions. |
| [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) (`mkdssp` binary) | Secondary structure assignment | The reference implementation Kabsch & Sander wrote the paper about. Ferritin ships its own DSSP port; agreement on H/E/G/I assignments per residue is a trivially checkable oracle. |
| [MolProbity](http://molprobity.biochem.duke.edu) | Clash detection, rotamer outliers, Ramachandran quality | Community-standard structure-validation suite. Bundles reduce + probe. Useful once ferritin starts producing prepped structures at scale — sanity-check that ferritin-prepped PDBs pass MolProbity at the same rate as OpenMM- or BALL-prepped ones. |
| [PDB2PQR](https://pdb2pqr.readthedocs.io) | Protonation states, pKa assignment, charge/radius parameters | Standard upstream for electrostatics prep. A natural oracle for anything ferritin does with charges at non-default pH or on titratable residues. |

Each of these is a real gap — the component exists in ferritin, the oracle
exists in the wild, nobody has wired them up. When one gets wired in, move
the row up into the main install table and delete it from here.

**Intent — reduce is the next one we plan to wire.** Open design questions
to resolve before writing the test, so the first PR isn't a false-green:

- Ferritin's `add_hydrogens` does polar-only H for CHARMM19 (united-atom
  carbons absorb C–H) but a fuller set for AMBER96. reduce places the full
  set by default. A naive "every H ferritin placed exists in reduce's
  output" works both ways; "every H reduce placed exists in ferritin's
  output" only works for AMBER96. Test needs to parametrize over FF.
- Reduce resolves asymmetric ambiguities (e.g. Asn/Gln/His flips) that
  ferritin currently doesn't. For an apples-to-apples comparison the test
  should disable reduce's flip search (`-NOFLIP`) or compare only against
  the set of H atoms whose placement isn't flip-dependent.
- Tolerance: coordinate-level RMSD per H across a small curated set
  (crambin + ubiquitin, no waters) vs. a looser "positional bucket" match.
  RMSD is tighter and more diagnostic; start there.

Once those three are answered, wiring is a ~100-line
`test_reduce_hydrogen_oracle.py` + one table-row promotion.

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
