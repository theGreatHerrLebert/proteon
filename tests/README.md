# Test suite

This directory holds ~650 Python test functions covering the proteon
public surface. The tree is flat — one `test_<module>.py` per Python
module where it made sense, plus the `oracle/` and `corpus/`
sub-directories for tests that share fixtures.

Canonical docs:

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — how to set up the dev
  environment and run the suite.
- [`oracle/README.md`](oracle/README.md) — oracle-test coverage table
  and authoring pattern.
- [`../docs/ORACLE_SETUP.md`](../docs/ORACLE_SETUP.md) — reproducibility
  recipe for the external oracle tools.

This file covers the stuff that falls between those: markers, fixtures,
and the canonical pytest invocations.

## Markers

Two custom markers are registered in [`conftest.py`](conftest.py) and
enforced with `--strict-markers`.

### `@pytest.mark.oracle("<tool>")`

Tags tests that compare proteon output against an independent,
externally-implemented tool. One argument, lowercase — `openmm`,
`ball`, `biopython`, `gemmi`, `usalign`, `mmseqs2`, `pydssp`, `reduce`,
`freesasa`. Lets you run the oracle layer specifically:

```bash
pytest -m oracle                     # every oracle
pytest -m oracle -k reduce           # filter by tool via test-id keyword
pytest --collect-only -m oracle -q   # inspect the coverage
```

Full pattern + conventions: [`oracle/README.md`](oracle/README.md).

### `@pytest.mark.slow`

Tags tests that take >10 seconds wall-clock (full minimizer sweeps,
corpus pipelines, large-PDB parity runs). Skip them in the dev inner
loop with:

```bash
pytest -m "not slow and not oracle"  # fast feedback cycle (~10s)
pytest -m slow                       # the 19 slow tests only
```

Add this marker to any new test that breaks 10s — it keeps the inner
loop sharp. See `conftest.py` for the registration.

## Canonical invocations

```bash
# Everything
pytest tests/ -v

# Fast inner loop (skips slow + oracle)
pytest -m "not slow and not oracle"

# Only what changed under a path
pytest tests/test_alignment.py tests/test_forcefield.py

# Oracle suite specifically
pytest tests/oracle/ -v

# With coverage report (requires pytest-cov)
pytest --cov=proteon --cov-report=term-missing
```

## The registry fixture

[`conftest.py`](conftest.py) holds the **test registry** — a single
source of truth for which force fields, code paths, and reference
structures are exercised by parametrized tests:

- `FORCE_FIELDS` — `["amber96", "charmm19_eef1"]`
- `PATHS` — `("default", None)`, `("force_nbl", 0)`, `("forbid_nbl", 10^7)`
- `STRUCTURES` — `1crn`, `1ubq`, `1bpi`, `1ake` (the v1 invariant set,
  chosen to span the NBL auto-threshold)
- `STRUCT_FF_PARAMS`, `STRUCT_FF_PATH_PARAMS` — pre-built parameter
  lists; prefer these over re-computing the cross product per file.
- `loaded_structures`, `v1_energies` — session-scoped fixtures that
  load / compute once and reuse.

Any new test file that needs to sweep over force fields, code paths, or
reference structures should consume these constants rather than
hardcode its own list. The 2026-04-11 CHARMM+EEF1 bugs demonstrated
what happens when files hardcode their own `V1_PDBS`: a missing NBL
call went undetected for months because only one structure crossed the
threshold and only one path was exercised. The registry expands every
parametrized test the moment a new FF or PDB is added.

See `conftest.py`'s module docstring for the full rationale.

## Layout

- `test_<module>.py` — unit tests for `packages/proteon/src/proteon/<module>.py`.
- `oracle/` — oracle tests + shared fixtures (`conftest.py`,
  `ball_energy_raw.jl`, `ball_energy_oracle.jl`).
- `corpus/` — corpus-release pipeline test fixtures.
- `fixtures/` — shared test data (PDB files beyond `test-pdbs/` live
  here only if they're test-suite-specific).

## When to add a new test file vs extend an existing one

- New module in `packages/proteon/src/proteon/` → new `test_<module>.py`.
  Matches the invariant: every shipped binary / module has a same-named
  test file as a coverage signal, even if it starts with a single smoke
  test.
- New assertion for an existing module → extend its `test_<module>.py`.
- New oracle (comparison against an external tool) → new
  `tests/oracle/test_<thing>_oracle.py`, tagged `@pytest.mark.oracle("<tool>")`.
  Install pointers go in [`oracle/README.md`](oracle/README.md).
