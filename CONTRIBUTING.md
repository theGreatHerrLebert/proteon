# Contributing to ferritin

Thanks for considering a contribution. This file is short on purpose — it
captures the few conventions that matter and points at the docs that
explain them in depth.

## Where things live

| Layer | What goes here |
|---|---|
| `ferritin-align/`, `ferritin-io/`, `ferritin-arrow/`, `ferritin-search/` | Pure Rust, no Python dependency. Algorithm code lives here. |
| `ferritin-connector/` | PyO3 bridge — `#[pyclass]` wrappers, `#[pyfunction]` entry points. Thin layer. |
| `packages/ferritin/` | Pythonic wrapper. One module per subsystem (align, dssp, sasa, forcefield, …). Decorates the connector types with an ABC (`RustWrapperObject`). |
| `tests/` | Python test suite (matches the wrapper surface). |
| `tests/oracle/` | Tests that compare ferritin against independent external tools. **See [tests/oracle/README.md](tests/oracle/README.md).** |
| `validation/` | Large-scale benchmarks and reports — not part of CI, but reproducible. |
| `devdocs/` | Internal planning and philosophy — [`ORACLE.md`](devdocs/ORACLE.md) in particular. |

## Quality bar

ferritin is a scientific compute kernel, not a platform. That shapes what a
"good" contribution looks like:

1. **New features land with at least one oracle test.** An oracle is an
   independent, externally-implemented tool (OpenMM, BALL, MMseqs2, USAlign,
   Biopython, Gemmi, FreeSASA, DSSP, GROMACS, reduce, …). Agreement with
   the oracle — at a tolerance you deliberately chose and documented — is
   what lets reviewers trust the claim. See
   [`tests/oracle/README.md`](tests/oracle/README.md) for the pattern and
   the install pointers.
2. **Edge-case tests defending structural semantics, not happy-path.**
   "`compute_energy` returns a number on crambin" is not a test; "total is
   finite on 218 invariant PDBs, signs are correct on pre-minimization
   clashes, NaN propagates if and only if inputs have NaN" is. If a future
   refactor breaks the implicit contract, the test should fail before the
   oracle does.
3. **Cross-path parity is a hard requirement.** Every accelerated path
   (NBL, SIMD, GPU, rayon) needs a parity test against the slow path on
   every component, parametrized over force field. The 2026-04 CHARMM+EEF1
   bugs hid for months because NBL and default paths weren't cross-checked.
   Never add a fast path without the parity test.
4. **No features you don't need.** Don't pre-abstract; three similar lines
   is better than a premature helper. Don't add fallbacks for scenarios
   that can't happen — only validate at system boundaries.
5. **No comments that restate the code.** Comments earn their keep by
   documenting the *why*: a hidden constraint, a subtle invariant, a
   workaround for a specific bug. If removing the comment wouldn't confuse
   a future reader, don't write it.

## Adding an oracle

The full pattern is in [`tests/oracle/README.md`](tests/oracle/README.md);
summary:

1. Install the oracle tool locally (install pointers in that README).
2. Write `tests/oracle/test_<thing>_oracle.py` tagged
   `@pytest.mark.oracle("<tool>")`.
3. Pick a deliberate tolerance and document *what it encodes* — why this
   number, not a rounder one. See
   [`devdocs/ORACLE.md`](devdocs/ORACLE.md) § Tolerances.
4. If install is heavy (OpenMM, BALL, CUDA): gate with
   `pytest.importorskip` so dev loops stay fast; CI turns the gate on.
5. Update the install table in `tests/oracle/README.md`.

## Setup

```bash
# Rust side
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# Python side
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy pandas pyarrow
cd ferritin-connector && maturin develop --release && cd ..
pip install -e packages/ferritin/
pytest                              # skips oracle tests that need heavy installs
pytest -m oracle                    # runs every oracle you have installed
```

Minimum Python: **3.12**. Minimum Rust: **1.75**.

## Commit messages

- Present tense, imperative: "fix", "add", "strip", not "fixed" / "adds".
- First line ≤ 72 chars, a blank line, then body explaining the *why* (the
  "what" is in the diff).
- Reference incidents, issues, or memory items that motivated the change
  when it would help the future reader.

## Pull requests

- Feature branches off `main`, PRs back to `main`.
- Keep diffs focused — a bug fix and a refactor are two PRs.
- CI must pass (`Lint`, `Rust`, `Python 3.12 / 3.13`, `MMseqs2 oracle`,
  `CLI smoke`) before merge. `Lint` includes `cargo fmt --check` and
  `cargo clippy -D warnings`.
- If a CI failure is pre-existing and unrelated, flag it in the PR rather
  than silencing it.

## License

By contributing, you agree that your contribution is licensed under the MIT
License of the project (see [`LICENSE`](LICENSE)).
