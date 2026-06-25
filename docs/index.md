# Proteon

Rust-first structural bioinformatics toolkit for loading, aligning, analyzing,
and preparing macromolecular structures, with an experimental Foldseek-style
search stack.

Proteon is a **library**, not a platform. No service, no database, no
scheduler. It gives you fast structure I/O and heavy compute from Rust, with
Python and CLI entry points.

## What's in here

- **[Install](install.md)** — pip, source, and CUDA notes.
- **[Quickstart](quickstart.md)** — load, align, analyze, prepare in a few lines.
- **[Architecture](architecture.md)** — the workspace layout and the three-layer
  Rust → connector → Python stack.
- **[Subsystems](subsystems/index.md)** — one page per public surface, with
  auto-generated API reference for the `proteon` Python package.
- **[Validation](validation.md)** — oracle-parity numbers vs OpenMM, Biopython,
  USAlign, MMseqs2.
- **[Rust API](rust.md)** — `cargo doc` output for the workspace crates.
- **[Why Proteon](WHY.md)** — the motivation behind the project.

## API stability tiers

The top-level `proteon` namespace is split into two tiers — see
**[API stability](stability.md)** for the full contract:

- **Stable** (`proteon.__stable__`) — the strict pure-compute core: one
  oracle-validated quantity per call, fixed signature (alignment, SASA, DSSP,
  H-bonds, geometry, I/O, the structure model, energy/minimization). proteon
  promises to keep these working.
- **Experimental** (`proteon.__experimental__`, canonical path
  `proteon.experimental.*`) — research frontiers and multi-stage orchestration
  whose APIs/schemas may still change: the `prepare` + structure-supervision
  pipeline, structural search, MSA/templates, electrostatics, Vina, `run_md`.
  Validated but not contract-frozen; flat top-level access is retained for
  back-compat and will warn in a future minor release.

Names beginning with `_` and the underlying `proteon_connector` PyO3 bindings
are **not** part of either contract — depend on them at your own risk.

## Search

Use the search box (top of the page or `s`) to jump across prose and the
auto-generated API reference at once.
