# Architecture

Proteon is a Cargo workspace with a thin Python wrapper on top. There are
**three layers**, and each public Python call passes through all of them.

```
                ┌─────────────────────────────────────────┐
   Layer 3 →    │  packages/proteon  (Pythonic API)       │
                │  - proteon.align, proteon.dssp, ...     │
                │  - RustWrapperObject ABC                │
                └────────────────────┬────────────────────┘
                                     │ PyO3
                ┌────────────────────▼────────────────────┐
   Layer 2 →    │  proteon-connector  (PyO3 cdylib)       │
                │  - #[pyclass] wrappers, inner = Rust    │
                │  - DSSP, SASA, H-bonds, forcefield/MD,  │
                │    align, search, supervision           │
                └────────────────────┬────────────────────┘
                                     │ pure-Rust call
                ┌────────────────────▼────────────────────┐
   Layer 1 →    │  proteon-align / -io / -arrow / -search │
                │  - no Python dep, no I/O assumptions    │
                │  - reusable as plain Rust libraries     │
                └─────────────────────────────────────────┘
```

## Workspace crates

| Crate | Purpose |
|-------|---------|
| `proteon-align` | TM-align + US-align family. `core/` ports TMalign; `ext/` adds SOI, MM-align, FlexAlign. |
| `proteon-io` | PDB/mmCIF loader bridge over `pdbtbx`. |
| `proteon-arrow` | Arrow `RecordBatch` + Parquet export for atoms and structures. |
| `proteon-search` | MMseqs2-compatible search: DB I/O, k-mer prefilter, ungapped/gapped SW, PSSM, MSA. CUDA kernels behind feature `cuda`. |
| `proteon-bin` | CLI binaries: `tmalign`, `usalign`, `ingest`, `build_kmi`, `fasta_to_mmseqs_db`. |
| `proteon-connector` | PyO3 bridge — exposes alignment, DSSP, SASA, H-bonds, geometry, forcefield/MD, search, supervision to Python. |

`gpu-poc/` is a standalone exploratory crate, excluded from the workspace.

## Python package layout

`packages/proteon/src/proteon/` — one module per subsystem. The top-level
`proteon` namespace re-exports the curated public surface; submodules
(`proteon.align`, `proteon.dssp`, …) are also stable.

```
proteon/
├── align.py          # tm_align, soi_align, flex_align, mm_align (+ batch variants)
├── analysis.py       # dihedrals, contact maps, distance matrices, CA extraction
├── arrow.py          # to/from Arrow + Parquet
├── core.py           # RustWrapperObject ABC
├── dssp.py           # secondary-structure assignment
├── forcefield.py     # CHARMM19+EEF1, AMBER96, OBC GB; minimize, MD
├── geometry.py       # transforms, RMSD, kabsch
├── hbond.py          # H-bond detection
├── hydrogens.py      # add hydrogens
├── io.py             # batch_load, batch_save
├── msa.py            # MSA feature generation
├── prepare.py        # batch_prepare: hydrogens + minimize
├── sasa.py           # solvent-accessible surface area
├── search.py         # MMseqs2-compatible search frontend
├── structure.py      # Structure (Pythonic wrapper around PyPDB)
├── supervision.py    # geometric-DL data export
└── ...
```

## Key design choices

- `core::` = TM-align port; `ext::` = US-align extensions on top.
- `DPWorkspace` pre-allocates DP matrices, never shrinks.
- `nwdp_core()` = generic DP engine parameterized by a scoring closure;
  the four NW-DP variants share it.
- All functions return values — no out-param mutation.
- Float operation order is preserved from the C++ originals for numerical
  fidelity. TM-scores match to ~4–5 decimal places.
- GPU paths silently fall back to CPU when no usable device is detected
  (`OnceLock`-cached probe).
- Cross-path parity tests guard NBL / SIMD / GPU / rayon vs the slow path on
  every component, parametrized over force field.

## Numerical precision

C++ TMalign / USalign are compiled with `-ffast-math`, which Rust does not
permit. TM-scores match to ~4–5 decimal places. Key constants are preserved
exactly: Kabsch `epsilon=1e-8`, `tol=0.01`, `sqrt3=1.73205080756888`. The `d0`
formula uses `.powf(1.0/3.0)` (not `.cbrt()`) to match C++ `pow(x, 1.0/3)`
behaviour.

## What gets documented where

- **This site (mkdocs):** the public Python surface (`packages/proteon/src/proteon/`),
  prose, tutorials, validation.
- **`cargo doc`:** the Rust crates. Self-hosted under [`/rust/`](rust.md) on
  the same site since the `pdbtbx` git dep blocks docs.rs publication.
- **`devdocs/` in the repo:** internal roadmaps, schemas, and session notes.
  Not published — those are work-in-progress for maintainers.
