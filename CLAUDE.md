# CLAUDE.md — Proteon

Structural bioinformatics toolkit in Rust with Python bindings. Library, not a platform.

## Repository Overview

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `proteon-align/` | TM-align + US-align family (core/, ext/) — pure Rust, no I/O |
| `proteon-io/` | PDB/mmCIF loader bridge over pdbtbx |
| `proteon-arrow/` | Arrow RecordBatch + Parquet export for atoms/structures |
| `proteon-search/` | MMseqs2-compatible search: DB I/O, k-mer prefilter, ungapped/gapped SW, PSSM, MSA, GPU kernels (feature `cuda`) |
| `proteon-bin/` | CLI binaries: `tmalign`, `usalign`, `ingest`, `build_kmi`, `fasta_to_mmseqs_db` |
| `proteon-connector/` | PyO3 cdylib bridge — exposes alignment, DSSP, SASA, H-bonds, geometry, forcefield/MD, search, supervision (feature `cuda`) |

`gpu-poc/` is a standalone exploratory crate, excluded from the workspace.

### Python Package

| Package | Purpose |
|---------|---------|
| `packages/proteon/` | Pythonic wrapper; one module per subsystem (align, dssp, sasa, hbond, geometry, forcefield, prepare, search, msa, supervision, corpus_release, …). `RustWrapperObject` ABC over `proteon-connector` types. |

### Docs

Top-level:
- `README.md` — user-facing entry; also referenced via `readme.workspace = true`
- `CLAUDE.md` — this file

Under `devdocs/`:
- `ROADMAP.md`, `RELIABILITY_ROADMAP.md`, `SEARCH_ROADMAP.md`, `GEOMETRIC_DL_INFRA_ROADMAP.md` — phased plans
- `ORACLE.md` — oracle testing strategy (BALL, BALLJL, Biopython, Gemmi, OpenMM, MMseqs2, USAlign)
- `STRUCTURE_SUPERVISION_SCHEMA.md`, `RUST_BATCH_SUPERVISION_CONTRACT.md` — Layer 5 schemas
- `AMBER_UPDATE.md`, `TODO_PYPI.md`, `NEXT_SESSION.md`, `TODO_NEXT*.md` — session/release notes

### Reference C++ / external repos

Sibling directories under `/scratch/TMAlign/` (read-only oracles, not part of proteon build):
`TMAlign/`, `USAlign/`, `MMseqs2/`, `foldseek/`, `gromacs-2026.1/`, `ball/`, `BiochemicalAlgorithms.jl`, `freesasa`, `gemmi`, `openfold`, `alphafold`, `pdbtbx/`, `tm-align/`, `us-align/`.

### External Dependencies

| Dep | Source | Notes |
|-----|--------|-------|
| `pdbtbx` | `git = "douweschulte/pdbtbx" branch=main` | Main branch required (published 0.12.0 has bugs); blocks crates.io publish |
| `arrow`, `parquet` | crates.io v54 | Columnar export |
| `pyo3`, `numpy` | crates.io v0.23 | Python bindings |
| `cudarc` | crates.io v0.19 (cuda-12050) | Optional `cuda` feature |
| `memmap2`, `rayon`, `clap`, `anyhow`, `thiserror` | crates.io | Standard |

## Build Commands

```bash
cd proteon

# Build everything
cargo build

# Build specific crate
cargo build -p proteon-align
cargo build -p proteon-search --features cuda

# Run all tests (Rust)
cargo test --workspace

# Run a specific test module
cargo test -p proteon-align kabsch
cargo test -p proteon-search prefilter
cargo test -p proteon-connector forcefield

# Run binaries
cargo run --bin tmalign -- structure1.pdb structure2.pdb
cargo run --bin usalign -- a.pdb b.pdb --mm 1     # multi-chain
cargo run --bin ingest -- test-pdbs -o feats.parquet
cargo run --bin build_kmi -- corpus.parquet -o db.kmi
cargo run --bin fasta_to_mmseqs_db -- input.fa out_db

# Release
cargo build --release

# Python (PyO3 connector)
cd packages/proteon
maturin develop --release           # CPU-only
maturin develop --release --features cuda

# Python tests
pytest                              # unit + integration (skips oracle)
pytest tests/oracle                 # oracle parity (slow; needs OpenMM/BALL/MMseqs2 fixtures)
```

### Python venv

A ready-to-use venv lives at `/scratch/TMAlign/proteon/.venv` with
`proteon` already installed (Python 3.12). Activate it before running
Python / pytest / pip commands:

```bash
source /scratch/TMAlign/proteon/.venv/bin/activate
```

Install oracle dependencies into this venv (not conda, not a fresh one).

## Architecture

### proteon-align

```
src/
├── core/                # TM-align port
│   ├── types.rs         # Coord3D, Transform, TMParams, AlignResult, AlignOptions, DPWorkspace, StructureData
│   ├── kabsch.rs        # Kabsch optimal rotation (eigenvalue-based)
│   ├── tmscore.rs       # score_fun8, tmscore8_search, detailed_search, get_score_fast, standard_tmscore
│   ├── nwdp.rs          # 4 NW-DP variants over a generic nwdp_core
│   ├── secondary_structure.rs
│   ├── residue_map.rs
│   └── align/           # tmalign, cpalign, dp_iter, initial_*
└── ext/                 # US-align extensions
    ├── blosum.rs, nwalign.rs, se.rs, hwrmsd.rs, flexalign.rs
    ├── soialign/        # SOI: close_k, sec_bond, greedy, iter
    └── mmalign/         # multi-chain: chain_assign, complex_score, dimer, iter, trim
```

### proteon-search (MMseqs2 port)

```
src/
├── db/                  # MMseqs2 .ffindex/.ffdata/.dbtype I/O (reader, writer, source, lookup, index)
├── alphabet.rs, reduced_alphabet.rs   # 20→reduced AA encoding
├── kmer.rs, kmer_generator.rs, kmer_index_file.rs  # k-mer index (.kmi v2, sorted-keys array)
├── matrix.rs, pssm.rs   # BLOSUM/PSSM scoring
├── prefilter.rs         # k-mer prefilter
├── ungapped.rs, gapped.rs, sequence.rs   # SW alignment
├── padded_db.rs         # query batching
├── msa.rs               # MSA feature generation
└── gpu/                 # CUDA kernels (feature `cuda`)
    ├── diagonal, sw                          # Phase 4 baselines
    ├── pssm_diagonal, pssm_sw                # Phase 4.5 PSSM SW
    ├── pssm_sw_warp, pssm_sw_warp_multitile  # Phase 4.5a/b warp-collab (RTX 5090: up to 124× short queries)
    └── *.cu                                  # NVRTC sources, embedded via include_str!
```

### proteon-connector (PyO3 bridge)

Three-layer rustims-style architecture:
1. **Pure Rust** (`proteon-align`, `proteon-io`, `proteon-arrow`, `proteon-search`) — no Python dep
2. **PyO3 connector** (`proteon-connector`) — `#[pyclass]` wrappers, `inner: RustType`
3. **Python package** (`packages/proteon/`) — Pythonic API over the connector

Subsystems live in `proteon-connector/src/`:
- **Geometry / I/O**: `py_pdb.rs`, `py_io.rs`, `py_structure.rs`, `py_geometry.rs`, `py_transform.rs`, `py_arrow.rs`, `altloc.rs`
- **Analysis**: `py_analysis.rs`, `py_dssp.rs`/`dssp.rs`, `py_sasa.rs`/`sasa.rs`, `py_hbond.rs`/`hbond.rs`
- **Preparation**: `py_add_hydrogens.rs`/`add_hydrogens.rs`, `bond_order.rs`, `fragment_templates.rs`, `reconstruct.rs`
- **Forcefield / MD** (`forcefield/`): CHARMM19+EEF1, AMBER96, OBC GB Phase B (CPU+GPU), neighbor list, minimize, MD with SHAKE/RATTLE; CUDA kernels in `*.cu` (energy, OBC, SASA, bonded)
- **Alignment**: `py_align.rs`, `py_align_funcs.rs` — one-to-many / many-to-many over `proteon-align`
- **Search / MSA**: `py_search.rs`, `py_msa.rs` — bridges to `proteon-search`
- **Supervision (Layer 5)**: `py_supervision.rs` — geometric-DL data export, atom37 indexing, parity-tested against Python supervision pipeline
- **Parallelism**: `parallel.rs` — rayon thread budget (note: `n_threads=0` runs SERIAL; use -1 or None)

## Key Design Choices

- `core::` = TM-align port; `ext::` = US-align extensions on top
- `DPWorkspace` pre-allocates DP matrices, never shrinks
- `nwdp_core()` = generic DP engine parameterized by scoring closure (4 variants share it)
- All functions return values (no out-param mutation)
- Float operation order preserved from C++ for numerical fidelity (~4-5 dp)
- `d0` formula uses `.powf(1.0/3.0)` (not `.cbrt()`) to match C++ `-ffast-math`
- `TMParams::for_search()` / `for_final()` / `for_scale()` replace C++ parameter_set functions
- GPU paths are silent CPU fallback when no usable device is detected (`OnceLock`-cached probe)
- Cross-path parity tests guard NBL/SIMD/GPU/rayon vs slow path on every component, parametrized over force field

## Numerical Precision

C++ TMalign/USalign are compiled with `-ffast-math`. Rust does not allow this. TM-scores match to ~4-5 decimal places. Key constants: Kabsch `epsilon=1e-8`, `tol=0.01`, `sqrt3=1.73205080756888`.

## Validation Status

- 50K random PDB battle test: 99.1% correct in 3.5h on RTX 5090 (CHARMM19+EEF1 + SASA on CUDA)
- 45,100-PDB end-to-end: 17.9 min on 120 cores
- TM-align vs USAlign: 0.003 median TM drift on 4,656 pairs
- SASA vs Biopython: 0.17% median deviation (1,000 PDBs)
- AMBER96 vs OpenMM: ≤0.5% all components at NoCutoff (218/218 invariants pass)
- OBC GB vs OpenMM: ≤5% GB / ≤1% total on crambin; GPU matches CPU to 1e-11
- Fold preservation (1000 PDBs): proteon CHARMM19+EEF1 median TM=0.9945, 30× faster than OpenMM CHARMM36+OBC2

## CI / branch protection

`main` is gated by both classic branch protection and a ruleset. All 8 Tests
jobs (`Lint`, `Version Sync`, `Rust (ubuntu-latest)`, `CLI smoke
(ubuntu-latest)`, `MMseqs2 byte-exact round-trip oracle`, `Python 3.11 / 3.12 /
3.13 (ubuntu-latest)`) must be green before any change reaches `main` —
including direct pushes, not just PR merges. The Tests workflow takes ~17
minutes, so back-to-back direct pushes will block on each other.

Two ways to keep velocity under the gate:

1. **PR-only flow** (recommended) — open a PR, let checks accumulate while you
   work on the next branch.
2. **Local pre-push gate** — run `scripts/preflight.sh` (~30 s, fast tier) or
   `scripts/preflight.sh --full` (~15 min, full CI mirror) before pushing, so
   the post-push wait is the only one you eat.

Admin emergency override is available on PR merges only
(`bypass_mode: pull_request` on ruleset `15659866`); direct pushes cannot
bypass. Escape hatch if the gate becomes a problem:
`gh api -X DELETE repos/theGreatHerrLebert/proteon/rulesets/15659866`.

## MSRV

Rust 1.75+ (workspace), 1.67+ (pdbtbx)
