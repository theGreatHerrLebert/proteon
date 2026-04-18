# Ferritin Roadmap

**Last updated: 2026-04-19**

This file tracks **what's not done yet**. For what ferritin already covers,
read the top-level `README.md` (public surface), `devdocs/ORACLE.md`
(validation philosophy), and `tests/oracle/README.md` (currently-wired
oracles with per-component tolerances). Release-note history lives in
GitHub Releases, not here.

## Shipped in v0.1.0 (2026-04-18)

High-throughput I/O, the TM-align / US-align / SOI / FlexAlign / MM-align
family, SASA (Shrake-Rupley), DSSP (Kabsch-Sander), backbone H-bonds, geometry,
selection language, structural-alphabet search (MMseqs2 port + experimental
Foldseek-style alphabet, GPU Smith-Waterman kernels), Arrow/Parquet export,
batch parallelism throughout, CHARMM19+EEF1 / AMBER96 / OBC GB force fields
on CPU and CUDA, L-BFGS / CG / SD minimizers, Velocity-Verlet MD with
SHAKE/RATTLE and Berendsen thermostat, cell-list neighbor list wired into
the default energy path, `ferritin.prepare()` + `batch_prepare()` +
`load_and_prepare()`, the Layer-5 geometric-DL data export, PyPI wheels
for Python 3.12/3.13 on Linux + macOS-arm64 + Windows.

## In flight

### Validation depth
- MD energy-conservation study over longer NVE runs (drift analysis on
  10+ ns trajectories); CG-vs-SD minimizer convergence comparison on a
  fold-preservation set.
- Foldseek alphabet — close the ~15% recall gap vs upstream at TM ≥ 0.5
  (currently near-parity at TM ≥ 0.9; scripts under `validation/`).
- Reduce / DSSP / BALL oracles are wired and gating (see
  `tests/oracle/README.md`); `mkdssp` and MolProbity remain candidate
  oracles, PDB2PQR for non-default-pH electrostatics.

### Structure-preparation gaps (vs PDBFixer / reduce / Schrödinger)
- Asn / Gln / His flip optimization (amide/ring orientation; reduce does
  this, ferritin currently doesn't — see `test_reduce_hydrogen_oracle.py`
  §Known convention gaps).
- Rotatable-H optimization (Ser/Thr/Tyr OH, Cys SH, Lys NH3+) — template
  default today; reduce's H-bond scoring would close a 1.5–2 Å residual.
- Protonation-state assignment (pH-dependent His tautomers, Asp/Glu).
- Fragment variant selection (N-terminal, C-terminal, disulfide CYS
  variants).
- Nucleotide / non-standard residue templates (MSE, PTR, HYP, etc.).

### Force field / MD
- MMFF94 for small molecules / ligands.
- NPT ensemble (barostat).
- Trajectory file output (DCD / XTC).
- Replica exchange / parallel tempering.

### Python ergonomics
- Trajectory analysis module (RMSD over time, Rg, auto-correlation).
- Jupyter notebook examples covering end-to-end prep + analysis.
- Zenodo DOI on the next release (`CITATION.cff` ready).

### Performance
- SIMD vectorization for distance / energy inner loops (beyond the
  libmarv warp-collab kernel already shipped for search).

## Design principles (unchanged since v0.1.0)

1. **Compute kernel, not platform** — zero friction on any infrastructure.
2. **Batch-first** — every feature has a `batch_*` / `load_and_*` variant.
3. **Oracle-validated** — every numerical claim lands with an oracle test.
   Oracle failures gate CI, not just log (see `.github/workflows/oracle.yml`).
4. **Pipeline-native** — Arrow/Parquet output, zero-GIL, composable.
5. **No speculative abstractions** — build what's needed, test what's built.

## Where the oracle list lives

The old roadmap maintained a separate "Available Oracles" table that drifted
out of sync with the actual test suite. The authoritative list now lives
in `tests/oracle/README.md`, with per-file coverage, install pointers, and
candidate oracles not yet wired up.
