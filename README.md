# Ferritin

Rust-first structural bioinformatics toolkit for loading, aligning, analyzing, searching, and preparing macromolecular structures.

## TL;DR

- Ferritin is a **library**, not a platform. No service, no database, no scheduler.
- It gives you **fast structure I/O and heavy compute** from Rust, with Python and CLI entry points.
- Core jobs already wired in: **PDB/mmCIF loading, TM-align/US-align family alignment, SASA, DSSP, H-bonds, geometry, search, and structure preparation/minimization**.
- Outputs stay interoperable: **NumPy, Arrow, Parquet, pandas/polars-friendly tables**.
- The repo also contains **dataset/release utilities** for sequence and structure-supervision pipelines.

If you want a Python package that can sit inside your own pipeline and do the expensive structural-biology work without forcing a platform decision, this is what Ferritin is for.

## Quick Start

```bash
pip install ferritin
```

```python
import ferritin

s = ferritin.load("1crn.pdb")

tm = ferritin.tm_align(s, ferritin.load("1ubq.pdb"))
print(tm.tm_score_chain1, tm.rmsd)

print(ferritin.total_sasa(s))
print(ferritin.dssp(s)[:10])

prepared = ferritin.prepare(s)
```

Runnable examples live in [`examples/`](examples/).

## What Ferritin Covers

| Area | Examples |
|---|---|
| Structure I/O | `load`, `load_pdb`, `load_mmcif`, `save`, tolerant batch loading |
| Structural alignment | `tm_align`, `soi_align`, `flex_align`, `mm_align`, one-to-many and many-to-many variants |
| Structure analysis | SASA, DSSP, backbone dihedrals, H-bonds, contact maps, distance matrices, RMSD/TM-score |
| Preparation | hydrogen placement, minimization, `prepare`, batch preparation |
| Search | structural-alphabet encoding, database build/load/save, search primitives |
| Data export | NumPy-backed access, DataFrame export, Arrow IPC, Parquet |
| Dataset tooling | sequence examples, training examples, corpus manifests, supervision releases |

## Why This Repo Exists

Most structural-bioinformatics tooling still forces at least one bad trade:

- fast but hard to integrate
- Pythonic but slow in the hot path
- useful for one algorithm but not for end-to-end data preparation
- tied to a service or monolithic stack

Ferritin takes a different shape:

- **Rust core for throughput**
- **Python API for ergonomics**
- **CLI binaries for batch jobs**
- **columnar export for downstream analytics/ML**

That makes it useful both as a daily research library and as a compute kernel inside larger pipelines.

## When To Use It

Use Ferritin when you need one or more of:

- high-throughput local processing of many structures
- structure alignment from Python without shelling out to legacy binaries
- direct programmatic access to DSSP, SASA, H-bonds, and geometry features
- preparation/minimization as part of dataset generation
- Arrow/Parquet outputs for DuckDB, polars, pandas, Spark, or ML workflows

Ferritin is probably not the right repo if you want:

- a hosted platform
- a GUI workbench
- a full MD engine
- a batteries-included training framework

## Public Surfaces

**Python**

```python
import ferritin
```

The Python package is the main user-facing surface and exposes the Rust-backed APIs directly.

**CLI**

Release binaries are tested for:

- `tmalign`
- `usalign`
- `ingest`

Example:

```bash
cargo build --release
./target/release/tmalign test-pdbs/1ubq.pdb test-pdbs/1crn.pdb
./target/release/ingest test-pdbs -o features.parquet
```

**Rust crates**

The workspace is split into focused crates:

- `ferritin-align`
- `ferritin-io`
- `ferritin-arrow`
- `ferritin-search`
- `ferritin-connector`
- `ferritin-bin`

## Evidence It Works

Ferritin is not just unit-tested on toy inputs.

- End-to-end validation on **45,100 real PDB structures** completed in **17.9 minutes** on **120 cores** after size filtering from a 50k corpus.
- TM-align behavior is checked against **USAlign** on **4,656 pairs** with **0.003 median TM-score drift**.
- SASA is checked against **Biopython** with **0.17% median deviation** on a 1,000-PDB sample.
- AMBER96 components are checked against **BALL** on a crambin oracle set.
- Large-scale runs already surfaced and fixed multiple real correctness bugs that smaller tests missed.

For more detail, see the validation and roadmap material under [`validation/`](validation/), [`RELIABILITY_ROADMAP.md`](RELIABILITY_ROADMAP.md), and related docs in the repo root.

## Repo Shape

```text
ferritin-align      alignment algorithms
ferritin-io         PDB/mmCIF I/O
ferritin-arrow      Arrow/Parquet export
ferritin-search     search and structural-alphabet tooling
ferritin-connector  PyO3 bridge and compute kernels exposed to Python
packages/ferritin   Python package
ferritin-bin        CLI binaries
examples/           runnable examples
tests/              Python test suite
validation/         benchmarks, reports, and oracle checks
```

## References

Alignment:

- Zhang & Skolnick. "TM-align: a protein structure alignment algorithm based on the TM-score." *Nucleic Acids Research* 33, 2302-2309 (2005). https://doi.org/10.1093/nar/gki524
- Zhang et al. "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." *Nature Methods* 19(9), 1109-1115 (2022). https://doi.org/10.1038/s41592-022-01585-1

Structural analysis:

- Kabsch & Sander. "Dictionary of protein secondary structure." *Biopolymers* 22, 2577-2637 (1983). https://doi.org/10.1002/bip.360221211
- Shrake & Rupley. "Environment and exposure to solvent of protein atoms." *J Mol Biol* 79(2), 351-371 (1973). https://doi.org/10.1016/0022-2836(73)90011-9
- Tien et al. "Maximum allowed solvent accessibilities of residues in proteins." *PLoS ONE* 8(11), e80635 (2013). https://doi.org/10.1371/journal.pone.0080635

Infrastructure:

- Hildebrandt et al. "BALL - Biochemical Algorithms Library 1.3." *BMC Bioinformatics* 11, 531 (2010). https://doi.org/10.1186/1471-2105-11-531
- Schulte. "pdbtbx: A Rust library for reading, editing, and saving crystallographic PDB/mmCIF files." *JOSS* 7(77), 4377 (2022). https://doi.org/10.21105/joss.04377

## License

MIT. See [LICENSE](LICENSE).
