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

Ferritin is batch-first. The single-structure helpers are there, but the
default shape is "load many, compute many, prepare many":

```python
import ferritin

paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = ferritin.batch_load(paths, n_threads=-1)

sasa = ferritin.batch_total_sasa(structures, n_threads=-1)
dssp = ferritin.batch_dssp(structures, n_threads=-1)
prep = ferritin.batch_prepare(
    structures,
    hydrogens="backbone",
    minimize=True,
    n_threads=-1,
)
hits = ferritin.tm_align_one_to_many(structures[0], structures[1:], n_threads=-1)

print(sasa)
print(dssp[0][:10])
print(prep[0].final_energy)
print(hits[0].tm_score_chain1, hits[0].rmsd)
```

Runnable examples live in [`examples/`](examples/).

## Persisted Search DBs

For structural-alphabet search, the default persisted path is:

```python
import ferritin

db = ferritin.build_search_db(["1crn.pdb", "1ubq.pdb"], out="search_db", k=6)
hits = ferritin.search(ferritin.load("1crn.pdb"), "search_db", top_k=5, rerank=False)
```

That writes the Parquet corpus and the eager compiled serving layout together,
so later path-based queries load the faster serving representation by default.

If you intentionally want Parquet-only storage, opt in explicitly:

```python
ferritin.save_search_db(db, "search_db", write_compiled=False)
lazy = ferritin.load_search_db("search_db", prefer_compiled=False)
```

If you already have an older Parquet-only DB and want to upgrade it in place on
first use, use `auto_compile_missing=True`:

```python
hits = ferritin.search(
    ferritin.load("1crn.pdb"),
    "search_db",
    rerank=False,
    auto_compile_missing=True,
)
```

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

Short structured Agent Notes exist on selected public boundary functions where
misuse is easy or scaling/cost tradeoffs matter; they are not intended to
annotate the full public API uniformly.

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

Each of those numbers is produced by an **oracle test**: ferritin's output compared against an independent, externally-implemented tool (OpenMM, BALL, MMseqs2, USAlign, Biopython, Gemmi, FreeSASA) at a documented tolerance. Every new numerical claim in the codebase lands with an oracle test next to it. The pattern and the index of current oracles live in [`tests/oracle/README.md`](tests/oracle/README.md); the full philosophy — why oracles over unit tests, how to pick tolerances, what to do when the oracle is also wrong — is in [`devdocs/ORACLE.md`](devdocs/ORACLE.md).

For more detail, see the validation and roadmap material under [`validation/`](validation/), [`devdocs/RELIABILITY_ROADMAP.md`](devdocs/RELIABILITY_ROADMAP.md), and related docs in [`devdocs/`](devdocs/).

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

## Acknowledgements

ferritin is only possible because of the work of the groups whose tools
and papers it builds on. The structure-alignment core is a Rust port of
Yang Zhang's TM-align and US-align. The search layer is a port of Martin
Steinegger and Johannes Söding's MMseqs2, and the experimental structural
alphabet is inspired by Michel van Kempen and Martin Steinegger's Foldseek
(independently trained, no GPL-licensed code re-used). The force-field
implementations follow Martin Karplus, Themis Lazaridis, Peter Kollman,
David Case, and their collaborators (CHARMM19, EEF1, AMBER96, OBC GB).
OpenMM (Peter Eastman and the Pande Lab), BALL (Andreas Hildebrandt and
collaborators), Foldseek, MMseqs2, USAlign, Biopython, Gemmi, and FreeSASA
serve as oracles — ferritin's correctness claims are only as strong as
those reference implementations, and the tolerances in our test suite are
where that dependency is made explicit. I/O rides on Douwe Schulte's
pdbtbx. The citations below point at the original work behind each of
these components — please cite them too when you cite ferritin.

## References

Alignment:

- Zhang & Skolnick. "TM-align: a protein structure alignment algorithm based on the TM-score." *Nucleic Acids Research* 33, 2302-2309 (2005). https://doi.org/10.1093/nar/gki524
- Zhang et al. "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." *Nature Methods* 19(9), 1109-1115 (2022). https://doi.org/10.1038/s41592-022-01585-1

Structural analysis:

- Kabsch & Sander. "Dictionary of protein secondary structure." *Biopolymers* 22, 2577-2637 (1983). https://doi.org/10.1002/bip.360221211
- Shrake & Rupley. "Environment and exposure to solvent of protein atoms." *J Mol Biol* 79(2), 351-371 (1973). https://doi.org/10.1016/0022-2836(73)90011-9
- Tien et al. "Maximum allowed solvent accessibilities of residues in proteins." *PLoS ONE* 8(11), e80635 (2013). https://doi.org/10.1371/journal.pone.0080635

Force fields and implicit solvation:

- Neria, Fischer, & Karplus. "Simulation of activation free energies in molecular systems." *J Chem Phys* 105(5), 1902-1921 (1996). https://doi.org/10.1063/1.472061 — CHARMM19 parameter set used by ferritin.
- Lazaridis & Karplus. "Effective energy function for proteins in solution." *Proteins* 35(2), 133-152 (1999). https://doi.org/10.1002/(SICI)1097-0134(19990501)35:2%3C133::AID-PROT1%3E3.0.CO;2-N — EEF1 implicit solvation.
- Cornell et al. "A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules." *J Am Chem Soc* 117(19), 5179-5197 (1995). https://doi.org/10.1021/ja00124a002 — AMBER94 / AMBER96 parameters.
- Onufriev, Bashford, & Case. "Exploring protein native states and large-scale conformational changes with a modified generalized Born model." *Proteins* 55(2), 383-394 (2004). https://doi.org/10.1002/prot.20033 — OBC Generalized Born implicit solvation.

Sequence and structure search:

- Steinegger & Söding. "MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets." *Nat Biotechnol* 35(11), 1026-1028 (2017). https://doi.org/10.1038/nbt.3988 — k-mer prefilter, ungapped/gapped Smith-Waterman, and PSSM/MSA pipeline that ferritin-search ports.
- van Kempen et al. "Fast and accurate protein structure search with Foldseek." *Nat Biotechnol* 42(2), 243-246 (2024). https://doi.org/10.1038/s41587-023-01773-0 — the 3Di structural-alphabet idea that `ferritin-align/src/search/alphabet.rs` builds on. Ferritin ships an experimental, independently-trained 20-letter structural alphabet (no GPL-licensed code re-used); benchmarks under `validation/bench_foldseek_retrieval.py` are currently ~15% behind Foldseek at TM ≥ 0.5 and close to parity at TM ≥ 0.9.

Infrastructure:

- Hildebrandt et al. "BALL - Biochemical Algorithms Library 1.3." *BMC Bioinformatics* 11, 531 (2010). https://doi.org/10.1186/1471-2105-11-531
- Schulte. "pdbtbx: A Rust library for reading, editing, and saving crystallographic PDB/mmCIF files." *JOSS* 7(77), 4377 (2022). https://doi.org/10.21105/joss.04377

## License

MIT. See [LICENSE](LICENSE).
