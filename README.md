# Proteon

Rust-first structural bioinformatics toolkit for loading, aligning, analyzing, and preparing macromolecular structures, with an experimental Foldseek-style search stack.

## TL;DR

- Proteon is a **library**, not a platform. No service, no database, no scheduler.
- It gives you **fast structure I/O and heavy compute** from Rust, with Python and CLI entry points.
- Core jobs already wired in: **PDB/mmCIF loading, SASA, DSSP, H-bonds, geometry, and structure preparation/minimization**.
- Proteon also ships an **experimental structural search stack** aimed at becoming its Foldseek-style retrieval layer, but that part of the repo is still pre-product.
- Outputs stay interoperable: **NumPy, Arrow, Parquet, pandas/polars-friendly tables**.
- The repo also contains **dataset/release utilities** for sequence and structure-supervision pipelines.

If you want a Python package that can sit inside your own pipeline and do the expensive structural-biology work without forcing a platform decision, this is what Proteon is for.

## Quick Start

```bash
pip install proteon
```

Proteon is batch-first. The single-structure helpers are there, but the
default shape is "load many, compute many, prepare many":

```python
import proteon

paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = proteon.batch_load(paths, n_threads=-1)

sasa = proteon.batch_total_sasa(structures, n_threads=-1)
dssp = proteon.batch_dssp(structures, n_threads=-1)
prep = proteon.batch_prepare(
    structures,
    hydrogens="backbone",
    minimize=True,
    n_threads=-1,
)
hits = proteon.tm_align_one_to_many(structures[0], structures[1:], n_threads=-1)

print(sasa)
print(dssp[0][:10])
print(prep[0].final_energy)
print(hits[0].tm_score_chain1, hits[0].rmsd)
```

Runnable examples live in [`examples/`](examples/).

## Preparing structures for deep learning

Proteon's data layer is **NumPy + Parquet + JSONL first**. Tensors come back
as `numpy.ndarray`, corpora are written as Parquet, and per-row manifests are
JSONL. Framework-specific integration (PyTorch / PyG / DGL / JAX) lives in
satellite packages — [`proteon-graphein`](https://github.com/theGreatHerrLebert/proteon-graphein),
[`proteon-pyg`](https://github.com/theGreatHerrLebert/proteon-pyg), and
similar siblings — never in this core package. That separation keeps the
data contract framework-agnostic and lets satellites move at their own
dependency cadence.

From a list of PDB paths to a supervision release directory is three calls:

```python
import proteon as p

# 1. Load
structures = [p.load(path) for path in pdb_paths]

# 2. Prepare: reconstruct missing atoms, place hydrogens, minimize.
#    Mutates structures in place; returns one PrepReport per structure.
prep_reports = p.batch_prepare(structures)

# 3. Build the on-disk supervision release: writes
#    examples/tensors.parquet (one row per chain), examples/examples.jsonl
#    (per-row metadata), failures.jsonl (with the 10-class taxonomy), and
#    release_manifest.json.
p.build_structure_supervision_dataset_from_prepared(
    structures, prep_reports,
    out_dir="out/release",
    release_id="v1",
)
```

For the full pipeline including raw-input rescue, ingestion-failure capture,
and a top-level corpus manifest, use the one-shot helper:

```python
p.build_local_corpus_smoke_release(
    [pathlib.Path(p) for p in pdb_paths],
    pathlib.Path("out/release"),
    release_id="v1",
)
```

A worked end-to-end script is at
[`examples/10_corpus_release_smoke.py`](examples/10_corpus_release_smoke.py).

Each `StructureSupervisionExample` carries the AlphaFold-style supervision
tensors — `aatype`, `all_atom_positions` `(L, 37, 3)`, `all_atom_mask`,
`atom14_*`, `pseudo_beta`, `phi`/`psi`/`omega`, `chi_angles`, the eight
`rigidgroups_*` arrays — plus chain metadata and an optional
`StructureQualityMetadata` block tying the example back to its
preparation. The full per-field contract (dtype, shape, mask semantics,
nullability) is documented at
[`devdocs/STRUCTURE_SUPERVISION_SCHEMA.md`](devdocs/STRUCTURE_SUPERVISION_SCHEMA.md)
and is enforced in CI by
[`tests/test_structure_supervision_contract.py`](tests/test_structure_supervision_contract.py).

Reading back into memory:

```python
examples = p.load_structure_supervision_examples("out/release/examples")
# Or stream chunk-by-chunk if the corpus doesn't fit in RAM:
for batch in p.iter_structure_supervision_examples("out/release/examples", batch_size=128):
    ...  # batch is list[StructureSupervisionExample]
```

### Four-layer training corpus (sequence + structure + training + validation)

For DL workflows that need *more than* per-chain structure supervision —
explicitly sequence-side data, MSA features, a joined training table, and
release validation — proteon ships a four-layer release stack:

```python
import proteon as p

# Build a sequence release (with optional MSA wiring via build_sequence_dataset).
seq_release = p.build_sequence_dataset(
    structures,
    out_dir="out/sequence_release",
    release_id="v1",
)

# Build a structure-supervision release (same call as above).
struct_release = p.build_structure_supervision_dataset_from_prepared(
    structures, prep_reports,
    out_dir="out/structure_release",
    release_id="v1",
)

# Join into a training release (one row per example, ragged residue axis,
# embedded sequence + structure tensors, split assignment + crop bounds +
# per-example weight). One zstd-compressed Parquet artifact, SHA-256 pinned.
training_release = p.build_training_release(
    seq_release, struct_release,
    out_dir="out/training_release",
    release_id="v1",
    split_assignments={"chainA": "train", "chainB": "val"},  # or omit for default
)

# Validate the assembled corpus — checks count + split consistency,
# duplicate joined record_ids, and tensor completeness across all four
# layers. Returns a CorpusValidationReport you can serialize alongside the
# release manifest.
report = p.validate_corpus_release("out/corpus_release_dir")
```

`p.build_local_corpus_smoke_release` does all five steps (plus rescue +
ingestion-failure capture + corpus manifest + validation) in one call and
is the recommended starting point for new pipelines.

Reading a training release back, optionally filtered by split:

```python
for batch in p.iter_training_examples("out/training_release", splits=("train",), batch_size=64):
    ...  # batch is list[TrainingExample], each carrying its embedded
         #  SequenceExample and StructureSupervisionExample.
```

The training-side Parquet schema is pinned at
`p.TRAINING_EXPORT_FORMAT` ("proteon.training_example.parquet.v0",
version `p.TRAINING_PARQUET_SCHEMA_VERSION`); the sequence-side schema is
pinned at `p.SEQUENCE_EXPORT_FORMAT`. CI gates both contracts via
[`tests/test_training_corpus_factory_contract.py`](tests/test_training_corpus_factory_contract.py).

### Cluster-aware splits (consumer contract for upstream clusterers)

For leakage-controlled training corpora, proteon consumes externally-produced
cluster assignments (e.g. from `mmseqs cluster` or `foldseek easy-cluster`)
as a typed artifact:

```python
rows = [
    p.ClusterAssignmentRow(
        record_id="chainA",
        cluster_id="cluster-1",
        representative_record_id="chainA",
        is_representative=True,
        cluster_size=2,
    ),
    p.ClusterAssignmentRow(
        record_id="chainB",
        cluster_id="cluster-1",
        representative_record_id="chainA",
        is_representative=False,
        cluster_size=2,
    ),
]

release_dir = p.build_cluster_assignments_release(
    rows,
    out_dir="out/cluster_release",
    release_id="v1",
    tool="mmseqs2",
    tool_version="14.7e284",
    params={"min_seq_id": 0.3, "coverage": 0.8},
    sequence_id_namespace=p.NAMESPACE_PREPARED_RECORD_ID,
)
assignments = p.load_cluster_assignments(release_dir)
```

Proteon **does not own the clustering algorithm** — that lives in upstream
tools where it belongs. What proteon owns is the **typed contract**: rich
provenance (`tool`, `tool_version`, `params`, `input_digest`,
`record_id_digest`, `representative_selection`,
`sequence_id_namespace`, …) so `cluster_id` is auditable rather than just
a string column, plus structural validators that catch duplicate
`record_id` rows, mis-pointed representatives, denormalised `cluster_size`
drift, manifest count inconsistencies, and namespace mismatches before
the assignments reach downstream training-corpus code.

Once you have a `ClusterAssignments`, `cluster_aware_split` produces a
leakage-controlled train / val / test assignment:

```python
result = p.cluster_aware_split(
    assignments,
    record_ids=["chainA", "chainB", "chainC"],
    ratios={"train": 0.8, "val": 0.1, "test": 0.1},
)
# result.assignments: dict[record_id, split]
# result.bounded_skew: True iff actual ratios are within DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE
#                     of requested (informational — assignment is leakage-free either way)
```

All members of one cluster land in the same split (no leakage). When the
caller also passes `grouping_keys` — e.g. to keep sibling chains of the
same parent structure together — union-find merges both constraint sets
so the equivalence classes stack rather than competing. The wrapper is
strict-by-default: partial-coverage clusterings raise
`ClusterCoverageError`, and the unsafe `raw_pdb_id` / `uniprot_id`
namespaces are rejected unless explicitly opted into via
`allow_unsafe_namespaces=True` (chain expansion makes them many-to-one
and breaks the training-example join).

`build_local_corpus_smoke_release` accepts a `cluster_assignments` kwarg
and routes the whole pipeline through `cluster_aware_split`
automatically, surfacing the skew report into the training-release
manifest's provenance for audit.

`validate_corpus_release` accepts an optional `cluster_assignments` (or
`cluster_assignments_path`) kwarg and runs a **cluster-leakage check**
when provided: every cluster's members must land in exactly one split,
the assignments' namespace must match the expected training-join
namespace, and the coverage gap (training records lacking cluster
annotation) is reported. A leakage violation is an error-severity issue
that flips `report.ok=False`; a namespace mismatch is a warning. The
result is recorded on `report.cluster_leakage_check`
(`ClusterLeakageReport` dataclass).

`build_local_corpus_smoke_release` forwards its in-memory
`cluster_assignments` to the validator automatically, so the
`validation_report.json` emitted at the end of a smoke run includes the
leakage check whenever cluster-aware splitting was used.

The cluster-assignments Parquet schema is pinned at
`p.CLUSTER_ASSIGNMENTS_FORMAT` ("proteon.cluster_assignments.parquet.v0",
version `p.CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION`). CI gates the
contract via [`tests/test_cluster_assignments.py`](tests/test_cluster_assignments.py).

## GPU build (optional)

The PyPI wheel is **CPU-only**. Proteon has GPU-accelerated kernels
for CHARMM19+EEF1 / AMBER96 / OBC GB energy + forces, SASA, and the
MMseqs2-style Smith-Waterman / PSSM search, but packaging those into
a PyPI wheel would pin you to a specific CUDA runtime and balloon the
wheel size — so the CUDA path is an opt-in local build instead.

Runtime dispatch is silent-fallback: the CPU-only wheel runs fine on a
GPU box, it just ignores the GPU. To use the GPU, build from source:

```bash
git clone https://github.com/theGreatHerrLebert/proteon.git
cd proteon
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
cd proteon-connector
maturin develop --release --features cuda
cd ..
pip install -e packages/proteon/
```

Requirements: CUDA 12.5 runtime on `$LD_LIBRARY_PATH` (`cudarc` is
pinned to `cuda-12050`), an NVIDIA GPU, an NVCC / driver combination
that supports your card. Confirm the build picked up the GPU with:

```python
import proteon
print(proteon.gpu_available(), proteon.gpu_info())
```

Validated on RTX 5090 via a 50,000-PDB battle test (99.1% correct in
3.5h, CHARMM19+EEF1 minimization + SASA fully on CUDA). There is no
`pip install proteon[cuda]` today; if you hit friction with the
source build, open an issue — we'll prioritize a GPU-wheel variant
(likely published as a separate `proteon-cuda` package) if there's
real demand.

## Persisted Search DBs

Structural-alphabet search is available today, but it is still **experimental**.
The current API is useful for local prototyping and benchmarks, not yet a mature
search product. For the default persisted path:

```python
import proteon

db = proteon.build_search_db(["1crn.pdb", "1ubq.pdb"], out="search_db", k=6)
hits = proteon.search(proteon.load("1crn.pdb"), "search_db", top_k=5, rerank=False)
```

That writes the Parquet corpus and the eager compiled serving layout together,
so later path-based queries load the faster serving representation by default.

If you intentionally want Parquet-only storage, opt in explicitly:

```python
proteon.save_search_db(db, "search_db", write_compiled=False)
lazy = proteon.load_search_db("search_db", prefer_compiled=False)
```

If you already have an older Parquet-only DB and want to upgrade it in place on
first use, use `auto_compile_missing=True`:

```python
hits = proteon.search(
    proteon.load("1crn.pdb"),
    "search_db",
    rerank=False,
    auto_compile_missing=True,
)
```

## What Proteon Covers

| Area | Examples |
|---|---|
| Structure I/O | `load`, `load_pdb`, `load_mmcif`, `save`, tolerant batch loading |
| Structural alignment | `tm_align`, `soi_align`, `flex_align`, `mm_align`, one-to-many and many-to-many variants |
| Structure analysis | SASA, DSSP, backbone dihedrals, H-bonds, contact maps, distance matrices, RMSD/TM-score |
| Preparation | hydrogen placement, minimization, `prepare`, batch preparation |
| Search | experimental structural-alphabet encoding, database build/load/save, search primitives |
| Data export | NumPy-backed access, DataFrame export, Arrow IPC, Parquet |
| Dataset tooling | sequence examples, training examples, corpus manifests, supervision releases |

## Why This Repo Exists

Most structural-bioinformatics tooling still forces at least one bad trade:

- fast but hard to integrate
- Pythonic but slow in the hot path
- useful for one algorithm but not for end-to-end data preparation
- tied to a service or monolithic stack

Proteon takes a different shape:

- **Rust core for throughput**
- **Python API for ergonomics**
- **CLI binaries for batch jobs**
- **columnar export for downstream analytics/ML**

That makes it useful both as a daily research library and as a compute kernel inside larger pipelines.

The sharper thesis is this:

- Proteon's durable reason to exist is as a **trusted structural compute kernel** with strong oracle-backed validation and a clean Python boundary.
- The search stack is an **experimental Foldseek-style effort** built on top of that kernel.
- Search matters, but it should be treated as an area of active product discovery until retrieval quality, indexing, and benchmarks are strong enough to stand on their own.

## When To Use It

Use Proteon when you need one or more of:

- high-throughput local processing of many structures
- structure alignment from Python without shelling out to legacy binaries
- direct programmatic access to DSSP, SASA, H-bonds, and geometry features
- preparation/minimization as part of dataset generation
- Arrow/Parquet outputs for DuckDB, polars, pandas, Spark, or ML workflows

Proteon is probably not the right repo if you want:

- a hosted platform
- a GUI workbench
- a full MD engine
- a batteries-included training framework

## Public Surfaces

**Python**

```python
import proteon
```

The Python package is the main user-facing surface and exposes the Rust-backed APIs directly.
The curated top-level namespace is the default contract for normal use.

Short structured Agent Notes exist on selected public boundary functions where
misuse is easy or scaling/cost tradeoffs matter; they are not intended to
annotate the full public API uniformly.

API tiers:

- Top-level `proteon.*`: curated convenience surface for common loading, alignment, analysis, preparation, and selected search workflows. `proteon.__all__` defines this surface explicitly. Search-related APIs are currently experimental even when exposed from the top level.
- Advanced submodules: use direct submodule imports like `proteon.sequence_export`, `proteon.sequence_release`, `proteon.supervision_export`, `proteon.supervision_dataset`, `proteon.corpus_release`, `proteon.corpus_validation`, or `proteon.msa_backend` when you need format-specific or pipeline-builder control.
- Advanced dataset/release helpers are intentionally not available from the top-level namespace.
- Internal surfaces: underscore-prefixed names and non-exported helpers are not stable API.

Release versioning is also single-sourced now: the repo-root [`VERSION`](VERSION)
file is authoritative, `python tools/set_version.py <x.y.z>` updates the build
metadata, and CI enforces that the Rust workspace and both Python packages stay
in sync.

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

- `proteon-align`
- `proteon-io`
- `proteon-arrow`
- `proteon-search`
- `proteon-connector`
- `proteon-bin`

## Evidence It Works

Proteon is not just unit-tested on toy inputs.

- End-to-end validation on **45,100 real PDB structures** completed in **17.9 minutes** on **120 cores** after size filtering from a 50k corpus.
- TM-align behavior is checked against **USAlign** on **4,656 pairs** with **0.003 median TM-score drift**.
- SASA is checked against **Biopython** with **0.17% median deviation** on a 1,000-PDB sample.
- AMBER96 components are checked against **OpenMM** to **<0.5% on every component at NoCutoff** across a 1,000-PDB benchmark (primary force-field oracle). The v0.2 50k AMBER96 population run is tracked as an active release claim with known coverage and pass-rate gaps rather than a passed floor. **BALL** serves as a secondary cross-check on crambin with wider, component-specific tolerances; BALL has documented convention deviations from the OpenMM-canonical AMBER96 (improper-torsion matching, partial-charge dictionary), so tight BALL parity is not the right bar.
- DSSP secondary-structure assignment is checked against **pydssp** (independent Kabsch-Sander reimplementation) at **≥95% per-residue agreement** on 1crn / 1ubq / 1enh / 1ake / 4hhb.
- Hydrogen placement is checked against **reduce** (Richardson Lab): every geometrically-determined H atom (backbone N-H, Cα, methylene, aromatic C-H, sp3 methine) agrees within **0.1 Å** after optimal matching, across 724 atoms on 1crn + 1ubq in both CHARMM19 polar-only and AMBER96 full-H modes.
- Large-scale runs already surfaced and fixed multiple real correctness bugs that smaller tests missed.

Each of those numbers is produced by an **oracle test**: proteon's output compared against an independent, externally-implemented tool (OpenMM, BALL, MMseqs2, USAlign, Biopython, Gemmi, FreeSASA) at a documented tolerance. Every new numerical claim in the codebase lands with an oracle test next to it.

- [`docs/ORACLE_SETUP.md`](docs/ORACLE_SETUP.md) — reproducibility recipe: pinned versions + install commands + run invocations. Copy-paste your way from a clean machine to the published numbers.
- [`tests/oracle/README.md`](tests/oracle/README.md) — per-test coverage table and the oracle-authoring pattern.
- [`devdocs/ORACLE.md`](devdocs/ORACLE.md) — the tolerance philosophy: why oracles, how to pick tolerances, what to do when the oracle itself is wrong.

For more detail, see the validation and roadmap material under [`validation/`](validation/), [`devdocs/RELIABILITY_ROADMAP.md`](devdocs/RELIABILITY_ROADMAP.md), and related docs in [`devdocs/`](devdocs/).

## Repo Shape

```text
proteon-align      alignment algorithms
proteon-io         PDB/mmCIF I/O
proteon-arrow      Arrow/Parquet export
proteon-search     search and structural-alphabet tooling
proteon-connector  PyO3 bridge and compute kernels exposed to Python
packages/proteon   Python package
proteon-bin        CLI binaries
examples/           runnable examples
tests/              Python test suite
validation/         benchmarks, reports, and oracle checks
```

## Acknowledgements

Derived and paper-inspired components (upstream license preserved in each module):

- `proteon-align` — TM-align and US-align, Zhang group; permissive academic license (any-purpose use with attribution).
- `proteon-search` — MMseqs2, Steinegger & Söding (MIT).
- `proteon-search/src/gpu/pssm_sw*` — libmarv kernel design, Kallenborn et al. (2025); paper-inspired, no source-code lineage.
- `proteon-connector/src/reconstruct.rs` — BiochemicalAlgorithms.jl, Hildebrandt et al. (MIT).
- `proteon-align/src/search/alphabet.rs` — 3Di structural alphabet, van Kempen et al. / Foldseek; paper-inspired, encoder weights independently trained, no GPL-licensed code reused.
- `proteon-electrostatics` — NESSie.jl, Kemmer / Hildebrandt lab (MIT); boundary-element continuum-electrostatics port, gated against NESSie's fixtures.
- `proteon-vina` — AutoDock Vina, Trott / Forli lab, Scripps (Apache-2.0); scoring function + BFGS local optimiser + Monte-Carlo docking. This crate is Apache-2.0; the rest of proteon is MIT.
- `proteon-io` — pdbtbx, Schulte (MIT).

Force-field implementations (CHARMM19, EEF1, AMBER96, OBC generalized Born) are derived from the primary literature cited in §References; parameter files are reproduced from the standard distribution of each force field.

Correctness oracles (used for verification; not linked into proteon): OpenMM, BALL, Foldseek, MMseqs2, US-align, Biopython, Gemmi, FreeSASA. The tolerances in the oracle test suite make the dependency on each explicit.

Primary-literature citations for each component are in §References and should be cited alongside proteon.

## References

Alignment:

- Zhang & Skolnick. "TM-align: a protein structure alignment algorithm based on the TM-score." *Nucleic Acids Research* 33, 2302-2309 (2005). https://doi.org/10.1093/nar/gki524
- Zhang et al. "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." *Nature Methods* 19(9), 1109-1115 (2022). https://doi.org/10.1038/s41592-022-01585-1

Structural analysis:

- Kabsch & Sander. "Dictionary of protein secondary structure." *Biopolymers* 22, 2577-2637 (1983). https://doi.org/10.1002/bip.360221211
- Shrake & Rupley. "Environment and exposure to solvent of protein atoms." *J Mol Biol* 79(2), 351-371 (1973). https://doi.org/10.1016/0022-2836(73)90011-9
- Tien et al. "Maximum allowed solvent accessibilities of residues in proteins." *PLoS ONE* 8(11), e80635 (2013). https://doi.org/10.1371/journal.pone.0080635

Force fields and implicit solvation:

- Neria, Fischer, & Karplus. "Simulation of activation free energies in molecular systems." *J Chem Phys* 105(5), 1902-1921 (1996). https://doi.org/10.1063/1.472061 — CHARMM19 parameter set used by proteon.
- Lazaridis & Karplus. "Effective energy function for proteins in solution." *Proteins* 35(2), 133-152 (1999). https://doi.org/10.1002/(SICI)1097-0134(19990501)35:2%3C133::AID-PROT1%3E3.0.CO;2-N — EEF1 implicit solvation.
- Cornell et al. "A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules." *J Am Chem Soc* 117(19), 5179-5197 (1995). https://doi.org/10.1021/ja00124a002 — AMBER94 / AMBER96 parameters.
- Onufriev, Bashford, & Case. "Exploring protein native states and large-scale conformational changes with a modified generalized Born model." *Proteins* 55(2), 383-394 (2004). https://doi.org/10.1002/prot.20033 — OBC Generalized Born implicit solvation.

Sequence and structure search:

- Steinegger & Söding. "MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets." *Nat Biotechnol* 35(11), 1026-1028 (2017). https://doi.org/10.1038/nbt.3988 — k-mer prefilter, ungapped/gapped Smith-Waterman, and PSSM/MSA pipeline that proteon-search ports.
- Kallenborn, Chacon, Hundt, Sirelkhatim, Didi, Cha, Dallago, Mirdita, Schmidt, Steinegger. "GPU-accelerated homology search with MMseqs2." *Nat Methods* 22, 2024-2027 (2025). https://doi.org/10.1038/s41592-025-02819-8 — libmarv, the canonical GPU Smith-Waterman kernel design that `proteon-search/src/gpu/pssm_sw*.rs` follows (warp-collaborative PSSM SW, shared-mem PSSM staging, padded-DB coalesced target layout).
- van Kempen et al. "Fast and accurate protein structure search with Foldseek." *Nat Biotechnol* 42(2), 243-246 (2024). https://doi.org/10.1038/s41587-023-01773-0 — the 3Di structural-alphabet idea that `proteon-align/src/search/alphabet.rs` builds on. Proteon ships an experimental, independently-trained 20-letter structural alphabet (no GPL-licensed code re-used); benchmarks under `validation/bench_foldseek_retrieval.py` are currently ~15% behind Foldseek at TM ≥ 0.5 and close to parity at TM ≥ 0.9.

Electrostatics:

- Kemmer, Rjasanow & Hildebrandt. "NESSie.jl — Efficient and Intuitive Finite Element and Boundary Element Methods for Nonlocal Protein Electrostatics in the Julia Language." *J Comput Sci* 28, 193-203 (2018). https://doi.org/10.1016/j.jocs.2018.08.008 — the boundary-element local/nonlocal continuum-electrostatics solver that `proteon-electrostatics` ports (MIT) and is gated against.

Docking:

- Trott & Olson. "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." *J Comput Chem* 31(2), 455-461 (2010). https://doi.org/10.1002/jcc.21334 — the Vina scoring function + local optimiser that `proteon-vina` ports (Apache-2.0).
- Eberhardt, Santos-Martins, Tillack & Forli. "AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings." *J Chem Inf Model* 61(8), 3891-3898 (2021). https://doi.org/10.1021/acs.jcim.1c00203 — the v1.2 parity target.

Infrastructure:

- Hildebrandt et al. "BALL - Biochemical Algorithms Library 1.3." *BMC Bioinformatics* 11, 531 (2010). https://doi.org/10.1186/1471-2105-11-531
- Schulte. "pdbtbx: A Rust library for reading, editing, and saving crystallographic PDB/mmCIF files." *JOSS* 7(77), 4377 (2022). https://doi.org/10.21105/joss.04377

## License

MIT. See [LICENSE](LICENSE) for the proteon license text and the
TM-align / US-align attribution notice.

Upstream license notices for derived and paper-inspired components
(MMseqs2, BiochemicalAlgorithms.jl, OpenMM, pdbtbx, and others) are
consolidated in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
