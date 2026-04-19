# proteon

Python bindings to the proteon structural bioinformatics toolkit (Rust core +
pdbtbx I/O). Thin wrapper over `proteon-connector`, opinionated surface for
corpus generation, structure preparation, alignment, SASA/DSSP, hydrogen
placement, MSA/template retrieval, and supervision-tensor export.

## Quickstart: generate a training corpus in 5 lines

```python
import proteon
from pathlib import Path

proteon.build_local_corpus_smoke_release(
    list(Path("my_pdbs/").glob("*.pdb")),
    out_dir="corpus_v0",
    release_id="demo-v0",
    split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
    n_threads=-1,
    overwrite=True,
)
```

That call handles: parse → batch-prepare (hydrogens, minimization) → per-chain
expansion → sequence-supervision export → structure-supervision export
(AF2 contract) → deterministic hash-split → top-level corpus manifest →
validation report. Failures are captured as machine-readable `FailureRecord`
rows, not silently dropped.

Output tree:

```
corpus_v0/
├── corpus/{corpus_release_manifest,validation_report}.json
├── prepared/{prepared_structures.jsonl, supervision_release/...}
├── sequence/{release_manifest.json, examples/{manifest.json,examples.jsonl,tensors.npz}}
└── training/{release_manifest.json, training_examples.jsonl, training.parquet}
```

The training layer is emitted as **Parquet** — one row per example with a
ragged residue axis and no outer padding. Writer is row-group chunked (default
512 examples per group) so peak memory stays bounded no matter how big the
corpus grows. The earlier padded-NPZ training format was removed because
`max_len × n_examples × fields` allocation scaled past practical memory at a
few thousand chains.

A reference artifact (manifests only, large tensors stripped) lives at
`examples/sample_corpus_v0/`.

## Training artifact (Parquet)

The training release is a single `training.parquet` file, one row per
example. Ragged residue axis uses Arrow `list<...>`; per-position fixed
dimensions (atom count, coordinate axes, rigid-group frames) use nested
`FixedSizeList` so readers can reshape losslessly. Any Arrow-compatible
tool — polars, DuckDB, pandas, pyarrow, torch via pyarrow — can read it
without a proteon-specific adapter.

Streaming reader for downstream training loops:

```python
from proteon import iter_training_examples

for ex in iter_training_examples("corpus_v0/training", split="train"):
    positions = ex.structure.all_atom_positions   # (L, 37, 3) numpy
    phi = ex.structure.phi                        # (L,) numpy
    ...
```

`split=` applies predicate pushdown — row-groups whose `split` column
provably doesn't match are skipped by the Parquet reader. No framework
coupling in the public API; wrap it into whatever `Dataset` your trainer
expects.

## Supervision tensors (AF2 contract)

The `structure_supervision.v0` release writes `tensors.npz` with padded
batch-major arrays covering:

| Field | Shape | Purpose |
|-------|-------|---------|
| `aatype` | `(N, L)` | 20-class residue identity |
| `residue_index` | `(N, L)` | author residue numbering |
| `seq_mask` | `(N, L)` | valid-residue mask |
| `all_atom_positions` | `(N, L, 37, 3)` | atom37 coordinates |
| `all_atom_mask` | `(N, L, 37)` | atom37 presence |
| `atom14_gt_positions` | `(N, L, 14, 3)` | atom14 ground truth |
| `atom14_gt_exists`, `atom14_atom_exists`, `atom14_atom_is_ambiguous` | `(N, L, 14)` | atom14 masks |
| `residx_atom14_to_atom37`, `residx_atom37_to_atom14` | `(N, L, ...)` | index maps |
| `pseudo_beta`, `pseudo_beta_mask` | `(N, L, 3)`, `(N, L)` | CB-or-CA pseudo atoms |
| `phi`, `psi`, `omega` + masks | `(N, L)` | backbone torsions |
| `chi_angles`, `chi_mask` | `(N, L, 4)` | sidechain torsions |
| `rigidgroups_gt_frames`, `rigidgroups_gt_exists`, `rigidgroups_group_exists`, `rigidgroups_group_is_ambiguous` | `(N, L, 8, ...)` | 8-group rigid-body frames |

Format is **framework-neutral**: NumPy `.npz` + JSONL metadata. Read with
`np.load(...)` + `json.loads(...)` — no torch/jax dependency. Consumers build
their own `Dataset` / `DataLoader` on top. Loaders in the package
(`load_structure_supervision_examples`, `load_training_examples`) return
dicts of `np.ndarray`.

## Split strategies

`build_local_corpus_smoke_release` accepts three mutually-exclusive modes:

- **Default** (no split args): all records → `train` except the last → `val`.
  Only useful for the smallest smoke paths.
- **`split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}`**: deterministic
  blake2b hash-split on `record_id`. Same record always lands in the same
  split across runs, corpus sizes, and input orderings. Ratios may be
  unnormalized — they're renormalized to 1.0.
- **`split_assignments={"1crn_A": "train", "1ake_A": "val", ...}`**:
  explicit per-record-id mapping. Must cover every (expanded) record_id.

Chosen strategy is recorded in `corpus_release_manifest.json → provenance.split_strategy`.

## Multi-chain handling

Structure/sequence supervision v0 is chain-scoped. The smoke-release helper
expands each loaded structure into one record per chain, so a multi-chain
PDB like `1ake` (chains A + B) becomes two records (`1ake_A`, `1ake_B`)
with separate supervision rows, torsions, and rigidgroup frames. Single-chain
structures pass through unchanged with `record_id` preserved.

## Install

Published package path:

```bash
pip install proteon
```

Local checkout path:

```bash
cd proteon-connector
maturin develop --release
pip install -e ../packages/proteon/
```

## Search DB serving

Persisted search DBs now default to writing both:

- the canonical Parquet corpus files
- the eager compiled serving layout used for repeated low-latency queries

Typical path:

```python
import proteon

db = proteon.build_search_db(["1crn.pdb", "1ubq.pdb"], out="search_db", k=6)
hits = proteon.search(proteon.load("1crn.pdb"), "search_db", rerank=False)
```

If you explicitly want Parquet-only storage, keep it lazy on purpose:

```python
proteon.save_search_db(db, "search_db", write_compiled=False)
lazy = proteon.load_search_db("search_db", prefer_compiled=False)
```

If you're reopening an older Parquet-only DB and want it upgraded in place
without a separate `compile_search_db()` call, use:

```python
db = proteon.load_search_db("search_db", auto_compile_missing=True)
# or:
hits = proteon.search(query, "search_db", auto_compile_missing=True)
```

## Known v0 characteristics

- `rigidgroup_frame_fraction` and `chi_angle_fraction` on raw PDBs run
  ~40–50% because sidechain atoms and many hydrogens are absent until
  `batch_prepare` fills them in. The validation report exposes both
  fractions so downstream trainers can filter on completeness.
- The split primitive is record-id hashing, **not** MMseqs2-clustered
  redundancy removal. Homologues can land in the same *or* different
  splits depending on the id. For leakage-free splits use
  `split_assignments` from an external cluster file.

## Smaller building blocks

If the 5-line helper isn't the right shape, the same pipeline is available
layer-by-layer:

| Function | Layer |
|----------|-------|
| `batch_load_tolerant` / `batch_load_tolerant_with_rescue` | PDB/mmCIF intake |
| `batch_prepare` | hydrogens + FF-aware minimization |
| `build_structure_supervision_dataset_from_prepared` | per-chain AF2-contract tensors |
| `build_sequence_dataset` | per-chain sequence + optional MSA/templates |
| `build_training_release` | join + split assignment |
| `build_corpus_release_manifest` | top-level manifest + failure aggregation |
| `validate_corpus_release` | post-build validation report |

All of these can be called directly with any subset of the pipeline.
