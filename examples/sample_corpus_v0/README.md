# sample_corpus_v0

Reference artifact tree from `build_local_corpus_smoke_release` on 10 PDBs
(33 chain records after multi-chain expansion; zero failures; all three
splits populated).

Built 2026-04-16 with `split_ratios={train: 0.7, val: 0.15, test: 0.15}`.

## Layout

```
sample_corpus_v0/
├── corpus/
│   ├── corpus_release_manifest.json   # top-level
│   └── validation_report.json         # post-build validation (ok=true)
├── prepared/
│   ├── prepared_structures.jsonl
│   └── supervision_release/
│       ├── release_manifest.json
│       ├── failures.jsonl
│       └── examples/
│           ├── manifest.json          # format: proteon.structure_supervision.parquet.v0
│           ├── examples.jsonl         # one row per chain
│           └── tensors.parquet        # streamed Parquet, row-group chunked, zstd-3
├── sequence/
│   ├── release_manifest.json
│   ├── failures.jsonl
│   └── examples/
│       ├── manifest.json              # sequence layer stays NPZ (narrow + small)
│       ├── examples.jsonl
│       └── tensors.npz
└── training/
    ├── release_manifest.json          # format: proteon.training_example.parquet.v0
    ├── training_examples.jsonl        # per-row split + weight metadata
    └── training.parquet               # joined training artifact, streamed
```

## What's stripped from the committed tree

Bulk tensor files are stripped so the tree stays GitHub-browseable:

- `prepared/supervision_release/examples/tensors.parquet` — 5.1 MB
- `sequence/examples/tensors.npz` — 8 KB
- `training/training.parquet` — 5.1 MB

Manifest `tensor_sha256` / `parquet_sha256` entries are retained as a
record of what was produced.

## Regenerate

```bash
python examples/10_corpus_release_smoke.py \
    --out sample_corpus_v0_full \
    test-pdbs/1a*.pdb test-pdbs/1b*.pdb
```

Re-running may not reproduce exact hashes — floating-point ordering and
Parquet compression internals vary. The validator passes on completeness
and ordering invariants, not byte equality.

## Pipeline invariants

- Per-example peak memory is bounded by the Parquet writer's row-group
  size (default 512 examples). Scales to 50K+ chain corpora without
  OOM at the training or supervision layers.
- Supervision and training layers both carry SHA-256 checksums of
  their Parquet artifacts in their respective `manifest.json` files.
- Failures are never silently dropped — `ingestion_failures.jsonl`
  (parse errors) and `failures.jsonl` (per-chain build errors) are
  always present, even when empty.
