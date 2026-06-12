# Proteon Search Roadmap

**Last updated: 2026-04-09**

Proteon does not have a real Foldseek-equivalent search product yet.

What exists today:

- a substantial structural alphabet prototype in [proteon-align/src/search/alphabet.rs](../proteon-align/src/search/alphabet.rs)
- validation scripts that use Foldseek as an oracle / benchmark
- strong downstream geometric refinement capability via TM-align, SOI-align, FlexAlign, and MM-align

What is missing:

- trained structural alphabet weights integrated into the library
- a searchable database format
- fast prefilter / index
- sequence-like alignment over AA + structural alphabet
- public Python and CLI search interfaces
- end-to-end search benchmarks and regression tests

This roadmap focuses on one goal:

**Make proteon-search the fast, reliable structure-search interface for AI-assisted structure bioinformatics, with modern data plumbing and geometry-aware reranking.**

---

## Product Goal

The intended search stack is:

1. encode each structure into a per-residue structural alphabet
2. build a compact searchable corpus
3. run a very fast prefilter over alphabet / hybrid sequence features
4. rerank the top candidates with proteon’s stronger geometric alignment stack
5. expose the entire workflow through Python, CLI, and Arrow/Parquet outputs

Proteon should not try to beat Foldseek everywhere on day one.
It should first become:

- easy to use
- inspectable
- batch-native
- reliable
- tightly integrated with proteon’s existing wrangling and alignment APIs

---

## Current State

> **STATUS UPDATE (2026-06): this section was written against the groundwork stage
> and is now largely obsolete.** The structural search is a **working end-to-end
> product** — most of the "placeholder" list below is done. See the status notes.

### What Is Real

- `proteon-align/src/search/alphabet.rs`: virtual-center construction, nearest
  spatial-neighbor selection, 10D feature extraction, and the **trained** VQ-VAE
  encoder (weights in `validation/alphabet_vqvae_rust.txt` — 9407 structures, 20-state
  codebook, all 20 states used). Encoder is `10→10(ReLU)→10(ReLU)→2→VQ(20)`.
- **Python search product** (`packages/proteon/src/proteon/search.py`): `encode_alphabet`
  / `batch_encode_alphabet` (Rust-backed), `build_search_db` (k-mer postings + Arrow/
  Parquet persistence, versioned), `save_search_db` / `load_search_db`, and `search`
  (prefilter → diagonal voting → diagonal rescore → **TM-align rerank**) returning ranked
  `SearchHit`s with `tm_score`/`rmsd`/`n_aligned`/`seq_identity`.
- **CLI** (`proteon-search build | query | inspect`, `packages/proteon/src/proteon/search_cli.py`).
- Tests: `tests/test_search.py`, `tests/test_search_index.py`, `tests/test_search_cli.py`.
- `validation/train_alphabet.py` (training/distillation) + `validation/bench_foldseek.py`.

### What Is Still Placeholder / Missing

- ~~encoder weights are placeholders~~ → **done** (trained VQ-VAE).
- ~~no public API exposure~~ → **done** (Python `encode_alphabet`/`build_search_db`/`search`).
- ~~no database search implementation~~ → **done** (`search`, end-to-end with rerank).
- ~~no indexing layer~~ → **done** (k-mer postings + positional postings, persisted).
- ~~no CLI~~ → **done** (`proteon-search`).
- **no retrieval metrics in CI** — still missing (Recall@K / nDCG vs a Foldseek baseline
  on a labeled set; roadmap P5 §12–13). **This is now the main open gap.**
- domain/chain splitting, GPU encoding — open, lower priority.

Implication:

**Proteon now has a working structural-search product; the open work is retrieval-quality
benchmarking + reliability gates, not the core pipeline.**

---

## Design Principles

1. Search should be pipeline-native, not a separate silo.
2. The first fast stage should be approximate and cheap.
3. The final ranking stage should use proteon’s geometry strengths.
4. Every stage must be benchmarkable independently.
5. Every scientific approximation needs a retrieval benchmark behind it.
6. Public APIs should expose both convenience and inspectable intermediate artifacts.

---

## Architecture

### Stage A: Encoding

Input:

- backbone atom coordinates per residue
- optional residue identity sequence

Output:

- structural alphabet state per residue
- validity mask
- optional raw 10D features
- optional low-dimensional embedding

Primary artifact:

- `EncodedChain`
  - `id`
  - `aa_sequence`
  - `sa_sequence`
  - `valid_mask`
  - `n_residues`
  - optional metadata

### Stage B: Search Database

Store:

- encoded chains / domains
- metadata table
- prefilter index
- version / parameter metadata

Requirements:

- deterministic rebuilds
- explicit format versioning
- easy partial inspection
- Arrow/Parquet friendly metadata exports

### Stage C: Prefilter

First implementation should be simple:

- k-mer or hashed-shingle prefilter over structural alphabet
- optional hybrid AA + alphabet score
- top-K candidate generation only

Do not start with:

- complicated ANN systems
- distributed infrastructure
- speculative GPU acceleration

### Stage D: Reranking

Use existing proteon strengths:

- alphabet local/global alignment score
- TM-align / SOI-align / FlexAlign reranking
- optional MM-align for complexes later

Returned result object should include:

- prefilter score
- sequence/alphabet alignment score
- TM-score
- RMSD
- aligned length
- coverage

---

## Priority Roadmap

## P0: Make Structural Alphabet Real

### 1. Integrate Trained Weights

Deliverables:

- replace placeholder encoder weights in `alphabet.rs`
- replace placeholder centroids
- add explicit version metadata for the trained model
- make encoder training artifact reproducible from checked-in scripts

Definition of done:

- encoder no longer emits all-invalid placeholder states
- fixed structures produce stable non-trivial alphabet sequences
- training provenance is documented

### 2. Expose Encoding API

Deliverables:

- Rust public API for `encode_structure`
- Python API:
  - `proteon.encode_alphabet(structure)`
  - `proteon.extract_alphabet_features(structure)`
- result object with:
  - `states`
  - `alphabet`
  - `valid_mask`
  - `partners`
  - `features`

Definition of done:

- available in Python and documented
- roundtripable into DataFrame / Arrow form

### 3. Add Golden Encoding Tests

Deliverables:

- fixed corpus of small structures
- expected structural alphabet strings
- stability tests across rebuilds

Definition of done:

- one regression test per golden fixture
- no accidental drift in state assignments

---

## P1: Build the Search Corpus Format

### 4. Define Encoded Corpus Schema

Recommended fields:

- `id`
- `source_path`
- `chain_id`
- `aa_sequence`
- `sa_sequence`
- `valid_mask`
- `n_residues`
- optional taxonomy / domain labels later

Deliverables:

- schema definition
- writer / reader
- format versioning

Definition of done:

- deterministic build from a list of structure files
- loadable without Python-specific code

### 5. Domain / Chain Splitting Policy

Deliverables:

- initial decision: chain-level first
- explicit behavior for:
  - multi-chain structures
  - missing residues
  - non-protein chains
  - short chains

Definition of done:

- documented and tested
- no hidden heuristics

---

## P2: Fast Prefilter

### 6. Implement Alphabet k-mer Index

First version:

- fixed-length k-mers over structural alphabet
- count / overlap based prefilter
- top-K candidate retrieval

Optional second signal:

- hybrid AA + alphabet k-mer score

Definition of done:

- self-hit always retrieves itself near the top
- near neighbors are recalled on a curated benchmark
- index build and query latency are measurable

### 7. Add Search DB Builder

CLI:

- `proteon-search createdb input_dir/ --out db_dir/`

Python:

- `proteon.build_search_db(paths, out=...)`

Definition of done:

- search DB can be built from thousands of structures
- metadata and index are inspectable

---

## P3: Alignment and Reranking

### 8. Add Sequence-Like Reranking

Deliverables:

- local/global alignment over structural alphabet
- optional hybrid AA + alphabet scoring

Definition of done:

- reranking improves top-hit quality over prefilter alone

### 9. Geometric Refinement

Deliverables:

- rerank top-N candidates with TM-align
- optional SOI-align / FlexAlign mode for difficult cases

Definition of done:

- final ranking object includes geometry metrics
- users can choose cheap-only vs refined search

---

## P4: Public Product Surface

### 10. Python Search API

Target API:

- `proteon.encode_alphabet(structure)`
- `proteon.build_search_db(paths, out=...)`
- `proteon.search(query, db, top_k=100, refine_top_n=20)`
- `proteon.search_many(queries, db, ...)`

Result object:

- query id
- hit ids
- prefilter scores
- rerank scores
- alignment metadata

### 11. CLI

Target commands:

- `proteon-search createdb`
- `proteon-search search`
- `proteon-search inspect`
- `proteon-search benchmark`

Definition of done:

- one-line usability for common workflows
- machine-readable output via TSV / JSON / Parquet

---

## P5: Benchmarks and Reliability Gates

### 12. Retrieval Benchmark Suite

Need a pinned benchmark corpus with:

- self-hit checks
- close homologs
- remote structural analogs
- hard negatives

Metrics:

- Recall@K
- MRR
- nDCG
- query latency
- index build time

Comparators:

- Foldseek
- proteon prefilter only
- proteon prefilter + rerank

### 13. Oracle and Drift Tests

Deliverables:

- benchmark script that runs against Foldseek on a pinned subset
- saved baseline metrics
- CI threshold checks for gross retrieval regressions

Definition of done:

- search quality regressions are visible in CI
- model / weight changes cannot silently degrade retrieval

---

## Test Strategy

### Tier 0: Smoke

- API imports
- DB build on tiny corpus
- one query returns hits

### Tier 1: Deterministic Functional

- encoding invariants
- schema roundtrip
- index build/load
- self-hit retrieval
- rerank object shape and score ordering

### Tier 2: Golden

- fixed encoded sequences on known structures
- fixed top-hit rankings on tiny benchmark sets

### Tier 3: Benchmark / Oracle

- compare retrieval quality against Foldseek on curated datasets
- compare prefilter and rerank stages separately

### Tier 4: Performance

- corpus build throughput
- query latency
- memory growth with corpus size

---

## Suggested File / Module Plan

Rust:

- `proteon-align/src/search/alphabet.rs`
- `proteon-align/src/search/index.rs`
- `proteon-align/src/search/prefilter.rs`
- `proteon-align/src/search/rerank.rs`
- `proteon-align/src/search/db.rs`

Connector:

- `proteon-connector/src/py_search.rs`

Python:

- `packages/proteon/src/proteon/search.py`

CLI:

- `proteon-bin/src/bin/search.rs`

Tests:

- `tests/test_search.py`
- `tests/test_search_db.py`
- `tests/test_search_benchmark.py`

---

## Immediate Next Steps

> **All five original "first real search capability" steps are DONE** (trained weights,
> `encode_alphabet`, corpus/DB format, k-mer prefilter + top-K, TM-align rerank), plus a
> `proteon-search` CLI. Proteon has a usable Foldseek-style search.

The remaining work is **reliability + quality**:

1. **Retrieval benchmark suite** (P5 §12) — Recall@K / nDCG / MRR on a labeled set
   (SCOP/CATH families) against a Foldseek baseline, runnable from a script.
2. **Pin a small retrieval gate in CI** (P5 §13) — a tiny fixed corpus + expected top-K so
   regressions in the prefilter/rerank are caught.
3. Then, quality work guided by those metrics (prefilter sensitivity, `k`/weight tuning,
   domain splitting).
