# Proteon Geometric DL Infrastructure Roadmap

**Last updated: 2026-04-13**

This document is the design anchor for steering proteon toward a complete
infrastructure layer for geometric deep learning in protein folding, docking,
search, and corpus construction.

The intent is not to turn proteon into "just another model repo".

The intended role is:

**Proteon should become the reliable, high-throughput data and geometry layer
that prepares, validates, transforms, indexes, and exports protein structure
corpora for downstream geometric DL systems.**

That includes:

- structure preparation
- corpus normalization
- large-scale filtering and provenance
- sequence / structure retrieval
- feature generation for AlphaFold-like and related pipelines
- model-ready shard export

This repo should optimize for:

1. reproducible corpus generation
2. scientifically defensible preprocessing
3. batch throughput at archive scale
4. inspectable intermediate artifacts
5. compatibility with modern geometric DL training stacks

---

## 1. Product Direction

Proteon is already more than a geometry kernel, but it is not yet a complete
geometric-DL prep platform.

The long-term product surface should look like this:

1. ingest raw PDB / mmCIF / AFDB-like records
2. normalize them into a stable internal schema
3. prepare structures robustly and quickly
4. score quality / confidence / drift / completeness
5. cluster and filter for training use
6. build searchable corpora over sequence and structure signals
7. generate model-ready feature + supervision artifacts
8. export versioned releases for training, evaluation, and inference

The repo should not drift toward a pile of disconnected benchmarks,
validation scripts, and partial model experiments.

The north star is:

**A versioned corpus factory and retrieval backend for protein geometric DL.**

---

## 2. What Proteon Already Has

Proteon already has unusually strong building blocks for this direction.

### Core strengths

- tolerant PDB / mmCIF loading
- batch-first Rust kernels with Python bindings
- hydrogen placement and missing-atom reconstruction
- force fields, minimizers, and MD building blocks
- Arrow / Parquet export paths
- broad structure analysis functionality
- strong alignment stack
- an emerging sequence-search and retrieval stack

### Search / retrieval strengths

- MMseqs-style DB I/O
- alphabet and substitution-matrix handling
- k-mer index and prefilter
- similar-k-mer expansion
- reduced alphabet support
- ungapped extension
- gapped alignment
- end-to-end search orchestration
- oracle checks against upstream MMseqs behavior

### Structure-prep evidence already established

From [validation/report/report.html](./validation/report/report.html):

- Proteon CHARMM19+EEF1 minimization preserves fold strongly on the sampled
  corpus:
  median TM-score `0.9945`
- Proteon AMBER96 single-point energy agrees with OpenMM within about `0.2%`
  total-energy error on the oracle case
- Proteon runs the 1000-PDB prep benchmark in `14.9 min` versus `449.7 min`
  for OpenMM at the reported equal-parallelism setting
- Proteon accepts `94.9%` of raw sampled PDBs end-to-end, versus `92.8%`
  for OpenMM and `36.3%` for GROMACS under the compared setups

Interpretation:

**Proteon is already credible as a structure-prep kernel for corpus
construction.**

That matters more for the stated goal than an early model implementation.

---

## 3. What Is Still Missing

The missing pieces are not mainly "more geometry".

They are mostly:

- corpus lineage
- stable artifact schemas
- structure-supervision preprocessing
- explicit training-example contracts
- split / leakage control
- release/version discipline
- robust orchestration of large prep jobs

This repo currently feels closer to:

- a strong compute kernel
- a strong validation effort
- a growing retrieval engine

than to:

- a complete geometric-DL data platform

That gap is what this roadmap addresses.

---

## 4. Reference Models We Should Track

We cloned the canonical reference repos locally as read-only design inputs:

- AlphaFold: [/scratch/TMAlign/alphafold](/scratch/TMAlign/alphafold)
- OpenFold: [/scratch/TMAlign/openfold](/scratch/TMAlign/openfold)

Use them differently:

- AlphaFold is the semantic reference for feature meaning and canonical
  preprocessing choices.
- OpenFold is the implementation reference for training loops, batch objects,
  crop logic, supervision tensors, and PyTorch-oriented data flow.

Key reference files:

### AlphaFold

- [alphafold/model/config.py](/scratch/TMAlign/alphafold/alphafold/model/config.py)
- [alphafold/model/features.py](/scratch/TMAlign/alphafold/alphafold/model/features.py)
- [alphafold/model/all_atom.py](/scratch/TMAlign/alphafold/alphafold/model/all_atom.py)
- [alphafold/data/pipeline.py](/scratch/TMAlign/alphafold/alphafold/data/pipeline.py)
- [alphafold/data/feature_processing.py](/scratch/TMAlign/alphafold/alphafold/data/feature_processing.py)
- [alphafold/data/templates.py](/scratch/TMAlign/alphafold/alphafold/data/templates.py)
- [alphafold/data/mmcif_parsing.py](/scratch/TMAlign/alphafold/alphafold/data/mmcif_parsing.py)

### OpenFold

- [openfold/config.py](/scratch/TMAlign/openfold/openfold/config.py)
- [openfold/data](/scratch/TMAlign/openfold/openfold/data)
- [openfold/model](/scratch/TMAlign/openfold/openfold/model)
- [docs/source/OpenFold_Training_Setup.md](/scratch/TMAlign/openfold/docs/source/OpenFold_Training_Setup.md)

Rule:

Do not cargo-cult these repos wholesale.

Use them to extract:

- feature contracts
- supervision contracts
- crop / sampling logic
- data layout expectations

Then implement proteon-native artifact formats and transforms.

---

## 5. Architecture Target

Proteon should evolve into five layers.

### Layer 1: Raw Intake

Responsibilities:

- ingest PDB, mmCIF, AFDB-like sources
- preserve source metadata
- checksum and identify every raw record
- keep raw artifacts immutable

Outputs:

- `raw_record`
- source manifests

### Layer 2: Canonical Structure Records

Responsibilities:

- normalize parsed records into a stable schema
- preserve chain, residue, atom, and metadata semantics
- record parse warnings and fallbacks

Outputs:

- `normalized_record`
- parse/failure logs

### Layer 3: Prepared Structural Corpus

Responsibilities:

- reconstruct / hydrogenate / minimize structures
- compute quality metrics and drift metrics
- classify and record failure modes

Outputs:

- `prepared_structure`
- `failure_record`
- prep-run manifests

### Layer 4: Retrieval + Corpus Intelligence

Responsibilities:

- cluster and deduplicate
- build sequence and structure search indices
- compute retrieval candidates
- support dataset building and hard-negative mining

Outputs:

- `cluster_assignment`
- `retrieval_index`
- `retrieval_hitset`

### Layer 5: Training Export

Responsibilities:

- build model-ready sequence / structure examples
- apply split policy
- export versioned training shards

Outputs:

- `structure_supervision_example`
- `sequence_example`
- `training_example`
- `release_manifest`

---

## 6. Corpus Artifact Model

Every major stage should emit stable artifacts.

### Core artifact types

- `raw_record`
- `normalized_record`
- `prepared_structure`
- `failure_record`
- `filter_decision`
- `cluster_assignment`
- `split_assignment`
- `retrieval_index`
- `retrieval_hitset`
- `sequence_example`
- `structure_supervision_example`
- `training_example`
- `release_manifest`

### Minimum metadata for every artifact

Every artifact should carry:

- `artifact_id`
- `artifact_type`
- `parent_ids`
- `run_id`
- `status`
- `code_rev`
- `config_rev`
- `created_at`
- `checksum`
- `provenance`

### Minimum metadata for every run

- hardware
- host
- environment / tool versions
- random seed where relevant
- command or config path
- shard id
- input manifest reference
- output manifest reference

This is the foundation for reproducibility.

---

## 7. Failure Taxonomy

Failure modes must be machine-readable and stable over time.

Suggested top-level classes:

- `parse_error`
- `unsupported_chemistry`
- `missing_required_atoms`
- `residue_mapping_error`
- `hydrogen_placement_error`
- `forcefield_parameterization_error`
- `minimization_nonconvergence`
- `numerical_instability`
- `postprep_quality_failure`
- `internal_pipeline_error`

Why this matters:

- dataset quality trends become measurable
- release diffs become explainable
- failure-recovery pipelines become possible
- model-data issues can be traced back to prep stages

---

## 8. AF2 / OpenFold Feature Space: What "Full Mode" Usually Means

MSA coverage alone is not enough.

An AlphaFold-style full pipeline typically has three distinct layers of data.

### A. Raw input features

Typical raw monomer inputs include:

- `aatype`
- `residue_index`
- `msa`
- `deletion_matrix`
- `num_alignments`
- `between_segment_residues`
- optional template features:
  - `template_aatype`
  - `template_all_atom_positions`
  - `template_all_atom_masks`
  - `template_sum_probs`

### B. Derived model-input features

Common derived features include:

- `target_feat`
- `msa_feat`
- `extra_msa`
- `extra_msa_mask`
- `extra_deletion_value`
- `extra_has_deletion`
- `seq_mask`
- `msa_mask`
- `msa_row_mask`
- `template_mask`
- `template_pseudo_beta`
- `template_pseudo_beta_mask`
- template torsion-angle features

### C. Structure supervision labels

Typical structure-side supervision includes:

- `all_atom_positions`
- `all_atom_mask`
- `atom14_gt_positions`
- `atom14_gt_exists`
- `atom14_alt_gt_positions`
- `atom14_alt_gt_exists`
- `atom14_atom_exists`
- `atom14_atom_is_ambiguous`
- `atom37_atom_exists`
- `pseudo_beta`
- `pseudo_beta_mask`
- torsion-angle labels and masks
- rigid-group frames and ambiguity metadata
- quality metadata like resolution / distillation flags

Important distinction:

Proteon does not need to precompute every pair feature that a model uses
internally, but it does need to provide the raw and supervised tensors that
training code expects.

---

## 9. AF2 / OpenFold Gap Analysis Against Proteon

This is the current working assessment, not a final audit.

### Likely already covered or close

- robust structure parsing
- canonical atom coordinates
- hydrogen placement
- missing-atom reconstruction
- minimization / cleanup
- broad structure QC metrics
- search / retrieval infrastructure
- Arrow / Parquet-friendly data export

### Likely still missing or incomplete

- stable atom37 canonical export contract
- atom14 conversion and ambiguity bookkeeping
- pseudo-beta generation as a stable artifact
- torsion-angle supervision tensors
- rigid-group frame supervision tensors
- template feature generation pipeline
- deletion-aware MSA transform outputs
- extra-MSA branch features
- crop logic for training examples
- split/leakage metadata at the release level
- stable training-example schema

### Practical implication

Proteon likely already covers much of the **structure cleanup** problem,
but not yet the full **model-supervision data contract** problem.

That gap is where near-term infra work should concentrate.

---

## 10. Training-Loop Data Contract We Should Aim For

Proteon should not jump directly from raw PDBs to "model batch".

Use three explicit training-side artifact layers.

### `sequence_example`

Purpose:

- sequence-side and alignment-side data

Fields should likely include:

- sequence
- residue index
- chain / segment metadata
- MSA rows
- deletion features
- template hits / template features
- masks

Initial Proteon v0 implementation should stay primitive:

- `record_id`
- `source_id`
- `chain_id`
- `sequence`
- `aatype`
- `residue_index`
- `seq_mask`
- optional `msa`
- optional `deletion_matrix`
- optional `msa_mask`
- optional `template_mask`

And should mirror the structure side with:

- `manifest.json`
- `examples.jsonl`
- `tensors.npz`
- release wrapper with `release_manifest.json` and `failures.jsonl`

### `structure_supervision_example`

Purpose:

- geometry labels and masks

Fields should likely include:

- atom37 coordinates + mask
- atom14 coordinates + masks
- pseudo-beta + mask
- torsion angles + masks
- rigid-group frames
- ambiguity metadata
- quality metadata

### `training_example`

Purpose:

- cropped, batchable, model-ready join of sequence and supervision data

Fields should likely include:

- joined sequence + structure tensors
- crop metadata
- split assignment
- sampling metadata
- curriculum or weighting metadata if used

Rule:

The first two should be stable reusable dataset artifacts.

The third can be model-family-specific.

Proteon v0 should keep `training_example` deliberately thin:

- join by `record_id`
- point to one `sequence_example` and one `structure_supervision_example`
- carry split assignment
- carry optional crop metadata
- carry optional weighting metadata

It should not become a framework-native batch object.

---

## 11. Retrieval's Role in the Larger System

Retrieval is not just a search product here.

It is part of the data engine.

Proteon retrieval should support:

- deduplication / clustering
- nearest-neighbor corpus exploration
- hard-negative mining
- family-aware split construction
- template candidate generation
- retrieval-augmented dataset construction
- eventual docking / interaction candidate discovery

That means the retrieval stack should be designed as a reusable infrastructure
layer, not only a user-facing query API.

---

## 12. Release Discipline

Corpus releases should become first-class repo outputs.

Every release should state:

- source datasets used
- code revision
- config revisions
- prep policy version
- filtering policy version
- clustering policy version
- split policy version
- counts at each stage
- failure breakdown
- quality distributions
- output checksums

Release naming should be explicit, for example:

- `proteon-corpus-v0.1`
- `proteon-corpus-v0.2-afdbmix1`
- `proteon-supervision-v0.1`

Do not rely on ad hoc benchmark outputs as the corpus record of truth.

Proteon v0 should provide a top-level corpus release manifest that links:

- `prepared_structures.jsonl`
- sequence release directory
- structure supervision release directory
- training release directory

and records:

- policy versions
- counts at each stage
- split counts
- failure breakdown
- length summaries

On top of that, Proteon should provide a release validation report that checks:

- count consistency across layers
- split-count consistency
- duplicate joined record ids
- coarse tensor completeness
- failure-distribution summaries

---

## 13. Near-Term Priorities

### Priority 1: Treat structure prep as a versioned dataset product

Concretely:

- define a `prepared_structure` manifest schema
- emit one machine-readable row per processed structure
- record success/failure and drift/QC metrics

### Priority 2: Add structure-supervision export

Concretely:

- atom37 canonical export
- atom14 mapping + ambiguity metadata
- pseudo-beta
- torsions
- rigid-group frame labels

### Priority 3: Build explicit training-example schemas

Concretely:

- `sequence_example`
- `structure_supervision_example`
- `training_example`

### Priority 4: Build dataset lineage into the pipeline

Concretely:

- manifests
- checksums
- parent references
- run metadata
- stable failure taxonomy

### Priority 5: Separate fast tests from heavy oracle runs

Concretely:

- deterministic default CI
- explicit oracle jobs
- explicit large-sample validation jobs

### Priority 6: Make retrieval useful for dataset construction

Concretely:

- stable retrieval artifacts
- cluster assignments
- hard-negative candidate export
- family-aware split helpers

---

## 14. Six-Week Execution Slice

### Week 1-2

- define artifact schemas
- implement `prepared_structure` manifests
- implement `failure_record` logging
- formalize policy/version metadata

### Week 3-4

- export structure supervision tensors:
  atom37, atom14, pseudo-beta, torsions, masks
- add quality reports over prepared outputs
- pin split between fast CI and oracle validation

### Week 5

- define and export `sequence_example` and `structure_supervision_example`
- add a first release manifest and release directory layout

### Week 6

- build first end-to-end reproducible corpus release
- run one baseline training pipeline or training-data smoke path against it
- identify the next true bottleneck:
  data quality, throughput, retrieval quality, or supervision completeness

---

## 15. Concrete Repo Additions We Should Expect

Likely top-level additions over time:

- `GEOMETRIC_DL_INFRA_ROADMAP.md` (this file)
- `DATASET_SCHEMA.md`
- `TRAINING_EXAMPLE_SCHEMA.md`
- `RELEASE_POLICY.md`
- `FAILURE_TAXONOMY.md`

Likely code additions:

- structured manifest writers
- stable supervision exporters
- corpus release builder
- split / clustering helpers
- dataset QC report generation

Likely validation additions:

- release validation suite
- leakage checks
- distribution drift checks
- supervision completeness checks

---

## 16. Non-Goals

At least for now, proteon should not optimize for:

- becoming a giant end-to-end training framework
- duplicating every model architecture in public repos
- chasing inference UX before dataset correctness
- implementing every possible DL feature before artifact contracts are stable

The correct order is:

1. reliable corpus infrastructure
2. stable supervision and feature exports
3. retrieval and split tooling
4. model-facing training integration
5. only then deeper model-specific specialization

---

## 17. Summary

Proteon should be driven toward a single coherent outcome:

**a reproducible, high-throughput, geometry-aware infrastructure layer for
building and serving protein datasets for geometric deep learning.**

The project is already credible on structure prep and increasingly credible on
retrieval.

The next maturity step is not "more isolated algorithms".

It is:

- explicit data contracts
- versioned corpus artifacts
- full structure-supervision preprocessing
- release discipline
- training-ready exports

If we keep the repo pointed at those goals, proteon can become the data and
geometry backbone for folding, docking, retrieval, and related protein DL
systems.
