# Changelog

All notable changes to proteon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The CHANGELOG is the **human narrative**; the machine-readable claims
manifest at `evident/evident.yaml` and per-release bundle at
`evident/reports/<tag>/manifest.json` are the **audit trail**. Each
release tag has a paired EVIDENT bundle pinned by sha256.

## [Unreleased]

## [0.3.0] — 2026-06-17

### GPU k-mer prefilter: from 0.32× to beating CPU, wired into `search()`

The resident GPU prefilter went from *slower-than-CPU-and-degrading* to a real
win, in four steps (benchmarked on an RTX 2070, every step bit-exact vs the CPU
oracle `diagonal_prefilter`):

- **Benchmark (#163)** — an isolated `prefilter_bench` showed the GPU path at
  0.49×→0.32× across 5k→100k targets (degrading with corpus size) because the
  reduction wrote a dense `best[#targets]` array, copied back and host-scanned
  per query.
- **Seq-keyed reduction (#164)** — replaced that dense array with a second
  open-addressing hash table keyed by `seq_id` plus on-device compaction, so
  copyback is O(hits) and nothing scales with target count. N-degradation
  *reversed*: 0.40×→0.87×.
- **Persistent scratch (#165)** — hoisted the per-query device buffers + stream
  into a caller-owned, grow-not-shrink `PrefilterScratch` reused across a batch.
  **GPU now beats CPU at 50k+ targets (1.23× at 100k).**
- **`search()` wiring (#166)** — `SearchEngine::search` is single-query, so it
  caches one scratch and gates the GPU prefilter on a measured target-count
  crossover (`SearchOptions::gpu_prefilter_min_targets`, default 75k), routing
  to CPU below it.

### Reliability: CLI tests, regression corpus, and canonical-tool oracles

- **CLI integration tests (#167)** for `tmalign` / `usalign` (`--outfmt 2`
  tabular contract) and `ingest` (valid Parquet, `--per-structure`, failure
  isolation). Fixed `ingest` to **exit nonzero when *every* input fails** rather
  than silently writing an empty Parquet.
- **Regression corpus (#168)** — added `waters` and `missing-backbone-atom`
  edge-case fixtures (with behaviour probed before asserting).
- **Canonical 8-class DSSP oracle (#169, #170)** — proteon's full H/G/I/E/B/T/S/C
  assignment vs `mkdssp` (DSSP 4.x, via Biopython) / `gmx dssp`, id-aligned,
  93–100% agreement, zero helix↔strand confusion. Vendored multi-chain fixtures
  so it runs on 4hhb in CI.
- **Backbone H-bond oracle (#171)** — `backbone_hbonds` vs `mkdssp`'s
  Kabsch–Sander H-bonds (unordered residue-id pairs + energy agreement);
  93.6–100% precision, 91.1–100% recall. Both oracles container-validated
  against real mkdssp 4.2.2 before shipping.

### EVIDENT: GROMACS AMBER96 fold-preservation oracle (#37)

Re-produces the GROMACS fold-preservation artifact (previously a partial run
dominated by uncategorised `pdb2gmx` errors) and adds it as a release-tier
claim — a third independent C lineage for AMBER96 fold preservation,
alongside the OpenMM-based `fold_preservation_amber`.

- **Runner fix** (`validation/tm_fold_preservation_gromacs.py`): `pdb2gmx`-stage
  failures (missing heavy atoms, incomplete rings, nonstandard / nucleic
  residues) are now classified as principled `skipped` records — out of the
  well-resolved-standard-protein population — rather than `error`. This
  mirrors the OpenMM AMBER arm's skip-on-missing (no `addMissingAtoms`, which
  hangs deterministically, PR #47), so the two arms compare the same
  population. Post-topology failures (grompp / mdrun / NMR-ensemble CA
  mismatch) remain `error`.
- **Re-run** (1000 PDBs): 363 ok / 438 skip / 199 error; GROMACS AMBER96
  median TM 0.9995.
- **New claim** `proteon-amber96-fold-preservation-vs-gromacs-release-1k-pdbs`
  with a #85-style scoring block: joins proteon-AMBER and GROMACS-AMBER per
  PDB (`validation/fold_preservation/join_gromacs_pair.py`). Gated on the
  relative median agreement (`|median(gromacs − proteon)| = 0.0036 < 0.01`)
  and proteon TM > 0.99 on ≥50% (87.3%); the 35.4% coverage is documented,
  not gated (same population-narrowing scoping as the other 50K/1K claims).

### EVIDENT claim integrity — honest rescoping of four release claims (#84, #86, #87, #88)

Follow-up to the tolerance scorer (#85), which made the recorded bands
enforceable and surfaced where claims overstated their evidence. The
underlying science was sound in every case; the claims were rewritten to
match what the evidence actually supports (fix-down, not evidence
regeneration). No tolerance was widened to turn a failing gate green.

- **#84 sasa** — rescoped to **Biopython-only**. FreeSASA diverges ~4.5%
  from proteon at the median (an inter-oracle atom-radius / probe-
  discretisation convention gap, not a proteon error) and the pinned
  artifact never carried FreeSASA evidence, so the cross-tool-parity claim
  was unsupported. FreeSASA divergence is now a documented failure mode.
  `inputs.class` corrected `random-sample` → `convenience-sample` (the
  runner takes the alphabetically-first 1000 files); the ~163 `.cif` files
  Biopython cannot parse are explicitly excluded from the median and
  per-structure pass-rate rather than silently counted as agreeing.
- **#86 dssp** — headline scoped to the **median agreement (0.953 ≥ 0.95)**
  only. The two failing `pass_rate` gates were demoted to documented
  coverage / distribution stats: n_ok/n_attempted (45.3%) is a
  population-narrowing artifact of comparing a gemmi-reencoded mmCIF
  against a raw-PDB parse (the false "identical PDB inputs" assumption is
  corrected), and a median of 0.953 cannot mathematically put 80% of
  structures above 0.95. The pi-helix (`I`) vs PP-helix (`P`) normalisation
  asymmetry is now documented.
- **#87 charmm** — the relative-error prose now discloses the
  `max(|BALL|, 1 kJ/mol)` denominator floor (was `/|BALL|`) and its
  low-magnitude-suppressing effect on improper torsion. The amber-openmm
  50K coverage `pass_rate` was demoted to a documented stat (same pattern).
  Heavy improper-torsion tails (p99 ~200%) remain documented, not gated.
- **#88 fold-preservation** — claim titles reframed as **relative agreement
  between two minimizers** (gated at 0.01 TM-score, not the headline 0.005).
  The AMBER assumption corrected: 10 Å cutoff (not 15 Å) and proteon-vacuum
  vs OpenMM-OBC implicit solvent (not "pure implementation drift"). Stale
  "CHARMM36+OBC2" labels in the AMBER runner
  (`validation/tm_fold_preservation_openmm_amber.py`) fixed to AMBER96+OBC.

All seven release-tier claims now pass their recorded bands honestly under
`score_claim.py --all --release-only`.

### v0.3.0 Phase D — Cluster-leakage check inside `validate_corpus_release`

Extends `corpus_validation.validate_corpus_release` with a new
cluster-leakage check that downstream consumers can use to **prove**
no cluster spans more than one split on a released corpus.

The check is the audit-side complement of Phase C — Phase C *prevents*
leakage at split time; Phase D *detects* leakage at release-validation
time. Together they make leakage-controlled training corpora a
release-tier claim rather than a hope.

### Added

- **`proteon.ClusterLeakageReport` dataclass** with fields
  `cluster_release_id`, `expected_namespace`, `actual_namespace`,
  `namespace_ok`, `no_leakage`, `leaking_clusters` (sample of
  cluster_ids → {split → count}), `coverage_fraction`,
  `cluster_size_summary` (min/max/mean/median), `unavoidable_skew`,
  `actual_ratios`.
- **`CorpusValidationReport.cluster_leakage_check` field** —
  `Optional[ClusterLeakageReport]`, populated when the validator runs
  with cluster info.
- **`validate_corpus_release` accepts two new mutually-exclusive
  kwargs**: `cluster_assignments_path` (loads from a Phase B0 release
  dir, for standalone audits) and `cluster_assignments` (in-memory
  ClusterAssignments, used by `corpus_smoke` when the smoke pipeline
  already has the object in hand). Passing both raises `ValueError`.
- **`expected_cluster_namespace` kwarg** with default
  `"prepared_record_id"` — the canonical training-join namespace.
- **`build_local_corpus_smoke_release` integration** (single-shot +
  chunked paths): when the smoke pipeline runs with a
  `cluster_assignments` kwarg (Phase C wiring), the validator
  invocation automatically forwards that object and the resulting
  `validation_report.json` carries the leakage check. No extra caller
  effort needed.
- **Issue codes recorded** (per the established `ValidationIssue`
  taxonomy at `corpus_validation.py:29`):
    - `cluster_spans_splits` (error) — any cluster spanning > 1 split
    - `cluster_leakage_namespace_mismatch` (warning) — assignments and
      corpus use different ID namespaces; leakage check is run but
      result is not load-bearing
    - `cluster_leakage_load_failed` (error) — Phase B0 release dir
      cannot be loaded
    - `cluster_leakage_bad_in_memory_type` (error) — `cluster_assignments`
      kwarg is not a `ClusterAssignments` instance
    - `cluster_leakage_skipped_no_training_release` (warning) —
      corpus has no training release to audit against
    - `cluster_partial_coverage` (warning) — some training records
      have no cluster annotation; reported via
      `coverage_fraction < 1.0`
- **`ClusterLeakageReport` exported** at top level
  (`proteon.ClusterLeakageReport`). `proteon.__all__` now at 216
  unique entries.
- **New test** `tests/test_cluster_leakage_validation.py` —
  10 assertions across 8 test classes: public-API surface, clean
  leakage-free case (in-memory and from-disk), leakage-detection
  (cluster spans 2 splits flagged as error, `report.ok=False`),
  namespace-mismatch warning (`namespace_ok=False`, check still runs),
  mutual-exclusion of the two kwargs, bad in-memory type, coverage
  gap reporting, backward compatibility (no cluster kwargs →
  `cluster_leakage_check` stays None).

### Notes

- No behavior change to existing validator paths.
  `_check_count_consistency`, `_check_training_release`,
  `_check_structure_tensor_completeness` are unmodified.
- No Parquet schema bumps.
- No version bump (bundles into the v0.3.0 tag at Phase G).

### v0.3.0 Phase C — `cluster_aware_split` leakage-controlled split

Family-aware split helper that consumes the Phase B0 `ClusterAssignments`
artifact and produces train / val / test assignments with the leakage
invariant: **no cluster spans more than one split**. Implemented as a
thin wrapper over the existing
`corpus_smoke._hash_split_assignments(grouping_keys=...)` machinery, not
a new split engine — per the codex review on
`TO_V030_TRAINING_CORPUS_FACTORY.md` (catch #2).

### Added

- **`proteon.cluster_aware_split`** — takes a `ClusterAssignments`, a
  list of record_ids, and optional ratios / seed / grouping_keys;
  returns a `ClusterAwareSplitResult` carrying both the per-record
  split and a skew report. Defaults reflect both codex reviews:
    - `strict_coverage=True` — partial coverage raises
      `ClusterCoverageError` (catch #6 on parent plan)
    - `allow_unsafe_namespaces=False` — rejects `raw_pdb_id` /
      `uniprot_id` namespaces (catch #5 on Phase B0 plan; chain
      expansion makes them many-to-one and breaks the training-example
      join)
    - `skew_tolerance=0.10` via `DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE`
      (catch #9 on parent plan: bounded deviation, not "approximated")
- **`proteon.ClusterAwareSplitResult` dataclass** with fields
  `assignments`, `requested_ratios`, `actual_ratios`, `skew`,
  `max_skew`, `skew_tolerance`, `bounded_skew`. The skew report is
  informational — assignments are leakage-free regardless of the
  reported skew. A dominant cluster will produce
  `bounded_skew=False` so callers can decide whether unavoidable
  drift is acceptable.
- **Composite grouping by union-find**: when both `cluster_id` and
  `grouping_keys` (e.g. sibling-chain parent ID) constrain a record,
  both apply — the equivalence classes stack rather than one winning.
  Per codex Q5 nuance on the parent plan.
- **`build_local_corpus_smoke_release` integration**: new optional
  `cluster_assignments: ClusterAssignments | None = None` kwarg in
  both single-shot and chunked paths. When provided, the smoke
  pipeline routes the split through `cluster_aware_split` and surfaces
  the skew report into the training-release manifest's `provenance`
  block under the `cluster_aware_split` key for audit.
- **3 new top-level exports** (`cluster_aware_split`,
  `ClusterAwareSplitResult`, `DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE`)
  bringing `proteon.__all__` to 215 unique entries.
- **New contract test** `tests/test_cluster_aware_split.py` —
  22 assertions across 9 test classes: public-API surface, leakage
  invariant (no cluster spans splits + singletons split independently
  + all inputs covered), determinism (same/different seeds,
  order-invariant), composite-grouping union-find (sibling-chain
  stacking, cluster-only path, mismatched-length rejection),
  strict-coverage default (partial coverage raises), unsafe-namespace
  rejection (`raw_pdb_id` and `uniprot_id` rejected by default,
  opt-in works), skew report (balanced corpus stays within tolerance,
  dominant cluster trips `bounded_skew=False`), default 80/10/10
  ratios, result dataclass shape.

### Notes

- No edits to `_hash_split_assignments` itself — Phase C wraps it,
  adding seeding via a `f"{seed}:{root}"` prefix on grouping keys so
  identical seeds reproduce splits without touching the existing
  function.
- No Parquet schema bumps. No version bump (bundles into v0.3.0 tag
  at Phase G).

### v0.3.0 Phase B0 — `ClusterAssignments` artifact contract

Keystone phase for the v0.3.0 data-engine layer. The cluster artifact
that Phase C (`cluster_aware_split`), Phase D (cluster-leakage check
inside `validate_corpus_release`), and Phase E (hard-negative mining
with same-cluster exclusion) all consume.

Per `feedback_compute_kernel` and the v0.3.0 plan, proteon owns the
**typed contract, joins, validation, and leakage checks**, but does
NOT own the clustering algorithm. Upstream tools (`mmseqs cluster`,
`foldseek easy-cluster`, …) produce the rows; proteon validates and
joins them.

### Added

- **`proteon.cluster_assignments` module** with three dataclasses
  (`ClusterAssignmentRow`, `ClusterAssignmentsManifest`,
  `ClusterAssignments` — the last is `frozen=True` with eager
  `record_id → cluster_id` and `cluster_id → members` indexes built
  in `__post_init__` via `object.__setattr__`).
- **Rich provenance on the manifest** so `cluster_id` is authoritative
  rather than "just a string column": `tool`, `tool_version`, `params`,
  `input_digest` paired with `input_digest_kind`, `record_id_digest`
  (proteon-side reproducibility), `representative_selection` policy,
  `sequence_id_namespace` with closed-enum validator,
  `custom_namespace_description` (required when namespace == "custom"),
  `created_from_release_id`.
- **Validators** that catch every shape of structural invalidity an
  external clusterer can introduce:
  `validate_cluster_record_id_uniqueness`,
  `validate_cluster_representative_consistency` (now also checks the
  representative row's pointer is consistent across all cluster members),
  `validate_cluster_size_consistency`,
  `validate_manifest_consistency` (counts vs actual rows),
  `validate_cluster_namespace`, `validate_cluster_coverage` (default
  `strict=False` for Phase D's report-style use; Phase C will default
  `strict=True`).
- **Canonical entry point** `build_cluster_assignments_release` runs
  all structural validators before writing any artifact; emits both
  JSONL (human-inspectable) and Parquet (joinable, SHA-256 checksummed)
  plus a `manifest.json` with the full provenance.
- **`load_cluster_assignments` / `iter_cluster_assignments`** for
  whole-corpus and streaming reads; both re-run
  `validate_manifest_consistency` so returned objects are guaranteed
  self-consistent.
- **`compute_record_id_digest`** helper exposed for callers that want
  to compute the canonical sorted-unique digest themselves.
- **30 new top-level exports** in a new `_CLUSTER_ASSIGNMENTS_API`
  tuple covering dataclasses, format/version constants, namespace
  + digest-kind enums, I/O, and validators.
- **README "Cluster-aware splits" subsection** documenting the
  consumer contract and forward-referencing Phases C and D.
- **New contract test** `tests/test_cluster_assignments.py` with 82
  assertions across 10 test classes: public API surface, schema
  source-of-truth sweep (`CLUSTER_ASSIGNMENT_FIELDS` mirrors
  `TENSOR_FIELDS`), release-round-trip with full provenance,
  `__post_init__` manifest validators, structural validators
  (including the codex-flagged duplicate-`record_id` and
  representative-pointer-drift cases), manifest consistency,
  namespace and coverage checks (including strict-mode
  `ClusterCoverageError`), frozen-container behavior + O(1) lookups,
  edge cases (empty / all-singletons / single-cluster / unicode IDs),
  and the streaming `iter_cluster_assignments` loader.

### Notes

- No behavior changes to existing modules. New module, additive
  exports only.
- No Parquet schema bumps to existing schemas. The new
  `CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION` starts at 1.
- No version bump (bundles into the v0.3.0 tag at Phase G per the
  parent plan `TO_V030_TRAINING_CORPUS_FACTORY.md`).

### v0.3.0 Phase A — Training-corpus factory surface

Same shape as v0.2.1: plumbing release that lifts already-shipped Layer 5
release-builder symbols into `proteon.__all__`. No behavior change, no
Parquet schema bump. This is Phase A of the multi-PR v0.3.0 milestone
(`TO_V030_TRAINING_CORPUS_FACTORY.md`); Phases B0 (`ClusterAssignments`
contract), C (`cluster_aware_split`), D (cluster-leakage validation
extension), E (hard-negative mining), and G (release tag) follow.

### Added

- **Sequence-export surface promoted to top-level**: `SequenceParquetWriter`,
  `export_sequence_examples`, `iter_sequence_examples`,
  `load_sequence_examples`, `SequenceReleaseManifest`,
  `build_sequence_release`, `build_sequence_dataset`,
  `SEQUENCE_EXPORT_FORMAT`, `SEQUENCE_PARQUET_SCHEMA_VERSION`. The
  MSA-wired `build_sequence_dataset` is now the canonical entry point for
  sequence-release construction (a previous test asserting the opposite
  has been replaced — see `tests/test_public_api_surface.py:33`).
- **Training-example surface promoted to top-level**: `TrainingExample`,
  `TrainingReleaseManifest`, `build_training_release`,
  `join_training_examples`, `iter_training_examples`,
  `load_training_examples`, `TRAINING_EXPORT_FORMAT`,
  `TRAINING_PARQUET_SCHEMA_VERSION`. `TrainingExample` is deliberately a
  thin join over (`SequenceExample`, `StructureSupervisionExample`) plus
  split/crop/weight metadata; crop / curriculum logic stays in model code
  per `devdocs/GEOMETRIC_DL_INFRA_ROADMAP.md` §10.
- **Release validation surface promoted to top-level**:
  `CorpusValidationReport`, `ValidationIssue`, `validate_corpus_release`.
  The validator was already shipping in v0.2.0+ and being called by
  `build_local_corpus_smoke_release`, but wasn't reachable as
  `proteon.validate_corpus_release` until now.
- **New contract test** `tests/test_training_corpus_factory_contract.py`
  pinning the new exports, the schema-version constants, and the
  `SequenceExample` Parquet round-trip surface. Documents one pre-existing
  dataclass-vs-schema drift (`msa_profile` is on `SequenceExample` but
  not serialized in the Parquet schema — closing requires schema v2,
  deferred per `reference_supervision_parquet_reader_not_version_aware`).
- **README four-layer DL section** documenting the
  `sequence_release → structure_release → training_release →
  validate_corpus_release` flow with worked snippets.

### Notes

- No version bump; this lands under `[Unreleased]` and rolls up into the
  v0.3.0 tag when all phases are complete. Codex review on the v0.3.0
  plan explicitly OK'd bundling: "if v0.3.0 is near, bundle; there is
  no technical dependency."
- No Parquet schema bumps. All advertised schemas stay at v1. Memory
  `reference_supervision_parquet_reader_not_version_aware` still applies
  to any future v2 work.

## [0.2.1] — 2026-05-24

Release theme: make the existing NumPy + Parquet-first DL-prep surface
discoverable from `proteon.*`. Pure plumbing release — no behavior
changes, no schema bumps, no force-field or geometry changes.

### Added

- **DL-prep surface promoted to the top-level `proteon` namespace**: the
  release-builder API (`build_structure_supervision_dataset` and
  `build_structure_supervision_dataset_from_prepared` — the canonical
  "structures + reports → release" path), the Parquet export layer
  (`SupervisionParquetWriter`, `export_structure_supervision_examples`,
  `iter_structure_supervision_examples`,
  `load_structure_supervision_examples`,
  `SUPERVISION_EXPORT_FORMAT`, `SUPERVISION_PARQUET_SCHEMA_VERSION`),
  the manifest types (`PreparedStructureRecord`, `FailureRecord`,
  `StructureSupervisionReleaseManifest`, `CorpusReleaseManifest`) and
  their build / load functions, and the one-shot
  `build_local_corpus_smoke_release`. All of these were already shipped
  inside submodules in v0.2.0; v0.2.1 just lifts them so
  `import proteon as p; p.build_structure_supervision_dataset_from_prepared(...)`
  works without knowing the submodule layout.
- **"Preparing structures for deep learning" section in `README.md`**
  documenting the three-call canonical flow and pointing at
  `examples/10_corpus_release_smoke.py` and
  `devdocs/STRUCTURE_SUPERVISION_SCHEMA.md`.
- **Exhaustive contract test**
  (`tests/test_structure_supervision_contract.py`): iterates every entry
  in `supervision_export.TENSOR_FIELDS` and asserts dtype, shape, and
  bit-equal Parquet round-trip on both a synthetic and a real
  (`test-pdbs/1crn.pdb`) example. 60 assertions; runs in <1 s in the
  default tier. Catches schema-vs-dataclass drift on the next CI run
  for new fields.

### Notes

- No change to `batch_prepare`'s signature; it still returns
  `List[PrepReport]` and mutates structures in place. The README and the
  top-level docstring now state this explicitly.
- Parquet schema is unchanged at v1; rigid-group frames have been in v1
  since v0.2.0. Any future v1 → v2 bump needs a dedicated
  version-aware-reader migration PR (the current reader inspects
  `format` but assumes columns exist).

## [0.2.0] — 2026-05-11

Release theme: comprehensive 50K-scale release-tier oracles + EVIDENT
v0.2.0 data-mount contract (image bundles tools, user mounts data via
`--bind`). Three new release-tier 50K claim locks (AMBER96-SP, DSSP
8-class, CHARMM fold-preservation) and one major DSSP algorithm fix
(T-classification interior-only) that moved DSSP-vs-mkdssp median
agreement from 0.8582 → 0.9527.

AMBER fold-preservation 50K is deferred to v0.2.1. SASA 50K is
deferred (runner refactor needed). GROMACS fold-pres regen (#37)
remains deferred.

### Changed

- **Lock CHARMM fold-preservation 50K claim with first-run headline**
  (`evident/claims/fold_preservation_charmm_50k.yaml`). Headline from
  the 2026-05-11 monster3 run (proteon-side finished 2026-05-08,
  OpenMM-side 2026-05-11 02:17 local): n_attempted = 47 183.
  proteon-side: n_ok = 44 464 (94.2%), median TM = 1.0000.
  OpenMM-side: n_ok = 8 029 (17.0%), median TM = 0.9990. Paired
  records = 7 791. **median tm_diff (openmm − proteon) = −0.0006**
  (passes the `<0.01` drift band by ~17× and the title's tighter
  0.001 by ~1.7×). |tm_diff| < 0.005 covers 85.0% of paired records.
  All three claim tolerances pass on the proteon-side. The
  OpenMM-side pass_rate floor (n_ok/n_attempted ≥ 0.85) does NOT
  clear — same population-narrowing pattern as the AMBER96-SP 50K
  claim (ff14SB-era typer rejects residues with non-canonical
  atom names after PDBFixer prep). Tolerance values left unchanged
  pending a recalibration decision; the failure_modes section
  documents the actual breakdown.


- **Lock AMBER96 single-point 50K claim with first-run data**
  (`evident/claims/forcefield_amber_openmm_50k.yaml`). Headline from
  the 2026-05-08 run (post-histidine PR #60 + NoCutoff PR #59):
  n_attempted=47 183, n_ok=7 775, **median rel_diff = 3.41e-04**
  (passes the `<0.01` median band by ~30×). Median |diff| =
  20.3 kJ/mol. p95 rel_diff = 0.856 — long tail driven by residues
  OpenMM's ff96 typer rejects after PDBFixer prep. Pass-rate
  (n_ok/n_attempted = 16.5%) sits well below the claim's 0.85
  floor; the gap is dominated by population-coverage limits
  (44% missing-heavy-atom skips + 40% ff96 typer fails), not by
  algorithm quality. Title now reflects both outcomes honestly.
  Tolerance values left unchanged pending an explicit recalibration
  decision; failure_modes section documents the actual breakdown.

- **Lock DSSP 8-class 50K claim with first-run headline (post-T-fix)**
  (`evident/claims/dssp_8class_50k.yaml`). Headline from the
  2026-05-10 run with the T-interior fix (PR #73) and PP-helix
  post-process (PR #71): n_attempted=47 183, n_ok=21 370,
  **median agreement_rate = 0.9527** (passes the ≥0.95 median band).
  ≥0.95 = 54.8% / ≥0.90 = 94.6% / ≥0.85 = 97.9% / ≥0.80 = 98.4%.
  Median moved +9.45 pp from the pre-T-fix 0.8582. Pass-rate floors
  fail at population scale (n_ok/n_attempted = 45.3% vs 0.85;
  ≥0.95/n_ok = 54.8% vs 0.80) — driven by length-mismatch skips
  and residual T/S boundary + π-helix disagreement. Tolerance
  values left unchanged pending recalibration; failure_modes
  section documents the actual breakdown and the per-class
  composition deltas.

### Fixed

- **DSSP: stop classifying turn endpoints as T (turn)**
  (`proteon-connector/src/dssp.rs`). proteon's T-classification rule
  treated *any* non-space turn marker as "in turn", which over-counts
  by including the H-bond donor (`>`/`X`) and acceptor (`<`/`X`)
  positions. Canonical DSSP only classifies the *interior* residues
  of an n-turn (markers `3`, `4`, `5`) as T; donor/acceptor positions
  fall through to S (bend) or `-` (loop) unless separately covered
  by another turn. The 50K mkdssp oracle made the over-emission
  visible: proteon emitted T 88.76% more often than mkdssp, and
  66% of all per-residue diffs (1.8M residues) traced to spurious
  T's at boundary positions adjacent to helices and strands. With
  this fix, T-classification matches the canonical "interior only"
  rule. No existing tests broke; impact on the median agreement_rate
  vs mkdssp at 50K scale will be measured in the next oracle run.

- **DSSP runner: normalise mkdssp's PP-helix class to loop in comparison**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 emits a "P" (PP-helix)
  class via its default `--min-pp-stretch=3`, but proteon's DSSP doesn't
  emit P at all — every PP-helix residue therefore counted as a
  per-residue mismatch even when both implementations agreed on the
  underlying H-bond geometry. Map `P → '-'` on the mkdssp side at
  comparison time; the raw `mkdssp_ss` field preserves the unmodified
  P chars per record for forensic reanalysis. Both arms now compare
  on the same 8-class alphabet `{H,G,I,E,B,T,S,-}`. Surfaced by the
  first 50K run (median agreement_rate 0.8447 / only 1.0% structures
  clearing 0.95) — the systematic gap accounts for the bulk of the
  per-residue diff. Runtime disabling via `--min-pp-stretch=N` is
  not viable — mkdssp v4.6.1 / libcifpp 10.0.3 has a CLI bug where
  any non-default value triggers `terminate called without an
  active exception` (SIGABRT) on every PDB. Runner also now
  persists the full proteon/mkdssp SS strings per record (~10 MB
  at 50K) so future reclassifications don't need the 80-min compute.
- **DSSP runner: pass `--output-format dssp` explicitly to mkdssp**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 picks output format
  based on the output filename's extension; we pipe through
  `/dev/stdout` (no extension) so it defaulted to mmCIF, which the
  runner's parser doesn't handle — resulting in `mkdssp_n=0` for every
  PDB and 100% length-mismatch skips. One-line fix: pass the format
  explicitly. Surfaced when the gemmi-bridge run produced records but
  every record was a skip with empty `mkdssp_composition`.
- **DSSP runner: bridge through gemmi for mkdssp compatibility**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 + libcifpp's strict
  validator rejects many PDB-derived datablocks at parse time
  ("Duplicate Key violation" on the internal `_refine` table when a
  PDB has multiple `REMARK 3` records). The runner now pre-converts
  PDB → mmCIF via `gemmi.read_pdb().make_mmcif_document().write_file()`
  before invoking mkdssp on the mmCIF. Pipeline verified on 5 PDBs
  that previously failed (12e8, 8rqw, 7vsc, 1aaj, 7nz7) — all now
  produce valid DSSP output. Sub-second per-PDB conversion overhead.
- **EVIDENT image: enable CCD download in libcifpp build**
  (`evident/Dockerfile{,.cuda}`). Both Dockerfiles previously set
  `-DCIFPP_DOWNLOAD_CCD=OFF` based on an outdated assumption that PDB
  input didn't need the Chemical Component Dictionary. mkdssp v4 in
  fact requires the CCD at runtime for every per-residue lookup ("ALA",
  "GLY", ...) regardless of input format; without it, every PDB fails
  with `compound information not found in /usr/local/share/libcifpp`.
  Surfaced when the v0.2.0 50K DSSP-vs-mkdssp run failed on every
  attempted record. Build-time download adds ~70 MB to the image
  (components.cif.gz); the runtime cost is one mmap per process.

### Added

- **DSSP 8-class vs mkdssp release-tier 50K claim** — completes the v0.2.0
  trust pyramid's DSSP rung. mkdssp v4.6.1 is the wider biology
  community's canonical Kabsch-Sander 1983 reference; proteon ports the
  same algorithm. Per-residue agreement at 50K corpus scale closes
  the gap left by the existing CI (per-PDB fixture) and the existing
  release-tier (proteon-vs-pydssp 1k) claims.
  - `evident/claims/dssp_8class_50k.{yaml,md}` — claim wired with
    `last_verified: null` pending the first monster3 artifact.
  - `validation/dssp_mkdssp_oracle.py` — new runner. Calls
    `proteon.dssp` on one side, `mkdssp` via subprocess on the other,
    parses mkdssp's per-residue 8-class column, computes per-PDB
    agreement rate. Loop characters canonicalised to '-' on both
    sides so the space-vs-character convention isn't a confounder.
    pebble per-task isolation, resume + PROTEON_PDB_LIST support,
    v0.2.0 bind-mount contract — same shape as the AMBER96 and CHARMM
    oracles.
  - `evident/evident.yaml` includes list extended with the new claim.
- **Per-structure batch primitives for SASA, energy, and H-bond counts.**
  Adds `proteon.batch_atom_sasa`, `batch_residue_sasa`, `batch_relative_sasa`,
  `batch_hbond_count`, and `batch_compute_energy` — rayon-parallel wrappers
  that produce one result per input structure. Each new function is
  parity-tested against a Python loop of the corresponding per-structure
  call (exact float equality; same dict shape for energy components and
  topology counts). Unblocks proteon-graphein's batch entry point — see
  proteon-graphein/CHANGELOG once that ships.
  - `batch_atom_sasa`: existing Rust function, previously unexposed in
    Python (`py_sasa.rs::batch_atom_sasa` was registered but never wrapped);
    now surfaced through `proteon.sasa`.
  - `batch_residue_sasa` and `batch_relative_sasa`: new Rust functions in
    `py_sasa.rs`. Pre-extract a `StructureView` (coords + radii + per-residue
    atom counts + residue names) on the main thread so the rayon body
    holds only owned data.
  - `batch_hbond_count`: new Rust function in `py_hbond.rs`. Mirrors the
    existing `batch_backbone_hbonds` data-flow but reduces to per-residue
    counts (matches the single-call `hbond_count_per_residue` semantics).
  - `batch_compute_energy`: new Rust function in `py_forcefield.rs`. Routes
    the same FF aliases as `compute_energy` (`charmm19_eef1`, `amber96`,
    `amber96_obc`) and forwards `nbl_threshold` / `nonbonded_cutoff` per
    structure. Uses the same chunk-clone pattern as `batch_minimize_hydrogens`
    to bound peak memory.
- **HID/HIE/HIP histidine protonation-state variants in AMBER96 data**
  (#60, PR 1 of 3). proteon's AMBER96 oracle previously showed 7-12%
  rel_diff vs OpenMM AMBER96 on every PDB containing histidines
  because only one HIS template shipped (with HIP-like charges by
  accident). This PR adds the three canonical AMBER96 templates as
  separate residue names — mirroring the CHARMM19 HSD precedent
  rather than extending the position-only variant suffix system:
  - `proteon-connector/data/amber96.ini` gains 9 new charge sections
    (HID/HIE/HIP × mid-chain / -N / -C terminal variants), 165 atom
    entries total. Charges and atom-class names extracted directly
    from OpenMM's vendored `amber96.xml` (the v0.2.0 oracle's parity
    target) via the new `scripts/extract_amber96_histidine_charges.py`
    script — hand-authoring 165 partial charges would invite
    sub-percent drift no reviewer can catch.
  - `proteon-connector/data/fragments/HID.json`, `HIE.json`, `HIP.json`
    — fragment templates mirroring `HIS.json`, with each variant's
    `delete` list extended to drop the H atom that doesn't belong in
    that tautomer (HID drops Hε2; HIE drops Hδ1; HIP keeps both).
  - `proteon-connector/src/fragment_templates.rs` — 3 new entries in
    the static `TEMPLATES` array.
  Existing structures with residue name "HIS" continue to load and
  energy-compute identically — no behaviour change on the existing
  HIS path. Structures providing residue name HID/HIE/HIP get the
  correct per-tautomer AMBER96 charges. The detection logic that
  decides which name to apply (based on which Hδ1 / Hε2 atoms are
  present in the PDB) lands in PR 2.

### Changed

- `validation/amber96_oracle.py` now calls
  `proteon.normalize_histidine_tautomers` between PDBFixer-write and
  `proteon.load`, so the runner's proteon arm sees HID/HIE/HIP residue
  names (matching what OpenMM AMBER96 detects internally on the same
  topology). Resolves the 7-12% rel_diff on every histidine-containing
  structure that surfaced in v0.2.0 contract smoke (#60).
  Local end-to-end on 1ubq (1 HIS, HD1-only): rel_diff 0.026% (down from
  ~10% pre-fix). The runner's per-record JSONL now also carries a
  `histidine_tautomer_counts` field for observability.
- `proteon` package exports `normalize_histidine_tautomers` at the top
  level — `proteon.normalize_histidine_tautomers(in, out)` is the public
  call site.

### Added

- **`proteon.prepare.normalize_histidine_tautomers(in_path, out_path)`**
  (#60, PR 2 of 3). Reads a PDB, walks each HIS residue, inspects which
  of Hδ1 / Hε2 are present, and writes a copy with the residue name
  updated to HID / HIE / HIP. Idempotent on already-renamed inputs.
  Returns a per-residue tally so the caller can log / observe the
  classification. The Rust loader is unchanged — proteon's existing
  residue-name-driven dispatch picks up the correct AMBER96 charges
  via the data added in PR #62.
- **`UserWarning` once per process when AMBER96's `compute_energy` sees
  any residue still named "HIS"** — flags the systematic ~7-12% drift
  this issue caused, and points the user at
  `proteon.prepare.normalize_histidine_tautomers`.
- **`tests/test_amber_invariants.py::TestHistidineTautomers`** — 6 tests
  pinning the renaming behaviour: HD1-only→HID, HE2-only→HIE,
  both→HIP, neither→stays HIS, idempotent on second call, and an
  end-to-end check that compute_energy with renamed residues hits the
  HID template's electrostatic charges (i.e. PR #62's data is wired
  into the typer correctly).

- **HID/HIE/HIP histidine protonation-state variants in AMBER96 data**
  (#60, PR 1 of 3). proteon's AMBER96 oracle previously showed 7-12%
  rel_diff vs OpenMM AMBER96 on every PDB containing histidines
  because only one HIS template shipped (with HIP-like charges by
  accident). This PR adds the three canonical AMBER96 templates as
  separate residue names — mirroring the CHARMM19 HSD precedent
  rather than extending the position-only variant suffix system:
  - `proteon-connector/data/amber96.ini` gains 9 new charge sections
    (HID/HIE/HIP × mid-chain / -N / -C terminal variants), 165 atom
    entries total. Charges and atom-class names extracted directly
    from OpenMM's vendored `amber96.xml` (the v0.2.0 oracle's parity
    target) via the new `scripts/extract_amber96_histidine_charges.py`
    script — hand-authoring 165 partial charges would invite
    sub-percent drift no reviewer can catch.
  - `proteon-connector/data/fragments/HID.json`, `HIE.json`, `HIP.json`
    — fragment templates mirroring `HIS.json`, with each variant's
    `delete` list extended to drop the H atom that doesn't belong in
    that tautomer (HID drops Hε2; HIE drops Hδ1; HIP keeps both).
  - `proteon-connector/src/fragment_templates.rs` — 3 new entries in
    the static `TEMPLATES` array.
  Existing structures with residue name "HIS" continue to load and
  energy-compute identically — no behaviour change on the existing
  HIS path. Structures providing residue name HID/HIE/HIP get the
  correct per-tautomer AMBER96 charges. The detection logic that
  decides which name to apply (based on which Hδ1 / Hε2 atoms are
  present in the PDB) lands in PR 2.

### Changed

- **Fold-preservation runners now support resume + `PROTEON_PDB_LIST`**.
  All four runners (proteon CHARMM, proteon AMBER, OpenMM CHARMM,
  OpenMM AMBER) previously opened OUT in `"w"` mode and re-walked the
  full directory glob on every invocation. At 50K scale this means a
  single SIGTERM / OOM / NFS hiccup during the multi-day run dropped
  every record, so the entire run had to restart from zero.
  Mirrors the pattern already shipped in `charmm19_eef1_ball_oracle.py`
  (#44):
  - On startup: read existing OUT, build `done_names` set of PDBs
    already in the JSONL.
  - Compute `pending = sample − done_names`, skip what's already done.
  - Open OUT in `"a"` mode so the resume seed is preserved.
  - `PROTEON_PDB_LIST=path/to/list.txt` overrides the directory glob,
    so 50K runs can use the same `validation/protein_only_50k.txt`
    pre-filter the CHARMM 50K oracle does (drops ~6K non-protein
    structures the FF can't parametrise).
  Surfaced when the first 50K fold-pres CHARMM run on monster3 hit
  proteon-side rate 0.32/s → ~40 hr per side, so the full chain is
  ~3 days. At that scale, no resume is unacceptable.

### Added

- **Three new 50K release-tier claims** (covering the v0.2.0 #42
  vision — full-corpus capture for every release-tier capability where
  monster3 compute permits):
  - `proteon-amber96-vs-openmm-corpus-50k-pdbs`
    (`evident/claims/forcefield_amber_openmm_50k.{yaml,md}`) —
    single-point AMBER96 energy at full corpus scale. Cross-implementation
    oracle counterpart to the fold-preservation AMBER claim, sensitive
    to a different bug class (per-term coefficient mismatches that don't
    surface in TM-score after minimization).
  - `proteon-charmm19-fold-preservation-vs-openmm-release-50k-pdbs`
    (`evident/claims/fold_preservation_charmm_50k.{yaml,md}`) — proteon
    CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2 minimization at 50K. Confirms
    the 1k headline (median tm_diff +0.0040) holds at population scale.
  - `proteon-amber96-fold-preservation-vs-openmm-release-50k-pdbs`
    (`evident/claims/fold_preservation_amber_50k.{yaml,md}`) — proteon
    AMBER96 vs OpenMM AMBER96 at 50K. The cleanest cross-implementation
    oracle proteon ships, since both arms claim the same parameter set.
  Each claim's `last_verified` block is null pending the first monster3
  run; tolerances and assumptions are wired in so the manifest validates
  immediately and the run only has to fill in the headline numbers.
- `evident/evident.yaml` includes list extended with the three new
  claim files.
- **SASA release-tier claim now triangulates against TWO oracles** —
  Biopython AND FreeSASA. `validation/run_validation.py::test_sasa`
  computes a FreeSASA Shrake-Rupley result alongside Biopython's, records
  `freesasa_total` / `freesasa_time_ms` / `freesasa_relative_diff` per
  structure, and downgrades to `warn` if EITHER oracle disagrees by >5%.
  Two independent C lineages agreeing on Shrake-Rupley is much stronger
  evidence than against either alone, per the
  `feedback_triangulate_with_two_oracles` principle (a shared bug in
  proteon vs Biopython couldn't also be present vs FreeSASA's
  independently-implemented core).
- `evident/claims/sasa.yaml` release-tier claim title + claim text +
  tolerances + oracle list updated for the two-oracle gate. Adds:
  - `median_relative_error < 0.005` vs Biopython (existing gate, unchanged)
  - `median_relative_error < 0.02` vs FreeSASA (new — looser because
    Biopython and FreeSASA themselves disagree by ~0.5–1% on most
    structures from atom-radius and probe-discretisation conventions)
  - `pass_rate >= 0.95` (existing gate, now keyed on either oracle
    triggering the warn downgrade)
  - Capability `sasa-cross-tool-parity` added alongside
    `sasa-accuracy-distributional`.
- `validation/report/render_sasa_release.py` extended: new
  `plot_freesasa_distribution` histogram, two new headline-table rows
  (median / p95 / p99 / band-pass count for the FreeSASA side), new
  HTML section explaining the two-oracle framing.

### Fixed

- `validation/amber96_oracle.py` now passes `nonbonded_cutoff=1e6` to
  `proteon.compute_energy(ff="amber96", ...)` to match OpenMM's NoCutoff
  convention. Without this, proteon truncates long-range Coulomb at 15 Å
  while the OpenMM arm goes full-range, producing a systematic ~5%
  median rel_diff. The 0.2% agreement the AMBER96 oracle ought to ship
  was being silently masked. Surfaced by the v0.2.0 contract end-to-end
  smoke on monster3 (50-PDB sample under apptainer bind-mount) — the
  bind-mount itself worked, but the runner output flagged the cutoff
  mismatch via proteon's own UserWarning. PR #56 shipped this gap; this
  PR closes it before any 50K AMBER96 run lands a real artifact.

### Changed

- All four fold-preservation runners now honour `N_PDBS` env override
  (was hardcoded `N = 1000`). Mirrors the pattern in the AMBER96 and
  CHARMM oracles. Required for the new 50K fold-preservation claims to
  scale beyond 1000 structures without code edits.
- `validation/amber96_oracle.py` migrated from
  `concurrent.futures.ProcessPoolExecutor` to `pebble.ProcessPool` for
  per-task subprocess isolation, mirroring the v0.1.4 CHARMM oracle
  pattern (#44). Pre-empts the `BrokenProcessPool` cascade at 50K
  scale that limited the v0.1.3 CHARMM run's `n_ok` to 807 of 44,210.
  Adds `TASK_TIMEOUT_S` env knob (default 60s) to terminate hung
  workers individually instead of stalling the whole run, hoists
  heavy imports (proteon, openmm, pdbfixer, pebble) to module
  top-level so cold-start cost is amortised across tasks, and adds
  `PROTEON_PDB_LIST` support so 50K runs can use a pre-filtered
  protein-only list. Also adds the v0.2.0 universal env var synonyms
  (`PROTEON_CORPUS_DIR`, `PROTEON_OUTPUT_DIR`,
  `PROTEON_AMBER_ORACLE_OUT`) and resume-from-existing-OUT logic to
  match the CHARMM oracle. Unblocks the AMBER96 vs OpenMM 50K
  extension on monster3 (#42).
- `tests/test_evident_runner_contract.py` extended with v0.2.0
  contract checks for `amber96_oracle.py` (env synonym + legacy-wins
  precedence) — same shape as the existing CHARMM tests.

### Added

- **v0.2.0 data-mount contract** — every release-tier oracle runner now
  reads its corpus directory and output directory from
  `PROTEON_CORPUS_DIR` and `PROTEON_OUTPUT_DIR` environment variables.
  The EVIDENT image entrypoint auto-exports them when the well-known
  `/data/pdbs` and `/data/out` bind mounts exist, so the golden replay
  becomes:
  ```bash
  docker run --rm \
    -v $(pwd)/pdbs:/data/pdbs \
    -v $(pwd)/out:/data/out \
    ghcr.io/thegreatherrlebert/proteon-evident:<tag> \
    replay <claim-id>
  ```
  No `-e` flags required. Closes the v0.1.x unreplayability gap (#38)
  where runners hardcoded monster3-only paths and couldn't be replayed
  from the image alone.
- `evident/CAPTURE_SCHEMA.md` gains a "Running in containers" section
  formalising the contract: which env vars exist, which paths take
  precedence, and what the runner authoring rule is. Includes an
  Apptainer / SLURM example for HPC replayability.
- `tests/test_evident_runner_contract.py` guards every release-tier
  runner against contract regression — pure path-resolution check,
  no oracle calls, runs in milliseconds under the existing Python
  test job.
- `USALIGN_BIN` env var on the SASA runner (`validation/run_validation.py`)
  for analogous reasons — image vendors USAlign at
  `/usr/local/bin/USalign`, source-tree dev keeps the
  `/scratch/TMAlign/USAlign/` default.

### Changed

- The four fold-preservation runners
  (`validation/tm_fold_preservation{,_amber,_openmm,_openmm_amber}.py`)
  no longer hardcode
  `/globalscratch/dateschn/proteon-benchmark/pdbs_50k`. Existing monster3
  invocations stay green (the legacy paths are now the unset-env
  fallback); the same scripts run cleanly inside the EVIDENT image
  against any bind-mounted corpus.
- `validation/fold_preservation/join_fold_preservation.py` reads its
  per-side JSONLs from `PROTEON_OUTPUT_DIR` (matching where the runners
  wrote them) when set, falls back to the historical
  `validation/fold_preservation/` location otherwise.
- `validation/charmm19_eef1_ball_oracle.py` accepts both the legacy
  `PROTEON_PDB_DIR` / `PROTEON_CHARMM_ORACLE_OUT` env vars and the new
  universal `PROTEON_CORPUS_DIR` / `PROTEON_OUTPUT_DIR` synonyms,
  legacy first.
- **Skip-missing-atoms fix extended to all PDBFixer-using oracle runners**
  (#48, follow-up to PR #47). The `addMissingAtoms()` deadlock that
  bottlenecked the v0.1.3 50K corpus oracle isn't unique to CHARMM — every
  runner that preprocesses wwPDB inputs through PDBFixer hits it. This
  applies the PR #47 skip pattern to `validation/amber96_oracle.py`,
  `validation/amber96_oracle_triangulate.py`,
  `validation/amber96_obc_oracle.py`,
  `validation/tm_fold_preservation_openmm.py`,
  `validation/tm_fold_preservation_openmm_amber.py`, and
  `validation/diag_obc_params.py`. Each runner detects missing heavy
  atoms via `fixer.findMissingAtoms()` and skips rather than invoking
  the deadlocking `fixer.addMissingAtoms()`. Comparison surface narrows
  to "well-resolved wwPDB" — the same population the v0.1.4 CHARMM
  corpus oracle adopted, and the more defensible scientific scope
  (modeled-back atoms have ad-hoc geometry). Pre-empts v0.2.0 50K
  extensions (#42) from re-discovering the same 79%-timeout regression
  class.

### Compatibility

All existing monster3 batch invocations of the release-tier runners
keep working without any change. The contract is additive: if you set
`PROTEON_CORPUS_DIR` / `PROTEON_OUTPUT_DIR`, runners use them; if you
don't, runners use the same paths they always did.

## [0.1.4] — 2026-05-04

Second EVIDENT release. Trust pyramid completed: the dropped
fold-preservation claims (proteon CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2,
proteon AMBER96 vs OpenMM AMBER96) are back, fully wired with
per-claim renderers. The 50K corpus oracle re-runs cleanly under
`pebble.ProcessPool` with **n_ok = 5,309 / 44,210** vs the v0.1.3
bundle's 807 — a 6.5× improvement on the same population. Manifest
trimmed from 20 → 18 force-of-evidence claims, then re-introduced the
two fold-preservation claims, landing at 20 release-tier claims with
real evidence.

### Added

- **Fold-preservation claims** (#50) — both proteon force fields are
  gated against OpenMM minimization on the same 1000-PDB random
  sample at TM-score level:
  - `proteon-charmm19-fold-preservation-vs-openmm-release-1k-pdbs`:
    proteon CHARMM19+EEF1 median TM=0.9944, OpenMM CHARMM36+OBC2
    median TM=0.9991, median tm_diff +0.0040, n_ok=886/1000.
  - `proteon-amber96-fold-preservation-vs-openmm-release-1k-pdbs`:
    proteon AMBER96 median TM=0.9959, OpenMM AMBER96 median
    TM=0.9992, median tm_diff +0.0028, n_ok=864/1000. Cleanest
    cross-implementation oracle proteon ships — both arms claim
    AMBER96, so the diff is purely implementation, not parameter
    set.
  - `validation/fold_preservation/join_fold_preservation.py` joins
    proteon-side and OpenMM-side per-PDB JSONLs into the canonical
    artifact `{pdb, n_ca, proteon: {...}, openmm: {...}, tm_diff,
    rmsd_diff_A}`.
  - `validation/report/render_fold_preservation.py` — per-claim HTML
    with TM distributions per side, tm_diff histogram against the
    claim's tolerance band, scatter, and top-20 outliers.
- **Three-tier capture schema** (#46) — `evident/CAPTURE_SCHEMA.md`
  defines `minimal` / `extended` / `full` capture levels for oracle
  runners, the per-PDB filesystem layout, the `hardware.json`
  schema, and the renderer contract. `PROTEON_CAPTURE_LEVEL` env
  controls the level.
- `validation/charmm19_eef1_ball_oracle.py` migrates from
  `concurrent.futures.ProcessPoolExecutor` to `pebble.ProcessPool`
  for true per-task subprocess isolation. Closes the
  `BrokenProcessPool` cascade that limited the v0.1.3 50K corpus
  run's `n_ok` to 807 of 44,210 attempted. (#44)
- New env knob `TASK_TIMEOUT_S` (default 60 s) on the corpus oracle
  runner — terminates hung BALL setup tasks individually instead of
  stalling the whole run.
- `pebble` added to both EVIDENT image variants' pip oracle install
  layer.

### Changed

- **Population definition for the 50K oracle is now "well-resolved
  wwPDB"** (#47). The runner skips PDBs with missing heavy atoms
  instead of running `PDBFixer.addMissingAtoms()`, which hangs
  deterministically on certain inputs. The defensible population
  shifts from "everything PDBFixer can repair" → "structures
  resolved well enough to score directly". 18,912 / 44,210 records
  on the v0.1.4 run skip for this reason — they are explicitly
  out-of-population, not failures.
- mkdssp build chain in both `evident/Dockerfile` and
  `evident/Dockerfile.cuda` is now version-pinned: `libmcfp v1.4.2`,
  `libcifpp v10.0.3`, `dssp v4.6.1`. Earlier `--depth 1 main` clones
  caused three breaking-upstream incidents during v1 prep. ARGs are
  exposed (`LIBMCFP_VERSION`, `LIBCIFPP_VERSION`, `DSSP_VERSION`)
  for downstream override. Closes #13.
- `evident/Dockerfile.cuda` switched from `jammy + 2 PPAs` to
  `noble + 0 PPAs` (#45) — eliminates the Launchpad/PPA flake class
  that broke 4 of the v0.1.3 cuda image builds.
- `libglib2.0-0` added to both runtime stages (#45). Apptainer smoke
  on monster3 caught `import ball` failing on
  `libglib-2.0.so.0: cannot open shared object file` in the cuda
  image; same fix applied to slim defensively.

### Removed

- 2 release-tier claims trimmed from `evident/evident.yaml` (#49)
  that asserted evidence we don't currently have:
  - `forcefield_amber_openmm` — removed; AMBER96 vs OpenMM AMBER96
    is now carried by the fold-preservation pair (#50).
  - `msa.yaml` — removed; MSA feature parity is held in unit-test
    scope, not framework-tier.

### Fixed

- 50K corpus run pass rate climbs from 807 / 44,210 (v0.1.3) to
  5,309 / 44,210 (v0.1.4) on the same input population. The lift
  is the sum of: pebble per-task isolation killing the
  `BrokenProcessPool` cascade, the `addMissingAtoms` hang fix,
  and the 60 s task timeout terminating the long tail of stuck
  BALL setups.

## [0.1.3] — 2026-05-03

First EVIDENT release: end-to-end claim / replay / report
architecture with sha256-pinned audit trail.

### Added

- **EVIDENT framework end-to-end** (Phases 1–4 across PRs
  #28 / #30 / #11 / #31):
  - `evident/scripts/lock_release_replays.py` walks every claim
    YAML under `evident/claims/`, sha256-pins each artifact, and
    emits an immutable bundle at `evident/reports/<tag>/`.
    Tier-aware: `pytest console output` style claims correctly
    surface as `ci-only` in the manifest, not `missing`.
  - Lock-time environment snapshot embedded in every manifest:
    Python version, platform, full `pip freeze`, sha256 of
    `cargo metadata`. Closes #36.
  - `evident/scripts/build_index.py` — multi-release aggregator
    served at `<pages-url>/evident/reports/`.
  - `evident/scripts/cut_release.sh` — one-command flow: scp
    artifacts from monster3 → lock bundle → print git tag steps.
  - Per-claim HTML renderers for the corpus oracle, SASA-vs-Biopython
    1k, and 50K battle test. Reports embed plots as base64,
    self-contained, no CDN dependency.
- **Reproducibility images** on GHCR
  (`ghcr.io/thegreatherrlebert/proteon-evident:v0.1.3`):
  - `:slim` — Debian trixie + Python 3.13. Bundles proteon,
    `ball-py`, `openmm`, `pdbfixer`, `biopython`, `gemmi`,
    `freesasa`, `pydssp`, `gromacs`, `mmseqs2`, `mkdssp`,
    `USalign`, `reduce`.
  - `:cuda` — nvidia/cuda 12.8.2 + Ubuntu noble + Python 3.12.
    Same payload, GPU-enabled `proteon-connector` + `cudarc`.
  - Entrypoint dispatches `replay <claim-id>`,
    `render <release-tag>`, plus pass-through to the vendored
    `evident` CLI.
- **CHARMM19+EEF1 oracle** (PRs #16–#26):
  - All seven force-field components (bond_stretch, angle_bend,
    proper_torsion, improper_torsion, vdw, electrostatic,
    EEF1 solvation) gated against BALL on crambin within
    per-component bands.
  - 50k-ready corpus runner with chunked-pool segfault isolation
    (later replaced by pebble in v0.1.4) and a protein-only
    filter that drops nucleic-acid PDBs the force field can't
    parametrise.
  - Distance-dependent dielectric (ε ∝ r) — the canonical
    CHARMM19 convention — replaces the prior constant-ε
    implementation.
  - PHE/TYR aromatic-ring para-diagonal LJ exclusion port from
    BALL's `charmmNonBonded.C:547-565`.
- **CHARMM19+EEF1 corpus oracle**:
  - 1k-PDB BALL oracle on the curated `validation/pdbs/`. Released
    alongside the unit oracle as `proteon-charmm19-vs-ball-corpus-1k-pdbs`.
  - 50K-PDB BALL oracle on a random wwPDB sample, run on monster3.
    Headline `n_ok=807 of 44 210 attempted` reflects the
    `BrokenProcessPool` cascade; per-component bands on the 807
    successful records match the unit oracle (median 0.05%
    bond_stretch, 0.001% torsion, 0.02% electrostatic, 1.62%
    EEF1 solvation, 0.41% improper). Cascade noise framed
    honestly in the claim's `failure_modes`; pebble migration
    is the v0.1.4 follow-up.
- **EVIDENT manifest** (`evident/evident.yaml`) wires 20 claims
  across all proteon subsystems (force fields, alignment, SASA,
  DSSP, hydrogens, GB OBC, supervision, MSA, I/O, pipeline batch).
  Each claim names its oracle, tolerance, replay command, and
  artifact path.

### Changed

- 11 of 13 CI-tier oracle claims now pass against external
  references (BALL, Biopython, USAlign, pydssp, reduce). The two
  pending — `proteon-amber96-vs-openmm-release-1k-pdbs` and
  `proteon-msa-vs-mmseqs2-research` — have monster3-side artifacts
  not yet mirrored.

### Removed

- 6 low-signal claims trimmed in PR #43 (`acceleration_paths`,
  `gpu_cpu_parity`, `forcefield_charmm19_internal`,
  `sasa_freesasa`, both fold-preservation entries). The two
  fold-preservation claims drop until #37 (PDBFixer pre-pass on
  the GROMACS runner) lands real artifacts. Manifest goes from
  26 → 20 claims; every "missing" row is now a real coverage gap,
  not a documentation artefact.

### Known gaps tracked for follow-up

- **#37** — fold-preservation GROMACS artifact regen needs
  PDBFixer pre-pass + monster3 GROMACS install.
- **#42** — v0.2.0 vision: every release-tier claim at 50K scale
  across all oracles.
- **#44** — pebble runner migration (resolved in [Unreleased]).

## [0.1.2] — 2026-04-24

First clean-room MIT release after the 2026-04-24 third-party
lineage audit.

### Changed

- Phase 3 hydrogen placer in `proteon-connector/src/add_hydrogens.rs`
  rewritten clean-room from standard crystallographic geometry; the
  prior dispatcher was structurally derived from BALL's
  `AddHydrogenProcessor` (LGPL-2.1).
- `reconstruct.rs` reattributed to its actual upstream: MIT
  `BiochemicalAlgorithms.jl`, not LGPL BALL C++.
- `forcefield/md.rs` and `forcefield/gb_obc.rs` cite primary
  literature / OpenMM's MIT Reference Platform.

### Added

- `THIRD_PARTY_NOTICES.md` consolidates verbatim upstream license
  texts for TM-align/US-align, MMseqs2, BiochemicalAlgorithms.jl,
  OpenMM, and pdbtbx.

### Performance

- SASA GPU auto-dispatch threshold raised from 500 → 10 000 atoms.
  Small-protein batch throughput recovered ~10× against the
  pre-2026-04-12 regression (monster3 5K: 18.6/s → 192.5/s; numerical
  output bit-identical).

### Benchmark

- `benchmark/run_benchmark.py` now tracks `n_negative_energy`
  alongside `n_converged`. After the 2026-04-11 CHARMM19 heavy-atom
  relaxation default, negative final energy is the documented
  correctness invariant; `converged` is preserved for back-compat.

### No public API changes.

## [0.1.1] — 2026-04-18

First clean PyPI release with populated package metadata.

### Added

- `pip install proteon` installs the Pythonic wrapper; the
  PyO3 `proteon-connector` is pulled in transitively.
- Wheels published for Linux x86_64 / aarch64, macOS x86_64 /
  aarch64, and Windows x86_64.

### Validation highlights at the time of release

- AMBER96 matches OpenMM to within 0.2% on every energy component
  at NoCutoff.
- CHARMM19+EEF1 pipeline: 50K random PDBs at 99.1% end-to-end
  success in 3.5h on RTX 5090.
- TM-align port: 0.003 median TM-score drift from the reference
  C++ USalign across 4 656 pairs.
- SASA: 0.17% median deviation vs Biopython on 1 000 structures.

### No runtime code changes since 0.1.0.

## [0.1.0]

Initial PyPI publication (one-shot to claim the project name; pages
rendered with empty descriptions, superseded by 0.1.1).

[Unreleased]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/theGreatHerrLebert/proteon/releases/tag/v0.1.0
