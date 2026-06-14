# Supervision pipeline wiring — templates into the release, end-to-end

## Motivation

proteon's AlphaFold/Layer-5 supervision pipeline is *almost* end-to-end. The
release machinery exists and is tested:

- `build_structure_supervision_release` — atom37/atom14, pseudo_beta, all
  torsions, rigidgroups, quality. (`supervision_release.py`)
- `build_sequence_release` / `build_sequence_dataset` — aatype/residue_index +
  MSA (depth-ragged) with OpenFold deletion features. (`sequence_release.py`)
- `build_training_release` — thin join of sequence × structure with split,
  crop metadata, weight; streaming `training.parquet`. (`training_example.py`)
- `build_corpus_release_manifest` — aggregating top manifest. (`corpus_release.py`)

**The one structural hole:** templates are fully *computed* but completely
*orphaned*. Both `build_template_features` (sequence/CIGAR path) and
`build_structure_template_features` (TM-align path) return rich `TemplateFeatures`
— `template_aatype`, `template_all_atom_positions/masks`, `template_sum_probs`,
and (structure path) derived `pseudo_beta` + `torsion_angles_sin_cos`. None of it
has a serialization schema and none of it reaches any release artifact. The only
template field on `SequenceExample` is `template_mask` — a per-template
confidence placeholder, not the tensors. A consumer reading a training release
today cannot get template features at all.

This plan closes that hole: **make `TemplateFeatures` a first-class, serialized
part of the training release**, so a training example carries its templates
through Parquet round-trip, with crop and lineage handled correctly.

## Scope

**In scope**

1. A `TemplateFeatures` Parquet schema + writer/reader (doubly-ragged: variable
   `N_templates` and variable `L`), mirroring the existing MSA depth-ragged
   encoding in `sequence_export.py`.
2. Attach templates to the training example: a new optional `templates:
   TemplateFeatures | None` on `TrainingExample` (NOT on `SequenceExample` —
   templates are a training-time join input, like the structure label), written
   into / read from the training release.
3. A builder seam: `build_training_release(...)` gains an optional
   `templates: Dict[record_id -> TemplateFeatures]` (precomputed by the caller
   via either template path), analogous to the existing `crop_metadata` /
   `weights` dicts. The release does **not** run retrieval itself — retrieval
   (search hits / candidate pools) stays the caller's job, consistent with the
   existing separation.
4. Crop correctness: when `crop_start/stop` is present, the template `L` axis
   must crop in lockstep with the example. Since crop is currently lazy
   (metadata only; tensors uncropped at rest), templates are stored **uncropped**
   and a single `crop_training_example` helper (new, thin) crops sequence +
   structure + template tensors together — so the loader has one call that keeps
   all three axis-aligned. (No eager cropping in the release; we only make the
   lazy crop *complete*.)
5. Lineage tightening (small, same PR): emit `tensor_sha256` on the structure
   and sequence releases (training already has `parquet_sha256`), and have
   `build_corpus_release_manifest` carry the child `parquet_sha256`/`tensor_sha256`
   forward so a corpus manifest pins exact bytes.

**Out of scope (explicitly deferred, separate projects)**

- Extra-MSA stochastic clustering branch (roadmap §8; needs the clustering
  pipeline — substantial, standalone).
- 3Di / FoldSeek-style structural prefilter for template retrieval-at-scale
  (STRUCTURE_TEMPLATES_PLAN §6).
- Eager cropping / a curriculum scheduler. Lazy crop-at-load is the AF-standard
  design; we complete it, we don't replace it.
- Split/leakage *enforcement* (policies are recorded; auto-enforcement is its
  own policy-engine project).

## Design

### Template Parquet schema

`TemplateFeatures` tensors are doubly-ragged: `(N_templates, L, ...)`. Reuse the
MSA precedent (`sequence_export.MSA_FIELDS`, `list<list<T>>`). For each template
tensor we store a `list<list<...>>` column: outer list = templates, inner list =
residues, leaf = the fixed inner shape (`FixedSizeList` for `(37,3)` etc.).
Per-template scalars (`template_sum_probs`, length `N_templates`) store as a
single `list<float32>`. A template-less example writes all-null (the MSA
null-handling pattern; `test_sequence_parquet_*` already exercises ragged
null-vs-non-null in one row group).

Fields (names mirror `TemplateFeatures`):

| column | shape | dtype | nullable |
|---|---|---|---|
| `template_aatype` | (N, L) | int32 | yes |
| `template_all_atom_positions` | (N, L, 37, 3) | float32 | yes |
| `template_all_atom_masks` | (N, L, 37) | float32 | yes |
| `template_sum_probs` | (N,) | float32 | yes |
| `template_pseudo_beta` | (N, L, 3) | float32 | yes |
| `template_pseudo_beta_mask` | (N, L) | float32 | yes |
| `template_torsion_angles_sin_cos` | (N, L, 7, 2) | float32 | yes |
| `template_alt_torsion_angles_sin_cos` | (N, L, 7, 2) | float32 | yes |
| `template_torsion_angles_mask` | (N, L, 7) | float32 | yes |

`n_templates` is recoverable from the outer list length; `query_len` from the
inner. Store a schema-version bump on the training Parquet
(`TRAINING_PARQUET_SCHEMA_VERSION += 1`) and read older artifacts (no template
columns) as `templates=None` — same backward-compat contract the MSA
`has_deletion`/`deletion_value` columns established
(`test_sequence_parquet_reads_artifact_missing_new_msa_columns`).

### Where templates live

`TrainingExample.templates: TemplateFeatures | None`. Rationale: templates are a
per-(query-example) *input* the model consumes alongside the MSA and the
structure label; they join at the training-example layer, exactly where
`structure` and `sequence` already join. Putting them on `SequenceExample` would
wrongly bind them to the sequence artifact (and force a sequence-release schema
change). The training release is the join layer — that's the natural home.

### Builder seam

```python
build_training_release(
    sequence_release_dir, structure_release_dir, out_dir, release_id,
    *, split_assignments, crop_metadata=None, weights=None,
    templates: Dict[str, TemplateFeatures] | None = None,   # NEW, by record_id
    ...,
)
```

The release looks up `templates.get(record_id)` per example, validates
`template.query_len == example_length` (reject/skip-with-warning on mismatch,
mirroring the structure-templates skip-visibility contract), serializes, and
records `count_with_templates` in the manifest. Callers produce the dict by
running either template path over their retrieval results — unchanged, separate.

### Crop correctness

Add `crop_training_example(example, start, stop) -> TrainingExample` in
`supervision_crop.py`, composing the existing `crop_sequence_example` +
`crop_structure_supervision_example` and adding template `L`-axis cropping
(slice axis 1 of every `(N, L, ...)` template tensor; `template_sum_probs` and
`n_templates` unchanged). This is the single lazy-crop entry the loader calls so
sequence, structure, **and** template residue axes stay aligned. Unit-tested for
axis alignment (a crop that drops residue k drops it from all three).

### Lineage tightening

- `supervision_export` / `sequence_export`: compute and store `tensor_sha256` in
  the release manifest (training already does this via `parquet_sha256`; lift the
  same helper). Load path verifies when `verify_checksum=True` (the existing flag).
- `build_corpus_release_manifest`: read child manifests' `tensor_sha256` /
  `parquet_sha256` and surface them (`sequence_tensor_sha256`,
  `structure_tensor_sha256`, `training_parquet_sha256`) so a corpus release pins
  exact child bytes — closing the "lineage back to artifacts" gap.

## Test plan

- **Schema round-trip**: a `TrainingExample` with templates (N=2, both at the
  query's `L` — all templates of one query share the query-projected length;
  varied `L` only *across* examples) survives `build_training_release` → load
  with all template tensors bit-identical; a template-less example loads
  `templates=None`; an `N=0` bundle round-trips as `N=0`, not `None`. Mixed
  null/non-null in one row group (the MSA ragged test's analogue).
- **Backward-compat**: an artifact written without template columns loads
  `templates=None`, not `KeyError` (drop-columns test, like the MSA one).
- **query_len mismatch**: a template whose `query_len != example length` is
  rejected with a visible warning and `templates=None` (skip-visibility).
- **Crop alignment**: `crop_training_example(ex, a, b)` yields sequence,
  structure, and template tensors all of length `b-a`, and the dropped residue
  index is dropped consistently across all three (assert a known column value
  moves correctly).
- **Lineage**: structure + sequence releases emit `tensor_sha256`; corrupting a
  byte fails `verify_checksum=True`; corpus manifest carries child shas.
- **Empty release**: zero examples → no template columns written, manifest
  `count_with_templates=0`, loader returns `[]` (the empty-release contract the
  sequence release already pins).

## Effort / risk

- Schema + writer/reader: the doubly-ragged encoding is the only genuinely new
  bit; the MSA path is a working template for it. ~Medium.
- Crop helper + lineage: small, compositional.
- Risk: doubly-ragged Parquet null handling (templates absent) is the sharp
  edge — covered by reusing and explicitly testing the MSA null pattern.
- Backward compat: additive columns + schema-version bump + missing-column→None;
  no existing artifact breaks, no oracle re-validation (tensors unchanged for
  non-template examples).

## Claudex review outcome (adopted)

Codex reviewed v1. All six findings adopted; both forks resolved its way.
Two claims were independently verified against the code before adopting:

- **MSA never reaches the training reader** (`training_example.py:603` —
  `seq_kwargs.setdefault("msa", None)`). The training parquet schema doesn't
  carry MSA, so a `TrainingExample` rebuilt from `training.parquet` has
  `msa=None`. Serializing templates alone does **not** make the training example
  OpenFold-complete — MSA is the other missing half. *Verified true.*
- **Pre-existing crop-boundary torsion bug** (`supervision_crop.py:41`).
  `crop_structure_supervision_example` blanket-slices every ndarray, so the first
  cropped residue keeps `pre_omega`/`phi` (mask + value) computed from the now-
  discarded residue `start-1`; the last cropped residue keeps a `psi` computed
  from the discarded residue `stop`. *Verified true* — independent of templates.

**Adopted decisions:**

1. **`None` vs `N=0` are distinct and both preserved.** `None` = retrieval not
   run / unavailable; `N=0` = retrieval ran, no usable templates (a valid empty
   bundle). All-null columns encode `None`; a present-but-empty outer list
   encodes `N=0`. Manifest tracks `count_with_templates` *and*
   `count_zero_templates` separately.
2. **query_len mismatch RAISES at release build** (not warn-and-discard) — a
   mismatch is a bad record-id join and silently dropping it contaminates
   training. Candidate-level retrieval failures still warn (that's the
   featurizer); release-level join misalignment is a hard error unless an
   explicit `permissive=True` is passed.
3. **Storage: separate `templates.parquet`** keyed by `record_id` (own checksum
   + schema-version + referential-integrity check), exposed on `TrainingExample`
   only *after* a left-join at load. Templates are large/sparse/optional/
   regenerable; a separate artifact avoids rewriting the big training parquet,
   lets template-free jobs skip the columns, and versions independently. The
   builder seam takes an **iterable / indexed artifact**, not a corpus-sized
   `Dict`.
4. **Encoding: doubly-ragged on disk, pad at collation.** Persist natural `N`;
   `max_templates` sampling + `template_mask` creation + padding happen in the
   reader/collator (matches OpenFold `make_fixed_size`). Needs a **new nullable
   doubly-ragged column builder** (the existing `_make_ragged_column` has no
   validity bitmap) with explicit shape/dtype/finite validation:
   all mandatory tensors agree on `(N, L)`, every non-None derived tensor agrees,
   `template_sum_probs.shape == (N,)`.
5. **Persisted geometry: raw inputs always; derived geometry as optional
   columns.** Persist `template_aatype` / `template_all_atom_positions` /
   `_masks` / `template_sum_probs` (both paths produce these). Derived
   `pseudo_beta` / torsions stay **Optional** columns (null for the sequence
   path, populated for the structure path) — matching the existing
   `TemplateFeatures` Optional fields, so the schema isn't path-inconsistent. We
   persist (not re-derive at load) because re-deriving template torsions needs
   the continuity signal, which the query-grid atom37 alone can't fully
   reconstruct.
6. **Crop stays lazy-only** this work. `crop_training_example` is the single
   load-time call; no eager curriculum materialization.

**Re-sequenced into reviewable PRs:**

- **PR-A (first, standalone):** Fix the crop-boundary torsion bug in
  `crop_structure_supervision_example` — clear `pre_omega`/`phi` (and classic
  `phi`/`omega`) masks on the first cropped residue and `psi`/`psi_mask` on the
  last. Verified latent bug; independent; unblocks correct cropping for
  everything downstream. Small.
- **PR-B:** Template artifact — `templates.parquet` writer/reader (nullable
  doubly-ragged), `None`-vs-`N=0`, checksum + schema-version, query_len-raises,
  referential integrity; `crop_training_example` extended to the template L axis
  (reusing PR-A's boundary cleanup); training manifest references it.
- **PR-C:** Complete-feature training loader — left-join MSA (from the sequence
  release) **and** templates onto the training example so one loader call yields
  an OpenFold-complete feature dict; lineage `tensor_sha256` propagation into the
  release + corpus manifests (the hashes already exist on inner exporters —
  propagate, don't recompute). This is what finally makes "end-to-end" true.

**Test-plan additions (from review):** malformed tensor-shape combos; `None`
vs `N=0`; optional derived fields independently null; differing `L` *across*
examples (never within one bundle — all templates of one query share the query-
projected `L`); zero-length crops; **crop-boundary torsion masks**; template
count above `max_templates`; pointer-only loading; split-filtered template joins;
duplicate/missing template keys; schema-version rejection vs additive compat; a
real loader→OpenFold feature-dict integration test.
