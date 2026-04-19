# Proteon Structure Supervision Schema

**Status:** draft v0  
**Last updated:** 2026-04-13

This document defines the first explicit contract for
`structure_supervision_example`.

The purpose of this artifact is:

- provide canonical structural supervision derived from a prepared protein
  structure
- remain independent of any concrete DL framework
- serve as the stable boundary between proteon and downstream JAX / PyTorch /
  NumPy consumers

This schema is intentionally **NumPy-first** and **framework-neutral**.

Proteon should export plain Python metadata plus `numpy.ndarray` tensors.

It should **not** depend on:

- PyTorch
- JAX
- TensorFlow
- framework-specific dataloaders

Those adapters can be built later on top of this contract.

The corresponding Rust/PyO3 batch implementation target is defined in
[RUST_BATCH_SUPERVISION_CONTRACT.md](/scratch/TMAlign/proteon/RUST_BATCH_SUPERVISION_CONTRACT.md).

---

## 1. Design Rule

`structure_supervision_example` is derived from the **post-prep** structure.

That means the source order is:

1. raw record
2. normalized record
3. prepared structure
4. structure supervision example

The supervision artifact must preserve provenance back to all earlier stages.

It must not overwrite the history of:

- raw parsing
- canonical normalization
- reconstruction / hydrogen placement
- minimization and QC

---

## 2. Scope of v0

v0 is deliberately narrow.

### Included in v0

- protein-only examples
- chain-level examples
- post-prep structures only
- NumPy arrays plus plain Python metadata
- atom37 coordinates and masks
- atom14 coordinates and masks
- pseudo-beta coordinates and masks
- backbone torsions and masks
- sidechain chi torsions and masks
- atom ambiguity metadata
- prep / quality metadata

### Explicitly deferred from v0

- ligand supervision
- nucleic acid supervision
- multimer/interface-specific supervision
- template features
- MSA features
- crop/sample logic
- framework-native tensor objects

### Conditional in v0

Rigid-group frames are desirable and should be included in v0 **if**
implementation effort remains bounded and does not destabilize the rest of the
 export contract.

If they delay the first usable artifact, they may land in v0.1 immediately
after the base export.

---

## 3. Core Principles

1. The artifact boundary is NumPy, not a DL framework.
2. The artifact should be derived from the prepared structure, not the raw one.
3. Shape, dtype, and mask semantics must be explicit.
4. Missing data must be represented by masks, not silent omission.
5. Ambiguity must be represented explicitly where relevant.
6. Provenance and prep quality are part of the supervision contract.
7. Public APIs should remain compatible with Rust-side batch extraction.

---

## 4. Object Shape

The Python-side object should be a plain dataclass-like container.

Recommended public API shape:

```python
example = proteon.build_structure_supervision_example(
    structure,
    prep_report=prep_report,
    record_id="...",
    source_id="...",
)
```

Recommended conceptual structure:

```python
StructureSupervisionExample(
    metadata=...,
    sequence=...,
    tensors=...,
    quality=...,
)
```

The exact class layout may vary, but the field contract below should remain
stable.

---

## 5. v0 Field Contract

### 5.1 Metadata fields

Plain Python scalars / strings / lists.

- `record_id: str`
- `source_id: str`
- `prep_run_id: str | None`
- `chain_id: str`
- `sequence: str`
- `length: int`
- `code_rev: str | None`
- `config_rev: str | None`

### 5.2 Index / residue identity fields

- `aatype: np.ndarray`
  - shape: `(N,)`
  - dtype: `int32`
  - semantics: canonical residue-type index per residue

- `residue_index: np.ndarray`
  - shape: `(N,)`
  - dtype: `int32`
  - semantics: residue numbering in the example coordinate order

- `seq_mask: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`
  - semantics: 1.0 for valid residues in this example

### 5.3 Atom37 supervision

- `all_atom_positions: np.ndarray`
  - shape: `(N, 37, 3)`
  - dtype: `float32`
  - semantics: canonical atom37 coordinates in Angstrom

- `all_atom_mask: np.ndarray`
  - shape: `(N, 37)`
  - dtype: `float32`
  - semantics: 1.0 where atom37 coordinate exists and is valid

- `atom37_atom_exists: np.ndarray`
  - shape: `(N, 37)`
  - dtype: `float32`
  - semantics: canonical atom existence by residue type, independent of
    whether a coordinate is present

### 5.4 Atom14 supervision

- `atom14_gt_positions: np.ndarray`
  - shape: `(N, 14, 3)`
  - dtype: `float32`

- `atom14_gt_exists: np.ndarray`
  - shape: `(N, 14)`
  - dtype: `float32`
  - semantics: 1.0 where a valid atom14 coordinate exists

- `atom14_atom_exists: np.ndarray`
  - shape: `(N, 14)`
  - dtype: `float32`
  - semantics: canonical atom14 existence by residue type

- `residx_atom14_to_atom37: np.ndarray`
  - shape: `(N, 14)`
  - dtype: `int32`

- `residx_atom37_to_atom14: np.ndarray`
  - shape: `(N, 37)`
  - dtype: `int32`

### 5.5 Ambiguity metadata

- `atom14_atom_is_ambiguous: np.ndarray`
  - shape: `(N, 14)`
  - dtype: `float32`
  - semantics: 1.0 where atom naming is symmetry-ambiguous

Optional for later:

- `atom14_alt_gt_positions: np.ndarray`
- `atom14_alt_gt_exists: np.ndarray`

These may be deferred from the very first implementation if the ambiguity
flags already ship and the alt-position path would delay delivery.

### 5.6 Pseudo-beta

- `pseudo_beta: np.ndarray`
  - shape: `(N, 3)`
  - dtype: `float32`
  - semantics: CB position, or CA for glycine

- `pseudo_beta_mask: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

### 5.7 Torsion supervision

For v0, torsions should be exposed in a simple NumPy format.

#### Backbone torsions

- `phi: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`
  - units: radians or degrees, must be fixed and documented

- `psi: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

- `omega: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

- `phi_mask: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

- `psi_mask: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

- `omega_mask: np.ndarray`
  - shape: `(N,)`
  - dtype: `float32`

#### Sidechain chi torsions

- `chi_angles: np.ndarray`
  - shape: `(N, 4)`
  - dtype: `float32`

- `chi_mask: np.ndarray`
  - shape: `(N, 4)`
  - dtype: `float32`

Preferred long-term representation:

- `torsion_angles_sin_cos: np.ndarray`
  - shape: `(N, 7, 2)` if following AF-style packing later

But v0 may begin with explicit scalar angles plus masks for clarity.

### 5.8 Rigid-group frames

Desirable in v0 if implementation remains bounded.

If included:

- `rigidgroups_gt_frames: np.ndarray`
  - shape: `(N, G, 4, 4)` or `(N, G, 7)` depending representation
  - dtype: `float32`

- `rigidgroups_gt_exists: np.ndarray`
  - shape: `(N, G)`
  - dtype: `float32`
  - semantics: 1.0 where the base atoms required to define the frame are
    present in the prepared structure

- `rigidgroups_group_exists: np.ndarray`
  - shape: `(N, G)`
  - dtype: `float32`

- `rigidgroups_group_is_ambiguous: np.ndarray`
  - shape: `(N, G)`
  - dtype: `float32`

Representation rule:

v0 should prefer a simple, explicit representation even if not maximally
compact. A `4x4` homogeneous frame tensor is acceptable if it simplifies the
consumer contract.

### 5.9 Quality / prep metadata

This is not optional.

The supervision artifact should carry the prep context used to produce it.

- `prep_success: bool`
- `force_field: str`
- `minimizer: str | None`
- `minimizer_steps: int | None`
- `converged: bool | None`
- `atoms_reconstructed: int`
- `hydrogens_added: int`
- `hydrogens_skipped: int`
- `n_unassigned_atoms: int`
- `skipped_no_protein: bool`
- `initial_energy: float | None`
- `final_energy: float | None`
- `energy_components: dict[str, float] | None`

Optional but strongly recommended:

- `source_format: str`
- `parse_warnings: list[str]`
- `prep_warnings: list[str]`
- `structure_checksum: str | None`

### 5.10 Export Layout

The recommended v0 on-disk export is:

- `manifest.json`
  - dataset-level metadata
  - format identifier
  - example count
  - tensor file name
  - metadata file name
- `examples.jsonl`
  - one JSON object per example
  - metadata and quality only
  - no dense tensor payloads
- `tensors.npz`
  - padded batch-major NumPy arrays
  - one array per tensor field
  - `seq_mask` and per-field masks define valid regions

This keeps the artifact:

- inspectable without a DL framework
- easy to adapt to JAX / PyTorch later
- efficient enough for local iteration and dataset packaging
- compatible with future Rust-side direct writers

Recommended release wrapper on top of the raw example export:

- `release_manifest.json`
- `failures.jsonl`
- `examples/`
  - `manifest.json`
  - `examples.jsonl`
  - `tensors.npz`

Recommended high-level builder:

```python
proteon.build_structure_supervision_dataset(
    structures,
    out_dir,
    release_id="...",
    prep_reports=prep_reports,
)
```

This builder should:

- emit successful supervision examples
- capture failures as structured `failure_record`s
- write one deterministic release directory

Recommended prep-to-supervision handoff:

- `prepared_structures.jsonl`
  - one `prepared_structure` record per prepared input
  - prep metrics, warnings, and provenance
- `supervision_release/`
  - supervision release directory derived from the same prepared inputs

---

## 6. Hydrogens Policy

Hydrogens are part of the prep process, but they should **not** be part of the
main v0 supervision tensors unless explicitly requested later.

Rationale:

- they improve prep quality
- they stabilize minimization
- they are not part of the standard AF2/OpenFold supervision contract
- including them in v0 would complicate canonical atom mappings early

Rule:

Hydrogens may influence the final prepared coordinates and quality metrics, but
the primary supervision export is heavy-atom canonical protein geometry.

---

## 7. Chain-Level Example Policy

v0 uses **chain-level** examples.

Rationale:

- simpler than full multimer semantics
- aligns with many existing training regimes
- easier to canonicalize and validate
- avoids premature interface-specific complexity

Rule:

If the source structure is multichain, v0 may emit one supervision example per
eligible protein chain after prep, with provenance back to the parent
structure-level record.

---

## 8. NumPy Boundary Rule

Proteon should expose:

- Python dataclasses or plain objects
- `numpy.ndarray`
- plain Python metadata

Proteon should not expose in-core:

- `torch.Tensor`
- `jax.Array`
- `tf.Tensor`
- framework-native dataset abstractions

Downstream adapters are explicitly out of scope for the proteon core.

---

## 9. Serialization Guidance

The in-memory schema is NumPy-first.

On-disk representation may differ for scale reasons.

Recommended storage split:

- metadata tables in Arrow / Parquet
- dense tensor payloads in NPZ, Zarr, or another chunk-friendly numeric format

Rule:

The on-disk layout may change over time, but the **logical field contract**
should remain stable.

---

## 10. v0 Public API Surface

Recommended first public API:

- `proteon.build_structure_supervision_example(structure, *, prep_report=None, record_id=None, source_id=None, chain_id=None)`
- `proteon.batch_build_structure_supervision_examples(structures, *, prep_reports=None, ...)`

Design intent:

- the single-example API is convenience
- the batch API is the production surface
- heavy tensor extraction should eventually run Rust-side in batch without
  changing the Python contract

Possible later export API:

- `proteon.export_structure_supervision_examples(examples, out_dir, format=...)`

The first milestone is the in-memory contract, not the final storage engine.

---

## 11. Validation Requirements

Before calling v0 complete, add tests for:

- atom37 shapes and masks
- atom14 mapping consistency
- glycine pseudo-beta behavior
- backbone torsion mask correctness at termini / chain breaks
- chi-mask correctness by residue type
- ambiguity flags on known ambiguous residues
- metadata propagation from `PrepReport`
- stable behavior on current test fixtures

---

## 12. Summary

The v0 `structure_supervision_example` should be:

- post-prep
- protein-only
- chain-level
- NumPy-first
- framework-neutral
- provenance-aware
- quality-aware

This is the correct stable boundary between proteon and any future
JAX/PyTorch training layer.
