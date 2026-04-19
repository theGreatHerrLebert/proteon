# Proteon Rust Batch Supervision Contract

**Status:** draft v0  
**Last updated:** 2026-04-13

This document defines the first Rust-side extraction contract for
`structure_supervision_example`.

The goal is straightforward:

- keep Proteon framework-neutral
- keep the public Python boundary NumPy-first
- move heavy structural extraction into Rust
- make batching the default production path

This is the intended shape for the fast path:

1. Rust parses / traverses prepared structures in bulk
2. Rust computes canonical supervision tensors in bulk
3. PyO3 exposes those results as `numpy.ndarray`
4. Python assembles thin metadata objects and convenience wrappers

Proteon should not expose PyTorch, JAX, or framework-native tensors here.

---

## 1. Design Rule

The public semantic contract is defined by:

- [STRUCTURE_SUPERVISION_SCHEMA.md](/scratch/TMAlign/proteon/STRUCTURE_SUPERVISION_SCHEMA.md)
- [supervision.py](/scratch/TMAlign/proteon/packages/proteon/src/proteon/supervision.py)

Rust is an implementation backend for that contract, not a second schema.

That means:

- field names stay stable
- shapes stay stable
- dtypes stay stable
- mask semantics stay stable

The Python layer may remain a small orchestration layer, but the heavy geometry
path should move to Rust when practical.

---

## 2. Production Surface

Proteon should expose two conceptual layers:

### 2.1 Public Python API

Single-example convenience:

```python
ex = proteon.build_structure_supervision_example(
    structure,
    prep_report=prep_report,
    record_id="...",
    source_id="...",
)
```

Batch production path:

```python
batch = proteon.batch_build_structure_supervision_examples(
    structures,
    prep_reports=prep_reports,
    record_ids=record_ids,
    source_ids=source_ids,
)
```

### 2.2 Internal Rust backend

The batch path should eventually delegate tensor extraction to Rust.

Python should still own:

- provenance fields
- record/source/prep identifiers
- prep report attachment
- final dataclass assembly

Rust should own:

- residue traversal
- canonical atom mapping
- coordinate extraction
- mask construction
- torsion geometry
- ambiguity flags
- future rigid-group frame extraction

---

## 3. Recommended Rust API Shape

The Rust boundary should be batch-first.

Recommended PyO3-exposed entry points:

```python
_extract_structure_supervision_batch(...)
_extract_structure_supervision_chain(...)
```

Recommended Python-side use:

```python
tensors = _connector.extract_structure_supervision_batch(
    structures,
    chain_ids=chain_ids,
)
```

Single-example extraction should be implemented as a thin wrapper over the
batch path, not the other way around.

---

## 4. Input Contract

The Rust extractor should consume already prepared Proteon structures.

v0 assumptions:

- protein-only residues are extracted
- examples are chain-level
- input structure order is preserved
- residue order inside a selected chain is preserved
- hydrogens may exist in the structure, but are ignored for canonical export

Required per-example inputs:

- `structure`
- optional `chain_id`

Optional later:

- explicit residue selection / cropping
- multimer example assembly
- export flags for deferred tensor groups

If a structure contains multiple chains and no `chain_id` is provided, Rust
must fail with the same semantics as Python.

---

## 5. Output Contract

Rust should return a batch tensor container with one leading batch dimension.

Recommended batch output fields:

- `aatype: int32[B, N]`
- `residue_index: int32[B, N]`
- `seq_mask: float32[B, N]`
- `all_atom_positions: float32[B, N, 37, 3]`
- `all_atom_mask: float32[B, N, 37]`
- `atom37_atom_exists: float32[B, N, 37]`
- `atom14_gt_positions: float32[B, N, 14, 3]`
- `atom14_gt_exists: float32[B, N, 14]`
- `atom14_atom_exists: float32[B, N, 14]`
- `residx_atom14_to_atom37: int32[B, N, 14]`
- `residx_atom37_to_atom14: int32[B, N, 37]`
- `atom14_atom_is_ambiguous: float32[B, N, 14]`
- `pseudo_beta: float32[B, N, 3]`
- `pseudo_beta_mask: float32[B, N]`
- `phi: float32[B, N]`
- `psi: float32[B, N]`
- `omega: float32[B, N]`
- `phi_mask: float32[B, N]`
- `psi_mask: float32[B, N]`
- `omega_mask: float32[B, N]`
- `chi_angles: float32[B, N, 4]`
- `chi_mask: float32[B, N, 4]`
- `rigidgroups_gt_frames: float32[B, N, 8, 4, 4]`
- `rigidgroups_gt_exists: float32[B, N, 8]`
- `rigidgroups_group_exists: float32[B, N, 8]`
- `rigidgroups_group_is_ambiguous: float32[B, N, 8]`

### Variable length handling

Batch extraction should support variable chain lengths by padding to `N_max`
within the batch and using masks.

That means:

- `seq_mask` marks valid residues
- atom/torsion masks remain zero for padded positions
- `residue_index` for padded rows may be zero-filled
- callers must rely on masks, not sentinel coordinates

This is the correct shape for Rust-side batching and later NumPy-to-JAX/PyTorch
adaptation.

---

## 6. Dtype and Layout Rules

These rules should be fixed early and treated as part of the ABI contract:

- integer indices: `int32`
- masks: `float32`
- coordinates: `float32`
- angles: `float32`
- C-contiguous arrays on the Python side

Do not use:

- Python lists for tensor payloads
- object arrays
- framework-native tensors

If Rust uses internal `Vec<f32>` or `ndarray`-style storage, the PyO3 bridge
should still expose contiguous NumPy arrays.

---

## 7. Error Semantics

Rust and Python should fail the same way for schema-level errors.

Expected hard failures:

- no protein residues in selected chain
- multi-chain structure without explicit `chain_id`
- requested `chain_id` missing
- unsupported structural invariants that violate the schema contract

Expected soft handling:

- missing canonical atoms
- unknown residues
- missing sidechain torsions
- ambiguous sidechain naming

These soft cases must produce masked outputs, not raise.

---

## 8. Phased Rust Implementation

### Phase 1

Move the highest-value dense extraction first:

- atom37
- atom14
- atom mappings
- ambiguity flags
- pseudo-beta

This gives immediate speedup and keeps the implementation bounded.

### Phase 2

Add:

- backbone torsions
- chi torsions

### Phase 3

Add:

- rigid-group frames
- rigid-group ambiguity/existence tensors

### Phase 4

Add:

- direct Arrow/Parquet/export integration if needed
- zero-copy or minimized-copy dataset export paths

---

## 9. Python Assembly Rule

Even after Rust extraction lands, Python should keep the public object model.

That means:

- Rust returns raw NumPy-compatible tensors
- Python attaches:
  - `record_id`
  - `source_id`
  - `prep_run_id`
  - `code_rev`
  - `config_rev`
  - `StructureQualityMetadata`
- Python returns `StructureSupervisionExample`

This preserves a stable user-facing API while allowing the implementation to
become fast.

---

## 10. Non-Goals

This contract does not imply:

- adding a DL library to Proteon
- exposing framework-specific dataset types
- binding Proteon to one training stack
- putting crop/sample logic into the Rust extractor immediately

Proteon’s role here is supervision generation and data-contract enforcement,
not framework ownership.

---

## 11. Immediate Next Step

The next concrete implementation step should be:

1. add an internal batch tensor container spec matching this document
2. sketch the PyO3 function signatures in the connector layer
3. port atom37 + atom14 + pseudo-beta extraction first
4. keep Python tests as the semantic oracle for Rust parity
