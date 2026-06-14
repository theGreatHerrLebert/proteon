"""Crop application for supervision/sequence examples.

The schema carries `crop_start`/`crop_stop` as *metadata*, but the tensors ship
full-length; a training consumer still has to slice every per-residue field
consistently to a fixed crop. That is error-prone — the structure example has
~30 residue-axis tensors, and on the **sequence** side the MSA fields are
`(N_seq, L)`, so the residue axis is **axis 1**, not axis 0. These helpers do it
once, correctly: a contiguous `[start, stop)` window applied to every per-residue
axis, preserving depth/metadata.

Deterministic slicing — gated by invariants (consistency vs a manual slice, the
no-op crop, bounds), not an external oracle (a crop has no ground truth beyond
self-consistency). `sample_contiguous_crop` provides the AF-style contiguous crop
region (seedable).
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np

from .sequence_example import SequenceExample
from .supervision import StructureSupervisionExample
from .templates import TemplateFeatures

# Sequence-side residue-axis-0 fields (one row per residue) vs MSA fields whose
# residue axis is axis 1 (`(N_seq, L)`). `template_mask` is per-template, not
# per-residue, so it is left untouched.
_SEQ_RESIDUE_AXIS0 = ("aatype", "residue_index", "seq_mask", "msa_profile")
_SEQ_MSA_AXIS1 = ("msa", "deletion_matrix", "msa_mask", "has_deletion", "deletion_value")

# Template tensors whose residue axis is axis 1 (`(N_templates, L, *)`). The
# per-template `template_sum_probs` and `n_templates` are residue-independent.
_TEMPLATE_AXIS1 = (
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_masks",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_torsion_angles_sin_cos",
    "template_alt_torsion_angles_sin_cos",
    "template_torsion_angles_mask",
)


def _validate(start: int, stop: int, length: int) -> None:
    if not (0 <= start <= stop <= length):
        raise ValueError(
            f"crop window [{start}, {stop}) out of range for length {length}"
        )


def _copy_and_zero(updates: dict, name: str, indexer) -> None:
    """Zero `updates[name][indexer]` on a *copy* — the sliced arrays in `updates`
    are views into the input example, so an in-place write would corrupt the
    original. No-op if the field is absent (Optional tensors)."""
    arr = updates.get(name)
    if arr is None:
        return
    arr = np.array(arr, copy=True)
    arr[indexer] = 0.0
    updates[name] = arr


def crop_structure_supervision_example(
    example: StructureSupervisionExample, start: int, stop: int
) -> StructureSupervisionExample:
    """Slice a structure example to the residue window `[start, stop)`. Every
    per-residue tensor (all have the residue count as axis 0) and the sequence
    are sliced; `length` is updated; scalar/quality metadata is preserved.

    Crop-boundary torsion correction: `pre_omega`/`phi` (AF-format and classic)
    at the first kept residue were computed from residue `start-1`, now discarded;
    the classic `psi` at the last kept residue used residue `stop`, also
    discarded. A blanket slice would keep those masks at 1, asserting a peptide
    bond to a residue no longer in the crop. So we clear them at the boundaries
    (only when a neighbour was actually dropped). The AF-format `psi` is
    within-residue (carbonyl O), so it needs no correction."""
    _validate(start, stop, example.length)
    updates = {"sequence": example.sequence[start:stop], "length": stop - start}
    for field in dataclasses.fields(example):
        value = getattr(example, field.name)
        if isinstance(value, np.ndarray):
            updates[field.name] = value[start:stop]

    if stop > start:
        if start > 0:  # a preceding residue was dropped → first row's pre_omega/phi stale
            _copy_and_zero(updates, "torsion_angles_mask", (0, slice(0, 2)))
            _copy_and_zero(updates, "phi_mask", 0)
            _copy_and_zero(updates, "omega_mask", 0)
        if stop < example.length:  # a following residue was dropped → last row's classic psi stale
            _copy_and_zero(updates, "psi_mask", -1)

    return dataclasses.replace(example, **updates)


def crop_sequence_example(
    example: SequenceExample, start: int, stop: int
) -> SequenceExample:
    """Slice a sequence example to the residue window `[start, stop)`. Residue-axis
    fields slice on axis 0; the MSA fields (`(N_seq, L)`) slice on **axis 1**, so
    the alignment depth is preserved. `template_mask` (per-template) is untouched."""
    _validate(start, stop, example.length)
    updates = {"sequence": example.sequence[start:stop], "length": stop - start}
    for name in _SEQ_RESIDUE_AXIS0:
        value = getattr(example, name)
        if value is not None:
            updates[name] = value[start:stop]
    for name in _SEQ_MSA_AXIS1:
        value = getattr(example, name)
        if value is not None:
            updates[name] = value[:, start:stop]
    return dataclasses.replace(example, **updates)


def crop_template_features(
    features: TemplateFeatures, start: int, stop: int
) -> TemplateFeatures:
    """Slice a template bundle's residue (query) axis to `[start, stop)`. Each
    `(N_templates, L, *)` tensor slices on **axis 1**; `template_sum_probs` and
    `n_templates` are residue-independent and untouched; `query_len` is updated.

    Crop-boundary torsion correction (mirrors the structure path): the first kept
    residue's `pre_omega`/`phi` were projected from residue `start-1`, now
    discarded, so `template_torsion_angles_mask[:, 0, 0:2]` is cleared when a
    preceding residue was dropped. The AF-format `psi` is within-residue."""
    _validate(start, stop, int(features.query_len))
    updates = {"query_len": stop - start}
    for name in _TEMPLATE_AXIS1:
        value = getattr(features, name)
        if value is not None:
            updates[name] = value[:, start:stop]
    cropped = dataclasses.replace(features, **updates)
    if stop > start and start > 0 and cropped.template_torsion_angles_mask is not None:
        tmask = np.array(cropped.template_torsion_angles_mask, copy=True)
        tmask[:, 0, 0:2] = 0.0  # pre_omega, phi of every template's first kept residue
        cropped = dataclasses.replace(cropped, template_torsion_angles_mask=tmask)
    return cropped


def sample_contiguous_crop(
    length: int, crop_size: int, rng: np.random.Generator
) -> Tuple[int, int]:
    """AlphaFold-style contiguous crop region `[start, stop)`. Chains no longer
    than `crop_size` are returned whole; otherwise a uniformly-random contiguous
    window of exactly `crop_size` residues. `rng` makes it reproducible."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if length <= crop_size:
        return 0, length
    start = int(rng.integers(0, length - crop_size + 1))
    return start, start + crop_size
