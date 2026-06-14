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

# Sequence-side residue-axis-0 fields (one row per residue) vs MSA fields whose
# residue axis is axis 1 (`(N_seq, L)`). `template_mask` is per-template, not
# per-residue, so it is left untouched.
_SEQ_RESIDUE_AXIS0 = ("aatype", "residue_index", "seq_mask", "msa_profile")
_SEQ_MSA_AXIS1 = ("msa", "deletion_matrix", "msa_mask", "has_deletion", "deletion_value")


def _validate(start: int, stop: int, length: int) -> None:
    if not (0 <= start <= stop <= length):
        raise ValueError(
            f"crop window [{start}, {stop}) out of range for length {length}"
        )


def crop_structure_supervision_example(
    example: StructureSupervisionExample, start: int, stop: int
) -> StructureSupervisionExample:
    """Slice a structure example to the residue window `[start, stop)`. Every
    per-residue tensor (all have the residue count as axis 0) and the sequence
    are sliced; `length` is updated; scalar/quality metadata is preserved."""
    _validate(start, stop, example.length)
    updates = {"sequence": example.sequence[start:stop], "length": stop - start}
    for field in dataclasses.fields(example):
        value = getattr(example, field.name)
        if isinstance(value, np.ndarray):
            updates[field.name] = value[start:stop]
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
