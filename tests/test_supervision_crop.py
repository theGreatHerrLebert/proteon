"""Crop application for structure + sequence supervision examples.

Deterministic slicing — checked by invariants (consistency vs a manual slice, the
no-op crop, the MSA axis, bounds). Pure NumPy + proteon, so it runs in CI.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

try:
    import proteon
    from proteon.sequence_example import build_sequence_example
    from proteon.supervision import build_structure_supervision_example
    from proteon.supervision_crop import (
        crop_sequence_example,
        crop_structure_supervision_example,
        sample_contiguous_crop,
    )
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"proteon unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_chain(pdb: str):
    path = REPO_ROOT / "test-pdbs" / pdb
    if not path.exists():
        pytest.skip(f"{path} not found")
    pairs = proteon.batch_load_tolerant([str(path)])
    st = pairs[0][1] if isinstance(pairs[0], tuple) else pairs[0]
    return st


def test_crop_structure_slices_every_residue_tensor_consistently():
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    start, stop = 5, 25
    cropped = crop_structure_supervision_example(ex, start, stop)

    assert cropped.length == stop - start
    assert cropped.sequence == ex.sequence[start:stop]
    for field in dataclasses.fields(ex):
        v = getattr(ex, field.name)
        if isinstance(v, np.ndarray):
            cv = getattr(cropped, field.name)
            assert cv.shape[0] == stop - start, field.name
            np.testing.assert_array_equal(cv, v[start:stop], err_msg=field.name)
    # Scalar metadata is preserved.
    assert cropped.record_id == ex.record_id and cropped.chain_id == ex.chain_id


def test_crop_structure_full_window_is_identity():
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    same = crop_structure_supervision_example(ex, 0, ex.length)
    assert same.length == ex.length and same.sequence == ex.sequence
    np.testing.assert_array_equal(same.all_atom_positions, ex.all_atom_positions)
    np.testing.assert_array_equal(
        same.torsion_angles_sin_cos, ex.torsion_angles_sin_cos
    )


def test_crop_structure_rejects_out_of_range():
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    with pytest.raises(ValueError):
        crop_structure_supervision_example(ex, -1, 5)
    with pytest.raises(ValueError):
        crop_structure_supervision_example(ex, 3, ex.length + 1)
    with pytest.raises(ValueError):
        crop_structure_supervision_example(ex, 10, 4)


def test_crop_sequence_slices_msa_on_residue_axis():
    st = _load_chain("1crn.pdb")
    n = sum(1 for r in st.chains[0].residues if r.is_amino_acid)
    seq = "".join(
        proteon.THREE_TO_ONE.get(r.name, "X")
        for r in st.chains[0].residues
        if r.is_amino_acid
    ) if hasattr(proteon, "THREE_TO_ONE") else "A" * n
    depth = 4
    msa = [seq for _ in range(depth)]
    deletion = [[float((i + j) % 4) for j in range(n)] for i in range(depth)]
    ex = build_sequence_example(st, msa=msa, deletion_matrix=deletion)

    start, stop = 3, 20
    cropped = crop_sequence_example(ex, start, stop)
    width = stop - start

    assert cropped.length == width
    assert cropped.sequence == ex.sequence[start:stop]
    # Residue-axis-0 fields.
    np.testing.assert_array_equal(cropped.aatype, ex.aatype[start:stop])
    np.testing.assert_array_equal(cropped.msa_profile, ex.msa_profile[start:stop])
    # MSA fields slice on axis 1; depth is preserved, residue axis is cropped.
    for name in ("msa", "deletion_matrix", "msa_mask", "has_deletion", "deletion_value"):
        cv = getattr(cropped, name)
        assert cv.shape == (depth, width), name
        np.testing.assert_array_equal(cv, getattr(ex, name)[:, start:stop], err_msg=name)


def test_sample_contiguous_crop():
    rng = np.random.default_rng(0)
    # Short chain: returned whole.
    assert sample_contiguous_crop(10, 256, rng) == (0, 10)
    # Long chain: a window of exactly crop_size, in range, reproducible by seed.
    a = sample_contiguous_crop(1000, 256, np.random.default_rng(7))
    b = sample_contiguous_crop(1000, 256, np.random.default_rng(7))
    assert a == b
    start, stop = a
    assert stop - start == 256 and 0 <= start and stop <= 1000
    with pytest.raises(ValueError):
        sample_contiguous_crop(100, 0, rng)
    with pytest.raises(ValueError):
        sample_contiguous_crop(-1, 256, rng)
