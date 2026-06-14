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


# Fields the crop corrects at the boundary (a torsion that referenced a dropped
# neighbour) — excluded from the blanket exact-slice check, asserted separately.
_BOUNDARY_CORRECTED = {"torsion_angles_mask", "phi_mask", "omega_mask", "psi_mask"}


def test_crop_structure_slices_every_residue_tensor_consistently():
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    start, stop = 5, 25
    cropped = crop_structure_supervision_example(ex, start, stop)

    assert cropped.length == stop - start
    assert cropped.sequence == ex.sequence[start:stop]
    for field in dataclasses.fields(ex):
        v = getattr(ex, field.name)
        if isinstance(v, np.ndarray) and field.name not in _BOUNDARY_CORRECTED:
            cv = getattr(cropped, field.name)
            assert cv.shape[0] == stop - start, field.name
            np.testing.assert_array_equal(cv, v[start:stop], err_msg=field.name)
    # Boundary-corrected fields are a faithful slice in the interior (only the
    # corrected boundary entries differ).
    np.testing.assert_array_equal(
        cropped.torsion_angles_mask[1:], ex.torsion_angles_mask[start + 1 : stop]
    )
    # Scalar metadata is preserved.
    assert cropped.record_id == ex.record_id and cropped.chain_id == ex.chain_id


def test_crop_clears_boundary_torsions_that_referenced_dropped_neighbours():
    """A mid-structure crop drops the residues just outside the window, so the
    first kept residue's pre_omega/phi (computed from start-1) and the last kept
    residue's classic psi (computed from stop) are stale. They must be masked,
    not blanket-sliced through (the crop-boundary bug)."""
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    start, stop = 5, 25
    # Precondition: in the full example these boundary torsions ARE bonded
    # (interior residues, real peptide bonds) — so the crop has something to fix.
    assert ex.torsion_angles_mask[start, 0] == 1.0 and ex.torsion_angles_mask[start, 1] == 1.0
    assert ex.psi_mask[stop - 1] == 1.0

    cropped = crop_structure_supervision_example(ex, start, stop)
    # First kept residue: pre_omega (0) and phi (1) cleared; AF psi (2) untouched.
    assert cropped.torsion_angles_mask[0, 0] == 0.0
    assert cropped.torsion_angles_mask[0, 1] == 0.0
    assert cropped.torsion_angles_mask[0, 2] == ex.torsion_angles_mask[start, 2]
    assert cropped.phi_mask[0] == 0.0 and cropped.omega_mask[0] == 0.0
    # Last kept residue: classic psi (uses next residue) cleared.
    assert cropped.psi_mask[-1] == 0.0
    # The input example must be untouched (slices are views — copy-on-write).
    assert ex.torsion_angles_mask[start, 0] == 1.0 and ex.psi_mask[stop - 1] == 1.0


def test_crop_from_start_and_to_end_leaves_unbroken_boundaries():
    """Cropping that does NOT drop a neighbour (start=0 / stop=length) must not
    spuriously mask: those boundaries have no discarded neighbour."""
    ex = build_structure_supervision_example(_load_chain("1crn.pdb"))
    # start=0: no residue before the window → nothing to clear at the front.
    head = crop_structure_supervision_example(ex, 0, 20)
    np.testing.assert_array_equal(head.torsion_angles_mask[0], ex.torsion_angles_mask[0])
    np.testing.assert_array_equal(head.phi_mask[:1], ex.phi_mask[:1])
    # stop=length: no residue after the window → last residue's psi unchanged.
    tail = crop_structure_supervision_example(ex, 20, ex.length)
    assert tail.psi_mask[-1] == ex.psi_mask[-1]


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
