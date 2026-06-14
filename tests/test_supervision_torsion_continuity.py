"""`compute_torsion_angles_sin_cos` must not bond `pre_omega`/`phi` across a
residue-index gap. Pure NumPy + proteon (no torch/openfold), so it runs in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from proteon.supervision_constants import ATOM_ORDER
    from proteon.supervision_geometry import (
        compute_torsion_angles_sin_cos,
        continuity_index,
        continuity_index_from_keys,
    )
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"proteon supervision modules unavailable: {exc}", allow_module_level=True
    )


class _StubResidue:
    """Minimal residue for continuity tests: just the primary-structure identity."""

    def __init__(self, serial_number, insertion_code=None, chain_id="A"):
        self.serial_number = serial_number
        self.insertion_code = insertion_code
        self.chain_id = chain_id


def _backbone_atom37(n: int):
    """`n` residues, each with N/CA/C/O present at distinct (non-degenerate)
    positions so every backbone torsion is geometrically defined."""
    pos = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=np.float32)
    for i in range(n):
        base = float(i) * 3.8
        for atom, offset in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 2.5)):
            idx = ATOM_ORDER[atom]
            pos[i, idx] = (base + offset, float(idx) * 0.5, float(i) * 0.3)
            mask[i, idx] = 1.0
    return pos, mask


def test_pre_omega_phi_masked_across_residue_gap():
    pos, mask = _backbone_atom37(3)
    resnames = ["ALA", "ALA", "ALA"]
    # residues 0->1 are bonded (idx 1,2); 1->2 has a gap (idx 2 then 4).
    residue_index = np.array([1, 2, 4], dtype=np.int32)

    out = compute_torsion_angles_sin_cos(pos, mask, resnames, residue_index)
    m = out["torsion_angles_mask"]

    # pre_omega (0) and phi (1) need the prev residue.
    assert m[0, 0] == 0.0 and m[0, 1] == 0.0  # no prev at the start
    assert m[1, 0] == 1.0 and m[1, 1] == 1.0  # bonded to residue 0
    assert m[2, 0] == 0.0 and m[2, 1] == 0.0  # gap → not bonded to residue 1
    # psi (2) is within-residue and unaffected by the gap.
    assert m[1, 2] == 1.0 and m[2, 2] == 1.0

    # Without residue_index, row-adjacency is used (OpenFold-exact): the gap row
    # is (incorrectly) treated as bonded — the behaviour the parity test relies on.
    out_rowadj = compute_torsion_angles_sin_cos(pos, mask, resnames)
    assert out_rowadj["torsion_angles_mask"][2, 0] == 1.0


def _diffs(idx):
    return np.diff(np.asarray(idx))


def test_continuity_index_bonds_insertion_codes():
    """Insertion-coded residues (100, 100A, 100B, 101) are one peptide chain —
    every step must read as bonded (+1), unlike raw serial_number which collapses
    100/100A and would break there. This is the codex catch, fixed for good."""
    residues = [
        _StubResidue(100),
        _StubResidue(100, "A"),
        _StubResidue(100, "B"),
        _StubResidue(101),
    ]
    idx = continuity_index(residues)
    assert np.all(_diffs(idx) == 1), idx  # all peptide-bonded → no masking


def test_continuity_index_breaks_on_number_gap_and_chain_change():
    residues = [
        _StubResidue(10),
        _StubResidue(12),          # number gap (11 missing) → break
        _StubResidue(13),          # bonded to 12
        _StubResidue(1, chain_id="B"),  # new chain → break
    ]
    d = _diffs(continuity_index(residues))
    assert d[0] != 1 and d[1] == 1 and d[2] != 1


def test_continuity_index_matches_serial_pattern_without_insertion_codes():
    """Regression equivalence: with no insertion codes the +1/break pattern is
    identical to the old raw-serial_number adjacency, so non-insertion structures
    (every validated fixture) are byte-for-byte unchanged."""
    serials = [1, 2, 3, 7, 8, 20]
    idx = continuity_index([_StubResidue(s) for s in serials])
    old_bonded = [serials[i] == serials[i - 1] + 1 for i in range(1, len(serials))]
    new_bonded = [bool(d == 1) for d in _diffs(idx)]
    assert new_bonded == old_bonded


def test_insertion_code_residue_keeps_pre_omega_phi_unmasked():
    """End-to-end: feeding the insertion-code-aware continuity index into
    `compute_torsion_angles_sin_cos` leaves pre_omega/phi *unmasked* at an
    inserted residue (atoms present, genuine peptide bond)."""
    pos, mask = _backbone_atom37(3)
    resnames = ["ALA", "ALA", "ALA"]
    residues = [_StubResidue(10), _StubResidue(10, "A"), _StubResidue(11)]
    idx = continuity_index(residues)

    m = compute_torsion_angles_sin_cos(pos, mask, resnames, idx)["torsion_angles_mask"]
    # 10 -> 10A (row 1) and 10A -> 11 (row 2) are both bonded → not masked.
    assert m[1, 0] == 1.0 and m[1, 1] == 1.0
    assert m[2, 0] == 1.0 and m[2, 1] == 1.0
    # Raw serial_number (10, 10, 11) would have masked row 1 (10 == 10, not +1).
    raw = compute_torsion_angles_sin_cos(
        pos, mask, resnames, np.array([10, 10, 11], dtype=np.int64)
    )["torsion_angles_mask"]
    assert raw[1, 0] == 0.0  # the bug this fix removes


def test_continuity_index_from_keys_treats_none_as_break():
    """Gap rows (unaligned template positions) are `None` keys → always a break."""
    keys = [("A", 5, None), None, ("A", 6, None), ("A", 7, None)]
    d = _diffs(continuity_index_from_keys(keys))
    # row1 None → break; row2 (after None) → break; row3 bonded to row2.
    assert d[0] != 1 and d[1] != 1 and d[2] == 1


def test_disconnected_insertion_code_is_masked_by_geometry():
    """A numbering-adjacent insertion code that is *physically* disconnected
    (CA far away) must still mask pre_omega/phi — numbering says bonded, geometry
    vetoes. This is the corpus icode_interleave case (res 3 -> 3A ~80 A apart),
    the codex P1 the pure-numbering fix would have regressed."""
    pos, mask = _backbone_atom37(3)
    # Displace the middle residue ~80 A away (like the fixture's interleaved 3A):
    # both its boundaries (0->1 and 1->2) become breaks.
    pos[1] += np.array([80.0, 0.0, 0.0], dtype=np.float32)
    resnames = ["ALA", "ALA", "ALA"]
    # Numbering says 10 -> 10A -> 11 are all bonded (insertion-code-adjacent)...
    residues = [_StubResidue(10), _StubResidue(10, "A"), _StubResidue(11)]
    idx = continuity_index(residues)
    assert np.all(_diffs(idx) == 1)  # numbering layer alone would bond them

    m = compute_torsion_angles_sin_cos(pos, mask, resnames, idx)["torsion_angles_mask"]
    # ...but geometry vetoes both boundaries of the displaced residue.
    assert m[1, 0] == 0.0 and m[1, 1] == 0.0  # 10A disconnected from 10
    assert m[2, 0] == 0.0 and m[2, 1] == 0.0  # 11 disconnected from 10A
    # psi (within-residue) is unaffected by the break.
    assert m[1, 2] == 1.0 and m[2, 2] == 1.0
