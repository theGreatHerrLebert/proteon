"""`compute_torsion_angles_sin_cos` must not bond `pre_omega`/`phi` across a
residue-index gap. Pure NumPy + proteon (no torch/openfold), so it runs in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from proteon.supervision_constants import ATOM_ORDER
    from proteon.supervision_geometry import compute_torsion_angles_sin_cos
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"proteon supervision modules unavailable: {exc}", allow_module_level=True
    )


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
