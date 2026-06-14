"""Oracle parity: proteon's AlphaFold-format torsion supervision vs OpenFold.

`supervision_geometry.compute_torsion_angles_sin_cos` must reproduce OpenFold's
`atom37_to_torsion_angles` — the 7-torsion `(N, 7, 2)` sin/cos format, the
180°-symmetric `alt`, and the `(N, 7)` mask — on real structures, so proteon's
structure-supervision export is directly OpenFold-loadable.

Needs `torch` + a checked-out `openfold` (neither a proteon/CI dependency), so it
is skipped unless both import. Run locally with those installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_of = pytest.importorskip("openfold.data.data_transforms")
from openfold.np import residue_constants as rc  # noqa: E402

try:
    from proteon.supervision_constants import ATOM_ORDER
    from proteon.supervision_geometry import compute_torsion_angles_sin_cos
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"proteon supervision modules unavailable: {exc}", allow_module_level=True
    )

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_atom37(path: Path):
    """Minimal atom37 from a PDB (no connector dependency): first model, all
    chains' ATOM records in file order."""
    residues: dict = {}
    order: list = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            if line.startswith("ENDMDL"):
                break
            continue
        name = line[12:16].strip()
        key = (line[21], line[22:26].strip(), line[26])
        if key not in residues:
            residues[key] = (line[17:20].strip(), {})
            order.append(key)
        residues[key][1][name] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    n = len(order)
    pos = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=np.float32)
    for i, key in enumerate(order):
        for atom_name, xyz in residues[key][1].items():
            idx = ATOM_ORDER.get(atom_name)
            if idx is not None:
                pos[i, idx] = xyz
                mask[i, idx] = 1.0
    resnames = [residues[k][0] for k in order]
    return pos, mask, resnames


def _openfold_reference(pos, mask, resnames):
    def aaidx(rn: str) -> int:
        return rc.restype_order.get(rc.restype_3to1.get(rn, "X"), 20)

    prot = {
        "aatype": torch.tensor([aaidx(r) for r in resnames], dtype=torch.long),
        "all_atom_positions": torch.tensor(pos, dtype=torch.float64),
        "all_atom_mask": torch.tensor(mask, dtype=torch.float64),
    }
    out = _of.atom37_to_torsion_angles("")(prot)  # curried transform
    return {k: out[k].numpy() for k in (
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
        "torsion_angles_mask",
    )}


@pytest.mark.parametrize("pdb", ["1crn.pdb", "1ubq.pdb"])
def test_torsion_sin_cos_matches_openfold(pdb):
    path = REPO_ROOT / "test-pdbs" / pdb
    if not path.exists():
        pytest.skip(f"{path} not found")
    pos, mask, resnames = _parse_atom37(path)

    mine = compute_torsion_angles_sin_cos(pos, mask, resnames)
    ref = _openfold_reference(pos, mask, resnames)

    # The torsion mask is integer-valued — it must match exactly.
    np.testing.assert_array_equal(
        mine["torsion_angles_mask"], ref["torsion_angles_mask"]
    )
    # Compare sin/cos where a torsion is defined (proteon float32 vs OpenFold f64).
    sel = ref["torsion_angles_mask"].astype(bool)
    assert sel.any()
    np.testing.assert_allclose(
        mine["torsion_angles_sin_cos"][sel],
        ref["torsion_angles_sin_cos"][sel],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        mine["alt_torsion_angles_sin_cos"][sel],
        ref["alt_torsion_angles_sin_cos"][sel],
        atol=1e-5,
    )
