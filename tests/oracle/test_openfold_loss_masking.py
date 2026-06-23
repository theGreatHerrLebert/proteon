"""Oracle: proteon's trust mask provably protects OpenFold's REAL FAPE loss.

The end-to-end guarantee of the masking subsystem — fed into OpenFold's own
`backbone_loss`, an untrustworthy residue (altloc / severe clash) is excluded from
the training loss: corrupting its predicted coordinates does NOT change FAPE,
while corrupting a usable residue's does. proteon's masked
`rigidgroups_gt_exists` IS OpenFold's `backbone_rigid_mask`.

Needs `torch` + a checked-out `openfold` (neither a proteon/CI dependency), so it
is skipped unless both import. The sibling `openfold/` checkout (read-only oracle
under the TMAlign workspace) is added to the path if present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the sibling openfold checkout importable if it isn't already (namespace
# package at <workspace>/openfold, alongside the proteon repo).
_REPO = Path(__file__).resolve().parents[2]
for _cand in (_REPO.parent / "openfold",):
    if (_cand / "openfold").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

torch = pytest.importorskip("torch")
pytest.importorskip("openfold.utils.loss")

from openfold.data.data_transforms import atom37_to_frames  # noqa: E402
from openfold.utils.loss import backbone_loss  # noqa: E402
from openfold.utils.rigid_utils import Rigid  # noqa: E402

try:
    import proteon
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"proteon unavailable: {exc}", allow_module_level=True)

PDBS = _REPO / "test-pdbs"


def _example(name):
    res = proteon.prepare_for_supervision([str(PDBS / f"{name}.pdb")], minimize=False)[0]
    cid = next(ch.id for ch in res.structure.models[0].chains
               if any(r.is_amino_acid for r in ch.residues))
    return proteon.build_structure_supervision_example(
        res.structure, chain_id=cid, prep_report=res.report, mask_untrustworthy_coords=True)


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def _backbone(example):
    proto = {
        "aatype": _t(example.aatype, torch.long),
        "all_atom_positions": _t(example.all_atom_positions),
        "all_atom_mask": _t(example.all_atom_mask),
    }
    fr = atom37_to_frames(proto)
    gt44 = fr["rigidgroups_gt_frames"][..., 0, :, :]
    mask = fr["rigidgroups_gt_exists"][..., 0] * _t(example.rigidgroups_gt_exists[:, 0])
    return gt44, mask


def _fape(gt44, mask, traj7):
    return backbone_loss(
        backbone_rigid_tensor=gt44.unsqueeze(0),
        backbone_rigid_mask=mask.unsqueeze(0),
        traj=traj7.unsqueeze(0),
    ).item()


@pytest.mark.parametrize("name", ["1bpi", "4hhb"])  # altloc-masked, clash-masked
def test_trust_mask_protects_openfold_fape(name):
    ex = _example(name)
    gt44, mask = _backbone(ex)
    masked = (mask < 0.5).nonzero().flatten().tolist()
    usable = (mask >= 0.5).nonzero().flatten().tolist()
    assert masked, f"{name} should have at least one masked residue to test protection"

    gt7 = Rigid.from_tensor_4x4(gt44).to_tensor_7()
    base = _fape(gt44, mask, gt7)
    assert base < 0.05  # perfect prediction -> ~0

    def corrupt(idx):
        t = gt7.clone()
        t[idx, 4:] += 50.0  # shove the predicted frame 50 Å away
        return _fape(gt44, mask, t)

    # A MASKED residue is excluded -> FAPE unchanged.
    assert abs(corrupt(masked[0]) - base) < 1e-4
    # A USABLE residue is in the loss -> FAPE jumps.
    assert corrupt(usable[len(usable) // 2]) > base + 0.01
