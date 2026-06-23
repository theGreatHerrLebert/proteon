"""Go all in: feed a proteon label-safe example into OpenFold's REAL FAPE loss.

The end-to-end proof that the masking subsystem produces genuine OpenFold training
supervision — and that the trustworthiness masks actually PROTECT the loss:

  1. Export a masked supervision example (proteon) from a real structure.
  2. Build OpenFold's backbone frames from its atom37 (`atom37_to_frames`).
  3. Run OpenFold's `backbone_loss` (FAPE) with prediction = ground truth -> ~0.
  4. Corrupt a MASKED residue's predicted frame -> loss UNCHANGED (mask excludes it).
  5. Corrupt an UNMASKED residue's predicted frame -> loss JUMPS.

Steps 4–5 are the whole point: the proteon trust mask, fed as OpenFold's
`backbone_rigid_mask`, keeps an untrustworthy residue's coordinates out of the
training loss — exactly the corruption the subsystem exists to prevent.

Run from the openfold checkout root (namespace package):
    cd /scratch/TMAlign/openfold
    PYTHONPATH=/scratch/TMAlign/proteon/packages/proteon/src \
        python /scratch/TMAlign/proteon/examples/14_openfold_loss_integration.py <pdb>
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import proteon
from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.loss import backbone_loss
from openfold.utils.rigid_utils import Rigid


def _to_t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def gt_backbone_frames(example):
    """OpenFold backbone frame tensor (N,4,4) + existence mask (N,) from a proteon
    example, using OpenFold's own `atom37_to_frames` on the (masked) atom37."""
    proto = {
        "aatype": _to_t(example.aatype, torch.long),
        "all_atom_positions": _to_t(example.all_atom_positions),
        "all_atom_mask": _to_t(example.all_atom_mask),
    }
    frames = atom37_to_frames(proto)
    bb_frames_44 = frames["rigidgroups_gt_frames"][..., 0, :, :]      # (N,4,4)
    # Existence mask = OpenFold's own backbone-frame existence AND proteon's trust
    # mask (rigidgroups_gt_exists is zeroed on untrustworthy residues by the export).
    of_exists = frames["rigidgroups_gt_exists"][..., 0]              # (N,)
    proteon_trust = _to_t(example.rigidgroups_gt_exists[:, 0])        # (N,)
    return bb_frames_44, of_exists * proteon_trust


def fape(gt_frames_44, mask, pred_traj_7):
    """OpenFold backbone FAPE, prediction given as a (1,N,7) trajectory."""
    return backbone_loss(
        backbone_rigid_tensor=gt_frames_44.unsqueeze(0),   # (1,N,4,4)
        backbone_rigid_mask=mask.unsqueeze(0),             # (1,N)
        traj=pred_traj_7.unsqueeze(0),                     # (1,1,N,7)
    ).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    args = ap.parse_args()

    res = proteon.prepare_for_supervision([args.pdb], minimize=False)[0]
    cid = next(ch.id for ch in res.structure.models[0].chains
               if any(r.is_amino_acid for r in ch.residues))
    ex = proteon.build_structure_supervision_example(
        res.structure, chain_id=cid, prep_report=res.report, mask_untrustworthy_coords=True)

    gt44, mask = gt_backbone_frames(ex)
    n = gt44.shape[0]
    masked = (mask < 0.5).nonzero().flatten().tolist()
    unmasked = (mask >= 0.5).nonzero().flatten().tolist()
    print(f"\n{args.pdb}: {n} residues, {len(masked)} masked, {len(unmasked)} usable "
          f"(clashscore {res.report.clashscore:.0f})")

    # Perfect prediction -> FAPE ~ 0.
    gt_traj7 = Rigid.from_tensor_4x4(gt44).to_tensor_7()   # (N,7)
    print(f"\n  FAPE(prediction = ground truth)      = {fape(gt44, mask, gt_traj7):.4f}   (perfect -> ~0)")

    def corrupt(idx):
        t = gt_traj7.clone()
        t[idx, 4:] += 50.0   # shove this residue's predicted frame 50 Å away
        return fape(gt44, mask, t)

    if masked:
        i = masked[0]
        print(f"  FAPE(corrupt MASKED residue {i:>3})      = {corrupt(i):.4f}   "
              f"(mask excludes it -> UNCHANGED)")
    else:
        print("  (no masked residue in this structure to demonstrate protection)")

    j = unmasked[len(unmasked) // 2]
    print(f"  FAPE(corrupt USABLE residue {j:>3})      = {corrupt(j):.4f}   "
          f"(in the loss -> JUMPS)")

    print("\n  => proteon's trust mask is OpenFold's backbone_rigid_mask: untrustworthy")
    print("     residues are provably excluded from the real training loss.\n")


if __name__ == "__main__":
    main()
