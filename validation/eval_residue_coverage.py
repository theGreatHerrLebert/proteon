#!/usr/bin/env python
"""Per-residue COMPLETENESS coverage — calibration for the per-residue masking
gate (devdocs/PER_RESIDUE_MASKING_SKETCH.md).

The structure-level gate drops any structure with missing atoms. Per-residue
masking would instead keep the OBSERVED residues. To set the coverage threshold
(keep a structure iff >= X% of residues are complete), we need the distribution
of per-structure completeness on a diverse corpus.

A residue is "complete" iff every atom37 slot its type EXPECTS is present
(`mask` covers `exists`). This is the dominant driver of the structure-level
drops (reconstruct_failed:missing_atoms); clashes/altlocs touch far fewer
residues and are a second-order correction.

Usage: python validation/eval_residue_coverage.py [N]
"""

import glob
import json
import os
import sys

import numpy as np

import proteon
from proteon.supervision_geometry import ATOM_ORDER, extract_atom37

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdbs_10k")
#: atom37 slots for the backbone (N, CA, C, O) — NOT 0..3, which would include CB.
_BB_SLOTS = [ATOM_ORDER[a] for a in ("N", "CA", "C", "O")]


def coverage(path):
    """(n_res, n_complete, n_bb_complete) over model 0's amino-acid residues."""
    s = proteon.load(path)
    residues = []
    for ch in s.models[0].chains:
        residues.extend(r for r in ch.residues if r.is_amino_acid)
    if not residues:
        return None
    a = extract_atom37(residues)
    mask, exists = a["mask"], a["exists"]
    # complete: no EXPECTED atom is missing (exists==1 implies mask==1).
    per_res_missing = ((exists > 0) & (mask == 0)).sum(axis=1)
    complete = int((per_res_missing == 0).sum())
    # backbone-complete: N, CA, C, O all present.
    bb = int((mask[:, _BB_SLOTS].sum(axis=1) == 4).sum())
    return len(residues), complete, bb


if __name__ == "__main__":
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    paths = [p for p in sorted(glob.glob(os.path.join(CORPUS, "*.pdb"))) if os.path.exists(p)]
    if n:
        stride = max(1, len(paths) // n)
        paths = paths[::stride][:n]
    print(f"corpus: {len(paths)} structures", file=sys.stderr)

    per_struct_cov, tot_res, tot_complete, tot_bb = [], 0, 0, 0
    failed = 0
    for p in paths:
        try:
            r = coverage(p)
        except Exception:
            r = None
        if r is None:
            failed += 1
            continue
        nres, ncomp, nbb = r
        per_struct_cov.append(ncomp / nres)
        tot_res += nres
        tot_complete += ncomp
        tot_bb += nbb

    cov = np.array(per_struct_cov)
    out = {
        "structures": len(paths),
        "scored": len(per_struct_cov),
        "failed": failed,
        "residue_level_yield": {
            "total_residues": tot_res,
            "complete_residues": tot_complete,
            "frac_complete": round(tot_complete / max(tot_res, 1), 3),
            "frac_backbone_complete": round(tot_bb / max(tot_res, 1), 3),
        },
        "per_structure_coverage": {
            "median": round(float(np.median(cov)), 3),
            "p10": round(float(np.percentile(cov, 10)), 3),
            "p25": round(float(np.percentile(cov, 25)), 3),
            "mean": round(float(cov.mean()), 3),
        },
        # Fraction of structures whose per-residue completeness is >= each
        # candidate coverage gate (and the residues those structures contribute).
        "structures_at_coverage": {
            str(t): {
                "structures": int((cov >= t).sum()),
                "frac_structures": round(float((cov >= t).mean()), 3),
            }
            for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
        },
    }
    print(json.dumps(out, indent=2))
