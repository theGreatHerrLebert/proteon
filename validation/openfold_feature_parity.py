"""Feature parity oracle: proteon supervision vs OpenFold data pipeline.

Loads the same PDB through both pipelines and compares every tensor
field by field. This is the structural-supervision oracle that proves
proteon's output is consumable by OpenFold without silent corruption.

## Key known difference

Amino acid ordering:
  - OpenFold: ARNDCQEGHILKMFPSTWYV (sorted by 3-letter code)
  - Proteon:  ACDEFGHIKLMNPQRSTVWY  (alphabetical 1-letter code)

All other constants (atom37, atom14, chi angles, swaps) match exactly.
The aatype indices must be mapped when comparing.

## Usage

    source venv/bin/activate
    python validation/openfold_feature_parity.py --pdb test-pdbs/1crn.pdb
    python validation/openfold_feature_parity.py --pdb-dir pdbs_50k --max-structures 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OPENFOLD_ROOT = Path("/globalscratch/dateschn/proteon-benchmark/openfold")

# ---------------------------------------------------------------------------
# AA ordering maps
# ---------------------------------------------------------------------------

# OpenFold's restype order (from residue_constants.py)
_OF_RESTYPES = "ARNDCQEGHILKMFPSTWYV"
_OF_AA_TO_IDX = {aa: i for i, aa in enumerate(_OF_RESTYPES)}

# Proteon's restype order
_PT_RESTYPES = "ACDEFGHIKLMNPQRSTVWY"
_PT_AA_TO_IDX = {aa: i for i, aa in enumerate(_PT_RESTYPES)}

# Mapping: proteon aatype index → OpenFold aatype index
_PT_TO_OF = np.array([_OF_AA_TO_IDX[aa] for aa in _PT_RESTYPES] + [20], dtype=np.int32)
# Mapping: OpenFold aatype index → proteon aatype index
_OF_TO_PT = np.array([_PT_AA_TO_IDX[aa] for aa in _OF_RESTYPES] + [20], dtype=np.int32)


def proteon_aatype_to_openfold(aatype: np.ndarray) -> np.ndarray:
    """Map proteon aatype indices to OpenFold ordering."""
    return _PT_TO_OF[aatype]


def openfold_aatype_to_proteon(aatype: np.ndarray) -> np.ndarray:
    """Map OpenFold aatype indices to proteon ordering."""
    return _OF_TO_PT[aatype]


# ---------------------------------------------------------------------------
# OpenFold feature extraction (pure NumPy, no torch)
# ---------------------------------------------------------------------------


def _load_openfold_protein(pdb_path: Path, chain_id: str = "A"):
    """Load a PDB file using OpenFold's protein parser, single chain."""
    sys.path.insert(0, str(OPENFOLD_ROOT))
    from openfold.np import protein, residue_constants
    from openfold.np.protein import from_pdb_string

    pdb_str = pdb_path.read_text()
    prot = from_pdb_string(pdb_str, chain_id=chain_id)
    return prot, residue_constants


def _openfold_features_from_protein(prot, rc) -> Dict[str, np.ndarray]:
    """Extract OpenFold-style feature dict from a Protein object.

    Mirrors OpenFold's make_atom14_masks + make_atom14_positions logic
    but in pure NumPy (no torch dependency for the comparison).
    """
    n = prot.aatype.shape[0]
    features = {
        "aatype": prot.aatype.astype(np.int32),
        "all_atom_positions": prot.atom_positions.astype(np.float32),
        "all_atom_mask": prot.atom_mask.astype(np.float32),
        "residue_index": prot.residue_index.astype(np.int32),
    }

    # Build atom14 masks using OpenFold's residue_constants tables
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        atom14_to_37 = []
        atom37_to_14 = np.full(37, -1, dtype=np.int32)
        mask = []
        for i, name in enumerate(atom_names):
            if name:
                a37 = rc.atom_order[name]
                atom14_to_37.append(a37)
                atom37_to_14[a37] = i
                mask.append(1.0)
            else:
                atom14_to_37.append(0)
                mask.append(0.0)
        restype_atom14_to_atom37.append(atom14_to_37)
        restype_atom37_to_atom14.append(atom37_to_14.tolist())
        restype_atom14_mask.append(mask)

    # UNK type
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([-1] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    aatype = features["aatype"]
    atom14_to_37 = restype_atom14_to_atom37[aatype]
    atom37_to_14 = restype_atom37_to_atom14[aatype]
    atom14_mask = restype_atom14_mask[aatype]

    features["residx_atom14_to_atom37"] = atom14_to_37
    features["residx_atom37_to_atom14"] = atom37_to_14
    features["atom14_atom_exists"] = atom14_mask

    # atom37_atom_exists from the standard atom mask table
    restype_atom37_mask = np.zeros((21, 37), dtype=np.float32)
    for i, rt in enumerate(rc.restypes):
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        for name in atom_names:
            if name:
                restype_atom37_mask[i, rc.atom_order[name]] = 1.0
    features["atom37_atom_exists"] = restype_atom37_mask[aatype]

    # atom14_gt_positions and atom14_gt_exists
    atom14_pos = np.zeros((n, 14, 3), dtype=np.float32)
    atom14_exists = np.zeros((n, 14), dtype=np.float32)
    for i in range(n):
        for a14 in range(14):
            a37 = atom14_to_37[i, a14]
            if atom14_mask[i, a14] > 0:
                atom14_pos[i, a14] = features["all_atom_positions"][i, a37]
                atom14_exists[i, a14] = features["all_atom_mask"][i, a37]

    features["atom14_gt_positions"] = atom14_pos
    features["atom14_gt_exists"] = atom14_exists

    # Ambiguity flags
    ambiguous = np.zeros((n, 14), dtype=np.float32)
    for i in range(n):
        rt_idx = aatype[i]
        if rt_idx >= 20:
            continue
        rt_name = rc.restype_1to3[rc.restypes[rt_idx]]
        swap = rc.residue_atom_renaming_swaps.get(rt_name, {})
        if not swap:
            continue
        atom_names = rc.restype_name_to_atom14_names[rt_name]
        name_to_a14 = {name: j for j, name in enumerate(atom_names) if name}
        for src, dst in swap.items():
            if src in name_to_a14:
                ambiguous[i, name_to_a14[src]] = 1.0
            if dst in name_to_a14:
                ambiguous[i, name_to_a14[dst]] = 1.0
    features["atom14_atom_is_ambiguous"] = ambiguous

    # Alt positions (swapped)
    alt_pos = atom14_pos.copy()
    alt_exists = atom14_exists.copy()
    for i in range(n):
        rt_idx = aatype[i]
        if rt_idx >= 20:
            continue
        rt_name = rc.restype_1to3[rc.restypes[rt_idx]]
        swap = rc.residue_atom_renaming_swaps.get(rt_name, {})
        if not swap:
            continue
        atom_names = rc.restype_name_to_atom14_names[rt_name]
        name_to_a14 = {name: j for j, name in enumerate(atom_names) if name}
        for src, dst in swap.items():
            si, di = name_to_a14.get(src), name_to_a14.get(dst)
            if si is not None and di is not None:
                alt_pos[i, si] = atom14_pos[i, di]
                alt_pos[i, di] = atom14_pos[i, si]
                alt_exists[i, si] = atom14_exists[i, di]
                alt_exists[i, di] = atom14_exists[i, si]
    features["atom14_alt_gt_positions"] = alt_pos
    features["atom14_alt_gt_exists"] = alt_exists

    # Pseudo-beta
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    gly_idx = rc.restype_order["G"]
    pseudo_beta = np.zeros((n, 3), dtype=np.float32)
    pseudo_beta_mask = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        if aatype[i] == gly_idx:
            if features["all_atom_mask"][i, ca_idx] > 0:
                pseudo_beta[i] = features["all_atom_positions"][i, ca_idx]
                pseudo_beta_mask[i] = 1.0
        else:
            if features["all_atom_mask"][i, cb_idx] > 0:
                pseudo_beta[i] = features["all_atom_positions"][i, cb_idx]
                pseudo_beta_mask[i] = 1.0
    features["pseudo_beta"] = pseudo_beta
    features["pseudo_beta_mask"] = pseudo_beta_mask

    return features


# ---------------------------------------------------------------------------
# Proteon feature extraction
# ---------------------------------------------------------------------------


def _proteon_features(pdb_path: Path) -> Dict[str, np.ndarray]:
    """Extract features using proteon's supervision pipeline."""
    import proteon

    structure = proteon.load(str(pdb_path))
    # Use first chain
    chain = structure.chains[0]
    ex = proteon.build_structure_supervision_example(structure, chain_id=chain.id)

    return {
        "aatype": ex.aatype,
        "all_atom_positions": ex.all_atom_positions,
        "all_atom_mask": ex.all_atom_mask,
        "atom37_atom_exists": ex.atom37_atom_exists,
        "atom14_gt_positions": ex.atom14_gt_positions,
        "atom14_gt_exists": ex.atom14_gt_exists,
        "atom14_atom_exists": ex.atom14_atom_exists,
        "atom14_atom_is_ambiguous": ex.atom14_atom_is_ambiguous,
        "atom14_alt_gt_positions": ex.atom14_alt_gt_positions,
        "atom14_alt_gt_exists": ex.atom14_alt_gt_exists,
        "residx_atom14_to_atom37": ex.residx_atom14_to_atom37,
        "residx_atom37_to_atom14": ex.residx_atom37_to_atom14,
        "pseudo_beta": ex.pseudo_beta,
        "pseudo_beta_mask": ex.pseudo_beta_mask,
        "residue_index": ex.residue_index,
        "sequence": ex.sequence,
        "length": ex.length,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_features(
    pt: Dict[str, np.ndarray],
    of: Dict[str, np.ndarray],
    pdb_name: str,
) -> Dict:
    """Compare proteon and OpenFold features field by field."""
    report = {"pdb": pdb_name, "pass": True, "fields": {}, "warnings": []}

    # Map proteon aatype to OpenFold ordering for comparison
    pt_aatype_mapped = proteon_aatype_to_openfold(pt["aatype"])
    of_aatype = of["aatype"]

    # Length check
    if len(pt_aatype_mapped) != len(of_aatype):
        report["pass"] = False
        report["warnings"].append(
            f"length mismatch: proteon={len(pt_aatype_mapped)}, openfold={len(of_aatype)}"
        )
        return report

    n = len(pt_aatype_mapped)
    report["length"] = n

    # 1. aatype (after mapping)
    aa_match = np.array_equal(pt_aatype_mapped, of_aatype)
    report["fields"]["aatype"] = {"match": aa_match}
    if not aa_match:
        n_diff = int(np.sum(pt_aatype_mapped != of_aatype))
        report["fields"]["aatype"]["n_diff"] = n_diff
        report["warnings"].append(f"aatype: {n_diff}/{n} differ after mapping")

    # 2. all_atom_positions — coordinate comparison with tolerance
    _compare_array(report, "all_atom_positions",
                   pt["all_atom_positions"], of["all_atom_positions"],
                   atol=1e-3, mask=pt["all_atom_mask"])

    # 3. all_atom_mask
    _compare_array(report, "all_atom_mask",
                   pt["all_atom_mask"], of["all_atom_mask"], exact=True)

    # 4. atom37_atom_exists
    _compare_array(report, "atom37_atom_exists",
                   pt["atom37_atom_exists"], of["atom37_atom_exists"], exact=True)

    # 5. atom14 fields
    _compare_array(report, "atom14_gt_positions",
                   pt["atom14_gt_positions"], of["atom14_gt_positions"],
                   atol=1e-3, mask=pt["atom14_gt_exists"])
    _compare_array(report, "atom14_gt_exists",
                   pt["atom14_gt_exists"], of["atom14_gt_exists"], exact=True)
    _compare_array(report, "atom14_atom_exists",
                   pt["atom14_atom_exists"], of["atom14_atom_exists"], exact=True)
    _compare_array(report, "atom14_atom_is_ambiguous",
                   pt["atom14_atom_is_ambiguous"], of["atom14_atom_is_ambiguous"], exact=True)

    # 6. Alt positions
    _compare_array(report, "atom14_alt_gt_positions",
                   pt["atom14_alt_gt_positions"], of["atom14_alt_gt_positions"],
                   atol=1e-3, mask=pt["atom14_alt_gt_exists"])
    _compare_array(report, "atom14_alt_gt_exists",
                   pt["atom14_alt_gt_exists"], of["atom14_alt_gt_exists"], exact=True)

    # 7. Index mappings
    _compare_array(report, "residx_atom14_to_atom37",
                   pt["residx_atom14_to_atom37"], of["residx_atom14_to_atom37"],
                   exact=True, mask=pt["atom14_atom_exists"])
    _compare_array(report, "residx_atom37_to_atom14",
                   pt["residx_atom37_to_atom14"], of["residx_atom37_to_atom14"],
                   exact=True, mask=pt["atom37_atom_exists"])

    # 8. Pseudo-beta
    _compare_array(report, "pseudo_beta",
                   pt["pseudo_beta"], of["pseudo_beta"],
                   atol=1e-3, mask=pt["pseudo_beta_mask"])
    _compare_array(report, "pseudo_beta_mask",
                   pt["pseudo_beta_mask"], of["pseudo_beta_mask"], exact=True)

    # Overall pass
    for field, info in report["fields"].items():
        if not info.get("match", True):
            report["pass"] = False

    return report


def _compare_array(
    report: Dict,
    name: str,
    pt: np.ndarray,
    of: np.ndarray,
    *,
    exact: bool = False,
    atol: float = 1e-4,
    mask: Optional[np.ndarray] = None,
):
    """Compare two arrays and record result in report."""
    if pt.shape != of.shape:
        report["fields"][name] = {
            "match": False,
            "reason": f"shape mismatch: {pt.shape} vs {of.shape}",
        }
        report["warnings"].append(f"{name}: shape {pt.shape} vs {of.shape}")
        return

    if mask is not None:
        # Broadcast mask to match array shape
        m = mask
        while m.ndim < pt.ndim:
            m = m[..., np.newaxis]
        m = np.broadcast_to(m, pt.shape) > 0
        pt_masked = pt[m]
        of_masked = of[m]
    else:
        pt_masked = pt.ravel()
        of_masked = of.ravel()

    if exact:
        match = np.array_equal(pt_masked, of_masked)
        n_diff = int(np.sum(pt_masked != of_masked)) if not match else 0
        report["fields"][name] = {"match": match, "n_diff": n_diff}
        if not match:
            report["warnings"].append(f"{name}: {n_diff}/{len(pt_masked)} values differ")
    else:
        if len(pt_masked) == 0:
            report["fields"][name] = {"match": True, "n_compared": 0}
            return
        max_diff = float(np.max(np.abs(pt_masked - of_masked)))
        mean_diff = float(np.mean(np.abs(pt_masked - of_masked)))
        match = max_diff <= atol
        report["fields"][name] = {
            "match": match,
            "max_diff": round(max_diff, 6),
            "mean_diff": round(mean_diff, 6),
            "n_compared": len(pt_masked),
        }
        if not match:
            report["warnings"].append(f"{name}: max_diff={max_diff:.6f} > {atol}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Feature parity: proteon vs OpenFold")
    parser.add_argument("--pdb", type=Path, help="Single PDB file")
    parser.add_argument("--pdb-dir", type=Path, help="Directory of PDB files")
    parser.add_argument("--max-structures", type=int, default=10)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.pdb:
        pdb_files = [args.pdb]
    elif args.pdb_dir:
        pdb_files = sorted(args.pdb_dir.glob("*.pdb"))[:args.max_structures]
    else:
        pdb_files = [Path("test-pdbs/1crn.pdb")]

    print(f"Comparing {len(pdb_files)} structures")
    print()

    reports = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    for pdb_path in pdb_files:
        try:
            pt = _proteon_features(pdb_path)
            # Extract chain_id from proteon's sequence field
            # proteon loads chain A by default for single-chain, or first chain
            import proteon as _proteon
            _s = _proteon.load(str(pdb_path))
            chain_id = _s.chains[0].id
            prot, rc = _load_openfold_protein(pdb_path, chain_id=chain_id)
            of = _openfold_features_from_protein(prot, rc)

            report = compare_features(pt, of, pdb_path.stem)
            reports.append(report)

            status = "PASS" if report["pass"] else "FAIL"
            if report["pass"]:
                n_pass += 1
            else:
                n_fail += 1

            warnings = report.get("warnings", [])
            warn_str = f" ({'; '.join(warnings[:3])})" if warnings else ""
            print(f"  {pdb_path.stem}: {status} (length={report.get('length', '?')}){warn_str}")

        except Exception as e:
            n_error += 1
            print(f"  {pdb_path.stem}: ERROR ({e})")

    print(f"\n{'='*50}")
    print(f"Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    if args.out_json:
        import json
        args.out_json.write_text(json.dumps(reports, indent=2))
        print(f"Report: {args.out_json}")


if __name__ == "__main__":
    main()
