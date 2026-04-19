"""Geometry helpers for structure supervision export.

These helpers are pure-NumPy today so the public contract can stabilize
without introducing a DL dependency.

They are intentionally factored out from `supervision.py` so the future
Rust-side batch implementation has a clean semantic target.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .supervision_constants import (
    ATOM_ORDER,
    CHI_ANGLES_ATOMS,
    RESIDUE_ATOM_RENAMING_SWAPS,
    atom14_names_or_unk,
)


def extract_atom37(residues) -> Dict[str, NDArray]:
    n = len(residues)
    positions = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=np.float32)
    exists = np.zeros((n, 37), dtype=np.float32)
    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        atom_names = atom14_names_or_unk(resname)
        for atom_name in atom_names:
            if atom_name:
                exists[i, ATOM_ORDER[atom_name]] = 1.0
        observed = residue_atom_positions(residue)
        for atom_name, coord in observed.items():
            idx = ATOM_ORDER.get(atom_name)
            if idx is None:
                continue
            positions[i, idx] = coord
            mask[i, idx] = 1.0
    return {"positions": positions, "mask": mask, "exists": exists}


def extract_atom14(residues, atom37: Dict[str, NDArray]) -> Dict[str, NDArray]:
    n = len(residues)
    positions = np.zeros((n, 14, 3), dtype=np.float32)
    mask = np.zeros((n, 14), dtype=np.float32)
    exists = np.zeros((n, 14), dtype=np.float32)
    to_atom37 = np.zeros((n, 14), dtype=np.int32)
    from_atom37 = np.zeros((n, 37), dtype=np.int32)
    ambiguous = np.zeros((n, 14), dtype=np.float32)
    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        atom14_names = atom14_names_or_unk(resname)
        swap = RESIDUE_ATOM_RENAMING_SWAPS.get(resname, {})
        for a14, atom_name in enumerate(atom14_names):
            if not atom_name:
                continue
            a37 = ATOM_ORDER[atom_name]
            exists[i, a14] = 1.0
            to_atom37[i, a14] = a37
            from_atom37[i, a37] = a14
            positions[i, a14] = atom37["positions"][i, a37]
            mask[i, a14] = atom37["mask"][i, a37]
            if atom_name in swap or atom_name in swap.values():
                ambiguous[i, a14] = 1.0
    return {
        "positions": positions,
        "mask": mask,
        "exists": exists,
        "to_atom37": to_atom37,
        "from_atom37": from_atom37,
        "ambiguous": ambiguous,
    }


def compute_pseudo_beta(residues, atom37: Dict[str, NDArray]):
    n = len(residues)
    coords = np.zeros((n, 3), dtype=np.float32)
    mask = np.zeros((n,), dtype=np.float32)
    ca_idx = ATOM_ORDER["CA"]
    cb_idx = ATOM_ORDER["CB"]
    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        if resname == "GLY":
            if atom37["mask"][i, ca_idx] > 0:
                coords[i] = atom37["positions"][i, ca_idx]
                mask[i] = 1.0
        else:
            if atom37["mask"][i, cb_idx] > 0:
                coords[i] = atom37["positions"][i, cb_idx]
                mask[i] = 1.0
    return coords, mask


def compute_backbone_torsions(residues):
    n = len(residues)
    phi = np.zeros((n,), dtype=np.float32)
    psi = np.zeros((n,), dtype=np.float32)
    omega = np.zeros((n,), dtype=np.float32)
    phi_mask = np.zeros((n,), dtype=np.float32)
    psi_mask = np.zeros((n,), dtype=np.float32)
    omega_mask = np.zeros((n,), dtype=np.float32)
    atoms = [residue_atom_positions(r) for r in residues]
    for i in range(n):
        if i > 0:
            prev = atoms[i - 1]
            cur = atoms[i]
            if all(k in prev for k in ("C",)) and all(k in cur for k in ("N", "CA", "C")):
                phi[i] = dihedral(prev["C"], cur["N"], cur["CA"], cur["C"])
                phi_mask[i] = 1.0
            if all(k in prev for k in ("CA", "C")) and all(k in cur for k in ("N", "CA")):
                omega[i] = dihedral(prev["CA"], prev["C"], cur["N"], cur["CA"])
                omega_mask[i] = 1.0
        if i < n - 1:
            cur = atoms[i]
            nxt = atoms[i + 1]
            if all(k in cur for k in ("N", "CA", "C")) and all(k in nxt for k in ("N",)):
                psi[i] = dihedral(cur["N"], cur["CA"], cur["C"], nxt["N"])
                psi_mask[i] = 1.0
    return {
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "phi_mask": phi_mask,
        "psi_mask": psi_mask,
        "omega_mask": omega_mask,
    }


def compute_chi_angles(residues):
    n = len(residues)
    angles = np.zeros((n, 4), dtype=np.float32)
    mask = np.zeros((n, 4), dtype=np.float32)
    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        observed = residue_atom_positions(residue)
        for chi_i, atom_names in enumerate(CHI_ANGLES_ATOMS.get(resname, ())):
            if chi_i >= 4:
                break
            if all(name in observed for name in atom_names):
                angles[i, chi_i] = dihedral(
                    observed[atom_names[0]],
                    observed[atom_names[1]],
                    observed[atom_names[2]],
                    observed[atom_names[3]],
                )
                mask[i, chi_i] = 1.0
    return {"angles": angles, "mask": mask}


def compute_rigidgroups(residues):
    n = len(residues)
    frames = np.tile(np.eye(4, dtype=np.float32), (n, 8, 1, 1))
    gt_exists = np.zeros((n, 8), dtype=np.float32)
    group_exists = np.zeros((n, 8), dtype=np.float32)
    ambiguous = np.zeros((n, 8), dtype=np.float32)

    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        observed = residue_atom_positions(residue)
        group_exists[i, 0] = 1.0
        group_exists[i, 3] = 1.0

        for group_idx, atom_names in _rigidgroup_base_atoms(resname):
            if group_idx >= 4:
                group_exists[i, group_idx] = 1.0
            if all(name in observed for name in atom_names):
                frames[i, group_idx] = _homogeneous_frame(
                    observed[atom_names[0]],
                    observed[atom_names[1]],
                    observed[atom_names[2]],
                    mirror_backbone=(group_idx == 0),
                )
                gt_exists[i, group_idx] = 1.0

        if resname in RESIDUE_ATOM_RENAMING_SWAPS:
            last_chi = len(CHI_ANGLES_ATOMS.get(resname, ())) - 1
            if last_chi >= 0:
                ambiguous[i, 4 + last_chi] = 1.0

    return {
        "frames": frames,
        "gt_exists": gt_exists,
        "group_exists": group_exists,
        "ambiguous": ambiguous,
    }


def residue_atom_positions(residue) -> Dict[str, NDArray[np.float32]]:
    out: Dict[str, NDArray[np.float32]] = {}
    for atom in getattr(residue, "atoms", []):
        name = getattr(atom, "name", "").strip().upper()
        if not name or name.startswith("H") or name.startswith("D"):
            continue
        if name in out:
            continue
        if hasattr(atom, "pos"):
            coord = np.asarray(atom.pos, dtype=np.float32)
        else:
            coord = np.asarray([atom.x, atom.y, atom.z], dtype=np.float32)
        out[name] = coord
    return out


def dihedral(p0, p1, p2, p3) -> np.float32:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return np.float32(0.0)
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    b2_hat = b2 / max(np.linalg.norm(b2), 1e-8)
    m1 = np.cross(n1, b2_hat)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.float32(np.arctan2(-y, x))


def _rigidgroup_base_atoms(resname: str):
    out = [
        (0, ("C", "CA", "N")),
        (3, ("CA", "C", "O")),
    ]
    for chi_idx, atom_names in enumerate(CHI_ANGLES_ATOMS.get(resname, ())):
        if chi_idx >= 4:
            break
        out.append((4 + chi_idx, tuple(atom_names[1:])))
    return out


def _homogeneous_frame(point_on_neg_x_axis, origin, point_on_xy_plane, *, mirror_backbone: bool):
    point_on_neg_x_axis = np.asarray(point_on_neg_x_axis, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    point_on_xy_plane = np.asarray(point_on_xy_plane, dtype=np.float64)

    ex = origin - point_on_neg_x_axis
    ex_norm = np.linalg.norm(ex)
    if ex_norm < 1e-8:
        return np.eye(4, dtype=np.float32)
    ex = ex / ex_norm

    ey = point_on_xy_plane - origin
    ey = ey - np.dot(ey, ex) * ex
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-8:
        return np.eye(4, dtype=np.float32)
    ey = ey / ey_norm

    ez = np.cross(ex, ey)
    ez_norm = np.linalg.norm(ez)
    if ez_norm < 1e-8:
        return np.eye(4, dtype=np.float32)
    ez = ez / ez_norm

    if mirror_backbone:
        ex = -ex
        ez = -ez

    frame = np.eye(4, dtype=np.float32)
    frame[:3, 0] = ex.astype(np.float32)
    frame[:3, 1] = ey.astype(np.float32)
    frame[:3, 2] = ez.astype(np.float32)
    frame[:3, 3] = origin.astype(np.float32)
    return frame
