"""Geometry helpers for structure supervision export.

These helpers are pure-NumPy today so the public contract can stabilize
without introducing a DL dependency.

They are intentionally factored out from `supervision.py` so the future
Rust-side batch implementation has a clean semantic target.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .supervision_constants import (
    ATOM_ORDER,
    CHI_ANGLES_ATOMS,
    CHI_PI_PERIODIC,
    RESIDUE_ATOM_RENAMING_SWAPS,
    atom14_names_or_unk,
)

# Per-torsion sign convention for the 7 AlphaFold torsions
# [pre_omega, phi, psi, chi1, chi2, chi3, chi4]. psi is negated because it is
# computed from the carbonyl O (not the next residue's N) — matching OpenFold.
_TORSION_SIGN = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

# CA–CA above this (Å) is a backbone break, not a peptide bond — the same
# threshold proteon.backbone_dihedrals uses for chain-break NaN-ing. Used to veto
# a *numbering*-adjacent residue pair whose CAs are physically disconnected
# (e.g. an interleaved insertion code 80 Å away — corpus icode_interleave).
_CA_BOND_MAX = 4.5


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
    alt = compute_atom14_alt(residues, positions, mask)

    return {
        "positions": positions,
        "mask": mask,
        "exists": exists,
        "to_atom37": to_atom37,
        "from_atom37": from_atom37,
        "ambiguous": ambiguous,
        "alt_positions": alt["alt_positions"],
        "alt_mask": alt["alt_mask"],
    }


def compute_atom14_alt(
    residues,
    positions: NDArray,
    mask: NDArray,
) -> Dict[str, NDArray]:
    """Compute alternate atom14 positions by swapping symmetric atom pairs.

    For non-ambiguous residues, alt = original. For ASP/GLU/PHE/TYR,
    the swap pairs from RESIDUE_ATOM_RENAMING_SWAPS are exchanged.
    """
    alt_positions = positions.copy()
    alt_mask = mask.copy()
    for i, residue in enumerate(residues):
        resname = (residue.name or "UNK").strip().upper()
        swap = RESIDUE_ATOM_RENAMING_SWAPS.get(resname, {})
        if not swap:
            continue
        atom14_names = atom14_names_or_unk(resname)
        name_to_idx = {name: idx for idx, name in enumerate(atom14_names) if name}
        for src, dst in swap.items():
            src_idx = name_to_idx.get(src)
            dst_idx = name_to_idx.get(dst)
            if src_idx is not None and dst_idx is not None:
                alt_positions[i, src_idx] = positions[i, dst_idx]
                alt_positions[i, dst_idx] = positions[i, src_idx]
                alt_mask[i, src_idx] = mask[i, dst_idx]
                alt_mask[i, dst_idx] = mask[i, src_idx]
    return {"alt_positions": alt_positions, "alt_mask": alt_mask}


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


def _torsion_sin_cos(a0, a1, a2, a3) -> NDArray[np.float64]:
    """`(sin, cos)` of the dihedral about the `a1`–`a2` axis defined by `a0`,`a3`,
    via the AlphaFold/OpenFold frame projection (`Rigid.from_3_points` + invert +
    apply, taking the 4th atom's local `(z, y)`). Equivalent to the standard
    dihedral but returned as a normalized `(sin, cos)` pair without `atan2`."""
    a0, a1, a2, a3 = (np.asarray(p, dtype=np.float64) for p in (a0, a1, a2, a3))
    e0 = a2 - a1
    e0 = e0 / np.sqrt((e0 * e0).sum() + 1e-8)
    e1 = a0 - a2
    e1 = e1 - e0 * float(e1 @ e0)
    e1 = e1 / np.sqrt((e1 * e1).sum() + 1e-8)
    e2 = np.cross(e0, e1)
    d = a3 - a2
    sin, cos = float(e2 @ d), float(e1 @ d)
    norm = np.sqrt(sin * sin + cos * cos + 1e-8)
    return np.array([sin / norm, cos / norm], dtype=np.float64)


# A residue's primary-structure identity for peptide-bond adjacency:
# (chain_id, sequence_number, insertion_code). `None` marks a gap row (e.g. an
# unaligned query position in the template path) — never bonded to anything.
ResidueKey = Tuple[str, int, Optional[str]]


def _icode_rank(icode: Optional[str]) -> Optional[int]:
    """Ordinal of a PDB insertion code: blank/`None` → 0, `A` → 1, … `Z` → 26.
    An unrecognized code returns `None` → treated as a chain break (conservative:
    never forge a bond we can't order)."""
    if icode is None:
        return 0
    s = icode.strip().upper()
    if s == "":
        return 0
    if len(s) == 1 and "A" <= s <= "Z":
        return ord(s) - ord("A") + 1
    return None


def _peptide_adjacent(prev: ResidueKey, cur: ResidueKey) -> bool:
    """Whether `cur` directly follows `prev` in primary structure (a peptide bond
    is expected). Same chain, and either the next insertion code within the same
    sequence number (10 → 10A, 10A → 10B) or the next sequence number with no
    insertion code (… → 11). Insertion-code-aware, so antibody-style numbering
    (100, 100A, 100B, 101) stays bonded instead of breaking at every inserted
    residue — which raw `serial_number` does, since 100 and 100A share it."""
    pc, ps, pi = prev
    cc, cs, ci = cur
    if pc != cc:
        return False
    pr, cr = _icode_rank(pi), _icode_rank(ci)
    if cs == ps and pr is not None and cr is not None and cr == pr + 1:
        return True
    if cs == ps + 1 and cr == 0:
        return True
    return False


def residue_key(residue) -> ResidueKey:
    """`(chain_id, serial_number, insertion_code)` for a structure residue.
    `chain_id`/`insertion_code` are read defensively (default `""`/`None`) so
    minimal residue-like objects without them fall back to serial-number-only
    continuity — the pre-insertion-code behaviour."""
    return (
        getattr(residue, "chain_id", ""),
        int(residue.serial_number),
        getattr(residue, "insertion_code", None),
    )


def continuity_index_from_keys(keys: Sequence[Optional[ResidueKey]]) -> NDArray:
    """Monotonic per-residue index where a step of exactly `+1` marks a peptide
    bond to the previous row and any larger jump marks a chain break — the form
    `compute_torsion_angles_sin_cos` consumes as `residue_index` to mask
    `pre_omega`/`phi` at breaks. `keys[i] is None` (a gap/unaligned row) always
    breaks. Insertion-code-aware via `_peptide_adjacent`.

    Caveat: this is numbering-based. A residue physically missing between two
    rows whose numbers are still consecutive (e.g. 10 and 11 with 10A absent)
    cannot be detected here — only backbone geometry could. Documented, deferred.
    """
    out = np.empty(len(keys), dtype=np.int64)
    counter = 0
    prev: Optional[ResidueKey] = None
    for i, k in enumerate(keys):
        if i:
            if prev is not None and k is not None and _peptide_adjacent(prev, k):
                counter += 1
            else:
                counter += 2  # any non-bond: jump >1 so the +1 adjacency test fails
        out[i] = counter
        prev = k
    return out


def continuity_index(residues) -> NDArray:
    """`continuity_index_from_keys` over a residue list — the supervision-path
    continuity signal. Use instead of raw `serial_number`, which collapses
    insertion codes and so spuriously breaks `pre_omega`/`phi` at 10/10A."""
    return continuity_index_from_keys([residue_key(r) for r in residues])


def _ca_break_state(prev_pos, prev_mask, cur_pos, cur_mask, ica):
    """Tri-state CA–CA continuity between two residues: `True` (bonded), `False`
    (CAs too far → backbone break), or `None` (a CA missing → unmeasurable, defer
    to numbering). Lets a numbering-adjacent pair be vetoed when its CAs are
    physically disconnected, without over-masking when geometry can't be read."""
    if prev_mask[ica] <= 0 or cur_mask[ica] <= 0:
        return None
    d = prev_pos[ica] - cur_pos[ica]
    # Inclusive: only CA-CA strictly > _CA_BOND_MAX is a break, matching
    # proteon.backbone_dihedrals (so exactly 4.5 A stays bonded).
    return bool(float(d @ d) <= _CA_BOND_MAX * _CA_BOND_MAX)


def compute_torsion_angles_sin_cos(
    positions: NDArray, mask: NDArray, resnames, residue_index=None
) -> Dict[str, NDArray]:
    """AlphaFold/OpenFold-format torsion supervision from atom37.

    Returns `torsion_angles_sin_cos` / `alt_torsion_angles_sin_cos` `(N, 7, 2)`
    and `torsion_angles_mask` `(N, 7)` for the 7 torsions
    `[pre_omega, phi, psi, chi1, chi2, chi3, chi4]`, reproducing OpenFold's
    `atom37_to_torsion_angles` (frame projection, the `[1,1,-1,1,1,1,1]` sign
    convention, and `chi_pi_periodic` for the 180°-symmetric alt). `positions`
    `(N, 37, 3)` + `mask` `(N, 37)` are atom37; `resnames` are the 3-letter codes.

    `pre_omega`/`phi` are computed from residue `i-1`. When `residue_index` is
    given they are masked at a **chain break**: a break is either a numbering
    discontinuity (`residue_index[i] != [i-1]+1`) **or** a geometric one (CA–CA >
    `_CA_BOND_MAX`), so a pair that is numbering-adjacent but physically
    disconnected — e.g. an interleaved insertion code 80 Å away — is still masked.
    Pass a `continuity_index` (insertion-code-aware), *not* raw `serial_number`,
    so a genuinely bonded 10/10A doesn't read as a break. With `residue_index=None`
    the behaviour is row-adjacency with no masking (OpenFold-exact, used by the
    parity test on gap-free meshes).
    """
    n = len(resnames)
    iN, iCA, iC, iO = ATOM_ORDER["N"], ATOM_ORDER["CA"], ATOM_ORDER["C"], ATOM_ORDER["O"]
    sin_cos = np.zeros((n, 7, 2), dtype=np.float64)
    tmask = np.zeros((n, 7), dtype=np.float64)

    for i in range(n):
        rn = (resnames[i] or "UNK").strip().upper()
        cur = positions[i]
        cm = mask[i]
        # Bonded to the previous row iff numbering says adjacent AND geometry
        # doesn't veto it (CAs not physically disconnected). residue_index=None is
        # OpenFold-exact row-adjacency with no masking at all (the parity path).
        prev_bonded = i > 0 and (
            residue_index is None
            or (
                int(residue_index[i]) == int(residue_index[i - 1]) + 1
                and _ca_break_state(positions[i - 1], mask[i - 1], cur, cm, iCA) is not False
            )
        )
        if prev_bonded:
            prev, pm = positions[i - 1], mask[i - 1]
            p_ca, p_c, pm_ca, pm_c = prev[iCA], prev[iC], pm[iCA], pm[iC]
        else:
            p_ca = p_c = np.zeros(3)
            pm_ca = pm_c = 0.0

        # pre_omega [prev_CA, prev_C, N, CA]; phi [prev_C, N, CA, C]; psi [N, CA, C, O].
        sin_cos[i, 0] = _torsion_sin_cos(p_ca, p_c, cur[iN], cur[iCA])
        tmask[i, 0] = pm_ca * pm_c * cm[iN] * cm[iCA]
        sin_cos[i, 1] = _torsion_sin_cos(p_c, cur[iN], cur[iCA], cur[iC])
        tmask[i, 1] = pm_c * cm[iN] * cm[iCA] * cm[iC]
        sin_cos[i, 2] = _torsion_sin_cos(cur[iN], cur[iCA], cur[iC], cur[iO])
        tmask[i, 2] = cm[iN] * cm[iCA] * cm[iC] * cm[iO]

        for ci, atom_names in enumerate(CHI_ANGLES_ATOMS.get(rn, ())[:4]):
            idx = [ATOM_ORDER[a] for a in atom_names]
            sin_cos[i, 3 + ci] = _torsion_sin_cos(*(cur[k] for k in idx))
            tmask[i, 3 + ci] = float(np.prod([cm[k] for k in idx]))

    sin_cos *= _TORSION_SIGN[None, :, None]

    # Alt torsions: mirror (chi → chi+π, i.e. negate sin & cos) the 180°-symmetric
    # chi of ASP/GLU/PHE/TYR; backbone + non-symmetric chi are unchanged.
    pi_periodic = np.zeros((n, 7), dtype=np.float64)
    for i in range(n):
        per = CHI_PI_PERIODIC.get((resnames[i] or "UNK").strip().upper())
        if per is not None:
            pi_periodic[i, 3:7] = per
    mirror = 1.0 - 2.0 * pi_periodic
    alt = sin_cos * mirror[:, :, None]

    return {
        "torsion_angles_sin_cos": sin_cos.astype(np.float32),
        "alt_torsion_angles_sin_cos": alt.astype(np.float32),
        "torsion_angles_mask": tmask.astype(np.float32),
    }


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
