"""Combine a per-residue trustworthiness mask into a supervision example's
coordinate label masks (phase 2 of per-residue masking — the export last mile).

The coverage gate (phase 1) keeps mostly-complete structures, and the export's
*presence* masks already zero missing atoms per label. But presence cannot see
the *trustworthiness* hazards — a residue can have every atom present yet be a
severe clash, an arbitrary altloc pick, or a chirality outlier. Those corrupt
even the backbone, so they must zero the coordinate label masks. This module is
that step: ``apply_residue_trust_mask`` multiplies a per-residue trust bool into
each coordinate mask, with the neighbour dependency that mask actually has (NOT
one broadcast mask — torsions need valid neighbours, a residue label needs only
itself). See ``devdocs/PER_RESIDUE_MASKING_SKETCH.md``.

It does NOT touch ``seq_mask`` / ``aatype`` / ``residue_index`` (an untrustworthy
conformer's residue IDENTITY is still a valid sequence label), nor the type /
definition masks (``atom*_atom_exists``, ``rigidgroups_group_exists``) — only the
ground-truth coordinate label masks.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

#: torsion_angles_mask columns: [pre_omega, phi, psi, chi1, chi2, chi3, chi4].
#: pre_omega/phi read residue i-1. The AF/OpenFold `psi` torsion (col 2) is
#: RESIDUE-LOCAL — `compute_torsion_angles_sin_cos` builds it from the residue's
#: own [N, CA, C, O], NOT the next residue's N (that i+1 dependency belongs to the
#: separate classic `psi_mask` field). So col 2 and chi (3..6) use t[i] only.
_TORSION_PREV_COLS = (0, 1)  # pre_omega, phi

#: Ground-truth coordinate label masks that a trust bool zeroes, with dependency:
#: "self" → t[i]; "prev" → t[i]·t[i-1]; "next" → t[i]·t[i+1].
_SELF_MASKS = (
    "all_atom_mask",
    "atom14_gt_exists",
    "atom14_alt_gt_exists",
    "pseudo_beta_mask",
    "chi_mask",
    "rigidgroups_gt_exists",
)
_PREV_MASKS = ("phi_mask", "omega_mask")
_NEXT_MASKS = ("psi_mask",)


def _self_next_prev(trust: NDArray):
    """(t, t·t_next, t·t_prev) as float32 (N,). Endpoints keep their own value —
    the corresponding presence mask is already 0 there, so the absent neighbour
    term is vacuous."""
    t = trust.astype(np.float32)
    t_prev = t.copy()
    t_prev[1:] *= t[:-1]
    t_next = t.copy()
    t_next[:-1] *= t[1:]
    return t, t_next, t_prev


def apply_residue_trust_mask(example, trust: NDArray):
    """Return a copy of ``example`` with its coordinate label masks zeroed wherever
    ``trust[i]`` is False (with each mask's neighbour dependency).

    Args:
        example: a :class:`~proteon.supervision.StructureSupervisionExample`.
        trust: per-residue bool/0-1 array, length ``example.length``, aligned to
            ``example.residue_index`` (use :func:`proteon.residue_trustworthy`).

    Positions/angles are left intact — only the loss MASKS are zeroed, the
    standard convention. ``seq_mask`` and the type/definition masks are untouched.
    """
    trust = np.asarray(trust)
    if trust.shape != (example.length,):
        raise ValueError(
            f"trust shape {trust.shape} != (length,)=({example.length},)"
        )
    if trust.all():
        return example  # nothing untrustworthy — no copy needed

    t, t_next, t_prev = _self_next_prev(trust)
    updates = {}

    for name in _SELF_MASKS:
        m = getattr(example, name, None)
        if m is not None:
            mult = t.reshape((-1,) + (1,) * (m.ndim - 1))
            updates[name] = (m * mult).astype(m.dtype)
    for name, dep in ((n, t_prev) for n in _PREV_MASKS):
        m = getattr(example, name, None)
        if m is not None:
            updates[name] = (m * dep).astype(m.dtype)
    for name, dep in ((n, t_next) for n in _NEXT_MASKS):
        m = getattr(example, name, None)
        if m is not None:
            updates[name] = (m * dep).astype(m.dtype)

    tam = getattr(example, "torsion_angles_mask", None)
    if tam is not None:
        col = np.tile(t.reshape(-1, 1), (1, tam.shape[1]))  # default: self (psi, chi)
        for c in _TORSION_PREV_COLS:
            col[:, c] = t_prev  # pre_omega, phi read residue i-1
        updates["torsion_angles_mask"] = (tam * col).astype(tam.dtype)

    return dataclasses.replace(example, **updates)
