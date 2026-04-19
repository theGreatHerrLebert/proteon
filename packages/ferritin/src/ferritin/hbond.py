"""Hydrogen bond detection.

Two methods:
    backbone_hbonds     — Kabsch-Sander electrostatic energy (same as DSSP)
    geometric_hbonds    — distance-based detection for all polar atoms
    hbond_count         — per-residue H-bond participation count

Batch-parallel variants available for all functions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import ferritin_connector
    _hbond = ferritin_connector.py_hbond
except ImportError:  # pragma: no cover
    _hbond = None


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


def backbone_hbonds(
    structure,
    energy_cutoff: float = -0.5,
) -> NDArray[np.float64]:
    """Detect backbone hydrogen bonds using the Kabsch-Sander energy criterion.

    Same method used by DSSP. Computes electrostatic interaction energy
    between backbone C=O and N-H groups.

    Args:
        structure: A ferritin Structure.
        energy_cutoff: Energy threshold in kcal/mol (default -0.5).
            More negative = stricter. Good H-bonds are around -3 kcal/mol.

    Returns:
        Nx4 numpy array where each row is:
        [acceptor_residue_idx, donor_residue_idx, energy, O-N_distance]

    Examples:
        >>> hbonds = ferritin.backbone_hbonds(structure)
        >>> print(f"{len(hbonds)} backbone H-bonds")
        >>> strong = hbonds[hbonds[:, 2] < -2.0]  # filter strong bonds

    Agent Notes:
        PREFER: batch_backbone_hbonds() for multiple structures.
        WATCH: Backbone only (CO...HN). For sidechain H-bonds, use
            geometric_hbonds() instead.
    """
    return np.asarray(_hbond.backbone_hbonds(_get_ptr(structure), energy_cutoff))


def geometric_hbonds(
    structure,
    dist_cutoff: float = 3.5,
) -> NDArray[np.float64]:
    """Detect hydrogen bonds by donor-acceptor distance.

    Finds all N/O...O/S contacts below the distance cutoff.
    Works for backbone and sidechain atoms.

    Args:
        structure: A ferritin Structure.
        dist_cutoff: Donor-acceptor distance cutoff in Angstroms (default 3.5).

    Returns:
        Nx3 numpy array: [donor_atom_idx, acceptor_atom_idx, distance]

    Agent Notes:
        WATCH: Distance only, no angle criterion — produces false positives.
            For backbone H-bonds with proper energy criterion, use
            backbone_hbonds() instead.
    """
    return np.asarray(_hbond.geometric_hbonds(_get_ptr(structure), dist_cutoff))


def hbond_count(
    structure,
    energy_cutoff: float = -0.5,
) -> NDArray[np.uint32]:
    """Count backbone H-bonds per residue.

    Returns how many H-bonds each amino acid residue participates in
    (as either donor or acceptor).

    Args:
        structure: A ferritin Structure.
        energy_cutoff: Kabsch-Sander energy cutoff (default -0.5).

    Returns:
        1D array of length n_residues (amino acids only).

    Examples:
        >>> counts = ferritin.hbond_count(structure)
        >>> print(f"Max H-bonds per residue: {counts.max()}")
    """
    return np.asarray(_hbond.hbond_count_per_residue(_get_ptr(structure), energy_cutoff))


def batch_backbone_hbonds(
    structures: Sequence,
    energy_cutoff: float = -0.5,
    *,
    n_threads: Optional[int] = None,
) -> List[NDArray[np.float64]]:
    """Detect backbone H-bonds for many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        energy_cutoff: Kabsch-Sander energy cutoff (default -0.5).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of Nx4 arrays, one per structure.
    """
    ptrs = [_get_ptr(s) for s in structures]
    return [np.asarray(a) for a in _hbond.batch_backbone_hbonds(ptrs, energy_cutoff, n_threads)]
