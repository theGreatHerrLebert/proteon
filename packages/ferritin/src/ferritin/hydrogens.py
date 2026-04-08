"""Peptide backbone hydrogen placement.

Places amide H atoms on backbone nitrogen of non-N-terminal, non-proline
amino acid residues. Required for accurate Kabsch-Sander H-bond detection
and DSSP secondary structure assignment.

Functions:
    place_peptide_hydrogens     — add backbone N-H to a single structure
    batch_place_peptide_hydrogens — parallel placement on many structures
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import ferritin_connector
    _add_h = ferritin_connector.py_add_hydrogens
except ImportError:
    _add_h = None


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


def place_peptide_hydrogens(
    structure,
    *,
    return_coords: bool = False,
) -> Tuple[int, int] | Tuple[Tuple[int, int], NDArray[np.float64]]:
    """Place peptide backbone hydrogen atoms on a protein structure.

    Adds amide H atoms to backbone N for non-N-terminal, non-proline
    amino acid residues. Modifies the structure in place.

    Uses the DSSP bisector method: H is placed at 1.02 A from N along
    the bisector of vectors C(i-1)->N and CA->N.

    Idempotent: calling twice does not add duplicate H atoms.

    Args:
        structure: A ferritin Structure (modified in place).
        return_coords: If True, also return Nx3 array of placed H positions.

    Returns:
        (n_added, n_skipped) if return_coords is False.
        ((n_added, n_skipped), coords_array) if return_coords is True.

    Examples:
        >>> added, skipped = ferritin.place_peptide_hydrogens(structure)
        >>> print(f"Placed {added} backbone H atoms")

        >>> (added, skipped), coords = ferritin.place_peptide_hydrogens(
        ...     structure, return_coords=True
        ... )

    Agent Notes:
        PREFER: Call before backbone_hbonds() or dssp() to use real H
            positions instead of virtual ones.
        WATCH: Backbone amide H only. Does not place sidechain or
            terminal hydrogens.
    """
    ptr = _get_ptr(structure)
    if return_coords:
        return _add_h.place_peptide_hydrogens_with_coords(ptr)
    return _add_h.place_peptide_hydrogens(ptr)


def batch_place_peptide_hydrogens(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Place peptide backbone H atoms on many structures in parallel.

    Args:
        structures: List of ferritin Structure objects.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of (n_added, n_skipped) tuples, one per structure.
    """
    ptrs = [_get_ptr(s) for s in structures]
    return _add_h.batch_place_peptide_hydrogens(ptrs, n_threads)
