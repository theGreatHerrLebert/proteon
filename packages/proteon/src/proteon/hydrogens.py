"""Hydrogen placement and fragment reconstruction.

Functions:
    place_peptide_hydrogens      — backbone N-H only
    place_all_hydrogens          — backbone + sidechain (standard AA)
    place_general_hydrogens      — all atoms including ligands
    reconstruct_fragments        — add missing heavy atoms from templates
    batch_place_peptide_hydrogens — parallel backbone H placement
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import proteon_connector
    _add_h = proteon_connector.py_add_hydrogens
except ImportError:  # pragma: no cover
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
        structure: A proteon Structure (modified in place).
        return_coords: If True, also return Nx3 array of placed H positions.

    Returns:
        (n_added, n_skipped) if return_coords is False.
        ((n_added, n_skipped), coords_array) if return_coords is True.

    Examples:
        >>> added, skipped = proteon.place_peptide_hydrogens(structure)
        >>> print(f"Placed {added} backbone H atoms")

        >>> (added, skipped), coords = proteon.place_peptide_hydrogens(
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
        structures: List of proteon Structure objects.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (n_added, n_skipped) tuples, one per structure.
    """
    ptrs = [_get_ptr(s) for s in structures]
    return _add_h.batch_place_peptide_hydrogens(ptrs, n_threads)


def place_all_hydrogens(structure, polar_only: bool = False) -> Tuple[int, int]:
    """Place backbone + sidechain hydrogens on standard amino acids.

    Runs Phase 1 (backbone amide N-H) then Phase 2 (sidechain templates
    for all 20 standard amino acids). Modifies structure in place.
    Idempotent.

    Args:
        structure: A proteon Structure (modified in place).
        polar_only: If True, only place hydrogens bonded to N/O/S
            (guanidinium, amide, hydroxyl, thiol, imidazole, indole,
            and the N-terminal NH3+). Use this for polar-H united-atom
            force fields like CHARMM19+EEF1, where non-polar C-H atoms
            are absorbed into united carbon types and would fail FF
            type lookup. The C-terminal carboxylate is also kept
            deprotonated (COO-) under this mode, matching CHARMM's
            convention.

    Returns:
        (n_added, n_skipped) tuple.
    """
    return _add_h.place_all_hydrogens(_get_ptr(structure), polar_only)


def place_general_hydrogens(structure, include_water: bool = False) -> Tuple[int, int]:
    """Place all hydrogens including ligands and non-standard residues.

    Runs Phase 1 (backbone) + Phase 2 (sidechain templates) + Phase 3
    (general algorithm using bond orders, ring detection, MMFF94 bond
    lengths). Handles ligands, cofactors, and non-standard residues.

    Args:
        structure: A proteon Structure (modified in place).
        include_water: If True, also place H on water molecules.

    Returns:
        (n_added, n_skipped) tuple.
    """
    return _add_h.place_general_hydrogens(_get_ptr(structure), include_water)


def reconstruct_fragments(structure) -> int:
    """Reconstruct missing heavy atoms from fragment templates.

    Uses BALL FragmentDB templates for the 20 standard amino acids.
    Adds missing backbone and sidechain heavy atoms using 3-point
    rigid body superposition and BFS coordinate propagation.

    Args:
        structure: A proteon Structure (modified in place).

    Returns:
        Number of atoms added.
    """
    return _add_h.reconstruct_fragments(_get_ptr(structure))
