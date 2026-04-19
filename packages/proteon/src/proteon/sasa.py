"""Solvent Accessible Surface Area (SASA) — Shrake-Rupley algorithm.

Rust implementation with rayon parallelism. Computes per-atom and per-residue
SASA using the Shrake-Rupley numerical dot method.

Functions:
    atom_sasa          — per-atom SASA (Angstroms²)
    residue_sasa       — per-residue SASA (sum of atom contributions)
    relative_sasa      — RSA (residue SASA / max SASA for residue type)
    total_sasa         — total SASA of a structure
    batch_total_sasa   — total SASA for many structures in parallel
    load_and_sasa      — load files + compute SASA in one parallel call
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import proteon_connector

    _sasa = proteon_connector.py_sasa
except ImportError:  # pragma: no cover
    _sasa = None


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


def atom_sasa(
    structure,
    probe: float = 1.4,
    n_points: int = 960,
    radii: str = "bondi",
) -> NDArray[np.float64]:
    """Compute per-atom SASA using Shrake-Rupley algorithm.

    Args:
        structure: A proteon Structure.
        probe: Probe radius in Angstroms (default 1.4 for water).
        n_points: Test points per sphere (default 960; higher = more precise).
        radii: Atomic radii table. "bondi" (default, element-based) or
            "protor" (atom-type-based, matches FreeSASA/NACCESS).

    Returns:
        1D numpy array of per-atom SASA in Angstroms².

    Examples:
        >>> sasa = proteon.atom_sasa(structure)
        >>> sasa_protor = proteon.atom_sasa(structure, radii="protor")

    Agent Notes:
        PREFER: batch_total_sasa() for multiple structures. Do not loop in Python.
        COST: ~12ms for 327 atoms, ~230ms for 58K atoms.
    """
    return np.asarray(_sasa.atom_sasa(_get_ptr(structure), probe, n_points, radii))


def residue_sasa(
    structure,
    probe: float = 1.4,
    n_points: int = 960,
    radii: str = "bondi",
) -> NDArray[np.float64]:
    """Compute per-residue SASA (sum of atom contributions per residue).

    Args:
        structure: A proteon Structure.
        probe: Probe radius in Angstroms (default 1.4).
        n_points: Test points per sphere (default 960).
        radii: Atomic radii table ("bondi" or "protor").

    Returns:
        1D numpy array of per-residue SASA in Angstroms².
    """
    return np.asarray(_sasa.residue_sasa(_get_ptr(structure), probe, n_points, radii))


def relative_sasa(
    structure,
    probe: float = 1.4,
    n_points: int = 960,
    radii: str = "bondi",
) -> NDArray[np.float64]:
    """Compute relative solvent accessibility (RSA) per residue.

    RSA = residue_SASA / max_SASA_for_residue_type (Tien et al. 2013).
    Values > 0.25 are typically considered "exposed".

    Args:
        structure: A proteon Structure.
        probe: Probe radius in Angstroms (default 1.4).
        n_points: Test points per sphere (default 960).

    Returns:
        1D numpy array of RSA values (0.0–1.0+). NaN for non-standard residues.

    Agent Notes:
        WATCH: RSA can exceed 1.0 at chain termini — not an error.
            NaN = non-standard residue (no reference max SASA). Filter first.
    """
    return np.asarray(_sasa.relative_sasa(_get_ptr(structure), probe, n_points, radii))


def total_sasa(
    structure,
    probe: float = 1.4,
    n_points: int = 960,
    radii: str = "bondi",
) -> float:
    """Total SASA of a structure in Angstroms².

    Args:
        structure: A proteon Structure.
        probe: Probe radius in Angstroms (default 1.4).
        n_points: Test points per sphere (default 960).
        radii: Atomic radii table ("bondi" or "protor").

    Returns:
        Total SASA in Angstroms².

    Examples:
        >>> print(f"Total SASA: {proteon.total_sasa(structure):.0f} A²")
        >>> # Match FreeSASA exactly:
        >>> print(f"ProtOr SASA: {proteon.total_sasa(structure, radii='protor'):.0f} A²")
    """
    return _sasa.total_sasa(_get_ptr(structure), probe, n_points, radii)


def batch_total_sasa(
    structures: Sequence,
    probe: float = 1.4,
    n_points: int = 960,
    *,
    n_threads: Optional[int] = None,
    radii: str = "bondi",
) -> NDArray[np.float64]:
    """Compute total SASA for many structures in parallel (Rust + rayon).

    Args:
        structures: List of proteon Structure objects.
        probe: Probe radius in Angstroms (default 1.4).
        n_points: Test points per sphere (default 960).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.
        radii: Atomic radii table ("bondi" or "protor").

    Returns:
        1D numpy array of total SASA values.
    """
    ptrs = [_get_ptr(s) for s in structures]
    return np.asarray(_sasa.batch_total_sasa(ptrs, probe, n_points, n_threads, radii))


def load_and_sasa(
    paths: Sequence,
    probe: float = 1.4,
    n_points: int = 960,
    *,
    n_threads: Optional[int] = None,
    radii: str = "bondi",
) -> List[Tuple[int, float]]:
    """Load files and compute total SASA in one parallel call (zero GIL).

    Args:
        paths: List of file paths.
        probe: Probe radius (default 1.4).
        n_points: Test points per sphere (default 960).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (index, total_sasa) tuples for files that loaded.

    Examples:
        >>> results = proteon.load_and_sasa(pdb_files, n_threads=-1)
        >>> for idx, sasa in results:
        ...     print(f"{pdb_files[idx]}: {sasa:.0f} A²")

    Agent Notes:
        WATCH: Failed inputs are skipped. Use the returned indices to map SASA
            values back to the original path list.
        PREFER: Use this for large file batches to avoid Python-side load loops.
    """
    str_paths = [str(p) for p in paths]
    return _sasa.load_and_sasa(str_paths, probe, n_points, n_threads, radii)
