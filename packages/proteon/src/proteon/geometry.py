"""Geometry building blocks: Kabsch superposition, RMSD, secondary structure, TM-score.

These are general-purpose functions useful beyond alignment — for MD analysis,
docking post-processing, point cloud fitting, etc.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import proteon_connector

    _geo = proteon_connector.py_geometry
except ImportError:  # pragma: no cover
    _geo = None


def kabsch_superpose(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute optimal superposition of x onto y (Kabsch algorithm).

    Args:
        x: Nx3 coordinates (mobile).
        y: Nx3 coordinates (reference).

    Returns:
        (rmsd, rotation_3x3, translation_3) after optimal superposition.

    """
    return _geo.kabsch_superpose(np.ascontiguousarray(x, dtype=np.float64),
                                  np.ascontiguousarray(y, dtype=np.float64))


def rmsd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """RMSD after optimal Kabsch superposition."""
    return _geo.rmsd(np.ascontiguousarray(x, dtype=np.float64),
                     np.ascontiguousarray(y, dtype=np.float64))


def rmsd_no_super(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """RMSD without superposition (direct coordinate comparison)."""
    return _geo.rmsd_no_super(np.ascontiguousarray(x, dtype=np.float64),
                              np.ascontiguousarray(y, dtype=np.float64))


def apply_transform(
    coords: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply rotation + translation: y = R @ x + t.

    Args:
        coords: Nx3 coordinates.
        rotation: 3x3 rotation matrix.
        translation: Length-3 translation vector.

    Returns:
        Nx3 transformed coordinates.
    """
    return _geo.apply_transform(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(rotation, dtype=np.float64),
        np.ascontiguousarray(translation, dtype=np.float64),
    )


def assign_secondary_structure(coords: NDArray[np.float64]) -> str:
    """Assign secondary structure from CA coordinates (distance geometry).

    Returns string of H (helix), E (sheet), T (turn), C (coil).
    """
    return _geo.assign_secondary_structure(
        np.ascontiguousarray(coords, dtype=np.float64)
    )


def tm_score(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    invmap: NDArray[np.int32],
) -> Tuple[float, int, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute TM-score for a pre-existing alignment.

    Args:
        x: Nx3 structure 1 coordinates.
        y: Mx3 structure 2 coordinates (normalization reference).
        invmap: Length-M array. invmap[j] = i means y[j] aligns to x[i], -1 = gap.

    Returns:
        (tm_score, n_aligned, rmsd, rotation_3x3, translation_3).
    """
    return _geo.tm_score(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(invmap, dtype=np.int32),
    )
