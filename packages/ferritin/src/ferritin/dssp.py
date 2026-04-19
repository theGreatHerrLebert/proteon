"""DSSP secondary structure assignment (Kabsch-Sander algorithm).

Native Rust implementation — no external DSSP binary needed.

Assigns per-residue secondary structure from backbone coordinates using
the Kabsch-Sander hydrogen bond energy criterion.

Functions:
    dssp                — SS string (H/G/I/E/B/T/S/C per residue)
    dssp_array          — SS as numpy u8 array
    batch_dssp          — SS for many structures in parallel
    load_and_dssp       — load files + compute SS in one parallel call
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import ferritin_connector
    _dssp = ferritin_connector.py_dssp
except ImportError:  # pragma: no cover
    _dssp = None


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


def dssp(structure) -> str:
    """Assign DSSP secondary structure.

    Uses the Kabsch-Sander hydrogen bond energy criterion to assign
    one-letter SS codes per amino acid residue:
        H = alpha helix (4-turn helix)
        G = 3-10 helix (3-turn helix)
        I = pi helix (5-turn helix)
        E = extended strand (part of beta ladder)
        B = isolated beta bridge
        T = hydrogen-bonded turn
        S = bend (CA angle > 70 degrees)
        C = coil (none of the above)

    No external DSSP binary needed — pure Rust implementation.

    Args:
        structure: A ferritin Structure.

    Returns:
        String of SS codes, one per amino acid residue.

    Examples:
        >>> ss = ferritin.dssp(structure)
        >>> print(ss)
        'CCCSSHHHHHHHHHHHCCCTHHHHHHTCCEEEEECCCCCCCTEEEEC'
        >>> n_helix = ss.count('H')

    Agent Notes:
        PREFER: batch_dssp() for multiple structures. Do not loop in Python.
        WATCH: CA-only structures return empty string — use
            assign_secondary_structure() instead for CA-only approximation.
    """
    return _dssp.dssp(_get_ptr(structure))


def dssp_array(structure) -> NDArray[np.uint8]:
    """Assign DSSP as a numpy array of ASCII character codes.

    Same as dssp() but returns numpy array for vectorized operations.

    Examples:
        >>> ss = ferritin.dssp_array(structure)
        >>> is_helix = ss == ord('H')
        >>> helix_fraction = is_helix.mean()
    """
    return np.asarray(_dssp.dssp_array(_get_ptr(structure)))


def batch_dssp(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[str]:
    """Compute DSSP for many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of SS strings.
    """
    ptrs = [_get_ptr(s) for s in structures]
    return _dssp.batch_dssp(ptrs, n_threads)


def load_and_dssp(
    paths: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Load files and compute DSSP in one parallel call (zero GIL).

    Args:
        paths: List of file paths.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (index, ss_string) tuples.

    Agent Notes:
        WATCH: Failed inputs are skipped. Use the returned indices to map DSSP
            strings back to the original path list.
        PREFER: Use this for many files instead of load() + dssp() in Python.
    """
    str_paths = [str(p) for p in paths]
    return _dssp.load_and_dssp(str_paths, n_threads)
