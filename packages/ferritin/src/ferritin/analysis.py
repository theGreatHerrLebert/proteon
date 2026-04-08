"""Structural analysis functions — Rust-accelerated with batch parallelism.

Single-structure functions work on either Structure objects (via Rust) or
raw numpy arrays (pure Python fallback).

Batch functions use rayon for true multi-core parallelism with GIL release.

Single-structure:
    distance_matrix     — pairwise distance matrix (numpy or Rust)
    contact_map         — boolean contact matrix (numpy or Rust)
    backbone_dihedrals  — phi / psi / omega torsion angles (Rust)
    dihedral_angle      — general dihedral from four points (numpy)
    centroid            — geometric center (Rust or numpy)
    radius_of_gyration  — Rg (Rust or numpy)
    extract_ca_coords   — CA coordinates (Rust)
    to_dataframe        — export to pandas/polars

Batch-parallel (rayon, GIL released):
    batch_extract_ca        — CA coords for N structures
    batch_distance_matrices — distance matrices for N structures
    batch_contact_maps      — contact maps for N structures
    batch_dihedrals         — backbone dihedrals for N structures
    batch_radius_of_gyration — Rg for N structures
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import ferritin_connector

    _analysis = ferritin_connector.py_analysis
except ImportError:
    _analysis = None


# ---------------------------------------------------------------------------
# Rust-accelerated single-structure functions
# ---------------------------------------------------------------------------


def extract_ca_coords(structure) -> NDArray[np.float64]:
    """Extract CA (alpha carbon) coordinates from a structure.

    Uses Rust for speed. Returns Mx3 numpy array.

    Examples:
        >>> ca = ferritin.extract_ca_coords(structure)
        >>> print(f"{len(ca)} residues")

    """
    if _analysis is not None and hasattr(structure, 'get_py_ptr'):
        return np.asarray(_analysis.extract_ca_coords(structure.get_py_ptr()))
    # Fallback: pure Python
    names = structure.atom_names
    coords = structure.coords
    mask = np.array([n.strip() == "CA" for n in names])
    return coords[mask]


def backbone_dihedrals(
    structure,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute backbone phi, psi, omega angles for each residue.

    Uses Rust for speed. Returns (phi, psi, omega) arrays in degrees.
    NaN for undefined angles (chain termini).

    Examples:
        >>> phi, psi, omega = ferritin.backbone_dihedrals(structure)
        >>> valid = ~np.isnan(phi)
        >>> print(f"Mean phi: {phi[valid].mean():.1f}")

    Agent Notes:
        WATCH: NaN at chain termini. Always filter with ~np.isnan()
            before computing statistics.
    """
    if _analysis is not None and hasattr(structure, 'get_py_ptr'):
        phi, psi, omega = _analysis.backbone_dihedrals(structure.get_py_ptr())
        return np.asarray(phi), np.asarray(psi), np.asarray(omega)
    # Fallback: pure Python (kept for array-only usage)
    return _backbone_dihedrals_python(structure)


def centroid(coords_or_structure) -> NDArray[np.float64]:
    """Centroid (geometric center).

    Accepts a Structure (Rust path) or Nx3 numpy array (numpy path).
    """
    if _analysis is not None and hasattr(coords_or_structure, 'get_py_ptr'):
        return np.asarray(_analysis.centroid(coords_or_structure.get_py_ptr()))
    c = np.asarray(coords_or_structure, dtype=np.float64)
    return np.mean(c, axis=0)


def radius_of_gyration(coords_or_structure) -> float:
    """Radius of gyration in Angstroms.

    Accepts a Structure (Rust path) or Nx3 numpy array (numpy path).
    """
    if _analysis is not None and hasattr(coords_or_structure, 'get_py_ptr'):
        return _analysis.radius_of_gyration(coords_or_structure.get_py_ptr())
    c = np.asarray(coords_or_structure, dtype=np.float64)
    center = np.mean(c, axis=0)
    diff = c - center
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ---------------------------------------------------------------------------
# Functions that work on numpy arrays (pure Python)
# ---------------------------------------------------------------------------


def distance_matrix(
    coords: NDArray[np.float64],
    coords2: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Pairwise Euclidean distance matrix.

    Args:
        coords: Nx3 coordinate array.
        coords2: Optional Mx3 array for NxM cross-distances.

    Returns:
        Distance matrix in Angstroms.

    Agent Notes:
        COST: N*N float64. 1000 residues = 8MB, 10000 = 800MB.
        PREFER: contact_map() for boolean contacts (8x less memory).
    """
    a = np.asarray(coords, dtype=np.float64)
    b = a if coords2 is None else np.asarray(coords2, dtype=np.float64)
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True)
    dist_sq = a_sq + b_sq.T - 2.0 * (a @ b.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return np.sqrt(dist_sq)


def contact_map(
    coords: NDArray[np.float64],
    cutoff: float = 8.0,
    coords2: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.bool_]:
    """Boolean contact map at a distance cutoff.

    Args:
        coords: Nx3 coordinate array.
        cutoff: Distance threshold in Angstroms (default 8.0).
        coords2: Optional Mx3 array for cross-contacts.

    Returns:
        Boolean matrix where True means distance <= cutoff.

    """
    dm = distance_matrix(coords, coords2)
    return dm <= cutoff


def dihedral_angle(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dihedral angle defined by four points (vectorized).

    Returns angles in degrees.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True)
    n1_norm = np.where(n1_norm < 1e-10, 1.0, n1_norm)
    n2_norm = np.where(n2_norm < 1e-10, 1.0, n2_norm)
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    b2_hat = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-10)
    m1 = np.cross(n1, b2_hat)

    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)

    return np.degrees(np.arctan2(-y, x))


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def to_dataframe(structure, engine: str = "pandas"):
    """Export structure to a DataFrame (one row per atom).

    Args:
        structure: A ferritin Structure.
        engine: "pandas" or "polars".

    Returns:
        DataFrame with columns: atom_name, element, residue_name,
        residue_number, chain_id, x, y, z, b_factor, occupancy.

    Agent Notes:
        WATCH: atom_name has leading/trailing spaces (PDB format).
            Always use df.atom_name.str.strip() when filtering.
    """
    coords = structure.coords
    data = {
        "atom_name": structure.atom_names,
        "element": structure.elements,
        "residue_name": structure.residue_names,
        "residue_number": np.asarray(structure.residue_serial_numbers),
        "chain_id": structure.chain_ids,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "b_factor": np.asarray(structure.b_factors),
        "occupancy": np.asarray(structure.occupancies),
    }

    if engine == "pandas":
        import pandas as pd
        return pd.DataFrame(data)
    elif engine == "polars":
        import polars as pl
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'pandas' or 'polars'.")


# ---------------------------------------------------------------------------
# Batch-parallel functions (Rust + rayon, GIL released)
# ---------------------------------------------------------------------------


def _get_ptrs(structures):
    """Get PyPDB pointers from Structure objects."""
    return [s.get_py_ptr() for s in structures]


def batch_extract_ca(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[NDArray[np.float64]]:
    """Extract CA coordinates from many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of Mx3 numpy arrays.

    Examples:
        >>> all_ca = ferritin.batch_extract_ca(structures, n_threads=-1)
    """
    ptrs = _get_ptrs(structures)
    return [np.asarray(a) for a in _analysis.batch_extract_ca(ptrs, n_threads)]


def batch_distance_matrices(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[NDArray[np.float64]]:
    """Compute CA distance matrices for many structures in parallel (Rust + rayon).

    Returns:
        List of NxN numpy arrays.
    """
    ptrs = _get_ptrs(structures)
    return [np.asarray(a) for a in _analysis.batch_distance_matrices(ptrs, n_threads)]


def batch_contact_maps(
    structures: Sequence,
    cutoff: float = 8.0,
    *,
    n_threads: Optional[int] = None,
) -> List[NDArray[np.bool_]]:
    """Compute CA contact maps for many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        cutoff: Distance threshold in Angstroms (default 8.0).
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of NxN boolean arrays.
    """
    ptrs = _get_ptrs(structures)
    return [np.asarray(a) for a in _analysis.batch_contact_maps(ptrs, cutoff, n_threads)]


def batch_dihedrals(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """Compute backbone dihedrals for many structures in parallel (Rust + rayon).

    Returns:
        List of (phi, psi, omega) tuples.
    """
    ptrs = _get_ptrs(structures)
    results = _analysis.batch_dihedrals(ptrs, n_threads)
    return [(np.asarray(p), np.asarray(s), np.asarray(o)) for p, s, o in results]


def batch_radius_of_gyration(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute radius of gyration for many structures in parallel (Rust + rayon).

    Returns:
        1D numpy array of Rg values.
    """
    ptrs = _get_ptrs(structures)
    return np.asarray(_analysis.batch_radius_of_gyration(ptrs, n_threads))


# ---------------------------------------------------------------------------
# Load + Analyze (full pipeline in Rust, zero GIL)
# ---------------------------------------------------------------------------


def load_and_analyze(
    paths: Sequence,
    cutoff: float = 8.0,
    *,
    n_threads: Optional[int] = None,
) -> List[dict]:
    """Load files and compute all analysis in one parallel call.

    Entire pipeline runs in Rust with rayon — file I/O, parsing,
    CA extraction, distance matrix, contact map, dihedrals, Rg.
    Zero GIL contention. This is the fastest way to process many structures.

    Files that fail to load are silently skipped.

    Args:
        paths: List of file paths (.pdb, .cif).
        cutoff: Contact map cutoff in Angstroms (default 8.0).
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of dicts, each containing:
            index, path, n_atoms, n_chains, n_residues, n_ca, rg,
            ca_coords (Mx3), distance_matrix (MxM), contact_map (MxM),
            phi, psi, omega (1D arrays).

    Examples:
        >>> results = ferritin.load_and_analyze(pdb_files, n_threads=-1)
        >>> for r in results:
        ...     print(f"{r['path']}: {r['n_ca']} CA, Rg={r['rg']:.1f}")

    """
    str_paths = [str(p) for p in paths]
    return _analysis.load_and_analyze(str_paths, cutoff, n_threads)


def load_and_extract_ca(
    paths: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, NDArray[np.float64]]]:
    """Load files and extract CA coordinates in one parallel call.

    Args:
        paths: List of file paths.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of (index, Mx3 array) tuples.
    """
    str_paths = [str(p) for p in paths]
    return [(i, np.asarray(a)) for i, a in _analysis.load_and_extract_ca(str_paths, n_threads)]


def load_and_contact_maps(
    paths: Sequence,
    cutoff: float = 8.0,
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, NDArray[np.bool_]]]:
    """Load files and compute contact maps in one parallel call.

    Args:
        paths: List of file paths.
        cutoff: Distance cutoff in Angstroms.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of (index, MxM bool array) tuples.
    """
    str_paths = [str(p) for p in paths]
    return [(i, np.asarray(a)) for i, a in _analysis.load_and_contact_maps(str_paths, cutoff, n_threads)]


# ---------------------------------------------------------------------------
# Pure-Python backbone dihedrals fallback (for non-Structure inputs)
# ---------------------------------------------------------------------------


def _backbone_dihedrals_python(structure):
    """Pure-Python fallback for backbone dihedrals."""
    all_phi = []
    all_psi = []
    all_omega = []

    for chain in structure.chains:
        n_coords = []
        ca_coords = []
        c_coords = []

        for residue in chain.residues:
            if not residue.is_amino_acid:
                continue
            n_pos = ca_pos = c_pos = None
            for atom in residue.atoms:
                name = atom.name.strip()
                if name == "N":
                    n_pos = atom.pos
                elif name == "CA":
                    ca_pos = atom.pos
                elif name == "C":
                    c_pos = atom.pos
            if n_pos is not None and ca_pos is not None and c_pos is not None:
                n_coords.append(n_pos)
                ca_coords.append(ca_pos)
                c_coords.append(c_pos)

        n_res = len(n_coords)
        if n_res == 0:
            continue

        n_arr = np.array(n_coords)
        ca_arr = np.array(ca_coords)
        c_arr = np.array(c_coords)

        phi = np.full(n_res, np.nan)
        psi = np.full(n_res, np.nan)
        omega = np.full(n_res, np.nan)

        if n_res > 1:
            phi[1:] = dihedral_angle(c_arr[:-1], n_arr[1:], ca_arr[1:], c_arr[1:])
        if n_res > 1:
            psi[:-1] = dihedral_angle(n_arr[:-1], ca_arr[:-1], c_arr[:-1], n_arr[1:])
        if n_res > 1:
            omega[1:] = dihedral_angle(ca_arr[:-1], c_arr[:-1], n_arr[1:], ca_arr[1:])

        all_phi.append(phi)
        all_psi.append(psi)
        all_omega.append(omega)

    if not all_phi:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    return np.concatenate(all_phi), np.concatenate(all_psi), np.concatenate(all_omega)
