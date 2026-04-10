"""Pythonic wrappers for structure I/O.

Includes batch_load for parallel loading with rayon (GIL released).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .structure import Structure

try:
    import ferritin_connector

    _io = ferritin_connector.py_io
except ImportError:
    _io = None


def load(path: Union[str, Path]) -> Structure:
    """Load a structure from a PDB or mmCIF file.

    Format is auto-detected from the file extension.

    Args:
        path: Path to the file (.pdb, .cif, .mmcif).

    Returns:
        Structure: The parsed structure.

    Agent Notes:
        PREFER: batch_load() for multiple files. Do not loop in Python.
        WATCH: residue_count includes water and ligands. Amino acid count
            may be lower.
    """
    ptr = _io.load(str(path))
    return Structure.from_py_ptr(ptr)


def load_pdb(path: Union[str, Path]) -> Structure:
    """Load a structure, forcing PDB format."""
    ptr = _io.load_pdb(str(path))
    return Structure.from_py_ptr(ptr)


def load_mmcif(path: Union[str, Path]) -> Structure:
    """Load a structure, forcing mmCIF format."""
    ptr = _io.load_mmcif(str(path))
    return Structure.from_py_ptr(ptr)


def save(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure to a PDB or mmCIF file.

    Format is auto-detected from the file extension.

    Args:
        structure: The structure to save.
        path: Output file path (.pdb or .cif/.mmcif).
    """
    _io.save(structure.get_py_ptr(), str(path))


def save_pdb(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure, forcing PDB format."""
    _io.save_pdb(structure.get_py_ptr(), str(path))


def save_mmcif(structure: Structure, path: Union[str, Path]) -> None:
    """Save a structure, forcing mmCIF format."""
    _io.save_mmcif(structure.get_py_ptr(), str(path))


def batch_load(
    paths: Sequence[Union[str, Path]],
    *,
    n_threads: Optional[int] = None,
) -> List[Structure]:
    """Load many structures in parallel using rayon (GIL released).

    All file I/O, parsing, and PDB construction happens in Rust
    across multiple threads. Much faster than a Python loop.

    Args:
        paths: List of file paths (.pdb, .cif, .mmcif).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of Structure objects (same order as paths).

    Raises:
        IOError: If any file fails to load.

    Examples:
        >>> structures = ferritin.batch_load(glob.glob("pdbs/*.pdb"), n_threads=-1)
    """
    str_paths = [str(p) for p in paths]
    ptrs = _io.batch_load(str_paths, n_threads)
    return [Structure.from_py_ptr(ptr) for ptr in ptrs]


def batch_load_tolerant(
    paths: Sequence[Union[str, Path]],
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, Structure]]:
    """Load many structures in parallel, skipping failures.

    Same as batch_load but doesn't raise on individual failures.

    Args:
        paths: List of file paths.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (index, Structure) tuples for files that loaded successfully.
        The index refers to the position in the original paths list.

    Examples:
        >>> results = ferritin.batch_load_tolerant(all_files, n_threads=-1)
        >>> print(f"{len(results)}/{len(all_files)} loaded")

    Agent Notes:
        WATCH: Failures are skipped silently at the return-value level. Always use
            the returned indices to map loaded structures back to the original list.
        PREFER: Use this for archive-scale ingestion where partial success is acceptable.
    """
    str_paths = [str(p) for p in paths]
    pairs = _io.batch_load_tolerant(str_paths, n_threads)
    return [(i, Structure.from_py_ptr(ptr)) for i, ptr in pairs]
