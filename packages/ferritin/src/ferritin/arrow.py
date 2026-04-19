"""Arrow/Parquet export and import for protein structures.

Converts structures to columnar Arrow format for zero-copy interop with
pandas, polars, DuckDB, Spark, and Parquet files.

    >>> import ferritin
    >>> s = ferritin.load("1crn.pdb")
    >>> ipc = ferritin.to_arrow(s, "1crn")      # → bytes (Arrow IPC)
    >>> ferritin.to_parquet(s, "out.parquet")     # → direct Parquet file
    >>>
    >>> # Read with any Arrow-compatible tool:
    >>> import pyarrow as pa
    >>> table = pa.ipc.open_file(ipc).read_all()
    >>> df = table.to_pandas()

The IPC bytes can also be read by polars:

    >>> import polars as pl
    >>> df = pl.read_ipc(ipc)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from .structure import Structure

if TYPE_CHECKING:
    pass

try:
    import ferritin_connector

    _arrow = ferritin_connector.py_arrow
except ImportError:  # pragma: no cover
    _arrow = None


def _check_available():
    if _arrow is None:
        raise ImportError(
            "ferritin Arrow support requires the ferritin-connector native extension. "
            "Rebuild with: cd ferritin-connector && maturin develop --release"
        )


def _get_ptr(s):
    """Extract PyPDB pointer from Structure or raw PyPDB."""
    if isinstance(s, Structure):
        return s.get_py_ptr()
    return s


def to_arrow(
    structure: Structure,
    structure_id: Optional[str] = None,
) -> bytes:
    """Export a structure to Arrow IPC bytes (per-atom schema).

    Returns bytes readable by ``pyarrow.ipc.open_file()`` or
    ``polars.read_ipc()``.

    Args:
        structure: The structure to export.
        structure_id: Identifier for the structure in the batch.
            Defaults to "unknown" if not provided.

    Returns:
        Arrow IPC file bytes containing one RecordBatch with columns:
        structure_id, model, chain_id, residue_name, residue_serial,
        insertion_code, conformer_id, atom_name, atom_serial, element,
        x, y, z, b_factor, occupancy, is_hetero, is_backbone.
    """
    _check_available()
    return _arrow.to_arrow_ipc(_get_ptr(structure), structure_id)


def to_structure_arrow(
    structure: Structure,
    structure_id: Optional[str] = None,
) -> bytes:
    """Export a structure summary to Arrow IPC bytes (one row per structure).

    Args:
        structure: The structure to summarize.
        structure_id: Identifier.

    Returns:
        Arrow IPC bytes with columns: structure_id, atom_count,
        residue_count, chain_count, model_count, chains.
    """
    _check_available()
    return _arrow.to_structure_arrow_ipc(_get_ptr(structure), structure_id)


def from_arrow(data: bytes) -> List[Tuple[str, Structure]]:
    """Import structures from Arrow IPC bytes (per-atom schema).

    Round-trips with ``to_arrow()``: PDB → Arrow → PDB.

    Args:
        data: Arrow IPC bytes produced by ``to_arrow()``.

    Returns:
        List of (structure_id, Structure) tuples.

    Raises:
        RuntimeError: If the Arrow data does not have the expected atom schema.

    Agent Notes:
        WATCH: Expects ferritin's per-atom schema, not an arbitrary Arrow table.
        INVARIANT: Round-trip preserves atom order so per-atom arrays still align.
    """
    _check_available()
    pairs = _arrow.from_arrow_ipc(data)
    return [(sid, Structure.from_py_ptr(pdb)) for sid, pdb in pairs]


def to_parquet(
    structure: Structure,
    path: str,
    structure_id: Optional[str] = None,
) -> None:
    """Write a structure directly to a Parquet file (Zstd compressed).

    No Python dependencies required — writes from Rust directly.

    Args:
        structure: The structure to write.
        path: Output file path.
        structure_id: Identifier for the structure in the file.
    """
    _check_available()
    _arrow.to_parquet(_get_ptr(structure), path, structure_id)


def from_parquet(path: str) -> List[Tuple[str, Structure]]:
    """Import structures from a Parquet file (per-atom schema).

    Round-trips with ``to_parquet()``: PDB → Parquet → PDB.

    Args:
        path: Path to a Parquet file written by ``to_parquet()`` or
            ``ferritin-ingest``.

    Returns:
        List of (structure_id, Structure) tuples.

    Raises:
        RuntimeError: If the file cannot be read or lacks the expected schema.

    Agent Notes:
        WATCH: Expects a ferritin atom-schema Parquet file, not a generic Parquet table.
        COST: Reads entire file into memory. For large files, filter with
            DuckDB/Polars first, then load the subset via from_arrow().
    """
    _check_available()
    pairs = _arrow.from_parquet(str(path))
    return [(sid, Structure.from_py_ptr(pdb)) for sid, pdb in pairs]
