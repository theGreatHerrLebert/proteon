"""Optional Rust backend hooks for structure supervision extraction.

The public supervision API remains in `proteon.supervision`. This module
provides a stable place to discover and call the future Rust/PyO3 fast path
without coupling callers to connector import details.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    import proteon_connector

    _supervision = getattr(proteon_connector, "py_supervision", None)
except ImportError:  # pragma: no cover
    _supervision = None


def rust_supervision_available() -> bool:
    """Whether the optional Rust supervision backend is available."""
    return _supervision is not None


def extract_structure_supervision_chain(structure, *, chain_id: Optional[str] = None):
    """Call the Rust single-chain supervision extractor.

    Returns the raw tensor dict from the connector backend. Provenance and
    quality metadata stay in the Python `StructureSupervisionExample` layer.
    """
    if _supervision is None:
        raise RuntimeError("Rust supervision backend is not available")
    ptr = structure.get_py_ptr() if hasattr(structure, "get_py_ptr") else structure
    return _supervision.extract_structure_supervision_chain(ptr, chain_id)


def batch_extract_structure_supervision(
    structures: Sequence,
    *,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
):
    """Call the Rust batch supervision extractor.

    The batch contract is defined in `devdocs/RUST_BATCH_SUPERVISION_CONTRACT.md` and
    returns padded batch-major NumPy arrays.
    """
    if _supervision is None:
        raise RuntimeError("Rust supervision backend is not available")
    ptrs = [s.get_py_ptr() if hasattr(s, "get_py_ptr") else s for s in structures]
    return _supervision.batch_extract_structure_supervision(ptrs, chain_ids)
