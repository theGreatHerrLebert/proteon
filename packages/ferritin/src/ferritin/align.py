"""Pythonic wrappers for structural alignment.

Provides: TM-align, SOI-align (sequence-order independent),
FlexAlign (flexible hinge-based). Each with single-pair, one-to-many,
and many-to-many batch variants.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .core import RustWrapperObject
from .structure import Structure

try:
    import ferritin_connector

    _align = ferritin_connector.py_align_funcs
except ImportError:  # pragma: no cover
    _align = None


def _get_ptr(s):
    """Extract PyPDB pointer from either a Structure or raw PyPDB."""
    if isinstance(s, Structure):
        return s.get_py_ptr()
    return s


def _get_ptrs(lst):
    return [_get_ptr(s) for s in lst]


# ===========================================================================
# AlignResult (TM-align)
# ===========================================================================


class AlignResult(RustWrapperObject):
    """Result of a TM-align structural alignment."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> AlignResult:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def tm_score_chain1(self) -> float:
        return self._ptr.tm_score_chain1

    @property
    def tm_score_chain2(self) -> float:
        return self._ptr.tm_score_chain2

    @property
    def rmsd(self) -> float:
        return self._ptr.rmsd

    @property
    def n_aligned(self) -> int:
        return self._ptr.n_aligned

    @property
    def seq_identity(self) -> float:
        return self._ptr.seq_identity

    @property
    def rotation_matrix(self):
        return self._ptr.rotation_matrix

    @property
    def translation(self):
        return self._ptr.translation

    @property
    def aligned_seq_x(self) -> str:
        return self._ptr.aligned_seq_x

    @property
    def aligned_seq_y(self) -> str:
        return self._ptr.aligned_seq_y

    def __repr__(self) -> str:
        return repr(self._ptr)


# ===========================================================================
# SoiAlignResult (SOI-align)
# ===========================================================================


class SoiAlignResult(RustWrapperObject):
    """Result of a SOI-align (sequence-order independent) alignment."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> SoiAlignResult:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def tm_score_chain1(self) -> float:
        return self._ptr.tm_score_chain1

    @property
    def tm_score_chain2(self) -> float:
        return self._ptr.tm_score_chain2

    @property
    def rmsd(self) -> float:
        return self._ptr.rmsd

    @property
    def n_aligned(self) -> int:
        return self._ptr.n_aligned

    @property
    def seq_identity(self) -> float:
        return self._ptr.seq_identity

    @property
    def rotation_matrix(self):
        return self._ptr.rotation_matrix

    @property
    def translation(self):
        return self._ptr.translation

    @property
    def aligned_seq_x(self) -> str:
        return self._ptr.aligned_seq_x

    @property
    def aligned_seq_y(self) -> str:
        return self._ptr.aligned_seq_y

    def __repr__(self) -> str:
        return repr(self._ptr)


# ===========================================================================
# FlexAlignResult (FlexAlign)
# ===========================================================================


class FlexAlignResult(RustWrapperObject):
    """Result of a FlexAlign (flexible hinge-based) alignment."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> FlexAlignResult:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def tm_score_chain1(self) -> float:
        return self._ptr.tm_score_chain1

    @property
    def tm_score_chain2(self) -> float:
        return self._ptr.tm_score_chain2

    @property
    def rmsd(self) -> float:
        return self._ptr.rmsd

    @property
    def n_aligned(self) -> int:
        return self._ptr.n_aligned

    @property
    def seq_identity(self) -> float:
        return self._ptr.seq_identity

    @property
    def hinge_count(self) -> int:
        return self._ptr.hinge_count

    @property
    def rotation_matrices(self):
        """Kx3x3 numpy array (K = hinge_count + 1 segments)."""
        return self._ptr.rotation_matrices

    @property
    def translations(self):
        """Kx3 numpy array."""
        return self._ptr.translations

    @property
    def aligned_seq_x(self) -> str:
        return self._ptr.aligned_seq_x

    @property
    def aligned_seq_y(self) -> str:
        return self._ptr.aligned_seq_y

    def __repr__(self) -> str:
        return repr(self._ptr)


# ===========================================================================
# TM-align functions
# ===========================================================================


def tm_align(
    structure1: Structure,
    structure2: Structure,
    *,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
    fast: bool = False,
) -> AlignResult:
    """Align two structures using TM-align.

    Args:
        structure1: Query structure.
        structure2: Target structure.
        chain1: Chain ID from structure1 (None = auto).
        chain2: Chain ID from structure2 (None = auto).
        fast: Use fast approximate mode (fTM-align).

    Agent Notes:
        WATCH: Asymmetric — tm_align(A, B) != tm_align(B, A). For symmetric
            comparison, use max(result.tm_score_chain1, result.tm_score_chain2).
        PREFER: tm_align_one_to_many() or tm_align_many_to_many() for batch.
            mm_align() for multi-chain complexes.
    """
    r = _align.tm_align_pair(_get_ptr(structure1), _get_ptr(structure2), chain1, chain2, fast)
    return AlignResult.from_py_ptr(r)


def tm_align_one_to_many(
    query: Structure,
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[AlignResult]:
    """Align one query against many targets in parallel (TM-align)."""
    results = _align.tm_align_one_to_many(_get_ptr(query), _get_ptrs(targets), n_threads, chain, fast)
    return [AlignResult.from_py_ptr(r) for r in results]


def tm_align_many_to_many(
    queries: List[Structure],
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[Tuple[int, int, AlignResult]]:
    """Align all pairs between two lists in parallel (TM-align, Cartesian product)."""
    results = _align.tm_align_many_to_many(_get_ptrs(queries), _get_ptrs(targets), n_threads, chain, fast)
    return [(qi, ti, AlignResult.from_py_ptr(r)) for qi, ti, r in results]


# ===========================================================================
# SOI-align functions
# ===========================================================================


def soi_align(
    structure1: Structure,
    structure2: Structure,
    *,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
    fast: bool = False,
) -> SoiAlignResult:
    """Align two structures using SOI-align (sequence-order independent)."""
    r = _align.soi_align_pair(_get_ptr(structure1), _get_ptr(structure2), chain1, chain2, fast)
    return SoiAlignResult.from_py_ptr(r)


def soi_align_one_to_many(
    query: Structure,
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[SoiAlignResult]:
    """SOI-align one query against many targets in parallel."""
    results = _align.soi_align_one_to_many(_get_ptr(query), _get_ptrs(targets), n_threads, chain, fast)
    return [SoiAlignResult.from_py_ptr(r) for r in results]


def soi_align_many_to_many(
    queries: List[Structure],
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[Tuple[int, int, SoiAlignResult]]:
    """SOI-align all pairs between two lists in parallel (Cartesian product)."""
    results = _align.soi_align_many_to_many(_get_ptrs(queries), _get_ptrs(targets), n_threads, chain, fast)
    return [(qi, ti, SoiAlignResult.from_py_ptr(r)) for qi, ti, r in results]


# ===========================================================================
# FlexAlign functions
# ===========================================================================


def flex_align(
    structure1: Structure,
    structure2: Structure,
    *,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
    fast: bool = False,
) -> FlexAlignResult:
    """Align two structures using FlexAlign (flexible, hinge-based)."""
    r = _align.flex_align_pair(_get_ptr(structure1), _get_ptr(structure2), chain1, chain2, fast)
    return FlexAlignResult.from_py_ptr(r)


def flex_align_one_to_many(
    query: Structure,
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[FlexAlignResult]:
    """FlexAlign one query against many targets in parallel."""
    results = _align.flex_align_one_to_many(_get_ptr(query), _get_ptrs(targets), n_threads, chain, fast)
    return [FlexAlignResult.from_py_ptr(r) for r in results]


def flex_align_many_to_many(
    queries: List[Structure],
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
    chain: Optional[str] = None,
    fast: bool = False,
) -> List[Tuple[int, int, FlexAlignResult]]:
    """FlexAlign all pairs between two lists in parallel (Cartesian product)."""
    results = _align.flex_align_many_to_many(_get_ptrs(queries), _get_ptrs(targets), n_threads, chain, fast)
    return [(qi, ti, FlexAlignResult.from_py_ptr(r)) for qi, ti, r in results]


# ===========================================================================
# MM-align result wrappers
# ===========================================================================


class ChainPairResult(RustWrapperObject):
    """Per-chain-pair result within an MM-align result."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> ChainPairResult:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def query_chain(self) -> int:
        return self._ptr.query_chain

    @property
    def target_chain(self) -> int:
        return self._ptr.target_chain

    @property
    def tm_score(self) -> float:
        return self._ptr.tm_score

    @property
    def rmsd(self) -> float:
        return self._ptr.rmsd

    @property
    def n_aligned(self) -> int:
        return self._ptr.n_aligned

    @property
    def aligned_seq_x(self) -> str:
        return self._ptr.aligned_seq_x

    @property
    def aligned_seq_y(self) -> str:
        return self._ptr.aligned_seq_y


class MMAlignResult(RustWrapperObject):
    """Result of MM-align (multi-chain complex alignment)."""

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> MMAlignResult:
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @property
    def total_score(self) -> float:
        return self._ptr.total_score

    @property
    def chain_assignments(self) -> List[Tuple[int, int]]:
        return self._ptr.chain_assignments

    @property
    def chain_pairs(self) -> List[ChainPairResult]:
        return [ChainPairResult.from_py_ptr(p) for p in self._ptr.chain_pairs]

    def __repr__(self) -> str:
        return repr(self._ptr)


# ===========================================================================
# MM-align functions
# ===========================================================================


def mm_align(
    structure1: Structure,
    structure2: Structure,
) -> MMAlignResult:
    """Align two multi-chain complexes using MM-align.

    Automatically determines chain-to-chain correspondence.
    """
    r = _align.mm_align_pair(_get_ptr(structure1), _get_ptr(structure2))
    return MMAlignResult.from_py_ptr(r)


def mm_align_one_to_many(
    query: Structure,
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
) -> List[MMAlignResult]:
    """MM-align one query complex against many targets in parallel."""
    results = _align.mm_align_one_to_many(_get_ptr(query), _get_ptrs(targets), n_threads)
    return [MMAlignResult.from_py_ptr(r) for r in results]


def mm_align_many_to_many(
    queries: List[Structure],
    targets: List[Structure],
    *,
    n_threads: Optional[int] = None,
) -> List[Tuple[int, int, MMAlignResult]]:
    """MM-align all pairs between two lists of complexes (Cartesian product)."""
    results = _align.mm_align_many_to_many(_get_ptrs(queries), _get_ptrs(targets), n_threads)
    return [(qi, ti, MMAlignResult.from_py_ptr(r)) for qi, ti, r in results]
