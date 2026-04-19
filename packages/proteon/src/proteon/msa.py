"""Pythonic wrapper around the Rust MMseqs-style search + MSA assembly.

Wraps `proteon_connector.py_msa.SearchEngine` following the project's
`RustWrapperObject` pattern: a Python class holds a pointer to a PyO3
Rust object and offers a Pythonic API on top.

Typical usage:

    from proteon.msa import MsaSearch

    engine = MsaSearch.build(targets)            # targets: list[(int, str)]
    hits = engine.search(query)                  # ranked list of Hit dicts
    msa = engine.build_msa(query)                # AF2-style tensor dict
    example = engine.build_sequence_example(structure)  # full SequenceExample

The constructor parameters mirror upstream MMseqs2 protein-search
defaults (k=6, BLOSUM62 at bit_factor=2, gap_open=-11, gap_extend=-1,
13-letter reduction). All keyword args are overridable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .core import RustWrapperObject
from .msa_backend import (
    build_search_engine as _build_search_engine,
    build_sequence_example_with_msa as _build_sequence_example_with_msa,
    rust_msa_available,
    search as _search,
    search_and_build_msa as _search_and_build_msa,
)


class MsaSearch(RustWrapperObject):
    """Search engine over a target sequence corpus.

    Construction builds the k-mer index up-front; per-query search and
    MSA assembly reuse the cached index. One engine, many queries is
    the intended usage pattern (matches upstream's `createindex` +
    `search` split).
    """

    def __init__(self, ptr):
        self._ptr = ptr

    @classmethod
    def from_py_ptr(cls, ptr) -> "MsaSearch":
        return cls(ptr)

    def get_py_ptr(self):
        return self._ptr

    @classmethod
    def build(
        cls,
        targets: Sequence[Tuple[int, str]],
        *,
        k: int = 6,
        reduce_to: Optional[int] = 13,
        bit_factor: float = 2.0,
        gap_open: int = -11,
        gap_extend: int = -1,
        min_score: int = 0,
        max_prefilter_hits: Optional[int] = 1000,
        max_results: Optional[int] = None,
        use_gpu: bool = True,
    ) -> "MsaSearch":
        """Build a search engine over a target corpus.

        `targets` is a sequence of `(seq_id, sequence_str)` pairs.
        Sequences are FASTA-style ASCII protein letters. Defaults
        match upstream MMseqs2 protein-search settings.

        `use_gpu=False` forces the CPU path even when the connector
        is compiled with CUDA and a device is available — useful for
        debugging GPU issues or reproducing upstream CPU bit-exact
        output on a GPU host. With `use_gpu=True` (default) the engine
        silently falls back to CPU when no GPU is present.
        """
        if not rust_msa_available():
            raise RuntimeError(
                "Rust MSA backend is not available — install proteon_connector "
                "(maturin develop --release in proteon-connector/)"
            )
        ptr = _build_search_engine(
            list(targets),
            k=k,
            reduce_to=reduce_to,
            bit_factor=bit_factor,
            gap_open=gap_open,
            gap_extend=gap_extend,
            min_score=min_score,
            max_prefilter_hits=max_prefilter_hits,
            max_results=max_results,
            use_gpu=use_gpu,
        )
        return cls(ptr)

    @property
    def target_count(self) -> int:
        """Number of targets indexed."""
        return self._ptr.target_count()

    def search(self, query: str) -> List[dict]:
        """Run a query and return the ranked list of hit dicts.

        Each hit dict carries:
          target_id, prefilter_score, best_diagonal, ungapped_score,
          score, query_start, query_end, target_start, target_end, cigar.
        """
        return _search(self._ptr, query)

    def build_msa(
        self,
        query: str,
        *,
        max_seqs: int = 256,
        gap_idx: int = 21,
    ) -> dict:
        """Run a query and assemble an AF2-style MSA tensor bundle.

        Returns a dict with `aatype` `(L,)`, `seq_mask` `(L,)`,
        `msa` `(N, L)`, `deletion_matrix` `(N, L)`, `msa_mask` `(N, L)`,
        plus scalar `n_seqs` / `query_len` / `gap_idx`. See
        `msa_backend.search_and_build_msa` for full field details.
        """
        return _search_and_build_msa(
            self._ptr, query, max_seqs=max_seqs, gap_idx=gap_idx
        )

    def build_sequence_example(
        self,
        structure,
        *,
        record_id: Optional[str] = None,
        source_id: Optional[str] = None,
        chain_id: Optional[str] = None,
        code_rev: Optional[str] = None,
        config_rev: Optional[str] = None,
        template_mask: Optional[Sequence[float]] = None,
        max_seqs: int = 256,
        gap_idx: int = 21,
    ):
        """Structure → derived query → MSA → fully populated SequenceExample.

        One-call helper that combines the structure-side fields
        (aatype, residue_index, seq_mask) from the chain with the
        MSA-side fields (msa, deletion_matrix, msa_mask) from the
        engine. The query string is derived from the chain
        automatically. Raises `ValueError` if the engine returns an
        MSA whose `query_len` doesn't match the chain length.
        """
        return _build_sequence_example_with_msa(
            structure,
            self._ptr,
            record_id=record_id,
            source_id=source_id,
            chain_id=chain_id,
            code_rev=code_rev,
            config_rev=config_rev,
            template_mask=template_mask,
            max_seqs=max_seqs,
            gap_idx=gap_idx,
        )
