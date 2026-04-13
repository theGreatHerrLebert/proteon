"""Optional Rust backend hooks for MSA assembly from search hits.

Mirrors the `supervision_backend` pattern: discovers the optional
`ferritin_connector.py_msa` extension and exposes a stable Python API
for callers without coupling them to connector import details.

The Rust side is `ferritin-connector::py_msa::SearchEngine`, which wraps
`ferritin_search::search::SearchEngine` and adds a one-call
`search_and_build_msa(query, max_seqs, gap_idx)` method returning a
NumPy-array dict. The dict's field shapes match
`SequenceExample`'s expected layout so the consumer can splat it into
`build_sequence_example(structure, msa=..., deletion_matrix=...)`
without copying field by field.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import ferritin_connector

    _msa = getattr(ferritin_connector, "py_msa", None)
except ImportError:
    _msa = None


def rust_msa_available() -> bool:
    """Whether the optional Rust MSA backend is available."""
    return _msa is not None


def build_search_engine(
    targets: List[Tuple[int, str]],
    *,
    k: int = 6,
    reduce_to: Optional[int] = 13,
    bit_factor: float = 2.0,
    gap_open: int = -11,
    gap_extend: int = -1,
    min_score: int = 0,
    max_prefilter_hits: Optional[int] = 1000,
    max_results: Optional[int] = None,
):
    """Construct a Rust SearchEngine over a target corpus.

    `targets` is a list of `(seq_id, sequence_str)` pairs. Sequences are
    ASCII protein letters (FASTA-style). The engine builds the k-mer
    index up-front; per-query search uses the cached index.

    Returns the raw connector object. Most callers should prefer
    `search_and_build_msa(...)` below.
    """
    if _msa is None:
        raise RuntimeError("Rust MSA backend is not available")
    return _msa.SearchEngine(
        targets,
        k=k,
        reduce_to=reduce_to,
        bit_factor=bit_factor,
        gap_open=gap_open,
        gap_extend=gap_extend,
        min_score=min_score,
        max_prefilter_hits=max_prefilter_hits,
        max_results=max_results,
    )


def search(engine, query: str) -> list:
    """Run a query and return the ranked list of hit dicts.

    Each hit dict carries `target_id`, `score`, `query_start`,
    `query_end`, `target_start`, `target_end`, `cigar`, plus the
    intermediate `prefilter_score`, `best_diagonal`, `ungapped_score`.
    """
    return engine.search(query)


def search_and_build_msa(
    engine,
    query: str,
    *,
    max_seqs: int = 256,
    gap_idx: int = 21,
):
    """Run a query and assemble an AF2-style MSA tensor bundle.

    Returns a dict with keys matching `SequenceExample`'s field names:

    - `aatype` — `(L,)` `uint8`
    - `seq_mask` — `(L,)` `float32`
    - `msa` — `(N, L)` `uint8`, row 0 is the query, rows 1+ are
      homologs in query coordinates with `gap_idx` for uncovered cells
    - `deletion_matrix` — `(N, L)` `uint8` (AF2 deletion convention)
    - `msa_mask` — `(N, L)` `float32`
    - `n_seqs`, `query_len`, `gap_idx` — scalar metadata
    """
    return engine.search_and_build_msa(
        query, max_seqs=max_seqs, gap_idx=gap_idx
    )
