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

End-to-end integration is also offered here as
`build_sequence_example_with_msa`: takes a structure + a search
engine, derives the query from the structure, runs the search,
assembles the MSA, and returns a `SequenceExample` with all fields
populated.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

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


def build_sequence_example_with_msa(
    structure,
    engine,
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
    """End-to-end: structure → query → search → MSA → SequenceExample.

    Builds the structure-side fields (`aatype`, `residue_index`,
    `seq_mask`) the same way `build_sequence_example` does, derives the
    query string from the chain, runs `engine.search_and_build_msa`,
    and returns a `SequenceExample` with the MSA-side fields populated
    from the engine output.

    Parameters mirror `build_sequence_example` plus `max_seqs` /
    `gap_idx` for the MSA assembly. `chain_id` is required for
    multi-chain structures.

    Raises `ValueError` if the engine returns an MSA whose
    `query_len` doesn't match the structure-derived chain length —
    indicates the engine was built with a different alphabet or the
    structure has been edited since the engine was constructed.
    """
    # Defer import so this module loads even if sequence_example isn't
    # importable for some unrelated reason (matches the existing optional-
    # dependency pattern used elsewhere in this package).
    from .sequence_example import SequenceExample, build_sequence_example

    # Build the structure-side example first (without MSA). This reuses
    # all the existing chain selection, residue filtering, and field
    # validation in build_sequence_example so the structure-side
    # contract stays single-source.
    base = build_sequence_example(
        structure,
        record_id=record_id,
        source_id=source_id,
        chain_id=chain_id,
        code_rev=code_rev,
        config_rev=config_rev,
        msa=None,
        deletion_matrix=None,
        template_mask=template_mask,
    )

    msa_dict = search_and_build_msa(
        engine,
        base.sequence,
        max_seqs=max_seqs,
        gap_idx=gap_idx,
    )

    if int(msa_dict["query_len"]) != base.length:
        raise ValueError(
            f"engine returned MSA query_len={msa_dict['query_len']} "
            f"but structure chain has length={base.length} — engine and "
            f"structure must agree on the chain identity"
        )

    # Cast at the boundary to match the existing SequenceExample schema:
    #   msa             uint8 → int32
    #   deletion_matrix uint8 → float32
    #   msa_mask        float32 → float32 (no-op)
    # The Rust side keeps uint8 for memory; this pays one O(N*L) cast at
    # the Python edge, which is dwarfed by the search itself.
    msa_arr = np.asarray(msa_dict["msa"], dtype=np.int32)
    deletion_arr = np.asarray(msa_dict["deletion_matrix"], dtype=np.float32)
    msa_mask_arr = np.asarray(msa_dict["msa_mask"], dtype=np.float32)

    return SequenceExample(
        record_id=base.record_id,
        source_id=base.source_id,
        chain_id=base.chain_id,
        sequence=base.sequence,
        length=base.length,
        code_rev=base.code_rev,
        config_rev=base.config_rev,
        aatype=base.aatype,
        residue_index=base.residue_index,
        seq_mask=base.seq_mask,
        msa=msa_arr,
        deletion_matrix=deletion_arr,
        msa_mask=msa_mask_arr,
        template_mask=base.template_mask,
    )
