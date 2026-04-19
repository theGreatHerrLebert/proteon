"""Optional Rust backend hooks for MSA assembly from search hits.

Mirrors the `supervision_backend` pattern: discovers the optional
`proteon_connector.py_msa` extension and exposes a stable Python API
for callers without coupling them to connector import details.

The Rust side is `proteon-connector::py_msa::SearchEngine`, which wraps
`proteon_search::search::SearchEngine` and adds a one-call
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
    import proteon_connector

    _msa = getattr(proteon_connector, "py_msa", None)
except ImportError:  # pragma: no cover
    _msa = None


def rust_msa_available() -> bool:
    """Whether the optional Rust MSA backend is available."""
    return _msa is not None


def open_search_engine_from_mmseqs_db_with_kmi(
    db_prefix: "str | 'os.PathLike[str]'",
    kmi_path: "str | 'os.PathLike[str]'",
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
):
    """Open an engine backed by a memory-mapped DB + pre-built `.kmi`.

    Neither the DB nor the k-mer postings are loaded into RAM; both
    mmap and page in on demand. Peak resident memory stays bounded by
    the query's working set, which is what makes full-UniRef50 feasible
    on a 240 GB monster3.

    Consistency checks on the `.kmi` at open time — kmer_size,
    alphabet_size, and the full reducer mapping must all match what
    `(matrix, alphabet, reduce_to)` would produce fresh. Mismatch
    raises a ValueError from the Rust side pointing at the offending
    field. Build the `.kmi` once per corpus and reopen on every
    engine construction.
    """
    if _msa is None:
        raise RuntimeError("Rust MSA backend is not available")
    return _msa.SearchEngine.open_from_mmseqs_db_with_kmi(
        str(db_prefix),
        str(kmi_path),
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


def build_search_engine_from_mmseqs_db(
    prefix: "str | 'os.PathLike[str]'",
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
):
    """Construct a Rust SearchEngine from an on-disk MMseqs2-compatible DB.

    `prefix` is the path passed to `mmseqs createdb` (or any byte-compatible
    writer such as `proteon_search::db::DBWriter`). The DB sequences are
    streamed into the Rust engine without a Python `list[(id, str)]`
    materialize — required when target corpora don't fit as Python
    objects (UniRef30 ≈ 30 M seqs, BFD ≈ 65 M).

    All other kwargs mirror `build_search_engine` so the two paths are
    interchangeable once the engine is built.
    """
    if _msa is None:
        raise RuntimeError("Rust MSA backend is not available")
    return _msa.SearchEngine.from_mmseqs_db(
        str(prefix),
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
    use_gpu: bool = True,
):
    """Construct a Rust SearchEngine over a target corpus.

    `targets` is a list of `(seq_id, sequence_str)` pairs. Sequences are
    ASCII protein letters (FASTA-style). The engine builds the k-mer
    index up-front; per-query search uses the cached index.

    `use_gpu` controls GPU dispatch when the connector is compiled with
    the `cuda` feature and a device is present; silent CPU fallback
    otherwise. Pass `False` to force the CPU path for debugging or
    bit-reproducible runs even on GPU hosts. The CPU and GPU paths are
    parity-tested to return identical `SearchHit` ordering.

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
        use_gpu=use_gpu,
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

    Duck-types on the engine object so callers can pass either the
    Python `MsaSearch` wrapper (exposes `.build_msa`) or the raw
    `proteon_connector.py_msa.SearchEngine` (exposes
    `.search_and_build_msa`). Lets the release builder accept either
    without forcing callers to unwrap.
    """
    if hasattr(engine, "build_msa"):
        return engine.build_msa(query, max_seqs=max_seqs, gap_idx=gap_idx)
    return engine.search_and_build_msa(
        query, max_seqs=max_seqs, gap_idx=gap_idx
    )


def batch_build_sequence_examples_with_msa(
    structures: Sequence,
    engine,
    *,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    template_masks: Optional[Sequence[Optional[Sequence[float]]]] = None,
    max_seqs: int = 256,
    gap_idx: int = 21,
) -> List:
    """Batch variant of `build_sequence_example_with_msa`.

    For each structure: derive the chain sequence, run the search
    engine to get an MSA tensor bundle, and emit a
    `SequenceExample` with the MSA-side fields populated. Equivalent
    to calling `build_sequence_example_with_msa` in a loop, but
    packaged so release-layer callers don't have to hand-roll the
    loop + per-structure arg expansion themselves.

    `engine` is a `MsaSearch` / `proteon_connector.py_msa.SearchEngine`
    (whichever the caller has); both expose the
    `search_and_build_msa` method via duck typing. The engine is
    shared across all queries so the k-mer index is built once.
    """
    n = len(structures)

    def _expand(values):
        if values is None:
            return [None] * n
        values = list(values)
        if len(values) != n:
            raise ValueError(f"expected {n} items, got {len(values)}")
        return values

    record_ids = _expand(record_ids)
    source_ids = _expand(source_ids)
    chain_ids = _expand(chain_ids)
    template_masks = _expand(template_masks)

    return [
        build_sequence_example_with_msa(
            structure,
            engine,
            record_id=record_ids[i],
            source_id=source_ids[i],
            chain_id=chain_ids[i],
            code_rev=code_rev,
            config_rev=config_rev,
            template_mask=template_masks[i],
            max_seqs=max_seqs,
            gap_idx=gap_idx,
        )
        for i, structure in enumerate(structures)
    ]


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

    from .sequence_example import compute_msa_profile
    msa_profile_arr = compute_msa_profile(msa_arr, msa_mask_arr)

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
        msa_profile=msa_profile_arr,
        template_mask=base.template_mask,
    )
