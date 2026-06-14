"""Structural-retrieval adapter: 3Di prefilter -> template candidate shortlist.

`build_structure_template_features` takes an explicit candidate pool; at corpus
scale you can't TM-align a query against everything. This adapter shortlists with
proteon's existing structural search (`search()` — 3Di k-mer prefilter + diagonal
rescore + TM-align rerank), applies leakage filtering, loads the surviving hit
structures, and hands them to the featurizer. Completes STRUCTURE_TEMPLATES_PLAN
T2 ("retrieval adapter"). See `devdocs/TEMPLATE_RETRIEVAL_ADAPTER_PLAN.md`.

**Leakage is the whole point of doing this carefully.** Structural templating is
only honest when the template set excludes the query itself and its known
relatives (same entry under another id, post-cutoff-date depositions, sequence/
cluster homologs). This adapter is the *mechanism* (`exclude_self`, `exclude_ids`,
`exclude_candidate`); computing date/cluster/sequence blocklists is *policy* the
caller supplies.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from .search import SearchDB, SearchHit, search
from .structure_templates import build_structure_template_features
from .templates import TemplateFeatures


@dataclass
class RetrievalDiagnostics:
    """Per-query funnel counts, for auditing what the filter dropped."""

    n_searched: int = 0
    n_excluded_self: int = 0
    n_excluded_blocklist: int = 0
    n_excluded_predicate: int = 0
    n_below_min_tm: int = 0
    n_load_failed: int = 0
    n_returned: int = 0


@dataclass
class TemplateCandidate:
    """A retrieved candidate: the loaded structure, its `SearchHit`, and the
    original 0-based search rank (lower = more similar by the search's ordering,
    which already promotes TM-reranked hits)."""

    structure: object
    hit: SearchHit
    rank: int


def _norm_id(value: object) -> str:
    """Case/whitespace-insensitive id key. IDs are noncanonical across corpora
    (case, padding); normalize both sides so membership tests don't silently miss."""
    return str(value).strip().lower()


def _real(path: object) -> Optional[str]:
    try:
        return os.path.realpath(str(path)) if path else None
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None


def _resolve_query_id(query, query_id: Optional[str]) -> Optional[str]:
    if query_id is not None:
        return str(query_id)
    ident = getattr(query, "identifier", None)
    return str(ident) if ident else None


def _validate_caps(search_top_k: int, max_candidates: int) -> None:
    # search_top_k >= 1: search() itself requires top_k >= 1, so a 0 here would
    # crash downstream rather than return an empty shortlist (codex). max_candidates
    # may be 0 (a valid "retrieve nothing").
    if search_top_k < 1:
        raise ValueError(f"search_top_k must be >= 1, got {search_top_k}")
    if not (search_top_k >= max_candidates >= 0):
        raise ValueError(
            f"require search_top_k ({search_top_k}) >= max_candidates ({max_candidates}) >= 0"
        )


def _validate_min_tm_config(query, search_kwargs: dict) -> None:
    """A `min_tm` floor is only meaningful if every searched hit is TM-scored.
    Reject configs that leave `tm_score=None` — the filter would then drop hits on
    *missing* data, silently emptying the shortlist. `search()` reranks
    `max(top_k, rerank_top_k)` hits and this adapter passes `top_k=search_top_k`,
    so every returned hit is already TM-scored regardless of `rerank_top_k` (no
    depth check needed). The genuine holes are a disabled rerank or an encoded
    query (no structure to align)."""
    if isinstance(query, dict):
        raise ValueError("min_tm requires a structure query (an encoded query is never TM-scored)")
    if search_kwargs.get("rerank", True) is False:
        raise ValueError("min_tm requires rerank=True (TM scores come from reranking)")


def retrieve_template_candidate_hits(
    query,
    db: Union[SearchDB, str],
    *,
    search_top_k: int = 50,
    max_candidates: int = 25,
    exclude_ids: Optional[Iterable[str]] = None,
    exclude_candidate: Optional[Callable[[SearchHit], bool]] = None,
    exclude_self: bool = True,
    min_tm: Optional[float] = None,
    query_id: Optional[str] = None,
    query_path: Optional[str] = None,
    rerank_top_k: int = 5,
    **search_kwargs,
) -> Tuple[List[Tuple[SearchHit, int]], RetrievalDiagnostics]:
    """Hit-only retrieval (no structure loading): search, filter for leakage, and
    return surviving `(SearchHit, original_rank)` in search order, plus a
    `RetrievalDiagnostics`. Use this when structures aren't local files (remote /
    object-backed DBs) — you resolve `source_path`s yourself. Otherwise use
    `retrieve_template_candidates`.

    Caps apply in this order: search `search_top_k`, filter, keep ≤ `max_candidates`.
    `search_top_k` is an *effort* limit — self/blocklist/below-`min_tm` hits consume
    it, so set it well above `max_candidates` to avoid starving the pool.
    """
    _validate_caps(search_top_k, max_candidates)
    if min_tm is not None:
        _validate_min_tm_config(query, search_kwargs)

    self_ids = set()
    self_paths = set()
    if exclude_self:
        qid = _resolve_query_id(query, query_id)
        qpath = _real(query_path)
        if qid is None and qpath is None:
            raise ValueError(
                "exclude_self=True but the query has no resolvable id/path; "
                "pass query_id= / query_path=, or set exclude_self=False"
            )
        if qid is not None:
            self_ids.add(_norm_id(qid))
        if qpath is not None:
            self_paths.add(qpath)
    block = {_norm_id(x) for x in (exclude_ids or ())}

    hits = search(query, db, top_k=search_top_k, rerank_top_k=rerank_top_k, **search_kwargs)
    diag = RetrievalDiagnostics(n_searched=len(hits))
    kept: List[Tuple[SearchHit, int]] = []
    for rank, hit in enumerate(hits):
        # Classify every searched hit so the diagnostics funnel reconciles with
        # n_searched; only *append* up to the cap (codex). max_candidates=0 thus
        # returns nothing while still counting exclusions.
        hid = _norm_id(hit.id)
        if hid in self_ids or _real(hit.source_path) in self_paths:
            diag.n_excluded_self += 1
            continue
        if hid in block:
            diag.n_excluded_blocklist += 1
            continue
        if exclude_candidate is not None and exclude_candidate(hit):
            diag.n_excluded_predicate += 1
            continue
        if min_tm is not None and not (hit.tm_score is not None and hit.tm_score >= min_tm):
            diag.n_below_min_tm += 1
            continue
        if len(kept) < max_candidates:
            kept.append((hit, rank))
    diag.n_returned = len(kept)
    return kept, diag


def retrieve_template_candidates(
    query,
    db: Union[SearchDB, str],
    *,
    search_top_k: int = 50,
    max_candidates: int = 25,
    exclude_ids: Optional[Iterable[str]] = None,
    exclude_candidate: Optional[Callable[[SearchHit], bool]] = None,
    exclude_self: bool = True,
    min_tm: Optional[float] = None,
    query_id: Optional[str] = None,
    query_path: Optional[str] = None,
    rerank_top_k: int = 5,
    **search_kwargs,
) -> Tuple[List[TemplateCandidate], RetrievalDiagnostics]:
    """Retrieve and **load** template candidates: `retrieve_template_candidate_hits`
    then load each hit's `source_path` with the tolerant loader. A hit that fails
    to load does NOT consume the `max_candidates` budget — survivors keep loading
    until the cap is filled or the (pre-cap) shortlist is exhausted; aggregate
    load failures are warned once.

    Returns `(candidates, diagnostics)` in search order — symmetric with
    `retrieve_template_candidate_hits`. The diagnostics' `n_load_failed` /
    `n_returned` reflect the loaded set (codex: the loading path must expose its
    own funnel, not the pre-load hit counts)."""
    from . import batch_load_tolerant  # local import: avoids package import cycle

    # Validate the caller's real cap here — the hit helper below is called with an
    # over-fetch cap, which would otherwise bypass this check (codex).
    _validate_caps(search_top_k, max_candidates)

    # Over-fetch hits to the full search depth so load failures don't starve the
    # cap; we stop loading once max_candidates structures are in hand.
    hits, diag = retrieve_template_candidate_hits(
        query,
        db,
        search_top_k=search_top_k,
        max_candidates=search_top_k,  # keep all survivors; cap is applied at load
        exclude_ids=exclude_ids,
        exclude_candidate=exclude_candidate,
        exclude_self=exclude_self,
        min_tm=min_tm,
        query_id=query_id,
        query_path=query_path,
        rerank_top_k=rerank_top_k,
        **search_kwargs,
    )

    out: List[TemplateCandidate] = []
    failed: List[str] = []
    for hit, rank in hits:
        if len(out) >= max_candidates:
            break
        loaded = batch_load_tolerant([hit.source_path])
        if not loaded:
            failed.append(hit.source_path)
            continue
        first = loaded[0]
        structure = first[1] if isinstance(first, tuple) else first
        out.append(TemplateCandidate(structure=structure, hit=hit, rank=rank))
    if failed:
        warnings.warn(
            f"{len(failed)} template candidate(s) failed to load and were skipped: "
            f"{failed[0]}{'...' if len(failed) > 1 else ''}",
            stacklevel=2,
        )
    diag.n_load_failed = len(failed)
    diag.n_returned = len(out)  # the loaded count, overriding the pre-load hit count
    return out, diag


def build_structure_template_features_from_db(
    query,
    db: Union[SearchDB, str],
    *,
    top_k: int = 4,
    search_top_k: int = 50,
    max_candidates: int = 25,
    exclude_ids: Optional[Iterable[str]] = None,
    exclude_candidate: Optional[Callable[[SearchHit], bool]] = None,
    exclude_self: bool = True,
    min_tm: Optional[float] = None,
    query_id: Optional[str] = None,
    query_path: Optional[str] = None,
    rerank_top_k: int = 5,
    query_chain: Optional[str] = None,
    fast: bool = False,
    n_threads: Optional[int] = None,
    **search_kwargs,
) -> TemplateFeatures:
    """One-call structural-retrieval templating: shortlist candidates from `db`
    (leakage-filtered) and featurize. `top_k` templates are kept from up to
    `max_candidates` retrieved structures.

    Note the cost: `build_structure_template_features` TM-aligns *every* retrieved
    candidate (~`max_candidates` alignments) before keeping `top_k`, on top of the
    search's own `rerank_top_k` TM-aligns — the alignment cost scales with
    `max_candidates`, not `top_k`. An empty shortlist yields an empty
    `TemplateFeatures` (`n_templates == 0`), not an error.

    **Single-chain scope (v1).** Like `build_structure_template_features` itself,
    this path is single-chain. `search()` encodes the query structure as loaded,
    so a multi-chain query is searched on its full 3Di — pass a single-chain query
    (or one whose default chain is the one you intend) and the matching
    `query_chain` for featurization. A retrieved candidate that is multi-chain is
    skipped *with a warning* by the featurizer (visible, never silent), so a
    multi-chain template DB may yield fewer templates than retrieved. Per-chain DB
    encoding + chain-aware retrieval is deferred (it needs chain metadata the
    search DB doesn't carry today)."""
    candidates, _diag = retrieve_template_candidates(
        query,
        db,
        search_top_k=search_top_k,
        max_candidates=max_candidates,
        exclude_ids=exclude_ids,
        exclude_candidate=exclude_candidate,
        exclude_self=exclude_self,
        min_tm=min_tm,
        query_id=query_id,
        query_path=query_path,
        rerank_top_k=rerank_top_k,
        **search_kwargs,
    )
    return build_structure_template_features(
        query,
        [c.structure for c in candidates],
        query_chain=query_chain,
        top_k=top_k,
        fast=fast,
        n_threads=n_threads,
    )
