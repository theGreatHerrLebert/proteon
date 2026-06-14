"""Structural-retrieval template adapter: search -> leakage filter -> featurize.

Builds a small structural-search DB from test-pdbs (plus a copied fixture as a
controllable near-identical candidate) and exercises the adapter's leakage
filtering, cap/min_tm validation, diagnostics, and the end-to-end featurize path.
Needs the native search backend + pyarrow, so it skips where unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

try:
    import proteon
    from proteon.search import build_search_db
    from proteon.template_retrieval import (
        RetrievalDiagnostics,
        TemplateCandidate,
        build_structure_template_features_from_db,
        retrieve_template_candidate_hits,
        retrieve_template_candidates,
    )
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"proteon search unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture(name: str) -> Path:
    p = REPO_ROOT / "test-pdbs" / name
    if not p.exists():
        pytest.skip(f"{p} not found")
    return p


def _db_with_self_copy(tmp_path: Path):
    """A DB containing 1crn, a coordinate-identical copy of 1crn at a distinct
    path with its HEADER stripped (so its entry id falls back to the path stem
    `crn_copy`, distinct from the query's `1CR` — a near-identical candidate that
    is NOT same-entry, so self-exclusion keeps it), and a dissimilar structure."""
    crn = _fixture("1crn.pdb")
    copy = tmp_path / "crn_copy.pdb"
    # Drop HEADER so the copy gets a different identifier than 1crn (path stem).
    lines = [ln for ln in crn.read_text().splitlines(keepends=True) if not ln.startswith("HEADER")]
    copy.write_text("".join(lines))
    others = [_fixture(n) for n in ("1ubq.pdb", "1bpi.pdb") if (REPO_ROOT / "test-pdbs" / n).exists()]
    paths = [crn, copy, *others]
    db = build_search_db([str(p) for p in paths], k=6)
    return db, crn, copy


def test_self_excluded_by_path_leaves_near_duplicate(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn)
    )
    # The exact query file is excluded; the copy (same coords, other path) survives.
    assert diag.n_excluded_self >= 1
    surviving_paths = {h.source_path for h, _ in hits}
    assert str(crn) not in {__import__("os").path.realpath(p) for p in surviving_paths}
    assert any(h.tm_score is not None and h.tm_score > 0.99 for h, _ in hits), "near-duplicate lost"


def test_exclude_ids_blocklist_drops_hit(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    # Block the copy by its entry id; the copy carries 1crn's header id.
    copy_struct = proteon.load(str(copy))
    block_id = copy_struct.identifier or "crn_copy"
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn),
        exclude_ids=[block_id],
    )
    assert diag.n_excluded_blocklist >= 1
    assert all((h.id or "").lower() != str(block_id).lower() for h, _ in hits)


def test_exclude_candidate_predicate(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    # Drop everything via a predicate; shortlist is empty, counted under predicate.
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn),
        exclude_candidate=lambda h: True,
    )
    assert hits == []
    assert diag.n_excluded_predicate >= 1


def test_min_tm_ok_with_default_rerank_depth(tmp_path: Path):
    """min_tm with the default shallow rerank_top_k is VALID: search reranks
    max(top_k, rerank_top_k) and the adapter passes top_k=search_top_k, so every
    returned hit is TM-scored regardless of rerank_top_k (codex)."""
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    # rerank_top_k=5 << search_top_k=10, min_tm set — must NOT raise.
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn),
        min_tm=0.5, rerank_top_k=5,
    )
    assert all(h.tm_score is not None and h.tm_score >= 0.5 for h, _ in hits)


def test_cap_ordering_validated():
    with pytest.raises(ValueError, match="search_top_k"):
        retrieve_template_candidate_hits(
            object(), "unused-db", search_top_k=5, max_candidates=10, exclude_self=False,
        )


def test_zero_search_top_k_rejected():
    """search_top_k=0 would crash search() (top_k>=1); reject up front so the
    adapter's accepted configs all behave (codex)."""
    with pytest.raises(ValueError, match="search_top_k must be >= 1"):
        retrieve_template_candidate_hits(
            object(), "unused-db", search_top_k=0, max_candidates=0, exclude_self=False,
        )


def test_exclude_self_without_identity_raises():
    """A query with no resolvable id/path and exclude_self=True must raise — a
    silent self-leak is worse than a loud error."""
    class _NoId:
        identifier = None

    with pytest.raises(ValueError, match="exclude_self"):
        retrieve_template_candidate_hits(_NoId(), "unused-db", exclude_self=True)


def test_min_tm_boundary_retained(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    # rerank everything so all hits are TM-scored; the near-duplicate is ~1.0.
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn),
        min_tm=0.99, rerank_top_k=10,
    )
    assert all(h.tm_score is not None and h.tm_score >= 0.99 for h, _ in hits)
    assert any(h.tm_score > 0.99 for h, _ in hits)


def test_build_features_from_db_end_to_end(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    qres = [r for r in q.chains[0].residues if r.is_amino_acid]
    tf = build_structure_template_features_from_db(
        q, db, top_k=2, search_top_k=10, max_candidates=5,
        query_path=str(crn), rerank_top_k=10,
    )
    # The near-duplicate yields at least one template; query_len matches the query.
    assert tf.query_len == len(qres)
    assert 1 <= tf.n_templates <= 2
    assert tf.template_all_atom_positions.shape == (tf.n_templates, len(qres), 37, 3)


def test_returns_typed_candidates_with_rank(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    cands, diag = retrieve_template_candidates(
        q, db, search_top_k=10, max_candidates=5, query_path=str(crn),
    )
    assert all(isinstance(c, TemplateCandidate) for c in cands)
    # Rank is the original search order, strictly increasing as returned.
    ranks = [c.rank for c in cands]
    assert ranks == sorted(ranks)
    assert all(hasattr(c.structure, "chains") for c in cands)
    # Loading diagnostics are exposed and reflect the loaded set (codex).
    assert diag.n_returned == len(cands)
    assert diag.n_load_failed == 0


def test_max_candidates_zero_returns_nothing(tmp_path: Path):
    """max_candidates=0 must return an empty shortlist (cap checked before append,
    not after — codex)."""
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    hits, diag = retrieve_template_candidate_hits(
        q, db, search_top_k=10, max_candidates=0, query_path=str(crn)
    )
    assert hits == [] and diag.n_returned == 0
    # Exclusions are still classified at cap 0 (classify-all, append-up-to-cap):
    # the query's own entry is counted as self-excluded, not silently skipped.
    assert diag.n_searched >= 1 and diag.n_excluded_self >= 1


def test_loader_path_validates_real_cap():
    """retrieve_template_candidates must validate the *caller's* cap, not the
    over-fetch cap it forwards internally (codex)."""
    with pytest.raises(ValueError, match="max_candidates"):
        retrieve_template_candidates(
            object(), "unused-db", search_top_k=10, max_candidates=-1, exclude_self=False,
        )
    with pytest.raises(ValueError, match="search_top_k"):
        retrieve_template_candidates(
            object(), "unused-db", search_top_k=5, max_candidates=10, exclude_self=False,
        )


def test_min_tm_rejects_configs_that_cannot_tm_score():
    """min_tm with rerank disabled, or an encoded-dict query, would silently drop
    every hit (tm_score always None) — reject up front (codex)."""
    with pytest.raises(ValueError, match="rerank"):
        retrieve_template_candidate_hits(
            object(), "unused-db", search_top_k=5, max_candidates=5,
            min_tm=0.5, rerank_top_k=5, exclude_self=False, rerank=False,
        )
    with pytest.raises(ValueError, match="encoded"):
        retrieve_template_candidate_hits(
            {"states": []}, "unused-db", search_top_k=5, max_candidates=5,
            min_tm=0.5, rerank_top_k=5, exclude_self=False,
        )


def test_empty_shortlist_featurizes_to_zero(tmp_path: Path):
    db, crn, copy = _db_with_self_copy(tmp_path)
    q = proteon.load(str(crn))
    # Exclude everything -> empty pool -> n_templates == 0, not an error.
    tf = build_structure_template_features_from_db(
        q, db, top_k=4, search_top_k=10, max_candidates=5,
        query_path=str(crn), exclude_candidate=lambda h: True,
    )
    assert tf.n_templates == 0
