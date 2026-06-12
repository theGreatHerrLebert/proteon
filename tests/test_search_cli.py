"""Smoke tests for the `proteon-search` CLI (build / inspect / query).

The CLI is a thin wrapper over the library search API; these confirm the
build -> inspect -> query pipeline runs end-to-end and that the self-match comes
back first with a perfect TM-score after reranking.
"""

import json
import os

import pytest

from conftest import TEST_PDBS_DIR
from proteon.search_cli import build_parser, main


def _pdb(name: str) -> str:
    return os.path.join(TEST_PDBS_DIR, name)


def _corpus():
    names = ["1crn.pdb", "1ubq.pdb", "1ake.pdb"]
    paths = [_pdb(n) for n in names if os.path.exists(_pdb(n))]
    if len(paths) < 2:
        pytest.skip("need at least two test PDBs for the search CLI test")
    return paths


def test_build_inspect_query_roundtrip(tmp_path, capsys):
    corpus = _corpus()
    db = str(tmp_path / "db")

    # build
    assert main(["build", *corpus, "-o", db, "-k", "6"]) == 0
    assert os.path.isdir(db)

    # inspect (json)
    capsys.readouterr()
    assert main(["inspect", db, "--format", "json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_entries"] >= len(corpus)
    assert info["k"] == 6

    # query: the query structure must come back as its own top hit with TM ~ 1.
    capsys.readouterr()
    assert main(["query", db, corpus[-1], "--top-k", "5", "--format", "json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits, "query should return at least the self-match"
    top = hits[0]
    assert top["tm_score"] is not None and top["tm_score"] > 0.95, f"self-match TM low: {top}"
    assert top["rmsd"] is not None and top["rmsd"] < 0.5


def test_query_no_rerank_omits_tm(tmp_path, capsys):
    corpus = _corpus()
    db = str(tmp_path / "db")
    assert main(["build", *corpus, "-o", db]) == 0
    capsys.readouterr()
    assert main(["query", db, corpus[0], "--no-rerank", "--format", "json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits
    # Without reranking the TM-align fields are not populated.
    assert all(h["tm_score"] is None for h in hits)


def test_build_requires_inputs(tmp_path):
    # An empty directory yields no structures -> nonzero exit.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["build", str(empty), "-o", str(tmp_path / "db")]) == 2


def test_query_rerank_default_matches_library():
    # Pin the CLI --rerank-top-k default to the library search() default (5). The
    # effective rerank depth is max(top_k, rerank_top_k), so this is a floor, not a cap.
    args = build_parser().parse_args(["query", "db", "q.pdb"])
    assert args.rerank_top_k == 5
    assert args.top_k == 10
