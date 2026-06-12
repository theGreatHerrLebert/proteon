"""Pinned retrieval gate for the structural search pipeline (roadmap P5 §13).

A fast, deterministic CI gate over the whole encode -> prefilter -> diagonal ->
TM-align-rerank pipeline. It uses **self-retrieval**: every structure in a small
fixed corpus must come back as its own top-1 hit (TM ~ 1), and must score itself
highest in the k-mer prefilter. If the encoder weights, the prefilter, or the
rerank regress, this breaks.

Cross-fold *recall* quality (does the prefilter recover true homologs) is measured
separately, offline, by `validation/bench_retrieval.py` against all-vs-all TM-align
ground truth — that needs a larger labeled-by-TM corpus than the repo ships, so it
is not a CI gate.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from conftest import REPO_ROOT, TEST_PDBS_DIR

# Single-chain structures that load + encode cleanly (exclude the multi-model file).
_CORPUS = ["1aaj.pdb", "1ake.pdb", "1bpi.pdb", "1crn.pdb", "1ubq.pdb"]


@pytest.fixture(scope="module")
def search_db():
    import proteon

    paths = [os.path.join(TEST_PDBS_DIR, n) for n in _CORPUS]
    paths = [p for p in paths if os.path.exists(p)]
    if len(paths) < 3:
        pytest.skip("need >= 3 single-chain test PDBs for the retrieval gate")
    tmp = tempfile.mkdtemp()
    db = proteon.build_search_db(paths, out=os.path.join(tmp, "db"), k=6)
    return db, paths


def test_every_structure_self_retrieves_top1(search_db):
    import proteon

    db, paths = search_db
    for path in paths:
        q = proteon.load(path)
        hits = proteon.search(q, db, top_k=3, rerank=True, rerank_top_k=3)
        assert hits, f"{os.path.basename(path)}: search returned no hits"
        top = hits[0]
        assert os.path.basename(top.source_path) == os.path.basename(path), (
            f"{os.path.basename(path)}: top hit is {os.path.basename(top.source_path)}, not self"
        )
        assert top.tm_score is not None and top.tm_score > 0.99, f"self TM low: {top.tm_score}"
        assert top.rmsd is not None and top.rmsd < 0.5, f"self RMSD high: {top.rmsd}"


def test_self_scores_highest_in_prefilter(search_db):
    import proteon

    db, paths = search_db
    for path in paths:
        q = proteon.load(path)
        hits = proteon.search(q, db, top_k=5, rerank=False)
        assert hits
        self_hits = [h for h in hits if os.path.basename(h.source_path) == os.path.basename(path)]
        assert self_hits, f"{os.path.basename(path)}: self not in prefilter results"
        best_other = max(
            (h.prefilter_score for h in hits if os.path.basename(h.source_path) != os.path.basename(path)),
            default=0.0,
        )
        # A structure must be at least as similar to itself as to anything else.
        assert self_hits[0].prefilter_score >= best_other, (
            f"{os.path.basename(path)}: self prefilter {self_hits[0].prefilter_score} < other {best_other}"
        )


def test_retrieval_benchmark_runs_end_to_end(tmp_path):
    """Smoke-guard validation/bench_retrieval.py against API drift (it had bit-rotted
    on the BatchResult interface). Runs it on the tiny test-pdbs corpus into temp paths
    and checks it completes and emits the metrics block."""
    script = os.path.join(REPO_ROOT, "validation", "bench_retrieval.py")
    if not os.path.exists(script):
        pytest.skip("bench_retrieval.py not present")
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [
            sys.executable, script,
            "--pdb-dir", TEST_PDBS_DIR,
            "--n-targets", "5", "--n-queries", "3",
            "--db-path", str(tmp_path / "db"),
            "--output", str(out),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"benchmark failed:\n{proc.stderr[-2000:]}"
    report = json.loads(out.read_text())
    assert "metrics" in report and "recall_at_k" in report["metrics"], report.keys()
