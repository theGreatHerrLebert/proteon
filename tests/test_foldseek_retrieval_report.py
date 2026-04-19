from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "validation" / "report_foldseek_retrieval.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("report_foldseek_retrieval", REPORT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_benchmark() -> dict:
    return {
        "corpus": {"thresholds": [0.5, 0.7, 0.9]},
        "timing": {"foldseek_s": 1.0, "proteon_search_s": 2.0},
        "metrics": {},
        "per_query": [
            {
                "query": "q1.pdb",
                "best_truth_nonself": {"source_path": "a.pdb", "tm_score": 0.95},
                "foldseek_top_nonself": {"source_path": "a.pdb"},
                "proteon_top_nonself": {"source_path": "b.pdb"},
                "proteon_top_hits": [{"source_path": "b.pdb"}, {"source_path": "a.pdb"}],
                "thresholds": {
                    "0.5": {"recall_at_k": 1.0, "proteon_recall_at_k": 0.5},
                    "0.7": {"recall_at_k": 1.0, "proteon_recall_at_k": 0.0},
                    "0.9": {"recall_at_k": 1.0, "proteon_recall_at_k": 0.0},
                },
            },
            {
                "query": "q2.pdb",
                "best_truth_nonself": {"source_path": "c.pdb", "tm_score": 0.8},
                "foldseek_top_nonself": {"source_path": "d.pdb"},
                "proteon_top_nonself": {"source_path": "c.pdb"},
                "thresholds": {
                    "0.5": {"recall_at_k": 0.0, "proteon_recall_at_k": 1.0},
                    "0.7": {"recall_at_k": 0.0, "proteon_recall_at_k": 1.0},
                    "0.9": {"recall_at_k": 1.0, "proteon_recall_at_k": 1.0},
                },
            },
        ],
        "skipped_truth_candidates": {"q1.pdb": [{"source_path": "large.pdb"}]},
        "skipped_proteon_queries": {"bad.pdb": "load_failed"},
    }


def test_summarize_classifies_top1_and_recall():
    report = _load_report_module()

    summary = report.summarize(_sample_benchmark(), primary_threshold="0.7")

    assert summary["top1_counts"] == {"foldseek_only": 1, "proteon_only": 1}
    assert summary["recall_counts"] == {"foldseek_better": 1, "proteon_better": 1}
    assert summary["threshold_summary"]["0.7"]["proteon_minus_foldseek"] == 0.0
    assert summary["n_skipped_truth_candidates"] == 1
    assert summary["worst_proteon_queries"][0]["query"] == "q1.pdb"
    assert summary["worst_proteon_queries"][0]["truth_rank_in_proteon_trace"] == 2


def test_cli_writes_markdown_and_json(tmp_path: Path):
    input_path = tmp_path / "bench.json"
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    input_path.write_text(json.dumps(_sample_benchmark()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            str(input_path),
            "--threshold",
            "0.7",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Foldseek vs proteon Retrieval Diagnostics" in result.stdout
    assert json.loads(json_output.read_text(encoding="utf-8"))["primary_threshold"] == "0.7"
    assert "| q1.pdb | a.pdb |" in markdown_output.read_text(encoding="utf-8")
