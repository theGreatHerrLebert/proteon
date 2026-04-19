from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "validation" / "report_multi_retrieval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("report_multi_retrieval", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _structural_data():
    return {
        "per_query": [
            {
                "query": "shape.pdb",
                "best_truth_nonself": {"source_path": "truth.pdb", "tm_score": 0.8},
                "foldseek_top_nonself": {"source_path": "truth.pdb"},
                "proteon_top_nonself": {"source_path": "truth.pdb"},
                "thresholds": {"0.7": {"recall_at_k": 1.0, "proteon_recall_at_k": 1.0}},
            },
            {
                "query": "seq.pdb",
                "best_truth_nonself": {"source_path": "seq_truth.pdb", "tm_score": 0.9},
                "foldseek_top_nonself": {"source_path": "other.pdb"},
                "proteon_top_nonself": {"source_path": "other.pdb"},
                "thresholds": {"0.7": {"recall_at_k": 0.0, "proteon_recall_at_k": 0.0}},
            },
            {
                "query": "notruth.pdb",
                "best_truth_nonself": None,
                "thresholds": {"0.7": {"recall_at_k": 1.0, "proteon_recall_at_k": 1.0}},
            },
        ]
    }


def _sequence_data():
    return {
        "per_query": [
            {
                "query": "shape.pdb",
                "sequence_top_nonself": {"source_path": "other.pdb"},
                "thresholds": {"0.7": {"sequence_recall_at_k": 0.0}},
            },
            {
                "query": "seq.pdb",
                "sequence_top_nonself": {"source_path": "seq_truth.pdb"},
                "thresholds": {"0.7": {"sequence_recall_at_k": 1.0}},
            },
            {
                "query": "notruth.pdb",
                "sequence_top_nonself": None,
                "thresholds": {"0.7": {"sequence_recall_at_k": 1.0}},
            },
        ]
    }


def test_summarize_buckets_shape_and_sequence_only():
    report = _load_module()

    summary = report.summarize(_structural_data(), _sequence_data(), threshold="0.7")

    assert summary["counts"] == {
        "proteon_only_vs_sequence": 1,
        "sequence_only_vs_proteon": 1,
        "no_truth": 1,
    }
    assert summary["mean_recall"] == {"foldseek": 0.6667, "proteon": 0.6667, "sequence": 0.6667}


def test_cli_writes_outputs(tmp_path: Path):
    structural = tmp_path / "struct.json"
    sequence = tmp_path / "seq.json"
    json_output = tmp_path / "out.json"
    markdown_output = tmp_path / "out.md"
    structural.write_text(json.dumps(_structural_data()), encoding="utf-8")
    sequence.write_text(json.dumps(_sequence_data()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--structural",
            str(structural),
            "--sequence",
            str(sequence),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Multi-Retrieval Report" in result.stdout
    assert "shape.pdb" in markdown_output.read_text(encoding="utf-8")
    assert json.loads(json_output.read_text(encoding="utf-8"))["threshold"] == "0.7"
