"""Tests for `ferritin.corpus_validation`.

`validate_corpus_release` is the QA gate that produces a CorpusValidationReport
per release. The existing `test_validation_report_build.py` renders rescue-
summary tables from pre-built reports; this file exercises the validator
itself — specifically the three branches of checks
(`_check_count_consistency`, `_check_training_release`,
`_check_structure_tensor_completeness`) and the JSON-write path.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pytest

from ferritin.corpus_release import CorpusReleaseManifest
from ferritin.corpus_validation import (
    CorpusValidationReport,
    ValidationIssue,
    validate_corpus_release,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, manifest: CorpusReleaseManifest, name: str = "manifest.json") -> Path:
    """Serialize a CorpusReleaseManifest to disk the same way
    build_corpus_release_manifest does — validate_corpus_release
    only needs the JSON file path."""
    p = tmp_path / name
    p.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return p


def _default_manifest(
    **overrides,
) -> CorpusReleaseManifest:
    """Internally-consistent manifest (no issues expected)."""
    base = dict(
        release_id="rel-test-0001",
        count_prepared=100,
        count_sequence_examples=50,
        count_structure_examples=40,
        count_training_examples=30,
        split_counts={"train": 20, "val": 5, "test": 5},
    )
    base.update(overrides)
    return CorpusReleaseManifest(**base)


def _write_training_jsonl(
    path: Path,
    rows: List[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_validation_issue_carries_severity_code_message(self):
        issue = ValidationIssue(severity="error", code="abc", message="boom")
        assert issue.severity == "error"
        assert issue.code == "abc"
        assert issue.message == "boom"

    def test_report_defaults_to_ok_with_empty_issues(self):
        report = CorpusValidationReport(release_id="r")
        assert report.ok is True
        assert report.issues == []
        assert report.counts == {}
        # `created_at` is a timezone-aware ISO string — required because
        # downstream release diffs compare reports across runs.
        assert report.created_at.endswith("+00:00")


# ---------------------------------------------------------------------------
# Happy path — counts consistent, no training / structure release linked
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_counts_consistent_returns_ok_with_only_warnings(self, tmp_path: Path):
        """No training or structure release is linked, so the validator
        emits 'missing_training_release' + 'missing_structure_release'
        warnings. Neither is an error, so ok stays True."""
        manifest = _default_manifest()
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        codes = {issue.code for issue in report.issues}
        assert codes == {"missing_training_release", "missing_structure_release"}
        assert all(issue.severity == "warning" for issue in report.issues)
        assert report.ok is True
        assert report.counts == {
            "prepared": 100,
            "sequence_examples": 50,
            "structure_examples": 40,
            "training_examples": 30,
        }
        assert report.split_counts == {"train": 20, "val": 5, "test": 5}


# ---------------------------------------------------------------------------
# _check_count_consistency
# ---------------------------------------------------------------------------


class TestCountConsistency:
    def test_training_exceeds_sequence_is_error(self, tmp_path: Path):
        manifest = _default_manifest(
            count_sequence_examples=10,
            count_training_examples=20,  # > sequence
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "training_exceeds_sequence" and issue.severity == "error"
            for issue in report.issues
        )
        assert report.ok is False

    def test_training_exceeds_structure_is_error(self, tmp_path: Path):
        manifest = _default_manifest(
            count_structure_examples=10,
            count_training_examples=20,
            count_sequence_examples=50,  # keep sequence gate satisfied
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "training_exceeds_structure" and issue.severity == "error"
            for issue in report.issues
        )
        assert report.ok is False

    def test_structure_exceeds_prepared_is_warning_only(self, tmp_path: Path):
        """Unlike training-exceeds-{seq,struct}, this case is a warning
        — the validator can't tell whether the prepared count is stale
        or the structure count is wrong, so it flags but doesn't fail."""
        manifest = _default_manifest(
            count_prepared=5,
            count_structure_examples=10,
            count_training_examples=3,  # keep training gates satisfied
            count_sequence_examples=15,
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "structure_exceeds_prepared" and issue.severity == "warning"
            for issue in report.issues
        )
        # No errors present — ok remains True despite the warning.
        assert report.ok is True


# ---------------------------------------------------------------------------
# _check_training_release
# ---------------------------------------------------------------------------


class TestTrainingRelease:
    def test_training_release_row_count_mismatch_is_error(self, tmp_path: Path):
        training_dir = tmp_path / "training"
        _write_training_jsonl(
            training_dir / "training_examples.jsonl",
            [
                {"record_id": "a", "split": "train"},
                {"record_id": "b", "split": "train"},
            ],  # 2 rows
        )
        manifest = _default_manifest(
            training_release=str(training_dir),
            count_training_examples=5,  # manifest says 5 but only 2 rows
            split_counts={"train": 2},
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "training_row_count_mismatch" for issue in report.issues
        )
        assert report.ok is False

    def test_duplicate_record_ids_surface_first_five(self, tmp_path: Path):
        training_dir = tmp_path / "training"
        _write_training_jsonl(
            training_dir / "training_examples.jsonl",
            [
                {"record_id": "a", "split": "train"},
                {"record_id": "a", "split": "train"},  # duplicate
                {"record_id": "b", "split": "train"},
                {"record_id": "b", "split": "train"},  # duplicate
            ],
        )
        manifest = _default_manifest(
            training_release=str(training_dir),
            count_training_examples=4,
            split_counts={"train": 4},
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        dup_issues = [i for i in report.issues if i.code == "duplicate_training_ids"]
        assert len(dup_issues) == 1
        assert "a" in dup_issues[0].message
        assert "b" in dup_issues[0].message
        assert report.ok is False

    def test_split_count_mismatch_is_error(self, tmp_path: Path):
        training_dir = tmp_path / "training"
        _write_training_jsonl(
            training_dir / "training_examples.jsonl",
            [
                {"record_id": f"r{i}", "split": "train"} for i in range(3)
            ],
        )
        manifest = _default_manifest(
            training_release=str(training_dir),
            count_training_examples=3,
            split_counts={"train": 2, "val": 1},  # manifest disagrees with jsonl
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "split_count_mismatch" and issue.severity == "error"
            for issue in report.issues
        )
        assert report.ok is False

    def test_default_split_is_train_when_missing(self, tmp_path: Path):
        training_dir = tmp_path / "training"
        _write_training_jsonl(
            training_dir / "training_examples.jsonl",
            [
                {"record_id": "a"},  # no split field
                {"record_id": "b"},
            ],
        )
        manifest = _default_manifest(
            training_release=str(training_dir),
            count_training_examples=2,
            split_counts={"train": 2},
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        # No split_count_mismatch — the validator inferred split="train".
        assert not any(i.code == "split_count_mismatch" for i in report.issues)


# ---------------------------------------------------------------------------
# _check_structure_tensor_completeness
# ---------------------------------------------------------------------------


class TestStructureTensorCompleteness:
    def test_missing_structure_release_is_warning(self, tmp_path: Path):
        manifest = _default_manifest()  # structure_release is None
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "missing_structure_release" and issue.severity == "warning"
            for issue in report.issues
        )

    def test_zero_structure_examples_is_warning(self, tmp_path: Path):
        struct_dir = tmp_path / "structure"
        struct_dir.mkdir()
        manifest = _default_manifest(
            structure_release=str(struct_dir),
            count_structure_examples=0,
            count_training_examples=0,  # keep training gate satisfied
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "no_structure_examples" and issue.severity == "warning"
            for issue in report.issues
        )

    def test_missing_tensors_parquet_is_error(self, tmp_path: Path):
        struct_dir = tmp_path / "structure"
        (struct_dir / "examples").mkdir(parents=True)
        # Deliberately do NOT create tensors.parquet.
        manifest = _default_manifest(
            structure_release=str(struct_dir),
            count_structure_examples=5,
            count_training_examples=3,
            count_sequence_examples=10,
        )
        report = validate_corpus_release(_write_manifest(tmp_path, manifest))
        assert any(
            issue.code == "missing_structure_tensors" and issue.severity == "error"
            for issue in report.issues
        )
        assert report.ok is False


# ---------------------------------------------------------------------------
# JSON output roundtrip
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_out_path_writes_roundtrippable_json(self, tmp_path: Path):
        manifest = _default_manifest()
        manifest_path = _write_manifest(tmp_path, manifest)
        out = tmp_path / "report.json"
        report = validate_corpus_release(manifest_path, out_path=out)

        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["release_id"] == report.release_id
        assert loaded["ok"] == report.ok
        # `issues` is serialized as plain dicts (not asdict of dataclasses
        # left as-is), because downstream tools read the JSON without
        # needing the ValidationIssue import path.
        assert all(set(i.keys()) == {"severity", "code", "message"} for i in loaded["issues"])

    def test_out_path_omitted_does_not_touch_disk(self, tmp_path: Path):
        manifest = _default_manifest()
        before = set(p.name for p in tmp_path.iterdir())
        validate_corpus_release(_write_manifest(tmp_path, manifest))
        after = set(p.name for p in tmp_path.iterdir())
        # Only the manifest we wrote — no side-effect file.
        assert after == before | {"manifest.json"}
