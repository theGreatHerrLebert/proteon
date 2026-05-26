"""Contract test for v0.3.0 Phase D cluster-leakage validation.

Phase D extends ``validate_corpus_release`` with the leakage-detection
check: when a ``ClusterAssignments`` is supplied (path or in-memory),
every cluster's members must land in exactly one split. A violation
records an ``error``-severity issue and marks the report ``ok=False``;
a namespace mismatch records a ``warning`` because the leakage
assertion below it becomes meaningless without confirmed ID alignment.

The whole point of the v0.3.0 data-engine layer is that downstream
consumers can audit a release artifact and *prove* no cluster spans
splits. This file pins that audit contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import proteon
from proteon.cluster_assignments import (
    NAMESPACE_PREPARED_RECORD_ID,
    NAMESPACE_RAW_PDB_ID,
    ClusterAssignmentRow,
    ClusterAssignments,
    ClusterAssignmentsManifest,
    build_cluster_assignments_release,
)
from proteon.corpus_release import (
    CorpusReleaseManifest,
    build_corpus_release_manifest,
)
from proteon.corpus_validation import (
    ClusterLeakageReport,
    validate_corpus_release,
)


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #


def _two_cluster_rows(*, namespace: str = NAMESPACE_PREPARED_RECORD_ID):
    a_members = ["rec-a1", "rec-a2", "rec-a3", "rec-a4"]
    b_members = ["rec-b1", "rec-b2", "rec-b3", "rec-b4"]
    rows = []
    for m in a_members:
        rows.append(
            ClusterAssignmentRow(
                record_id=m,
                cluster_id="cluster-A",
                representative_record_id="rec-a1",
                is_representative=(m == "rec-a1"),
                cluster_size=4,
            )
        )
    for m in b_members:
        rows.append(
            ClusterAssignmentRow(
                record_id=m,
                cluster_id="cluster-B",
                representative_record_id="rec-b1",
                is_representative=(m == "rec-b1"),
                cluster_size=4,
            )
        )
    return rows, a_members, b_members


def _write_assignments_release(tmp_path: Path, *, namespace: str = NAMESPACE_PREPARED_RECORD_ID) -> Path:
    rows, _, _ = _two_cluster_rows(namespace=namespace)
    return build_cluster_assignments_release(
        rows,
        tmp_path / "cluster_release",
        release_id="phase-d-test",
        tool="mmseqs2",
        tool_version="14.7e284",
        sequence_id_namespace=namespace,
    )


def _make_in_memory_assignments(*, namespace: str = NAMESPACE_PREPARED_RECORD_ID) -> ClusterAssignments:
    rows, _, _ = _two_cluster_rows(namespace=namespace)
    return ClusterAssignments(
        manifest=ClusterAssignmentsManifest(
            release_id="in-memory",
            sequence_id_namespace=namespace,
            count_sequences=len(rows),
            count_clusters=2,
        ),
        rows=tuple(rows),
    )


def _build_fake_corpus_release(
    tmp_path: Path,
    *,
    splits: Dict[str, str],
) -> Tuple[Path, Dict[str, int]]:
    """Build a minimal corpus release directory whose training_release
    contains the given per-record splits.

    Only what ``validate_corpus_release`` actually reads is populated:
    a ``corpus_release_manifest.json`` with split_counts and a
    ``training_release/training_examples.jsonl`` with one row per
    record_id.
    """
    training_dir = tmp_path / "training_release"
    training_dir.mkdir()
    # training_examples.jsonl — minimum fields the validator parses.
    with (training_dir / "training_examples.jsonl").open("w", encoding="utf-8") as fh:
        for rid, split in splits.items():
            fh.write(json.dumps({"record_id": rid, "split": split}))
            fh.write("\n")
    # release_manifest.json — referenced by corpus_release.py's _load_json
    split_counts: Dict[str, int] = {}
    for split in splits.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    (training_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "release_id": "training-rel",
                "count_examples": len(splits),
                "split_counts": split_counts,
            }
        ),
        encoding="utf-8",
    )
    # Build the top-level manifest pointing at the training release.
    corpus_root = build_corpus_release_manifest(
        tmp_path / "corpus",
        release_id="phase-d-corpus",
        training_release=training_dir,
    )
    return corpus_root / "corpus_release_manifest.json", split_counts


# --------------------------------------------------------------------------- #
# 1. Public-API surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    def test_cluster_leakage_report_is_top_level(self):
        assert "ClusterLeakageReport" in proteon.__all__
        assert hasattr(proteon, "ClusterLeakageReport")

    def test_report_has_cluster_leakage_check_field(self):
        fields = proteon.CorpusValidationReport.__dataclass_fields__
        assert "cluster_leakage_check" in fields, (
            "CorpusValidationReport must carry cluster_leakage_check for Phase D"
        )


# --------------------------------------------------------------------------- #
# 2. Clean leakage-free case
# --------------------------------------------------------------------------- #


class TestNoLeakage:
    """All members of one cluster in one split → no_leakage=True, report.ok=True."""

    def test_in_memory_clean_split(self, tmp_path):
        assignments = _make_in_memory_assignments()
        # cluster-A → train, cluster-B → test (perfectly leakage-free)
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path, cluster_assignments=assignments
        )
        assert report.cluster_leakage_check is not None
        leak = report.cluster_leakage_check
        assert leak.no_leakage
        assert leak.namespace_ok
        assert leak.leaking_clusters == {}
        assert leak.coverage_fraction == pytest.approx(1.0)
        # No error issues from the leakage check on a clean corpus.
        error_codes = {i.code for i in report.issues if i.severity == "error"}
        assert "cluster_spans_splits" not in error_codes

    def test_from_disk_clean_split(self, tmp_path):
        release_dir = _write_assignments_release(tmp_path)
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path, cluster_assignments_path=release_dir
        )
        assert report.cluster_leakage_check is not None
        assert report.cluster_leakage_check.no_leakage


# --------------------------------------------------------------------------- #
# 3. Leakage detection
# --------------------------------------------------------------------------- #


class TestLeakageDetection:
    """A cluster spanning > 1 split must be flagged as an error."""

    def test_cluster_spans_two_splits_records_error(self, tmp_path):
        assignments = _make_in_memory_assignments()
        # Sabotage cluster-A: rec-a1 in train, rec-a2 in test.
        splits = {"rec-a1": "train", "rec-a2": "test"}
        splits.update({f"rec-a{i}": "train" for i in [3, 4]})
        splits.update({f"rec-b{i}": "test" for i in [1, 2, 3, 4]})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path, cluster_assignments=assignments
        )
        leak = report.cluster_leakage_check
        assert leak is not None
        assert not leak.no_leakage
        assert "cluster-A" in leak.leaking_clusters
        assert set(leak.leaking_clusters["cluster-A"].keys()) == {"train", "test"}
        # Hard failure on the overall report.
        assert not report.ok
        codes = {i.code for i in report.issues if i.severity == "error"}
        assert "cluster_spans_splits" in codes


# --------------------------------------------------------------------------- #
# 4. Namespace mismatch handling
# --------------------------------------------------------------------------- #


class TestNamespaceMismatch:
    """Mismatched namespace records a warning; leakage check still runs but
    the report flags namespace_ok=False so consumers know the result is
    not load-bearing."""

    def test_namespace_mismatch_warning_but_no_hard_failure(self, tmp_path):
        assignments = _make_in_memory_assignments(namespace=NAMESPACE_RAW_PDB_ID)
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path,
            cluster_assignments=assignments,
            expected_cluster_namespace=NAMESPACE_PREPARED_RECORD_ID,
        )
        leak = report.cluster_leakage_check
        assert leak is not None
        assert not leak.namespace_ok
        assert leak.actual_namespace == NAMESPACE_RAW_PDB_ID
        assert leak.expected_namespace == NAMESPACE_PREPARED_RECORD_ID
        # Warning, not error.
        warning_codes = {i.code for i in report.issues if i.severity == "warning"}
        assert "cluster_leakage_namespace_mismatch" in warning_codes


# --------------------------------------------------------------------------- #
# 5. Mutual-exclusion of the two kwargs
# --------------------------------------------------------------------------- #


class TestMutualExclusion:
    def test_passing_both_path_and_object_raises(self, tmp_path):
        release_dir = _write_assignments_release(tmp_path)
        assignments = _make_in_memory_assignments()
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        with pytest.raises(ValueError, match="at most one"):
            validate_corpus_release(
                manifest_path,
                cluster_assignments_path=release_dir,
                cluster_assignments=assignments,
            )


# --------------------------------------------------------------------------- #
# 6. Bad in-memory type
# --------------------------------------------------------------------------- #


class TestBadInMemoryType:
    def test_non_clusterassignments_object_records_error(self, tmp_path):
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path,
            cluster_assignments={"this": "is not a ClusterAssignments"},
        )
        codes = {i.code for i in report.issues if i.severity == "error"}
        assert "cluster_leakage_bad_in_memory_type" in codes
        assert not report.ok


# --------------------------------------------------------------------------- #
# 7. Coverage gap reporting
# --------------------------------------------------------------------------- #


class TestCoverageGap:
    """Cluster artifact doesn't cover all training records → warning."""

    def test_partial_coverage_warning(self, tmp_path):
        assignments = _make_in_memory_assignments()
        # Add a training record not present in the cluster artifact.
        splits = {f"rec-a{i}": "train" for i in range(1, 5)}
        splits.update({f"rec-b{i}": "test" for i in range(1, 5)})
        splits["rec-unclustered"] = "val"
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(
            manifest_path, cluster_assignments=assignments
        )
        leak = report.cluster_leakage_check
        assert leak is not None
        assert leak.coverage_fraction < 1.0
        warning_codes = {i.code for i in report.issues if i.severity == "warning"}
        assert "cluster_partial_coverage" in warning_codes


# --------------------------------------------------------------------------- #
# 8. Backward compatibility (no cluster artifact → unchanged behavior)
# --------------------------------------------------------------------------- #


class TestBackwardCompatibility:
    """The new kwargs are optional; absent → cluster_leakage_check stays None."""

    def test_no_cluster_kwargs_leaves_field_none(self, tmp_path):
        splits = {f"rec-{i}": "train" for i in range(5)}
        manifest_path, _ = _build_fake_corpus_release(tmp_path, splits=splits)
        report = validate_corpus_release(manifest_path)
        assert report.cluster_leakage_check is None


# --------------------------------------------------------------------------- #
# 9. ClusterLeakageReport surface (post-Codex-review trimming)
# --------------------------------------------------------------------------- #


class TestReportFieldSurface:
    """Phase D's first round of code review (codex on PR #93) caught that
    the originally-included ``unavoidable_skew`` / ``actual_ratios``
    fields computed skew from realised counts as both numerator and
    denominator — always yielding zero. They were dropped from the
    ClusterLeakageReport schema in favour of the existing
    ClusterAwareSplitResult provenance (Phase C) which records skew
    accurately at split time. Pin the trimmed surface here so the
    fields don't accidentally come back without a real implementation.
    """

    def test_report_does_not_advertise_skew_fields(self):
        fields = set(ClusterLeakageReport.__dataclass_fields__.keys())
        # Fields that SHOULD stay:
        assert "no_leakage" in fields
        assert "leaking_clusters" in fields
        assert "coverage_fraction" in fields
        assert "cluster_size_summary" in fields
        # Fields that were dropped after the codex review on PR #93:
        assert "unavoidable_skew" not in fields, (
            "unavoidable_skew was removed because the computation was "
            "trivially-zero; restore it only with a real requested-ratio "
            "source (see ClusterAwareSplitResult / training manifest "
            "provenance)"
        )
        assert "actual_ratios" not in fields, (
            "actual_ratios was removed for the same reason — see Phase C "
            "ClusterAwareSplitResult for the accurate per-split fractions"
        )
