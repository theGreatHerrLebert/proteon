"""Validation and QA reporting for corpus releases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .corpus_release import CorpusReleaseManifest, load_corpus_release_manifest

try:
    import pyarrow as pa
    _HAS_PYARROW = True

    def pa_types_is_nested(t) -> bool:
        return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False

    def pa_types_is_nested(t) -> bool:  # type: ignore
        return False


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class ClusterLeakageReport:
    """Result of the optional cluster-leakage check (v0.3.0 Phase D).

    Populated when ``validate_corpus_release`` is called with a
    ``cluster_assignments_path`` pointing at a Phase B0
    ``ClusterAssignments`` release. The ``no_leakage`` invariant is
    the load-bearing assertion: it is True iff every cluster's members
    landed in exactly one split, which is exactly what
    ``cluster_aware_split`` is supposed to guarantee.

    ``namespace_ok`` is True iff the loaded assignments' namespace
    matches ``expected_namespace`` (default
    ``prepared_record_id`` — the canonical training-join namespace).
    A False here means the cluster artifact was built against a
    different ID system than the corpus it's being audited against,
    and the leakage check below is meaningless.

    ``cluster_size_summary`` (min/max/mean/median) and
    ``unavoidable_skew`` mirror the report attached to
    ``ClusterAwareSplitResult`` at split time, so the validator catches
    drift between the assignments-at-split-time and the
    assignments-at-validation-time (e.g. a different cluster artifact
    was loaded into the validator than the one originally used to
    build the split).
    """

    cluster_release_id: str
    expected_namespace: str
    actual_namespace: str
    namespace_ok: bool
    no_leakage: bool
    leaking_clusters: Dict[str, Dict[str, int]] = field(default_factory=dict)
    coverage_fraction: float = 1.0
    cluster_size_summary: Dict[str, float] = field(default_factory=dict)
    unavoidable_skew: bool = False
    actual_ratios: Dict[str, float] = field(default_factory=dict)


@dataclass
class CorpusValidationReport:
    release_id: str
    artifact_type: str = "validation_report"
    format: str = "proteon.corpus_validation.v0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ok: bool = True
    counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    completeness: Dict[str, float] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    cluster_leakage_check: Optional[ClusterLeakageReport] = None


def validate_corpus_release(
    corpus_release_manifest: str | Path,
    *,
    out_path: str | Path | None = None,
    cluster_assignments_path: str | Path | None = None,
    cluster_assignments: object | None = None,
    expected_cluster_namespace: str = "prepared_record_id",
) -> CorpusValidationReport:
    """Validate a corpus release and optionally write a JSON QA report.

    When ``cluster_assignments_path`` **or** ``cluster_assignments``
    is supplied, the validator also runs the v0.3.0 Phase D
    **cluster-leakage check**: it loads (or accepts in-memory) the
    ``ClusterAssignments``, asserts the namespace matches
    ``expected_cluster_namespace`` (default ``prepared_record_id``,
    the canonical training-join namespace), and verifies that every
    cluster's members landed in exactly one split. The result is
    recorded under ``report.cluster_leakage_check``.

    A leakage failure (any cluster spanning > 1 split) records an
    ``error``-severity issue and marks the whole report ``ok=False``;
    a namespace mismatch records a ``warning`` because the leakage
    assertion below it becomes meaningless without confirmed ID
    alignment.

    ``cluster_assignments_path`` and ``cluster_assignments`` are
    mutually exclusive — pass at most one. The in-memory form is what
    ``corpus_smoke.build_local_corpus_smoke_release`` uses when the
    caller supplied an in-memory ClusterAssignments to the smoke
    pipeline; the path form is the standalone audit entry point.
    """
    if cluster_assignments_path is not None and cluster_assignments is not None:
        raise ValueError(
            "pass at most one of cluster_assignments_path / cluster_assignments"
        )
    manifest = load_corpus_release_manifest(corpus_release_manifest)
    report = CorpusValidationReport(
        release_id=manifest.release_id,
        counts={
            "prepared": manifest.count_prepared,
            "sequence_examples": manifest.count_sequence_examples,
            "structure_examples": manifest.count_structure_examples,
            "training_examples": manifest.count_training_examples,
        },
        split_counts=dict(manifest.split_counts),
        failure_breakdown=dict(manifest.failure_breakdown),
    )

    _check_count_consistency(manifest, report)
    _check_training_release(manifest, report)
    _check_structure_tensor_completeness(manifest, report)
    if cluster_assignments_path is not None or cluster_assignments is not None:
        _check_cluster_leakage(
            manifest,
            report,
            cluster_assignments_path=(
                Path(cluster_assignments_path)
                if cluster_assignments_path is not None
                else None
            ),
            cluster_assignments=cluster_assignments,
            expected_namespace=expected_cluster_namespace,
        )
    report.ok = not any(issue.severity == "error" for issue in report.issues)

    if out_path is not None:
        Path(out_path).write_text(
            json.dumps(
                {
                    **asdict(report),
                    "issues": [asdict(issue) for issue in report.issues],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def _check_count_consistency(manifest: CorpusReleaseManifest, report: CorpusValidationReport) -> None:
    if manifest.count_training_examples > manifest.count_sequence_examples:
        report.issues.append(
            ValidationIssue("error", "training_exceeds_sequence", "training example count exceeds sequence example count")
        )
    if manifest.count_training_examples > manifest.count_structure_examples:
        report.issues.append(
            ValidationIssue("error", "training_exceeds_structure", "training example count exceeds structure example count")
        )
    if manifest.count_prepared and manifest.count_structure_examples > manifest.count_prepared:
        report.issues.append(
            ValidationIssue("warning", "structure_exceeds_prepared", "structure example count exceeds prepared record count")
        )


def _check_training_release(manifest: CorpusReleaseManifest, report: CorpusValidationReport) -> None:
    if manifest.training_release is None:
        report.issues.append(ValidationIssue("warning", "missing_training_release", "no training release linked"))
        return
    rows = _load_jsonl(Path(manifest.training_release) / "training_examples.jsonl")
    if len(rows) != manifest.count_training_examples:
        report.issues.append(
            ValidationIssue("error", "training_row_count_mismatch", "training_examples.jsonl row count does not match manifest count")
        )
    split_counts: Dict[str, int] = {}
    duplicate_ids = set()
    seen = set()
    for row in rows:
        rid = row["record_id"]
        if rid in seen:
            duplicate_ids.add(rid)
        seen.add(rid)
        split = str(row.get("split", "train"))
        split_counts[split] = split_counts.get(split, 0) + 1
    if duplicate_ids:
        report.issues.append(
            ValidationIssue("error", "duplicate_training_ids", f"duplicate training record_ids found: {sorted(duplicate_ids)[:5]}")
        )
    if split_counts != manifest.split_counts:
        report.issues.append(
            ValidationIssue("error", "split_count_mismatch", "training split counts do not match corpus manifest")
        )


def _check_structure_tensor_completeness(manifest: CorpusReleaseManifest, report: CorpusValidationReport) -> None:
    if manifest.structure_release is None:
        report.issues.append(ValidationIssue("warning", "missing_structure_release", "no structure release linked"))
        return
    if manifest.count_structure_examples == 0:
        report.issues.append(
            ValidationIssue("warning", "no_structure_examples", "structure release contains no supervision examples")
        )
        return
    tensor_path = Path(manifest.structure_release) / "examples" / "tensors.parquet"
    if not tensor_path.exists():
        report.issues.append(ValidationIssue("error", "missing_structure_tensors", "structure tensors.parquet is missing"))
        return
    # Stream per-row-group so peak memory is bounded even on archive-
    # scale releases; accumulate only scalar sums we need for fractions.
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(tensor_path)
    valid_residues = 0.0
    pseudo_sum = 0.0
    rigid_sum = 0.0
    chi_sum = 0.0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(
            rg_idx,
            columns=["seq_mask", "pseudo_beta_mask", "rigidgroups_gt_exists", "chi_mask"],
        )
        # Each column is list<...>; list_flatten drops the outer list
        # dimension but preserves inner FixedSizeList dims, and subsequent
        # flattens unwrap those. Summing after the full flatten gives the
        # scalar total across all rows in the row-group.
        def _flat_sum(arr):
            while pa_types_is_nested(arr.type):
                arr = pc.list_flatten(arr)
            return float(pc.sum(arr).as_py() or 0.0)
        valid_residues += _flat_sum(rg.column("seq_mask"))
        pseudo_sum += _flat_sum(rg.column("pseudo_beta_mask"))
        rigid_sum += _flat_sum(rg.column("rigidgroups_gt_exists"))
        chi_sum += _flat_sum(rg.column("chi_mask"))

    if valid_residues <= 0:
        report.issues.append(ValidationIssue("error", "no_valid_structure_residues", "structure seq_mask has no valid residues"))
        return
    report.completeness["pseudo_beta_fraction"] = pseudo_sum / valid_residues
    report.completeness["rigidgroup_frame_fraction"] = rigid_sum / max(valid_residues * 8.0, 1.0)
    report.completeness["chi_angle_fraction"] = chi_sum / max(valid_residues * 4.0, 1.0)

    if report.completeness["pseudo_beta_fraction"] < 0.95:
        report.issues.append(
            ValidationIssue("warning", "low_pseudo_beta_fraction", "pseudo-beta completeness is below 95%")
        )


def _check_cluster_leakage(
    manifest: CorpusReleaseManifest,
    report: CorpusValidationReport,
    *,
    cluster_assignments_path: Optional[Path] = None,
    cluster_assignments: object | None = None,
    expected_namespace: str,
) -> None:
    """Verify no cluster spans more than one split.

    Accepts either a path (loaded fresh, the standalone-audit form) or
    an in-memory ``ClusterAssignments`` (already loaded — used by
    ``corpus_smoke.build_local_corpus_smoke_release`` when the caller
    supplied assignments to the smoke pipeline).

    Imported lazily because corpus_validation is consumed by
    corpus_smoke, and a top-level import of cluster_assignments would
    create a small import-time circle through validate_corpus_release.
    """
    # Lazy import — see docstring.
    from .cluster_assignments import (
        ClusterAssignments,
        DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE,
        load_cluster_assignments,
    )

    if manifest.training_release is None:
        report.issues.append(
            ValidationIssue(
                "warning",
                "cluster_leakage_skipped_no_training_release",
                "cannot run cluster-leakage check: no training release linked",
            )
        )
        return

    if cluster_assignments is not None:
        if not isinstance(cluster_assignments, ClusterAssignments):
            report.issues.append(
                ValidationIssue(
                    "error",
                    "cluster_leakage_bad_in_memory_type",
                    (
                        f"cluster_assignments kwarg must be a ClusterAssignments "
                        f"instance, got {type(cluster_assignments).__name__}"
                    ),
                )
            )
            return
        assignments = cluster_assignments
    else:
        try:
            assignments = load_cluster_assignments(cluster_assignments_path)
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            report.issues.append(
                ValidationIssue(
                    "error",
                    "cluster_leakage_load_failed",
                    f"failed to load cluster assignments at {cluster_assignments_path}: {exc}",
                )
            )
            return

    namespace_ok = assignments.manifest.sequence_id_namespace == expected_namespace
    if not namespace_ok:
        report.issues.append(
            ValidationIssue(
                "warning",
                "cluster_leakage_namespace_mismatch",
                (
                    f"cluster assignments use sequence_id_namespace="
                    f"{assignments.manifest.sequence_id_namespace!r}, expected "
                    f"{expected_namespace!r}; leakage check below assumes IDs "
                    f"align with training join key — interpret with caution"
                ),
            )
        )

    # Load the per-record split assignments from the training release.
    training_rows = _load_jsonl(
        Path(manifest.training_release) / "training_examples.jsonl"
    )
    record_to_split: Dict[str, str] = {row["record_id"]: str(row.get("split", "train")) for row in training_rows}

    cluster_to_splits: Dict[str, Dict[str, int]] = {}
    covered_ids = set()
    for row in assignments.rows:
        split = record_to_split.get(row.record_id)
        if split is None:
            # Record present in the cluster artifact but absent from
            # the training release. Surfaces as a low coverage_fraction
            # below; not itself a leakage failure.
            continue
        covered_ids.add(row.record_id)
        cluster_to_splits.setdefault(row.cluster_id, {}).setdefault(split, 0)
        cluster_to_splits[row.cluster_id][split] += 1

    leaking = {
        cid: split_dist
        for cid, split_dist in cluster_to_splits.items()
        if len(split_dist) > 1
    }
    no_leakage = not leaking

    if leaking:
        sample = dict(list(leaking.items())[:3])
        report.issues.append(
            ValidationIssue(
                "error",
                "cluster_spans_splits",
                (
                    f"{len(leaking)} cluster(s) span more than one split — "
                    f"leakage invariant violated. Sample: {sample}"
                ),
            )
        )

    coverage_fraction = (
        len(covered_ids) / len(record_to_split) if record_to_split else 1.0
    )
    if coverage_fraction < 1.0 and namespace_ok:
        report.issues.append(
            ValidationIssue(
                "warning",
                "cluster_partial_coverage",
                (
                    f"cluster assignments cover {coverage_fraction:.4f} of "
                    f"training records; some training records have no "
                    f"cluster annotation"
                ),
            )
        )

    # Cluster size summary + skew measurement against the realised
    # split distribution.
    sizes: Dict[str, int] = {}
    for cid, split_dist in cluster_to_splits.items():
        sizes[cid] = sum(split_dist.values())
    if sizes:
        sorted_sizes = sorted(sizes.values())
        n = len(sorted_sizes)
        median = (
            float(sorted_sizes[n // 2])
            if n % 2
            else float((sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2.0)
        )
        cluster_size_summary = {
            "min": float(sorted_sizes[0]),
            "max": float(sorted_sizes[-1]),
            "mean": float(sum(sorted_sizes) / n),
            "median": median,
        }
    else:
        cluster_size_summary = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    # Compare the realised per-split fractions against the corpus-
    # manifest split_counts ratios; surface as unavoidable_skew when
    # the deviation exceeds DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE.
    total_split = sum(manifest.split_counts.values()) or 1
    requested_ratios = {
        k: v / total_split for k, v in manifest.split_counts.items()
    }
    realised_per_split: Dict[str, int] = {}
    for split in record_to_split.values():
        realised_per_split[split] = realised_per_split.get(split, 0) + 1
    total_realised = sum(realised_per_split.values()) or 1
    actual_ratios = {
        k: v / total_realised for k, v in realised_per_split.items()
    }
    max_skew = max(
        (
            abs(actual_ratios.get(k, 0.0) - requested_ratios.get(k, 0.0))
            for k in set(actual_ratios) | set(requested_ratios)
        ),
        default=0.0,
    )
    unavoidable_skew = max_skew > DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE

    report.cluster_leakage_check = ClusterLeakageReport(
        cluster_release_id=assignments.manifest.release_id,
        expected_namespace=expected_namespace,
        actual_namespace=assignments.manifest.sequence_id_namespace,
        namespace_ok=namespace_ok,
        no_leakage=no_leakage,
        leaking_clusters=leaking,
        coverage_fraction=coverage_fraction,
        cluster_size_summary=cluster_size_summary,
        unavoidable_skew=unavoidable_skew,
        actual_ratios=actual_ratios,
    )


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
