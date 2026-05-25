"""Contract test for v0.3.0 Phase C ``cluster_aware_split``.

The leakage invariant — no cluster spans more than one split — is what
makes the whole training-corpus stack defensible against
contrastive-learning false positives and shortcut learning. The
codex reviews on the parent v0.3.0 plan (Q5 / catch #9) and the Phase
B0 plan (catch #5 / catch #6) drove four asymmetric defaults this
test pins:

1. **Strict coverage by default** — partial-coverage clusterings raise
   ``ClusterCoverageError`` rather than silently producing singleton-ish
   splits.
2. **Unsafe namespaces rejected by default** — ``raw_pdb_id`` and
   ``uniprot_id`` can be many-to-one after chain expansion and corrupt
   the training-example join; ``allow_unsafe_namespaces=True`` is the
   explicit opt-in.
3. **Composite grouping by union-find** — when both cluster_id and
   sibling-chain ``grouping_keys`` constrain a record, both apply (not
   one wins).
4. **Skew is informational, not failing** — large dominant clusters
   produce unavoidable skew; the result carries
   ``bounded_skew=False`` but the assignment is still leakage-free.
"""

from __future__ import annotations

from typing import List

import pytest

import proteon
from proteon.cluster_assignments import (
    DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE,
    NAMESPACE_PREPARED_RECORD_ID,
    NAMESPACE_RAW_PDB_ID,
    NAMESPACE_UNIPROT_ID,
    ClusterAssignmentRow,
    ClusterAssignments,
    ClusterAssignmentsManifest,
    ClusterAwareSplitResult,
    ClusterCoverageError,
    cluster_aware_split,
)


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #


def _assignments(
    rows: List[ClusterAssignmentRow],
    *,
    namespace: str = NAMESPACE_PREPARED_RECORD_ID,
) -> ClusterAssignments:
    """Build a ClusterAssignments in-memory without writing to disk.

    Avoids the full release-builder round-trip; tests that need
    structural validation already cover that in
    ``tests/test_cluster_assignments.py``.
    """
    return ClusterAssignments(
        manifest=ClusterAssignmentsManifest(
            release_id="test",
            sequence_id_namespace=namespace,
            count_sequences=len(rows),
            count_clusters=len({r.cluster_id for r in rows}),
        ),
        rows=tuple(rows),
    )


def _eight_record_two_clusters() -> ClusterAssignments:
    """Two clusters of size 4 each — clean 50/50 split is achievable."""
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
    return _assignments(rows)


# --------------------------------------------------------------------------- #
# 1. Public-API surface
# --------------------------------------------------------------------------- #


class TestPublicSurface:
    def test_cluster_aware_split_is_top_level(self):
        assert "cluster_aware_split" in proteon.__all__
        assert callable(proteon.cluster_aware_split)

    def test_result_type_is_top_level(self):
        assert "ClusterAwareSplitResult" in proteon.__all__
        assert isinstance(proteon.ClusterAwareSplitResult.__dataclass_fields__, dict)

    def test_default_skew_tolerance_constant_exported(self):
        assert "DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE" in proteon.__all__
        assert proteon.DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE == DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE


# --------------------------------------------------------------------------- #
# 2. Core leakage invariant
# --------------------------------------------------------------------------- #


class TestLeakageInvariant:
    def test_no_cluster_spans_splits(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        result = cluster_aware_split(a, rids, {"train": 0.5, "val": 0.25, "test": 0.25})
        # For every cluster, all members share a split.
        for cluster_id in {row.cluster_id for row in a.rows}:
            splits = {result.assignments[m] for m in a.members_of(cluster_id)}
            assert len(splits) == 1, (
                f"cluster {cluster_id} spans splits {splits} — leakage invariant violated"
            )

    def test_singletons_split_independently(self):
        rows = [
            ClusterAssignmentRow(
                record_id=f"rec-{i}",
                cluster_id=f"cluster-{i}",
                representative_record_id=f"rec-{i}",
                is_representative=True,
                cluster_size=1,
            )
            for i in range(20)
        ]
        a = _assignments(rows)
        result = cluster_aware_split(
            a, [r.record_id for r in rows], {"train": 0.8, "val": 0.1, "test": 0.1}
        )
        # 20 singletons shouldn't all land in one split.
        seen_splits = set(result.assignments.values())
        assert len(seen_splits) >= 2

    def test_result_assignments_cover_all_inputs(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        result = cluster_aware_split(a, rids)
        assert set(result.assignments.keys()) == set(rids)


# --------------------------------------------------------------------------- #
# 3. Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_same_seed_same_input_same_output(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        r1 = cluster_aware_split(a, rids, seed=42)
        r2 = cluster_aware_split(a, rids, seed=42)
        assert r1.assignments == r2.assignments

    def test_different_seed_can_yield_different_split(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        r1 = cluster_aware_split(a, rids, seed=0)
        r2 = cluster_aware_split(a, rids, seed=999)
        # With 2 clusters of size 4 and 3 splits, the split MIGHT be
        # identical by coincidence; assert at least one seed pair
        # diverges by trying a small range.
        diverged = False
        for s in [1, 2, 3, 4, 5, 7, 11, 13, 17, 19]:
            ri = cluster_aware_split(a, rids, seed=s)
            if ri.assignments != r1.assignments:
                diverged = True
                break
        assert diverged, "no seed in {1..19} diverged from seed=0 — seed input is being ignored"

    def test_input_order_does_not_affect_output(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        r1 = cluster_aware_split(a, rids, seed=7)
        r2 = cluster_aware_split(a, list(reversed(rids)), seed=7)
        assert r1.assignments == r2.assignments


# --------------------------------------------------------------------------- #
# 4. Composite grouping (union-find)
# --------------------------------------------------------------------------- #


class TestCompositeGrouping:
    """When both ``cluster_id`` and ``grouping_keys`` constrain a record,
    union-find merges the equivalence classes — both constraints apply.
    """

    def test_sibling_chains_with_same_grouping_key_share_split(self):
        # Two singletons whose grouping_key claims they're siblings.
        rows = [
            ClusterAssignmentRow(
                record_id="chainA",
                cluster_id="c-a",
                representative_record_id="chainA",
                is_representative=True,
                cluster_size=1,
            ),
            ClusterAssignmentRow(
                record_id="chainB",
                cluster_id="c-b",  # different cluster
                representative_record_id="chainB",
                is_representative=True,
                cluster_size=1,
            ),
        ]
        a = _assignments(rows)
        # Both come from the same parent PDB structure; grouping_keys
        # forces them together.
        result = cluster_aware_split(
            a,
            ["chainA", "chainB"],
            grouping_keys=["parent-1", "parent-1"],
        )
        assert result.assignments["chainA"] == result.assignments["chainB"]

    def test_cluster_constraint_dominates_when_no_grouping_keys(self):
        # Without sibling info, only cluster_id constrains.
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        result = cluster_aware_split(a, rids)
        # cluster-A members share a split, cluster-B members share a split
        a_split = result.assignments["rec-a1"]
        for m in ["rec-a2", "rec-a3", "rec-a4"]:
            assert result.assignments[m] == a_split

    def test_mismatched_grouping_keys_length_raises(self):
        a = _eight_record_two_clusters()
        rids = list(a.record_ids)
        with pytest.raises(ValueError, match="grouping_keys length"):
            cluster_aware_split(a, rids, grouping_keys=["only-one"])


# --------------------------------------------------------------------------- #
# 5. Strict coverage default
# --------------------------------------------------------------------------- #


class TestStrictCoverageDefault:
    def test_partial_coverage_raises_by_default(self):
        a = _eight_record_two_clusters()
        # Ask to split a record that isn't in the cluster artifact.
        with pytest.raises(ClusterCoverageError, match="missing"):
            cluster_aware_split(a, list(a.record_ids) + ["unknown-rec"])

    def test_partial_coverage_allowed_when_strict_disabled(self):
        a = _eight_record_two_clusters()
        # With strict_coverage=False, the validator returns a report;
        # the missing rid will then fail on cluster_id_for lookup at
        # the split step (KeyError) — proving why the strict default
        # is the right one.
        with pytest.raises(KeyError):
            cluster_aware_split(
                a, list(a.record_ids) + ["unknown-rec"], strict_coverage=False
            )


# --------------------------------------------------------------------------- #
# 6. Unsafe namespace rejection
# --------------------------------------------------------------------------- #


class TestUnsafeNamespaceRejection:
    def test_raw_pdb_id_namespace_rejected_by_default(self):
        a = _assignments(
            [
                ClusterAssignmentRow(
                    record_id="1crn",
                    cluster_id="c-1",
                    representative_record_id="1crn",
                    is_representative=True,
                    cluster_size=1,
                )
            ],
            namespace=NAMESPACE_RAW_PDB_ID,
        )
        with pytest.raises(ValueError, match="unsafe for training-example splits"):
            cluster_aware_split(a, ["1crn"])

    def test_uniprot_id_namespace_rejected_by_default(self):
        a = _assignments(
            [
                ClusterAssignmentRow(
                    record_id="P12345",
                    cluster_id="c-1",
                    representative_record_id="P12345",
                    is_representative=True,
                    cluster_size=1,
                )
            ],
            namespace=NAMESPACE_UNIPROT_ID,
        )
        with pytest.raises(ValueError, match="unsafe for training-example splits"):
            cluster_aware_split(a, ["P12345"])

    def test_allow_unsafe_namespaces_opt_in_bypasses_rejection(self):
        a = _assignments(
            [
                ClusterAssignmentRow(
                    record_id="1crn",
                    cluster_id="c-1",
                    representative_record_id="1crn",
                    is_representative=True,
                    cluster_size=1,
                )
            ],
            namespace=NAMESPACE_RAW_PDB_ID,
        )
        # Opt-in must work, otherwise tools that supply raw_pdb_id with
        # verified one-to-one mapping have no escape hatch.
        result = cluster_aware_split(a, ["1crn"], allow_unsafe_namespaces=True)
        assert "1crn" in result.assignments


# --------------------------------------------------------------------------- #
# 7. Skew report
# --------------------------------------------------------------------------- #


class TestSkewReport:
    def test_many_balanced_singletons_skew_below_tolerance(self):
        # Twenty singleton clusters — hash distribution averages out
        # close to the requested ratios. With only 2-4 clusters the
        # hash either over- or under-shoots by 25-50% per split, so
        # "bounded skew" is only a meaningful assertion at moderate N.
        rows = [
            ClusterAssignmentRow(
                record_id=f"rec-{i:02d}",
                cluster_id=f"c-{i:02d}",
                representative_record_id=f"rec-{i:02d}",
                is_representative=True,
                cluster_size=1,
            )
            for i in range(40)
        ]
        a = _assignments(rows)
        result = cluster_aware_split(
            a, [r.record_id for r in rows], {"train": 0.5, "val": 0.5}, seed=0
        )
        assert result.bounded_skew, (
            f"40 balanced singletons should yield bounded skew, got {result.skew}"
        )

    def test_one_dominant_cluster_produces_unavoidable_skew(self):
        # 10 records all in cluster-X (single cluster) → entire corpus
        # must land in one split → 100% in one bucket, 0% in others.
        # This is the failure mode the skew report exists to flag:
        # the assignment is still leakage-free, but the corpus is
        # unsplittable in the usual sense.
        rows = [
            ClusterAssignmentRow(
                record_id=f"rec-{i}",
                cluster_id="cluster-X",
                representative_record_id="rec-0",
                is_representative=(i == 0),
                cluster_size=10,
            )
            for i in range(10)
        ]
        a = _assignments(rows)
        result = cluster_aware_split(
            a, [r.record_id for r in rows], {"train": 0.8, "val": 0.1, "test": 0.1}
        )
        # Single cluster → all 10 records land in exactly one split.
        # That's the leakage-free outcome; "skew" simply reports the
        # gap between requested and actual ratios.
        assert len(set(result.assignments.values())) == 1
        # max_skew is at least the largest requested ratio (because one
        # bucket gets 100% — overshoots by 1.0 - 0.8 = 0.2 on train, or
        # undershoots by 0.8 on val/test). Float subtraction loses a
        # ulp, so use a tolerant lower bound.
        assert result.max_skew >= 0.19999
        # Skew tolerance is 0.10 by default; this dominant cluster
        # therefore trips bounded_skew=False, surfacing the issue to
        # callers without changing the assignment.
        assert not result.bounded_skew

    def test_skew_report_carries_actual_and_requested_ratios(self):
        a = _eight_record_two_clusters()
        result = cluster_aware_split(
            a, list(a.record_ids), {"train": 0.5, "val": 0.5}
        )
        assert set(result.requested_ratios.keys()) == {"train", "val"}
        assert set(result.actual_ratios.keys()) == {"train", "val"}
        assert sum(result.actual_ratios.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 8. Default ratios
# --------------------------------------------------------------------------- #


class TestDefaultRatios:
    def test_default_ratios_are_80_10_10_train_val_test(self):
        a = _eight_record_two_clusters()
        result = cluster_aware_split(a, list(a.record_ids))
        assert set(result.requested_ratios.keys()) == {"train", "val", "test"}
        assert result.requested_ratios["train"] == pytest.approx(0.8)
        assert result.requested_ratios["val"] == pytest.approx(0.1)
        assert result.requested_ratios["test"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# 9. Result dataclass shape
# --------------------------------------------------------------------------- #


class TestResultDataclassShape:
    def test_all_documented_fields_present(self):
        a = _eight_record_two_clusters()
        result = cluster_aware_split(a, list(a.record_ids))
        for field_name in (
            "assignments",
            "requested_ratios",
            "actual_ratios",
            "skew",
            "max_skew",
            "skew_tolerance",
            "bounded_skew",
        ):
            assert hasattr(result, field_name), f"ClusterAwareSplitResult missing {field_name}"
        assert isinstance(result.bounded_skew, bool)
        assert isinstance(result.assignments, dict)
