"""Contract test for the v0.3.0 Phase B0 cluster-assignments artifact.

The cluster-assignments artifact is the keystone contract Phase C
(``cluster_aware_split``), Phase D (cluster-leakage check inside
``validate_corpus_release``), and Phase E (hard-negative mining with
same-cluster exclusion) all consume. The codex review on
``TO_V030_PHASE_B0_CLUSTER_ASSIGNMENTS.md`` explicitly called out the
failure mode this file exists to prevent:

    > Externally supplied clusters can be scientifically meaningless
    > while structurally valid.

Per that review, the validators here must catch every shape of structural
invalidity (duplicate ``record_id`` rows, missing/multiple/mis-pointed
representatives, denormalised ``cluster_size`` drift, manifest count
inconsistencies, namespace mismatches), and the manifest's provenance
fields must round-trip through both JSONL and Parquet so consumers can
audit ``cluster_id`` authority.

Uses a ``CLUSTER_ASSIGNMENT_FIELDS``-driven sweep parallel to
``TENSOR_FIELDS`` in ``tests/test_structure_supervision_contract.py``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("pyarrow")

import proteon
from proteon.cluster_assignments import (
    ALL_CLUSTER_NAMESPACES,
    ALL_INPUT_DIGEST_KINDS,
    CLUSTER_ASSIGNMENTS_FORMAT,
    CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION,
    DIGEST_KIND_FASTA,
    DIGEST_KIND_RECORD_ID_LIST,
    NAMESPACE_CUSTOM,
    NAMESPACE_PREPARED_RECORD_ID,
    NAMESPACE_RAW_PDB_ID,
    ClusterAssignmentRow,
    ClusterAssignments,
    ClusterAssignmentsManifest,
    ClusterCoverageError,
    _cluster_assignment_fields_factory,
    build_cluster_assignments_release,
    build_cluster_assignments_schema,
    compute_record_id_digest,
    iter_cluster_assignments,
    load_cluster_assignments,
    load_cluster_assignments_jsonl,
    load_cluster_assignments_parquet,
    validate_cluster_coverage,
    validate_cluster_namespace,
    validate_cluster_record_id_uniqueness,
    validate_cluster_representative_consistency,
    validate_cluster_size_consistency,
    validate_manifest_consistency,
    write_cluster_assignments_jsonl,
    write_cluster_assignments_parquet,
)


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #


def _two_cluster_rows() -> List[ClusterAssignmentRow]:
    """Three records in two clusters: cluster-A has two members (with a
    declared representative), cluster-B has one (singleton).
    """
    return [
        ClusterAssignmentRow(
            record_id="rec-1",
            cluster_id="cluster-A",
            representative_record_id="rec-1",
            is_representative=True,
            cluster_size=2,
            source_id="rec-1.pdb",
        ),
        ClusterAssignmentRow(
            record_id="rec-2",
            cluster_id="cluster-A",
            representative_record_id="rec-1",
            is_representative=False,
            cluster_size=2,
            source_id="rec-2.pdb",
        ),
        ClusterAssignmentRow(
            record_id="rec-3",
            cluster_id="cluster-B",
            representative_record_id="rec-3",
            is_representative=True,
            cluster_size=1,
            source_id=None,
        ),
    ]


def _release(tmp_path: Path, *, namespace: str = NAMESPACE_PREPARED_RECORD_ID) -> Path:
    return build_cluster_assignments_release(
        _two_cluster_rows(),
        tmp_path / "release",
        release_id="b0-contract",
        tool="mmseqs2",
        tool_version="14.7e284",
        params={"min_seq_id": 0.3, "coverage": 0.8, "coverage_mode": "bidirectional"},
        input_digest="deadbeef" * 8,
        input_digest_kind=DIGEST_KIND_FASTA,
        representative_selection="mmseqs2_set_cover_default",
        sequence_id_namespace=namespace,
        created_from_release_id="seq-release-v1",
        code_rev="abc123",
        config_rev="cfg1",
        provenance={"engine": "ci"},
    )


# --------------------------------------------------------------------------- #
# 1. Top-level export surface
# --------------------------------------------------------------------------- #


class TestPublicAPISurface:
    """Every symbol Phase B0 promises is reachable as ``proteon.<name>``
    AND is listed in ``proteon.__all__``."""

    @pytest.mark.parametrize(
        "name",
        [
            # Dataclasses
            "ClusterAssignmentRow",
            "ClusterAssignments",
            "ClusterAssignmentsManifest",
            "ClusterCoverageReport",
            "ClusterValidationReport",
            "ClusterCoverageError",
            # Constants
            "CLUSTER_ASSIGNMENTS_FORMAT",
            "CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION",
            # Namespace enum
            "ALL_CLUSTER_NAMESPACES",
            "NAMESPACE_PREPARED_RECORD_ID",
            "NAMESPACE_RAW_PDB_ID",
            "NAMESPACE_UNIPROT_ID",
            "NAMESPACE_CUSTOM",
            # Digest-kind enum
            "ALL_INPUT_DIGEST_KINDS",
            "DIGEST_KIND_FASTA",
            "DIGEST_KIND_RECORD_ID_LIST",
            "DIGEST_KIND_CUSTOM",
            # I/O
            "build_cluster_assignments_release",
            "build_cluster_assignments_schema",
            "compute_record_id_digest",
            "load_cluster_assignments",
            "iter_cluster_assignments",
            "write_cluster_assignments_jsonl",
            "load_cluster_assignments_jsonl",
            "write_cluster_assignments_parquet",
            "load_cluster_assignments_parquet",
            # Validators
            "validate_cluster_namespace",
            "validate_cluster_coverage",
            "validate_cluster_representative_consistency",
            "validate_cluster_size_consistency",
            "validate_cluster_record_id_uniqueness",
            "validate_manifest_consistency",
        ],
    )
    def test_symbol_in_all_and_resolves(self, name: str):
        assert name in proteon.__all__, f"{name} missing from proteon.__all__"
        assert hasattr(proteon, name), f"proteon.{name} does not resolve"

    def test_schema_constants_pinned(self):
        assert proteon.CLUSTER_ASSIGNMENTS_FORMAT == "proteon.cluster_assignments.parquet.v0"
        assert proteon.CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION == 1

    def test_namespace_enum_is_closed(self):
        # Sanity check: the tuple membership is what the manifest's
        # __post_init__ validates against.
        assert NAMESPACE_PREPARED_RECORD_ID in ALL_CLUSTER_NAMESPACES
        assert NAMESPACE_RAW_PDB_ID in ALL_CLUSTER_NAMESPACES
        assert NAMESPACE_CUSTOM in ALL_CLUSTER_NAMESPACES
        assert "garbage" not in ALL_CLUSTER_NAMESPACES


# --------------------------------------------------------------------------- #
# 2. Schema source-of-truth sweep
# --------------------------------------------------------------------------- #


class TestSchemaFieldContract:
    """Every entry in ``CLUSTER_ASSIGNMENT_FIELDS`` appears in the Arrow
    schema with the declared type + nullability, and every field round-trips
    through both JSONL and Parquet."""

    def test_every_declared_field_in_arrow_schema(self):
        schema = build_cluster_assignments_schema()
        declared = _cluster_assignment_fields_factory()
        names_in_schema = {f.name: f for f in schema}
        for name, dtype, nullable, _attr in declared:
            assert name in names_in_schema, f"{name} missing from Parquet schema"
            field = names_in_schema[name]
            assert field.type == dtype, f"{name}: schema type {field.type} != declared {dtype}"
            assert field.nullable == nullable, (
                f"{name}: schema nullable={field.nullable} != declared {nullable}"
            )

    @pytest.mark.parametrize(
        "name,_dtype,_nullable,attr", _cluster_assignment_fields_factory()
    )
    def test_jsonl_round_trip_preserves_field(self, tmp_path, name, _dtype, _nullable, attr):
        original = _two_cluster_rows()
        path = tmp_path / "rows.jsonl"
        write_cluster_assignments_jsonl(original, path)
        loaded = load_cluster_assignments_jsonl(path)
        for o, r in zip(original, loaded):
            assert getattr(o, attr) == getattr(r, attr), (
                f"JSONL round-trip drifted on field {name}"
            )

    @pytest.mark.parametrize(
        "name,_dtype,_nullable,attr", _cluster_assignment_fields_factory()
    )
    def test_parquet_round_trip_preserves_field(self, tmp_path, name, _dtype, _nullable, attr):
        original = _two_cluster_rows()
        path = tmp_path / "rows.parquet"
        write_cluster_assignments_parquet(original, path)
        loaded = load_cluster_assignments_parquet(path)
        for o, r in zip(original, loaded):
            assert getattr(o, attr) == getattr(r, attr), (
                f"Parquet round-trip drifted on field {name}"
            )


# --------------------------------------------------------------------------- #
# 3. Release builder + provenance round-trip
# --------------------------------------------------------------------------- #


class TestReleaseAndProvenance:
    def test_release_round_trip_preserves_full_manifest(self, tmp_path):
        release_dir = _release(tmp_path)
        assignments = load_cluster_assignments(release_dir)
        m = assignments.manifest
        assert m.release_id == "b0-contract"
        assert m.tool == "mmseqs2"
        assert m.tool_version == "14.7e284"
        assert m.params == {
            "min_seq_id": 0.3,
            "coverage": 0.8,
            "coverage_mode": "bidirectional",
        }
        assert m.input_digest == "deadbeef" * 8
        assert m.input_digest_kind == DIGEST_KIND_FASTA
        assert m.representative_selection == "mmseqs2_set_cover_default"
        assert m.sequence_id_namespace == NAMESPACE_PREPARED_RECORD_ID
        assert m.created_from_release_id == "seq-release-v1"
        assert m.code_rev == "abc123"
        assert m.config_rev == "cfg1"
        assert m.count_sequences == 3
        assert m.count_clusters == 2
        assert m.count_singletons == 1
        assert m.cluster_size_summary["min"] == 1.0
        assert m.cluster_size_summary["max"] == 2.0
        assert m.assignments_parquet_sha256  # populated, non-empty
        assert m.provenance == {"engine": "ci"}

    def test_release_record_id_digest_is_deterministic(self, tmp_path):
        a = _release(tmp_path / "a")
        b = _release(tmp_path / "b")
        assert load_cluster_assignments(a).manifest.record_id_digest == (
            load_cluster_assignments(b).manifest.record_id_digest
        )

    def test_record_id_digest_helper_is_order_invariant(self):
        a = compute_record_id_digest(["rec-3", "rec-1", "rec-2"])
        b = compute_record_id_digest(["rec-1", "rec-2", "rec-3"])
        assert a == b


# --------------------------------------------------------------------------- #
# 4. Manifest __post_init__ validators
# --------------------------------------------------------------------------- #


class TestManifestPostInitValidators:
    def test_invalid_namespace_raises(self):
        with pytest.raises(ValueError, match="sequence_id_namespace"):
            ClusterAssignmentsManifest(
                release_id="r", sequence_id_namespace="bogus"
            )

    def test_invalid_input_digest_kind_raises(self):
        with pytest.raises(ValueError, match="input_digest_kind"):
            ClusterAssignmentsManifest(
                release_id="r", input_digest_kind="md5"
            )

    def test_custom_namespace_without_description_raises(self):
        with pytest.raises(ValueError, match="custom_namespace_description"):
            ClusterAssignmentsManifest(
                release_id="r",
                sequence_id_namespace=NAMESPACE_CUSTOM,
                custom_namespace_description=None,
            )
        # Empty string also rejected.
        with pytest.raises(ValueError, match="custom_namespace_description"):
            ClusterAssignmentsManifest(
                release_id="r",
                sequence_id_namespace=NAMESPACE_CUSTOM,
                custom_namespace_description="   ",
            )

    def test_custom_namespace_with_description_accepted(self):
        m = ClusterAssignmentsManifest(
            release_id="r",
            sequence_id_namespace=NAMESPACE_CUSTOM,
            custom_namespace_description="bespoke chain-renumber mapping v1",
        )
        assert m.sequence_id_namespace == NAMESPACE_CUSTOM


# --------------------------------------------------------------------------- #
# 5. Structural validators
# --------------------------------------------------------------------------- #


class TestStructuralValidators:
    def test_duplicate_record_id_raises(self):
        rows = _two_cluster_rows()
        rows.append(
            ClusterAssignmentRow(
                record_id="rec-1",  # duplicate
                cluster_id="cluster-C",
                representative_record_id="rec-1",
                is_representative=True,
                cluster_size=1,
            )
        )
        with pytest.raises(ValueError, match="duplicate record_id"):
            validate_cluster_record_id_uniqueness(rows)

    def test_empty_record_id_raises(self):
        rows = _two_cluster_rows()
        rows[0] = ClusterAssignmentRow(
            record_id="",
            cluster_id="cluster-A",
            representative_record_id="",
            is_representative=True,
            cluster_size=2,
        )
        with pytest.raises(ValueError, match="empty record_id"):
            validate_cluster_record_id_uniqueness(rows)

    def test_missing_representative_raises(self):
        rows = [
            ClusterAssignmentRow(
                record_id="rec-1",
                cluster_id="cluster-A",
                representative_record_id="rec-99",
                is_representative=False,
                cluster_size=1,
            ),
        ]
        with pytest.raises(ValueError, match="no representative"):
            validate_cluster_representative_consistency(rows)

    def test_multiple_representatives_in_one_cluster_raises(self):
        rows = [
            ClusterAssignmentRow(
                record_id="rec-1",
                cluster_id="cluster-A",
                representative_record_id="rec-1",
                is_representative=True,
                cluster_size=2,
            ),
            ClusterAssignmentRow(
                record_id="rec-2",
                cluster_id="cluster-A",
                representative_record_id="rec-2",
                is_representative=True,
                cluster_size=2,
            ),
        ]
        with pytest.raises(ValueError, match="multiple representatives"):
            validate_cluster_representative_consistency(rows)

    def test_representative_pointer_drift_raises(self):
        rows = [
            ClusterAssignmentRow(
                record_id="rec-1",
                cluster_id="cluster-A",
                representative_record_id="rec-1",
                is_representative=True,
                cluster_size=2,
            ),
            ClusterAssignmentRow(
                record_id="rec-2",
                cluster_id="cluster-A",
                representative_record_id="rec-99",  # mis-pointed
                is_representative=False,
                cluster_size=2,
            ),
        ]
        with pytest.raises(ValueError, match="claims representative_record_id"):
            validate_cluster_representative_consistency(rows)

    def test_cluster_size_mismatch_raises(self):
        rows = _two_cluster_rows()
        rows[0] = ClusterAssignmentRow(
            record_id="rec-1",
            cluster_id="cluster-A",
            representative_record_id="rec-1",
            is_representative=True,
            cluster_size=99,  # wrong
        )
        with pytest.raises(ValueError, match="cluster_size=99"):
            validate_cluster_size_consistency(rows)

    def test_cluster_size_zero_raises(self):
        rows = [
            ClusterAssignmentRow(
                record_id="rec-1",
                cluster_id="cluster-A",
                representative_record_id="rec-1",
                is_representative=True,
                cluster_size=0,
            ),
        ]
        with pytest.raises(ValueError, match="cluster_size"):
            validate_cluster_size_consistency(rows)


# --------------------------------------------------------------------------- #
# 6. Manifest-consistency validator
# --------------------------------------------------------------------------- #


class TestManifestConsistencyValidator:
    def test_wrong_count_sequences_raises(self):
        rows = _two_cluster_rows()
        m = ClusterAssignmentsManifest(
            release_id="r", count_sequences=99, count_clusters=2, count_singletons=1
        )
        with pytest.raises(ValueError, match="count_sequences"):
            validate_manifest_consistency(m, rows)

    def test_wrong_count_clusters_raises(self):
        rows = _two_cluster_rows()
        m = ClusterAssignmentsManifest(
            release_id="r", count_sequences=3, count_clusters=99, count_singletons=1
        )
        with pytest.raises(ValueError, match="count_clusters"):
            validate_manifest_consistency(m, rows)

    def test_wrong_count_singletons_raises(self):
        rows = _two_cluster_rows()
        m = ClusterAssignmentsManifest(
            release_id="r", count_sequences=3, count_clusters=2, count_singletons=99
        )
        with pytest.raises(ValueError, match="count_singletons"):
            validate_manifest_consistency(m, rows)


# --------------------------------------------------------------------------- #
# 7. Namespace + coverage validators
# --------------------------------------------------------------------------- #


class TestNamespaceAndCoverage:
    def test_namespace_match_returns_no_issues(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        assert validate_cluster_namespace(a, expected=NAMESPACE_PREPARED_RECORD_ID) == []

    def test_namespace_mismatch_returns_issue(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path, namespace=NAMESPACE_RAW_PDB_ID))
        issues = validate_cluster_namespace(a, expected=NAMESPACE_PREPARED_RECORD_ID)
        assert len(issues) == 1
        assert issues[0].code == "cluster_namespace_mismatch"

    def test_namespace_validator_rejects_unknown_expected(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        with pytest.raises(ValueError, match="not a known namespace"):
            validate_cluster_namespace(a, expected="garbage")

    def test_coverage_full_match(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        report = validate_cluster_coverage(a, ["rec-1", "rec-2", "rec-3"])
        assert report.is_full_coverage
        assert report.coverage_fraction == 1.0
        assert report.missing_record_ids == []

    def test_coverage_missing_ids_reported(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        report = validate_cluster_coverage(a, ["rec-1", "rec-2", "rec-3", "rec-4"])
        assert not report.is_full_coverage
        assert report.missing_record_ids == ["rec-4"]
        assert report.coverage_fraction == pytest.approx(0.75)

    def test_coverage_extra_singletons_flagged(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        # Expected set excludes rec-3, which is a singleton -> flagged
        # in singleton_record_ids_in_extras (benign).
        report = validate_cluster_coverage(a, ["rec-1", "rec-2"])
        assert "rec-3" in report.extra_record_ids
        assert "rec-3" in report.singleton_record_ids_in_extras

    def test_coverage_strict_raises_on_missing(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        with pytest.raises(ClusterCoverageError, match="missing=1"):
            validate_cluster_coverage(
                a, ["rec-1", "rec-2", "rec-3", "rec-4"], strict=True
            )


# --------------------------------------------------------------------------- #
# 8. ClusterAssignments frozen behavior + lookups
# --------------------------------------------------------------------------- #


class TestClusterAssignmentsContainer:
    def test_assignments_is_frozen(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        with pytest.raises(FrozenInstanceError):
            a.manifest = ClusterAssignmentsManifest(release_id="other")  # type: ignore[misc]

    def test_cluster_id_for_lookup(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        assert a.cluster_id_for("rec-1") == "cluster-A"
        assert a.cluster_id_for("rec-3") == "cluster-B"

    def test_members_of_lookup(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        assert set(a.members_of("cluster-A")) == {"rec-1", "rec-2"}
        assert a.members_of("cluster-B") == ("rec-3",)

    def test_unknown_record_id_raises_key_error(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        with pytest.raises(KeyError):
            a.cluster_id_for("rec-does-not-exist")

    def test_record_ids_property_in_declared_order(self, tmp_path):
        a = load_cluster_assignments(_release(tmp_path))
        assert a.record_ids == ("rec-1", "rec-2", "rec-3")


# --------------------------------------------------------------------------- #
# 9. Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_empty_clustering_builds_and_loads(self, tmp_path):
        release_dir = build_cluster_assignments_release(
            [],
            tmp_path / "empty",
            release_id="empty",
            tool="mmseqs2",
            tool_version="14.7e284",
        )
        a = load_cluster_assignments(release_dir)
        assert a.manifest.count_sequences == 0
        assert a.manifest.count_clusters == 0
        assert a.record_ids == ()

    def test_all_singletons(self, tmp_path):
        rows = [
            ClusterAssignmentRow(
                record_id=f"rec-{i}",
                cluster_id=f"cluster-{i}",
                representative_record_id=f"rec-{i}",
                is_representative=True,
                cluster_size=1,
            )
            for i in range(5)
        ]
        release_dir = build_cluster_assignments_release(
            rows,
            tmp_path / "singletons",
            release_id="singletons",
            tool="custom",
            tool_version="n/a",
        )
        a = load_cluster_assignments(release_dir)
        assert a.manifest.count_singletons == 5
        assert a.manifest.count_clusters == 5

    def test_single_cluster_all_members(self, tmp_path):
        rep = ClusterAssignmentRow(
            record_id="rec-rep",
            cluster_id="only",
            representative_record_id="rec-rep",
            is_representative=True,
            cluster_size=4,
        )
        members = [
            ClusterAssignmentRow(
                record_id=f"rec-{i}",
                cluster_id="only",
                representative_record_id="rec-rep",
                is_representative=False,
                cluster_size=4,
            )
            for i in range(3)
        ]
        release_dir = build_cluster_assignments_release(
            [rep] + members,
            tmp_path / "one_cluster",
            release_id="one_cluster",
            tool="custom",
            tool_version="n/a",
        )
        a = load_cluster_assignments(release_dir)
        assert a.manifest.count_clusters == 1
        assert a.manifest.count_singletons == 0
        assert len(a.members_of("only")) == 4

    def test_unicode_record_id_round_trip(self, tmp_path):
        rows = [
            ClusterAssignmentRow(
                record_id="récord-α-Δ",
                cluster_id="クラスタ-1",
                representative_record_id="récord-α-Δ",
                is_representative=True,
                cluster_size=1,
            ),
        ]
        release_dir = build_cluster_assignments_release(
            rows,
            tmp_path / "unicode",
            release_id="unicode",
            tool="custom",
            tool_version="n/a",
        )
        a = load_cluster_assignments(release_dir)
        assert a.record_ids == ("récord-α-Δ",)
        assert a.cluster_id_for("récord-α-Δ") == "クラスタ-1"


# --------------------------------------------------------------------------- #
# 10. Streaming loader
# --------------------------------------------------------------------------- #


class TestStreamingLoader:
    def test_iter_yields_rows_one_at_a_time(self, tmp_path):
        release_dir = _release(tmp_path)
        rows = list(iter_cluster_assignments(release_dir))
        assert len(rows) == 3
        assert all(isinstance(r, ClusterAssignmentRow) for r in rows)

    def test_iter_yields_chunks_when_batch_size_set(self, tmp_path):
        release_dir = _release(tmp_path)
        chunks = list(iter_cluster_assignments(release_dir, batch_size=2))
        # 3 rows, batch_size=2 -> [chunk-of-2, chunk-of-1]
        assert sum(len(c) for c in chunks) == 3
        assert all(isinstance(c, list) for c in chunks)
