"""Cluster-assignments artifact contract for the v0.3.0 data-engine layer.

Externally-produced clustering (e.g. ``mmseqs cluster``, ``foldseek
easy-cluster``) is consumed by proteon as a typed
``ClusterAssignments`` artifact. This module defines:

- ``ClusterAssignmentRow`` — one cluster membership row (per-row Parquet
  schema is pinned at ``CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION``).
- ``ClusterAssignmentsManifest`` — release-level provenance with the
  fields needed to make ``cluster_id`` authoritative rather than "just a
  string column" (``tool``, ``tool_version``, ``params``,
  ``input_digest`` + ``input_digest_kind``, ``record_id_digest``,
  ``representative_selection``, ``sequence_id_namespace``,
  ``custom_namespace_description``, ``created_from_release_id``).
- ``ClusterAssignments`` — frozen container bundling a manifest with a
  tuple of rows, plus O(1) ``cluster_id_for`` / ``members_of`` lookups
  built eagerly in ``__post_init__``.
- I/O: ``build_cluster_assignments_release`` (canonical) plus JSONL and
  Parquet read/write/iter helpers.
- Validators that catch the worst failure modes external consumers can
  introduce: duplicate ``record_id`` rows, missing or mis-pointed
  representatives, denormalised ``cluster_size``, manifest count
  inconsistencies, namespace mismatches, and partial-coverage joins.

Per ``feedback_compute_kernel`` and the v0.3.0 plan, proteon owns the
contract and validation — it does NOT own the clustering algorithm.
Upstream tools produce these artifacts; proteon consumes them. The
canonical join key is the post-chain-expansion ``record_id`` used by
``training_example.build_training_release``; the namespace validator
enforces alignment.

See ``TO_V030_PHASE_B0_CLUSTER_ASSIGNMENTS.md`` for the design rationale
and ``tests/test_cluster_assignments.py`` for the contract gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import blake2b
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np  # noqa: F401  -- consumed via pyarrow.from_numpy_dtype downstream

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAS_PYARROW = False

from ._artifact_checksum import sha256_file
from .corpus_validation import ValidationIssue


# --------------------------------------------------------------------------- #
# Format + schema constants
# --------------------------------------------------------------------------- #

CLUSTER_ASSIGNMENTS_FORMAT = "proteon.cluster_assignments.parquet.v0"
CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Closed enumerations (validated in __post_init__)
# --------------------------------------------------------------------------- #

NAMESPACE_PREPARED_RECORD_ID: str = "prepared_record_id"
NAMESPACE_RAW_PDB_ID: str = "raw_pdb_id"
NAMESPACE_UNIPROT_ID: str = "uniprot_id"
NAMESPACE_CUSTOM: str = "custom"
ALL_CLUSTER_NAMESPACES: Tuple[str, ...] = (
    NAMESPACE_PREPARED_RECORD_ID,
    NAMESPACE_RAW_PDB_ID,
    NAMESPACE_UNIPROT_ID,
    NAMESPACE_CUSTOM,
)

DIGEST_KIND_FASTA: str = "fasta_sha256"
DIGEST_KIND_RECORD_ID_LIST: str = "record_id_list_sha256"
DIGEST_KIND_CUSTOM: str = "custom"
ALL_INPUT_DIGEST_KINDS: Tuple[str, ...] = (
    DIGEST_KIND_FASTA,
    DIGEST_KIND_RECORD_ID_LIST,
    DIGEST_KIND_CUSTOM,
)


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #


class ClusterCoverageError(ValueError):
    """Raised by ``validate_cluster_coverage(..., strict=True)`` when the
    cluster artifact does not 1:1 cover the expected record_id set.

    Phase C's ``cluster_aware_split`` should default to strict mode so
    partial-coverage clusterings don't silently degenerate into
    singleton-ish split inputs — that's exactly the
    "scientifically meaningless cluster passing as valid" failure mode
    Phase B0 exists to prevent.
    """


# --------------------------------------------------------------------------- #
# Per-row dataclass + manifest
# --------------------------------------------------------------------------- #


@dataclass
class ClusterAssignmentRow:
    """One cluster membership.

    ``record_id`` is the post-chain-expansion identifier that matches
    ``training_example.TrainingExample.record_id``. ``cluster_id`` is the
    canonical cluster identifier; it is **not** required to equal
    ``representative_record_id`` so callers can relabel clusters for
    stability across upstream tool versions whose representative
    selection may drift. ``cluster_size`` is denormalised (the manifest
    only carries summary stats); structural validators enforce that the
    denormalisation matches actual member counts.
    """

    record_id: str
    cluster_id: str
    representative_record_id: str
    is_representative: bool
    cluster_size: int
    source_id: Optional[str] = None


@dataclass
class ClusterAssignmentsManifest:
    """Release-level provenance for one ``ClusterAssignments`` artifact.

    The provenance fields here are what give ``cluster_id`` authority —
    without them, two assignments with the same ``cluster_id`` strings
    have no way to be compared or audited. ``input_digest`` (paired with
    ``input_digest_kind``) and ``record_id_digest`` give upstream and
    proteon-side reproducibility respectively. ``representative_selection``
    documents the policy used to pick representatives; MMseqs / foldseek
    can change this between versions, so an identical (params, sequences)
    input can yield a different ``cluster_id`` set without this audit
    field.
    """

    release_id: str
    artifact_type: str = "cluster_assignments_release"
    format: str = CLUSTER_ASSIGNMENTS_FORMAT
    schema_version: int = CLUSTER_ASSIGNMENTS_PARQUET_SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None

    # === provenance ===
    tool: str = ""
    tool_version: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    input_digest: str = ""
    input_digest_kind: str = DIGEST_KIND_FASTA
    record_id_digest: str = ""
    representative_selection: str = ""
    sequence_id_namespace: str = NAMESPACE_PREPARED_RECORD_ID
    custom_namespace_description: Optional[str] = None
    created_from_release_id: Optional[str] = None

    # === counts + summaries ===
    count_sequences: int = 0
    count_clusters: int = 0
    count_singletons: int = 0
    cluster_size_summary: Dict[str, float] = field(default_factory=dict)

    # === file references ===
    assignments_file_jsonl: str = "assignments.jsonl"
    assignments_file_parquet: str = "assignments.parquet"
    assignments_parquet_sha256: Optional[str] = None

    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence_id_namespace not in ALL_CLUSTER_NAMESPACES:
            raise ValueError(
                f"sequence_id_namespace={self.sequence_id_namespace!r} not in "
                f"{ALL_CLUSTER_NAMESPACES}"
            )
        if self.input_digest_kind not in ALL_INPUT_DIGEST_KINDS:
            raise ValueError(
                f"input_digest_kind={self.input_digest_kind!r} not in "
                f"{ALL_INPUT_DIGEST_KINDS}"
            )
        if self.sequence_id_namespace == NAMESPACE_CUSTOM and not (
            self.custom_namespace_description or ""
        ).strip():
            raise ValueError(
                "sequence_id_namespace='custom' requires a non-empty "
                "custom_namespace_description"
            )


@dataclass(frozen=True)
class ClusterAssignments:
    """User-facing container: manifest + tuple of rows + eager indexes.

    ``frozen=True`` defends against accidental mutation when assignments
    are reused across multiple Phase C / Phase D / Phase E calls; the
    private lookup indexes are built once in ``__post_init__`` via
    ``object.__setattr__`` (the frozen contract forbids normal assignment).
    """

    manifest: ClusterAssignmentsManifest
    rows: Tuple[ClusterAssignmentRow, ...]

    def __post_init__(self) -> None:
        record_to_cluster: Dict[str, str] = {}
        cluster_to_members: Dict[str, List[str]] = {}
        for row in self.rows:
            record_to_cluster[row.record_id] = row.cluster_id
            cluster_to_members.setdefault(row.cluster_id, []).append(row.record_id)
        object.__setattr__(self, "_record_to_cluster", record_to_cluster)
        object.__setattr__(
            self,
            "_cluster_to_members",
            {k: tuple(v) for k, v in cluster_to_members.items()},
        )

    def cluster_id_for(self, record_id: str) -> str:
        """O(1) cluster lookup. Raises ``KeyError`` for unknown record_id."""
        return self._record_to_cluster[record_id]  # type: ignore[attr-defined]

    def members_of(self, cluster_id: str) -> Tuple[str, ...]:
        """O(1) members lookup. Raises ``KeyError`` for unknown cluster_id."""
        return self._cluster_to_members[cluster_id]  # type: ignore[attr-defined]

    @property
    def record_ids(self) -> Tuple[str, ...]:
        """All record_ids in declared row order (de-duplication is the
        caller's job; structural validators enforce uniqueness at
        build time)."""
        return tuple(row.record_id for row in self.rows)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


@dataclass
class ClusterCoverageReport:
    """Result of ``validate_cluster_coverage``. Non-strict mode returns
    this for the caller to inspect; ``Phase D`` accumulates the report
    into ``CorpusValidationReport.cluster_leakage_check`` (forward note).
    """

    missing_record_ids: List[str]
    extra_record_ids: List[str]
    coverage_fraction: float
    is_full_coverage: bool
    singleton_record_ids_in_extras: List[str]


@dataclass
class ClusterValidationReport:
    """Aggregate of all non-raising validators. ``is_ok`` is True iff
    every issue list is empty and coverage (if computed) is full.
    """

    namespace_issues: List[ValidationIssue] = field(default_factory=list)
    representative_issues: List[ValidationIssue] = field(default_factory=list)
    size_consistency_issues: List[ValidationIssue] = field(default_factory=list)
    record_id_uniqueness_issues: List[ValidationIssue] = field(default_factory=list)
    manifest_consistency_issues: List[ValidationIssue] = field(default_factory=list)
    coverage: Optional[ClusterCoverageReport] = None
    is_ok: bool = True


# --------------------------------------------------------------------------- #
# Parquet schema source-of-truth
# --------------------------------------------------------------------------- #


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for cluster-assignments Parquet I/O. "
            "Install with `pip install pyarrow`."
        )


def _cluster_assignment_fields_factory() -> Tuple[Tuple[str, object, bool, str], ...]:
    """Source-of-truth field descriptor list. Built lazily so the module
    imports cleanly even when pyarrow is missing.

    Tuple shape: ``(column_name, arrow_type, nullable, dataclass_attr)``.
    Mirrors the ``TENSOR_FIELDS`` pattern in
    ``supervision_export.py`` so tests can sweep declared fields against
    actual Parquet round-trip behaviour.
    """
    _require_pyarrow()
    return (
        ("record_id", pa.string(), False, "record_id"),
        ("cluster_id", pa.string(), False, "cluster_id"),
        ("representative_record_id", pa.string(), False, "representative_record_id"),
        ("is_representative", pa.bool_(), False, "is_representative"),
        ("cluster_size", pa.int32(), False, "cluster_size"),
        ("source_id", pa.string(), True, "source_id"),
    )


def build_cluster_assignments_schema() -> "pa.Schema":
    """Build the canonical Arrow schema for the assignments Parquet file."""
    _require_pyarrow()
    return pa.schema(
        [pa.field(name, dtype, nullable) for name, dtype, nullable, _ in _cluster_assignment_fields_factory()]
    )


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #


def validate_cluster_record_id_uniqueness(
    rows: Sequence[ClusterAssignmentRow],
) -> None:
    """Every ``record_id`` must appear exactly once.

    Duplicate ``record_id`` would silently make
    ``ClusterAssignments.cluster_id_for`` ambiguous (the
    ``__post_init__`` dict would overwrite earlier entries) and break
    Phase C's leakage guarantees. Structural integrity, not policy —
    raises ``ValueError`` with the offending IDs listed.
    """
    seen: Dict[str, int] = {}
    for row in rows:
        seen[row.record_id] = seen.get(row.record_id, 0) + 1
    duplicates = sorted(rid for rid, n in seen.items() if n > 1)
    if duplicates:
        raise ValueError(
            f"duplicate record_id rows in cluster assignments: "
            f"{duplicates[:20]}{'...' if len(duplicates) > 20 else ''}"
        )
    empties = [i for i, row in enumerate(rows) if not row.record_id]
    if empties:
        raise ValueError(
            f"empty record_id at row indexes: {empties[:20]}"
            f"{'...' if len(empties) > 20 else ''}"
        )


def validate_cluster_representative_consistency(
    assignments: "ClusterAssignments | Sequence[ClusterAssignmentRow]",
) -> None:
    """Each cluster has exactly one representative row, and that row's
    pointers are internally consistent.

    Specifically: for each cluster_id, exactly one member row has
    ``is_representative=True``; that row's ``record_id`` equals
    ``representative_record_id`` for every member of the cluster; and
    every member's ``representative_record_id`` is consistent across
    the cluster.
    """
    rows = assignments.rows if isinstance(assignments, ClusterAssignments) else assignments
    by_cluster: Dict[str, List[ClusterAssignmentRow]] = {}
    for row in rows:
        by_cluster.setdefault(row.cluster_id, []).append(row)
    for cluster_id, members in by_cluster.items():
        reps = [r for r in members if r.is_representative]
        if len(reps) == 0:
            raise ValueError(
                f"cluster {cluster_id!r} has no representative row"
            )
        if len(reps) > 1:
            raise ValueError(
                f"cluster {cluster_id!r} has multiple representatives: "
                f"{[r.record_id for r in reps]}"
            )
        rep = reps[0]
        # All members must agree on which record_id is the representative.
        for m in members:
            if m.representative_record_id != rep.record_id:
                raise ValueError(
                    f"cluster {cluster_id!r}: member {m.record_id!r} "
                    f"claims representative_record_id={m.representative_record_id!r}, "
                    f"but the cluster's is_representative=True row is "
                    f"{rep.record_id!r}"
                )


def validate_cluster_size_consistency(
    assignments: "ClusterAssignments | Sequence[ClusterAssignmentRow]",
) -> None:
    """Each row's denormalised ``cluster_size`` equals the actual count
    of rows sharing that ``cluster_id``."""
    rows = assignments.rows if isinstance(assignments, ClusterAssignments) else assignments
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.cluster_id] = counts.get(row.cluster_id, 0) + 1
    for row in rows:
        actual = counts[row.cluster_id]
        if row.cluster_size != actual:
            raise ValueError(
                f"row {row.record_id!r}: cluster_size={row.cluster_size} "
                f"but cluster {row.cluster_id!r} actually has {actual} members"
            )
        if row.cluster_size <= 0:
            raise ValueError(
                f"row {row.record_id!r}: cluster_size must be > 0, got {row.cluster_size}"
            )


def validate_manifest_consistency(
    manifest: ClusterAssignmentsManifest,
    rows: Sequence[ClusterAssignmentRow],
) -> None:
    """Manifest counts match actual row contents.

    Runs in ``build_cluster_assignments_release`` before writing and in
    ``load_cluster_assignments`` after reading; catches caller-supplied
    manifests whose counts drift from reality.
    """
    if manifest.count_sequences != len(rows):
        raise ValueError(
            f"manifest.count_sequences={manifest.count_sequences} but rows has "
            f"{len(rows)} entries"
        )
    cluster_ids = {row.cluster_id for row in rows}
    if manifest.count_clusters != len(cluster_ids):
        raise ValueError(
            f"manifest.count_clusters={manifest.count_clusters} but rows contain "
            f"{len(cluster_ids)} distinct cluster_ids"
        )
    sizes: Dict[str, int] = {}
    for row in rows:
        sizes[row.cluster_id] = sizes.get(row.cluster_id, 0) + 1
    actual_singletons = sum(1 for n in sizes.values() if n == 1)
    if manifest.count_singletons != actual_singletons:
        raise ValueError(
            f"manifest.count_singletons={manifest.count_singletons} but rows have "
            f"{actual_singletons} singleton clusters"
        )
    for row in rows:
        if not row.record_id:
            raise ValueError("empty record_id in row")
        if not row.cluster_id:
            raise ValueError(f"empty cluster_id at row record_id={row.record_id!r}")


def validate_cluster_namespace(
    assignments: ClusterAssignments,
    *,
    expected: str,
) -> List[ValidationIssue]:
    """Returns issues (does not raise) so the caller picks policy.

    Phase D's cluster-leakage check accumulates these into
    ``CorpusValidationReport.cluster_leakage_check``; Phase C should
    raise on mismatch (forward note in the Phase C plan).
    """
    if expected not in ALL_CLUSTER_NAMESPACES:
        raise ValueError(
            f"expected={expected!r} is not a known namespace; "
            f"choose from {ALL_CLUSTER_NAMESPACES}"
        )
    actual = assignments.manifest.sequence_id_namespace
    if actual != expected:
        return [
            ValidationIssue(
                severity="error",
                code="cluster_namespace_mismatch",
                message=(
                    f"cluster assignments use sequence_id_namespace={actual!r} "
                    f"but caller expected {expected!r}; joining on these IDs "
                    f"will be incorrect"
                ),
            )
        ]
    return []


def validate_cluster_coverage(
    assignments: ClusterAssignments,
    expected_record_ids: Iterable[str],
    *,
    strict: bool = False,
) -> ClusterCoverageReport:
    """Compare assignment coverage to an expected record_id set.

    Default ``strict=False`` returns a report; Phase D uses this form
    to accumulate into ``CorpusValidationReport``. Phase C should set
    ``strict=True`` to refuse splitting a partial-coverage clustering.
    """
    expected_set = set(expected_record_ids)
    actual_set = set(assignments.record_ids)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    overlap = expected_set & actual_set
    fraction = len(overlap) / len(expected_set) if expected_set else 1.0
    is_full = fraction == 1.0 and not extra
    singleton_extras = sorted(
        rid
        for rid in extra
        if assignments.members_of(assignments.cluster_id_for(rid)) == (rid,)
    )
    report = ClusterCoverageReport(
        missing_record_ids=missing,
        extra_record_ids=extra,
        coverage_fraction=fraction,
        is_full_coverage=is_full,
        singleton_record_ids_in_extras=singleton_extras,
    )
    if strict and not is_full:
        raise ClusterCoverageError(
            f"cluster assignments do not 1:1 cover the expected record_id set: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"coverage_fraction={fraction:.4f}"
        )
    return report


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #


def write_cluster_assignments_jsonl(
    rows: Iterable[ClusterAssignmentRow],
    path: str | Path,
) -> Path:
    """Write rows as JSONL, one row per line. Human-inspectable form."""
    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), separators=(",", ":")))
            handle.write("\n")
    return out


def load_cluster_assignments_jsonl(
    path: str | Path,
) -> List[ClusterAssignmentRow]:
    """Load JSONL rows into a list of ``ClusterAssignmentRow``."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [ClusterAssignmentRow(**row) for row in rows]


# --------------------------------------------------------------------------- #
# Parquet I/O
# --------------------------------------------------------------------------- #


def _rows_to_record_batch(
    rows: Sequence[ClusterAssignmentRow],
) -> "pa.RecordBatch":
    _require_pyarrow()
    fields = _cluster_assignment_fields_factory()
    arrays: List["pa.Array"] = []
    for name, dtype, _nullable, attr in fields:
        col = [getattr(row, attr) for row in rows]
        arrays.append(pa.array(col, type=dtype))
    return pa.RecordBatch.from_arrays(arrays, schema=build_cluster_assignments_schema())


def write_cluster_assignments_parquet(
    rows: Iterable[ClusterAssignmentRow],
    path: str | Path,
) -> Path:
    """Write rows as a single Parquet file. Joinable artifact (the
    canonical join column is ``record_id``).
    """
    _require_pyarrow()
    materialised = tuple(rows)
    batch = _rows_to_record_batch(materialised)
    table = pa.Table.from_batches([batch])
    out = Path(path)
    pq.write_table(table, out)
    return out


def load_cluster_assignments_parquet(
    path: str | Path,
) -> List[ClusterAssignmentRow]:
    """Load Parquet rows into a list of ``ClusterAssignmentRow``."""
    _require_pyarrow()
    table = pq.read_table(Path(path))
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    out: List[ClusterAssignmentRow] = []
    fields = _cluster_assignment_fields_factory()
    for i in range(n):
        kwargs = {attr: cols[name][i] for name, _dtype, _nullable, attr in fields}
        out.append(ClusterAssignmentRow(**kwargs))
    return out


# --------------------------------------------------------------------------- #
# Release builder + streaming loader
# --------------------------------------------------------------------------- #


def _compute_cluster_size_summary(
    rows: Sequence[ClusterAssignmentRow],
) -> Dict[str, float]:
    if not rows:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    sizes: Dict[str, int] = {}
    for row in rows:
        sizes[row.cluster_id] = sizes.get(row.cluster_id, 0) + 1
    values = sorted(sizes.values())
    n = len(values)
    median = (
        float(values[n // 2])
        if n % 2
        else float((values[n // 2 - 1] + values[n // 2]) / 2.0)
    )
    return {
        "min": float(values[0]),
        "max": float(values[-1]),
        "mean": float(sum(values) / n),
        "median": median,
    }


def compute_record_id_digest(record_ids: Iterable[str]) -> str:
    """SHA-256 over the sorted unique record_ids.

    Stable across different ordering of the same corpus content; detects
    corpus drift while ignoring serialization order. Used to populate
    ``ClusterAssignmentsManifest.record_id_digest``.
    """
    sorted_unique = sorted(set(record_ids))
    payload = "\n".join(sorted_unique).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_cluster_assignments_release(
    rows: Sequence[ClusterAssignmentRow],
    out_dir: str | Path,
    *,
    release_id: str,
    tool: str,
    tool_version: str,
    params: Optional[Mapping[str, object]] = None,
    input_digest: str = "",
    input_digest_kind: str = DIGEST_KIND_FASTA,
    representative_selection: str = "",
    sequence_id_namespace: str = NAMESPACE_PREPARED_RECORD_ID,
    custom_namespace_description: Optional[str] = None,
    created_from_release_id: Optional[str] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Mapping[str, object]] = None,
    overwrite: bool = False,
) -> Path:
    """Canonical entry point: validates, writes JSONL + Parquet + manifest.

    Structural validators (``validate_cluster_record_id_uniqueness``,
    ``validate_cluster_representative_consistency``,
    ``validate_cluster_size_consistency``,
    ``validate_manifest_consistency``) run before any file is written.
    """
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    materialised = tuple(rows)
    # Structural validators — raise before writing anything.
    validate_cluster_record_id_uniqueness(materialised)
    validate_cluster_representative_consistency(materialised)
    validate_cluster_size_consistency(materialised)

    cluster_ids = {row.cluster_id for row in materialised}
    sizes: Dict[str, int] = {}
    for row in materialised:
        sizes[row.cluster_id] = sizes.get(row.cluster_id, 0) + 1
    n_singletons = sum(1 for n in sizes.values() if n == 1)

    record_id_digest = compute_record_id_digest(row.record_id for row in materialised)

    jsonl_path = root / "assignments.jsonl"
    parquet_path = root / "assignments.parquet"
    write_cluster_assignments_jsonl(materialised, jsonl_path)
    write_cluster_assignments_parquet(materialised, parquet_path)
    parquet_sha = sha256_file(parquet_path)

    manifest = ClusterAssignmentsManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        tool=tool,
        tool_version=tool_version,
        params=dict(params or {}),
        input_digest=input_digest,
        input_digest_kind=input_digest_kind,
        record_id_digest=record_id_digest,
        representative_selection=representative_selection,
        sequence_id_namespace=sequence_id_namespace,
        custom_namespace_description=custom_namespace_description,
        created_from_release_id=created_from_release_id,
        count_sequences=len(materialised),
        count_clusters=len(cluster_ids),
        count_singletons=n_singletons,
        cluster_size_summary=_compute_cluster_size_summary(materialised),
        assignments_parquet_sha256=parquet_sha,
        provenance=dict(provenance or {}),
    )
    # Post-construction sanity check.
    validate_manifest_consistency(manifest, materialised)

    (root / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )
    return root


def load_cluster_assignments(release_dir: str | Path) -> ClusterAssignments:
    """Load a release directory back into a ``ClusterAssignments``.

    Prefers the Parquet artifact (canonical for joins); the JSONL file
    is the human-inspection sibling. Runs
    ``validate_manifest_consistency`` after loading so the returned
    object is guaranteed self-consistent.
    """
    root = Path(release_dir)
    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest = ClusterAssignmentsManifest(**manifest_data)
    parquet_path = root / manifest.assignments_file_parquet
    if parquet_path.exists():
        rows = tuple(load_cluster_assignments_parquet(parquet_path))
    else:
        rows = tuple(
            load_cluster_assignments_jsonl(root / manifest.assignments_file_jsonl)
        )
    validate_manifest_consistency(manifest, rows)
    return ClusterAssignments(manifest=manifest, rows=rows)


def iter_cluster_assignments(
    release_dir: str | Path,
    *,
    batch_size: Optional[int] = None,
) -> Iterator["ClusterAssignmentRow | List[ClusterAssignmentRow]"]:
    """Stream rows for very large clusterings.

    ``batch_size=None`` yields one ``ClusterAssignmentRow`` per
    iteration; a positive integer yields ``list[ClusterAssignmentRow]``
    chunks of length ≤ ``batch_size``. Reads the Parquet artifact one
    row group at a time so peak memory is bounded by the row group
    size set at write time.
    """
    _require_pyarrow()
    root = Path(release_dir)
    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest = ClusterAssignmentsManifest(**manifest_data)
    parquet_path = root / manifest.assignments_file_parquet
    pf = pq.ParquetFile(parquet_path)
    fields = _cluster_assignment_fields_factory()
    chunk: List[ClusterAssignmentRow] = []
    for rg_index in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_index)
        cols = {name: rg.column(name).to_pylist() for name in rg.column_names}
        for i in range(rg.num_rows):
            kwargs = {attr: cols[name][i] for name, _t, _n, attr in fields}
            row = ClusterAssignmentRow(**kwargs)
            if batch_size is None:
                yield row
            else:
                chunk.append(row)
                if len(chunk) >= batch_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# Phase C: cluster_aware_split — leakage-controlled split helper
# --------------------------------------------------------------------------- #

_DEFAULT_SPLIT_RATIOS: Mapping[str, float] = MappingProxyType(
    {"train": 0.8, "val": 0.1, "test": 0.1}
)

# Default skew tolerance for the bounded-deviation check. Per the codex
# review on TO_V030_TRAINING_CORPUS_FACTORY.md (catch #9 — "cluster
# splitting by hash of cluster ID can produce bad ratio skew when large
# clusters exist; tests should check bounded deviation and report
# unavoidable skew, not just 'approximated'"), the threshold is liberal
# enough that ordinary clusterings pass but pathological ones (one
# cluster dominating the corpus) get flagged.
DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE: float = 0.10

# Namespaces that are unsafe for direct training joins without an
# explicit ID-mapping artifact, per the codex review on the Phase B0
# plan (catch #5 — "raw_pdb_id and uniprot_id are not safe for
# training_example.py joins because they can be many-to-one after chain
# expansion"). cluster_aware_split rejects these unless the caller
# explicitly opts in via allow_unsafe_namespaces=True.
_UNSAFE_CLUSTER_NAMESPACES_FOR_SPLITTING: Tuple[str, ...] = (
    NAMESPACE_RAW_PDB_ID,
    NAMESPACE_UNIPROT_ID,
)


@dataclass
class ClusterAwareSplitResult:
    """Return value of ``cluster_aware_split``.

    Carries both the per-record split assignment and a skew report so
    callers can detect unavoidable distribution drift caused by lumpy
    cluster sizes — flagging that, rather than silently producing a
    badly-skewed corpus, is the point of running cluster-aware
    splitting in the first place.
    """

    assignments: Dict[str, str]
    requested_ratios: Dict[str, float]
    actual_ratios: Dict[str, float]
    skew: Dict[str, float]
    max_skew: float
    skew_tolerance: float
    bounded_skew: bool


def _union_find_roots(
    record_ids: Sequence[str],
    *equivalence_groups: Sequence[Sequence[str]],
) -> Dict[str, str]:
    """Build a canonical "effective grouping key" per record_id by
    union-finding across one or more equivalence-class groupings.

    Each entry in ``equivalence_groups`` is a sequence of clusters
    (lists of record_ids that belong together). Records that appear
    in the same cluster in *any* of the groupings end up in the same
    equivalence class. The returned mapping sends each record_id to
    the lexicographically smallest record_id in its equivalence
    class — the deterministic "effective grouping key" used for
    hash-splitting downstream.
    """
    parent: Dict[str, str] = {rid: rid for rid in record_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Pick lexicographically smaller as root so the result is
        # deterministic across input orderings.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for groups in equivalence_groups:
        for cluster in groups:
            if len(cluster) < 2:
                continue
            first = cluster[0]
            for other in cluster[1:]:
                union(first, other)

    return {rid: find(rid) for rid in record_ids}


def _split_skew_report(
    assignments: Mapping[str, str],
    requested_ratios: Mapping[str, float],
    *,
    skew_tolerance: float,
) -> ClusterAwareSplitResult:
    total = sum(requested_ratios.values()) or 1.0
    normalised_requested = {k: v / total for k, v in requested_ratios.items()}
    n = len(assignments)
    counts = Counter(assignments.values())
    actual = {k: counts.get(k, 0) / n if n else 0.0 for k in requested_ratios}
    skew = {k: abs(actual[k] - normalised_requested[k]) for k in requested_ratios}
    max_skew = max(skew.values()) if skew else 0.0
    return ClusterAwareSplitResult(
        assignments=dict(assignments),
        requested_ratios=dict(normalised_requested),
        actual_ratios=actual,
        skew=skew,
        max_skew=max_skew,
        skew_tolerance=skew_tolerance,
        bounded_skew=max_skew <= skew_tolerance,
    )


def cluster_aware_split(
    cluster_assignments: ClusterAssignments,
    record_ids: Sequence[str],
    ratios: Mapping[str, float] = _DEFAULT_SPLIT_RATIOS,
    *,
    seed: int = 0,
    grouping_keys: Optional[Sequence[str]] = None,
    strict_coverage: bool = True,
    allow_unsafe_namespaces: bool = False,
    skew_tolerance: float = DEFAULT_CLUSTER_SPLIT_SKEW_TOLERANCE,
) -> ClusterAwareSplitResult:
    """Family-aware split that prevents cluster members from spanning splits.

    All record_ids sharing a ``cluster_id`` land in the same split, so
    test-set leakage from same-family training examples is structurally
    impossible. Determinism is preserved: identical
    ``(cluster_assignments, record_ids, ratios, seed)`` always yields
    identical assignments regardless of input order.

    When ``grouping_keys`` is supplied (e.g. for keeping sibling chains
    of the same source structure together — the existing
    ``corpus_smoke._hash_split_assignments`` use case), the splitter
    runs union-find over **both** constraint sets: records sharing a
    cluster AND records sharing a grouping key end up in one
    equivalence class together, which then hash-splits as a unit. This
    is the "stack, don't absorb" interaction recommended by the codex
    review on the parent v0.3.0 plan (Q5 nuance).

    Defaults reflect the codex reviews of the parent plan and Phase B0:

    - ``strict_coverage=True`` (codex catch #6 on parent plan):
      partial-coverage clusterings raise ``ClusterCoverageError``
      rather than silently producing singleton-ish splits.
    - Unsafe namespaces (``raw_pdb_id``, ``uniprot_id``) are rejected
      unless ``allow_unsafe_namespaces=True`` because they can be
      many-to-one after chain expansion and corrupt the
      training-example join (codex catch #5 on Phase B0 plan).
    - ``skew_tolerance=0.10`` is liberal enough that ordinary
      clusterings pass; very lumpy clusters (one cluster
      dominating the corpus) trip ``bounded_skew=False`` so callers
      can decide whether the unavoidable skew is acceptable.

    The returned ``ClusterAwareSplitResult`` carries both the per-record
    assignment and the skew report. The skew is *informational* — even
    with a dominant cluster the assignment is still leakage-free.
    """
    namespace = cluster_assignments.manifest.sequence_id_namespace
    if not allow_unsafe_namespaces and namespace in _UNSAFE_CLUSTER_NAMESPACES_FOR_SPLITTING:
        raise ValueError(
            f"cluster_aware_split: sequence_id_namespace={namespace!r} is "
            f"unsafe for training-example splits (can be many-to-one after "
            f"chain expansion). Either supply assignments whose namespace is "
            f"{NAMESPACE_PREPARED_RECORD_ID!r}, or pass "
            f"allow_unsafe_namespaces=True if you have verified ID alignment."
        )
    if grouping_keys is not None and len(grouping_keys) != len(record_ids):
        raise ValueError(
            f"grouping_keys length {len(grouping_keys)} != record_ids length "
            f"{len(record_ids)}; they must align one-to-one"
        )

    # Coverage check — strict by default so partial clusterings raise.
    validate_cluster_coverage(
        cluster_assignments, record_ids, strict=strict_coverage,
    )

    # Build the per-record cluster-id list (now safe; coverage passed).
    cluster_ids_per_record = [
        cluster_assignments.cluster_id_for(rid) for rid in record_ids
    ]

    # Bucket record_ids by cluster_id to form the cluster equivalence
    # groups for union-find.
    cluster_groups: Dict[str, List[str]] = {}
    for rid, cid in zip(record_ids, cluster_ids_per_record):
        cluster_groups.setdefault(cid, []).append(rid)
    cluster_eq = tuple(cluster_groups.values())

    # If grouping_keys were supplied, bucket those too.
    sibling_eq: Tuple[List[str], ...] = ()
    if grouping_keys is not None:
        sibling_groups: Dict[str, List[str]] = {}
        for rid, gk in zip(record_ids, grouping_keys):
            sibling_groups.setdefault(gk, []).append(rid)
        sibling_eq = tuple(sibling_groups.values())

    # Union-find collapses both constraints into a single per-record
    # effective grouping key.
    effective_keys = _union_find_roots(record_ids, cluster_eq, *([sibling_eq] if sibling_eq else []))

    # Hand off to the existing hash-split engine. The seed is mixed
    # into the grouping key so different seeds produce different
    # splits without touching _hash_split_assignments itself.
    from .corpus_smoke import _hash_split_assignments

    seeded_keys = [f"{seed}:{effective_keys[rid]}" for rid in record_ids]
    raw_assignments = _hash_split_assignments(
        record_ids,
        ratios,
        grouping_keys=seeded_keys,
    )

    return _split_skew_report(
        raw_assignments, ratios, skew_tolerance=skew_tolerance,
    )
