"""Release-oriented wrappers around structure supervision exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .failure_taxonomy import ALL_FAILURE_CLASSES, INTERNAL_PIPELINE_ERROR
from .supervision import StructureSupervisionExample
from .supervision_export import (
    SUPERVISION_EXPORT_FORMAT,
    export_structure_supervision_examples,
)


@dataclass
class FailureRecord:
    """Structured failure row for supervision/release pipelines.

    `failure_class` must be one of the canonical values in
    `ferritin.failure_taxonomy.ALL_FAILURE_CLASSES` — dataset quality
    trends are only measurable if the class set stays closed. Writing
    free-text classes raises ValueError at construction time.
    """

    record_id: str
    artifact_type: str = "failure_record"
    stage: str = "structure_supervision"
    status: str = "failed"
    failure_class: str = INTERNAL_PIPELINE_ERROR
    message: str = ""
    source_id: Optional[str] = None
    prep_run_id: Optional[str] = None
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure_class not in ALL_FAILURE_CLASSES:
            raise ValueError(
                f"failure_class={self.failure_class!r} is not in the canonical "
                f"taxonomy; allowed values: {ALL_FAILURE_CLASSES}"
            )


@dataclass
class StructureSupervisionReleaseManifest:
    """Machine-readable manifest for one supervision dataset release."""

    release_id: str
    artifact_type: str = "release_manifest"
    format: str = SUPERVISION_EXPORT_FORMAT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    count_examples: int = 0
    count_failures: int = 0
    example_export_dir: str = "examples"
    examples_file: str = "examples/examples.jsonl"
    tensor_file: str = "examples/tensors.parquet"
    failure_file: str = "failures.jsonl"
    lengths: Dict[str, float] = field(default_factory=dict)
    sequence_lengths: List[int] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)


def build_structure_supervision_release(
    examples: Iterable[StructureSupervisionExample],
    out_dir: str | Path,
    *,
    release_id: str,
    failures: Optional[Iterable[FailureRecord]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
    row_group_size: int = 512,
) -> Path:
    """Write a supervision release directory with examples, failures, and manifest.

    `examples` may be an iterator; it is consumed once without
    materializing the full list. Peak memory is `O(row_group_size)`,
    set by the Parquet writer's buffer.
    """
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    # Stream examples through the Parquet writer. We track lengths on
    # the side so the outer release manifest can summarize without
    # re-reading the artifact.
    example_dir = root / "examples"
    from .supervision_export import SupervisionParquetWriter
    lengths: List[int] = []
    with SupervisionParquetWriter(example_dir, row_group_size=row_group_size) as writer:
        for ex in examples:
            writer.append(ex)
            lengths.append(int(ex.length))
    count_examples = writer.count

    # IMPORTANT: read the failures collection *after* examples have
    # been fully iterated — callers commonly pass the same list they
    # append to while the example generator runs (e.g. the dataset
    # builder). Snapshotting earlier would capture an empty list.
    failure_list = list(failures or [])

    failure_path = root / "failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failure_list:
            handle.write(json.dumps(asdict(failure), separators=(",", ":")))
            handle.write("\n")

    manifest = StructureSupervisionReleaseManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        count_examples=count_examples,
        count_failures=len(failure_list),
        lengths=_length_summary(lengths),
        sequence_lengths=lengths,
        provenance=dict(provenance or {}),
    )
    (root / "release_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )
    return root


def load_failure_records(path: str | Path) -> List[FailureRecord]:
    """Load failure records from a JSONL file."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [FailureRecord(**row) for row in rows]


def _length_summary(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
    }
