"""Top-level corpus release manifests linking sequence/structure/training layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CorpusReleaseManifest:
    """Top-level manifest for one coherent corpus release."""

    release_id: str
    artifact_type: str = "release_manifest"
    format: str = "proteon.corpus_release.v0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    prep_policy_version: Optional[str] = None
    filter_policy_version: Optional[str] = None
    clustering_policy_version: Optional[str] = None
    split_policy_version: Optional[str] = None
    prepared_manifest: Optional[str] = None
    sequence_release: Optional[str] = None
    structure_release: Optional[str] = None
    training_release: Optional[str] = None
    count_prepared: int = 0
    count_sequence_examples: int = 0
    count_structure_examples: int = 0
    count_training_examples: int = 0
    count_ingestion_failures: int = 0
    count_rescued_inputs: int = 0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    sequence_lengths: Dict[str, float] = field(default_factory=dict)
    structure_lengths: Dict[str, float] = field(default_factory=dict)
    rescued_inputs_manifest: Optional[str] = None
    provenance: Dict[str, object] = field(default_factory=dict)


def build_corpus_release_manifest(
    out_dir: str | Path,
    *,
    release_id: str,
    prepared_manifest: str | Path | None = None,
    sequence_release: str | Path | None = None,
    structure_release: str | Path | None = None,
    training_release: str | Path | None = None,
    ingestion_failures: str | Path | None = None,
    rescued_inputs_manifest: str | Path | None = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    prep_policy_version: Optional[str] = None,
    filter_policy_version: Optional[str] = None,
    clustering_policy_version: Optional[str] = None,
    split_policy_version: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
) -> Path:
    """Write a top-level manifest linking the child release layers.

    `ingestion_failures` points to a `FailureRecord` JSONL captured
    during raw intake (files that failed to parse / load). These are
    counted into `count_ingestion_failures` and merged into
    `failure_breakdown` by class — so a release that silently drops
    bad inputs can't look "clean" in the top-level manifest.
    """
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    prepared_count = _count_jsonl_rows(prepared_manifest)
    seq_manifest = _load_json(sequence_release, "release_manifest.json")
    struc_manifest = _load_json(structure_release, "release_manifest.json")
    train_manifest = _load_json(training_release, "release_manifest.json")

    failure_breakdown: Dict[str, int] = {}
    if sequence_release is not None:
        _merge_failure_breakdown(failure_breakdown, Path(sequence_release) / "failures.jsonl")
    if structure_release is not None:
        _merge_failure_breakdown(failure_breakdown, Path(structure_release) / "failures.jsonl")
    ingestion_count = 0
    if ingestion_failures is not None:
        _merge_failure_breakdown(failure_breakdown, Path(ingestion_failures))
        ingestion_count = _count_jsonl_rows(ingestion_failures)
    rescued_count = _count_jsonl_rows(rescued_inputs_manifest)

    manifest = CorpusReleaseManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        prep_policy_version=prep_policy_version,
        filter_policy_version=filter_policy_version,
        clustering_policy_version=clustering_policy_version,
        split_policy_version=split_policy_version,
        prepared_manifest=None if prepared_manifest is None else str(Path(prepared_manifest)),
        sequence_release=None if sequence_release is None else str(Path(sequence_release)),
        structure_release=None if structure_release is None else str(Path(structure_release)),
        training_release=None if training_release is None else str(Path(training_release)),
        count_prepared=prepared_count,
        count_sequence_examples=int((seq_manifest or {}).get("count_examples", 0)),
        count_structure_examples=int((struc_manifest or {}).get("count_examples", 0)),
        count_training_examples=int((train_manifest or {}).get("count_examples", 0)),
        count_ingestion_failures=ingestion_count,
        count_rescued_inputs=rescued_count,
        failure_breakdown=failure_breakdown,
        split_counts=dict((train_manifest or {}).get("split_counts", {})),
        sequence_lengths=dict((seq_manifest or {}).get("lengths", {})),
        structure_lengths=dict((struc_manifest or {}).get("lengths", {})),
        rescued_inputs_manifest=(
            None if rescued_inputs_manifest is None else str(Path(rescued_inputs_manifest))
        ),
        provenance=dict(provenance or {}),
    )
    (root / "corpus_release_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )
    return root


def load_corpus_release_manifest(path: str | Path) -> CorpusReleaseManifest:
    """Load a top-level corpus release manifest."""
    return CorpusReleaseManifest(**json.loads(Path(path).read_text(encoding="utf-8")))


def _load_json(root: str | Path | None, filename: str) -> Optional[dict]:
    if root is None:
        return None
    path = Path(root) / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: str | Path | None) -> int:
    if path is None:
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _merge_failure_breakdown(dst: Dict[str, int], path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("failure_class", "internal_pipeline_error"))
        dst[key] = dst.get(key, 0) + 1
