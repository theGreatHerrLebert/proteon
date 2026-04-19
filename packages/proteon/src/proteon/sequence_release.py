"""Release-oriented wrappers around sequence example exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .failure_taxonomy import classify_exception
from .sequence_example import SequenceExample, build_sequence_example
from .sequence_export import (
    SEQUENCE_EXPORT_FORMAT,
    SEQUENCE_PARQUET_SCHEMA_VERSION,
    export_sequence_examples,
)
from .supervision_release import FailureRecord


@dataclass
class SequenceReleaseManifest:
    release_id: str
    artifact_type: str = "release_manifest"
    format: str = SEQUENCE_EXPORT_FORMAT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    count_examples: int = 0
    count_failures: int = 0
    example_export_dir: str = "examples"
    examples_file: str = "examples/examples.jsonl"
    # `tensor_file` is None when `count_examples == 0` since no
    # tensors.parquet is written in that case; set to
    # "examples/tensors.parquet" only when a real artifact exists so
    # the manifest doesn't point at a nonexistent path.
    tensor_file: Optional[str] = None
    tensor_schema_version: int = SEQUENCE_PARQUET_SCHEMA_VERSION
    failure_file: str = "failures.jsonl"
    lengths: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)


def build_sequence_release(
    examples: Iterable[SequenceExample],
    out_dir: str | Path,
    *,
    release_id: str,
    failures: Optional[Iterable[FailureRecord]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
    row_group_size: int = 64,
) -> Path:
    """Build a sequence release by streaming examples through the Parquet writer.

    `examples` may be any iterable (including a generator). It is
    consumed once without materializing the full list — MSA blocks in
    particular are never corpus-sized-retained, only `row_group_size`
    at a time.

    `failures` is read *after* `examples` is fully iterated, so the
    canonical `_iter_examples` pattern (where a dataset builder appends
    to the same failures list while the generator runs) works
    correctly.
    """
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    example_dir = root / "examples"
    from .sequence_export import SequenceParquetWriter
    lengths: List[int] = []
    with SequenceParquetWriter(example_dir, row_group_size=row_group_size) as writer:
        for ex in examples:
            writer.append(ex)
            lengths.append(int(ex.length))
    count_examples = writer.count

    # Snapshot failures AFTER the example generator has been exhausted.
    # Upstream callers commonly pass a list they append to during
    # iteration; reading earlier would capture an empty snapshot.
    failure_list = list(failures or [])

    with (root / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for failure in failure_list:
            handle.write(json.dumps(asdict(failure), separators=(",", ":")))
            handle.write("\n")
    manifest = SequenceReleaseManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        count_examples=count_examples,
        count_failures=len(failure_list),
        tensor_file="examples/tensors.parquet" if count_examples > 0 else None,
        lengths=_length_summary(lengths),
        provenance=dict(provenance or {}),
    )
    (root / "release_manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return root


def build_sequence_dataset(
    structures: Sequence,
    out_dir: str | Path,
    *,
    release_id: str,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    msas: Optional[Sequence[Optional[Sequence[str]]]] = None,
    deletion_matrices: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,
    template_masks: Optional[Sequence[Optional[Sequence[float]]]] = None,
    msa_engine: Optional[object] = None,
    msa_max_seqs: int = 256,
    msa_gap_idx: int = 21,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
) -> Path:
    """Build a sequence-dataset release.

    `msa_engine` wires the GPU MSA search into the release: when
    provided (typically a `MsaSearch` / connector engine), each
    structure's MSA is computed on the fly from the engine's target
    corpus and baked into the `SequenceExample`. Overrides any
    explicit `msas` / `deletion_matrices` for structures where the
    engine returns a result; explicit inputs still win if the engine
    raises on a particular query (documented as an escape hatch for
    oddball queries that the search doesn't handle gracefully).

    `msa_max_seqs` / `msa_gap_idx` flow through to
    `engine.search_and_build_msa`. Defaults match the upstream
    MSA-assembly conventions (256 rows, gap_idx=21 matches the
    `X`-aware 21-letter protein alphabet).
    """
    n = len(structures)
    record_ids = _expand_optional(record_ids, n)
    source_ids = _expand_optional(source_ids, n)
    chain_ids = _expand_optional(chain_ids, n)
    msas = _expand_optional(msas, n)
    deletion_matrices = _expand_optional(deletion_matrices, n)
    template_masks = _expand_optional(template_masks, n)

    # Generator: yields one SequenceExample per structure and appends
    # failures to `failures` out-of-band. `build_sequence_release`
    # consumes the iterator and writes each example straight into the
    # Parquet row-group buffer — no intermediate `list[SequenceExample]`,
    # so MSA blocks are never corpus-sized-retained in Python memory.
    failures: List[FailureRecord] = []

    def _iter_examples():
        for i, structure in enumerate(structures):
            record_id = record_ids[i] or _default_record_id(structure, chain_ids[i])
            try:
                if msa_engine is not None and msas[i] is None and deletion_matrices[i] is None:
                    from .msa_backend import build_sequence_example_with_msa
                    yield build_sequence_example_with_msa(
                        structure,
                        msa_engine,
                        record_id=record_id,
                        source_id=source_ids[i],
                        chain_id=chain_ids[i],
                        code_rev=code_rev,
                        config_rev=config_rev,
                        template_mask=template_masks[i],
                        max_seqs=msa_max_seqs,
                        gap_idx=msa_gap_idx,
                    )
                    continue
                yield build_sequence_example(
                    structure,
                    record_id=record_id,
                    source_id=source_ids[i],
                    chain_id=chain_ids[i],
                    code_rev=code_rev,
                    config_rev=config_rev,
                    msa=msas[i],
                    deletion_matrix=deletion_matrices[i],
                    template_mask=template_masks[i],
                )
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        record_id=record_id,
                        stage="sequence_example",
                        failure_class=classify_exception(exc),
                        message=str(exc),
                        source_id=source_ids[i],
                        code_rev=code_rev,
                        config_rev=config_rev,
                        provenance={"exception_type": type(exc).__name__},
                    )
                )

    return build_sequence_release(
        _iter_examples(),
        out_dir,
        release_id=release_id,
        failures=failures,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance=provenance,
        overwrite=overwrite,
    )


def _length_summary(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(lengths)), "max": float(max(lengths)), "mean": float(sum(lengths) / len(lengths))}


def _default_record_id(structure, chain_id: Optional[str]) -> str:
    ident = getattr(structure, "identifier", None) or "structure"
    return f"{ident}:{chain_id}" if chain_id else str(ident)


def _expand_optional(values, n: int) -> list:
    if values is None:
        return [None] * n
    if len(values) != n:
        raise ValueError(f"expected {n} items, got {len(values)}")
    return list(values)
