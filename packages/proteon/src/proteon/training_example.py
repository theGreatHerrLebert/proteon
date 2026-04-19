"""Framework-neutral joined training examples.

The training release is written as a single Parquet file with one row per
example and variable-length residue axis (no padding). This replaces the
earlier padded NPZ format, which did not scale past a few thousand chains
because the padded-stack allocation grew with (n_examples * max_length *
fields). The Parquet writer streams in row-group chunks so peak memory is
bounded by the row-group size, not the corpus size.

Readers get Parquet's built-in predicate pushdown (filter by split or
length without materializing other rows) and any Arrow-compatible tool —
polars, DuckDB, pandas, pyarrow, torch via pyarrow — can load without a
framework-specific adapter. The streaming iterator
`iter_training_examples` materializes numpy arrays on the way out for
consumers that prefer that shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAS_PYARROW = False

from ._artifact_checksum import sha256_file, verify_sha256
from .sequence_example import SequenceExample
from .sequence_export import load_sequence_examples
from .supervision import StructureQualityMetadata, StructureSupervisionExample
from .supervision_export import (
    SEQUENCE_FIELDS as _SEQUENCE_FIELDS,
    STRUCTURE_FIELDS as _STRUCTURE_FIELDS,
    TENSOR_FIELDS as _TENSOR_FIELDS,
    iter_structure_supervision_examples,
    load_structure_supervision_examples,
)

TRAINING_EXPORT_FORMAT = "proteon.training_example.parquet.v0"
TRAINING_PARQUET_SCHEMA_VERSION = 1


@dataclass
class TrainingExample:
    """Thin join artifact over sequence and structure examples."""

    record_id: str
    source_id: Optional[str]
    chain_id: str
    split: str
    crop_start: Optional[int] = None
    crop_stop: Optional[int] = None
    weight: float = 1.0
    sequence: SequenceExample | None = None
    structure: StructureSupervisionExample | None = None


@dataclass
class TrainingReleaseManifest:
    """Shared manifest linking sequence and structure releases.

    `parquet_file` / `parquet_sha256` point to the streaming
    per-example Parquet artifact (one row per example, variable
    residue axis). `parquet_schema_version` pins the schema for
    readers to bail early on mismatches. `row_group_size` is
    informational — it controls the granularity of predicate
    pushdown when readers filter by split or length.
    """

    release_id: str
    artifact_type: str = "release_manifest"
    format: str = TRAINING_EXPORT_FORMAT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    sequence_release: str = ""
    structure_release: str = ""
    count_examples: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    examples_file: str = "training_examples.jsonl"
    parquet_file: Optional[str] = None
    parquet_sha256: Optional[str] = None
    parquet_schema_version: int = TRAINING_PARQUET_SCHEMA_VERSION
    parquet_fields: List[str] = field(default_factory=list)
    row_group_size: int = 512
    provenance: Dict[str, object] = field(default_factory=dict)


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for the training release Parquet path. "
            "Install with `pip install pyarrow` (>=14 recommended)."
        )


def _build_training_schema() -> "pa.Schema":
    """Build the Parquet schema for a training release.

    Ragged residue axis is stored as `list<...>`. Per-position fixed
    dimensions (e.g. 37 atoms x 3 coordinates) are stored as nested
    `FixedSizeList` so readers can reshape without per-row metadata.
    """
    _require_pyarrow()

    def _typ(inner_shape: Tuple[int, ...], dtype: type) -> "pa.DataType":
        pa_dtype = pa.from_numpy_dtype(np.dtype(dtype))
        current = pa_dtype
        for dim in reversed(inner_shape):
            current = pa.list_(current, dim)
        return pa.list_(current)

    fields = [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("chain_id", pa.string()),
        ("split", pa.string()),
        ("length", pa.int32()),
        ("weight", pa.float32()),
        ("crop_start", pa.int32()),
        ("crop_stop", pa.int32()),
        # Non-tensor scalars that must round-trip. Previously these were
        # zeroed on load: sequence="" and all revs/quality=None regardless
        # of what the source release carried. Consumers that reached
        # past the tensor fields got silently-degraded SequenceExample /
        # StructureSupervisionExample objects.
        ("sequence", pa.string()),
        ("code_rev", pa.string()),
        ("config_rev", pa.string()),
        ("prep_run_id", pa.string()),
        ("quality_json", pa.string()),
    ]
    for name, inner_shape, dtype, _ in _TENSOR_FIELDS:
        fields.append((name, _typ(inner_shape, dtype)))
    return pa.schema(fields)


def _make_ragged_column(
    arrays: Sequence[np.ndarray],
    inner_shape: Tuple[int, ...],
    dtype: type,
) -> "pa.Array":
    """Build a `list<...>` column from per-example arrays.

    Each `arrays[i]` has shape `(L_i,) + inner_shape`. Output is a
    pyarrow ListArray whose elements are reshaped per-example slabs —
    no outer padding. Memory is `sum(L_i) * prod(inner_shape)`, which
    is the fundamental lower bound.
    """
    _require_pyarrow()
    lengths = [int(a.shape[0]) for a in arrays]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    if lengths:
        np.cumsum(lengths, out=offsets[1:])
    if arrays:
        flat = np.concatenate([np.ascontiguousarray(a, dtype=dtype).reshape(-1) for a in arrays])
    else:
        flat = np.zeros(0, dtype=dtype)
    current = pa.array(flat, type=pa.from_numpy_dtype(np.dtype(dtype)))
    for dim in reversed(inner_shape):
        current = pa.FixedSizeListArray.from_arrays(current, dim)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), current)


def _training_examples_to_record_batch(
    batch: Sequence[TrainingExample],
    schema: "pa.Schema",
) -> "pa.RecordBatch":
    """Convert a chunk of TrainingExamples into one pyarrow RecordBatch."""
    _require_pyarrow()

    record_ids = [ex.record_id for ex in batch]
    source_ids = [ex.source_id for ex in batch]
    chain_ids = [ex.chain_id for ex in batch]
    splits = [ex.split for ex in batch]
    lengths = [int(ex.sequence.length if ex.sequence is not None else 0) for ex in batch]
    weights = [float(ex.weight) for ex in batch]
    crop_starts = [ex.crop_start for ex in batch]
    crop_stops = [ex.crop_stop for ex in batch]

    # Round-trip scalars. sequence/code_rev/config_rev are duplicated
    # between SequenceExample and StructureSupervisionExample; we take
    # them from the sequence side since the sequence release is the
    # canonical source for those (structure supervision mirrors them).
    sequences = [
        ex.sequence.sequence if ex.sequence is not None else ""
        for ex in batch
    ]
    code_revs = [
        (ex.sequence.code_rev if ex.sequence is not None else None)
        for ex in batch
    ]
    config_revs = [
        (ex.sequence.config_rev if ex.sequence is not None else None)
        for ex in batch
    ]
    prep_run_ids = [
        (ex.structure.prep_run_id if ex.structure is not None else None)
        for ex in batch
    ]
    quality_jsons: List[Optional[str]] = []
    for ex in batch:
        q = getattr(ex.structure, "quality", None) if ex.structure is not None else None
        if q is None:
            quality_jsons.append(None)
        else:
            quality_jsons.append(json.dumps(asdict(q), separators=(",", ":")))

    columns: List["pa.Array"] = [
        pa.array(record_ids, type=pa.string()),
        pa.array(source_ids, type=pa.string()),
        pa.array(chain_ids, type=pa.string()),
        pa.array(splits, type=pa.string()),
        pa.array(lengths, type=pa.int32()),
        pa.array(weights, type=pa.float32()),
        pa.array(crop_starts, type=pa.int32()),
        pa.array(crop_stops, type=pa.int32()),
        pa.array(sequences, type=pa.string()),
        pa.array(code_revs, type=pa.string()),
        pa.array(config_revs, type=pa.string()),
        pa.array(prep_run_ids, type=pa.string()),
        pa.array(quality_jsons, type=pa.string()),
    ]
    for name, inner_shape, dtype, attr in _SEQUENCE_FIELDS:
        per_row = [np.ascontiguousarray(getattr(ex.sequence, attr)) for ex in batch]
        columns.append(_make_ragged_column(per_row, inner_shape, dtype))
    for name, inner_shape, dtype, attr in _STRUCTURE_FIELDS:
        per_row = [np.ascontiguousarray(getattr(ex.structure, attr)) for ex in batch]
        columns.append(_make_ragged_column(per_row, inner_shape, dtype))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def join_training_examples(
    sequence_examples: Sequence[SequenceExample],
    structure_examples: Sequence[StructureSupervisionExample],
    *,
    split_assignments: Optional[Dict[str, str]] = None,
    crop_metadata: Optional[Dict[str, tuple[int, int]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[TrainingExample]:
    """Join sequence and structure artifacts by `record_id`."""
    seq_by_id = {ex.record_id: ex for ex in sequence_examples}
    struc_by_id = {ex.record_id: ex for ex in structure_examples}
    shared_ids = sorted(set(seq_by_id).intersection(struc_by_id))

    out: List[TrainingExample] = []
    for record_id in shared_ids:
        seq = seq_by_id[record_id]
        struc = struc_by_id[record_id]
        split = (split_assignments or {}).get(record_id, "train")
        crop = (crop_metadata or {}).get(record_id)
        out.append(
            TrainingExample(
                record_id=record_id,
                source_id=seq.source_id or struc.source_id,
                chain_id=seq.chain_id,
                split=split,
                crop_start=None if crop is None else int(crop[0]),
                crop_stop=None if crop is None else int(crop[1]),
                weight=float((weights or {}).get(record_id, 1.0)),
                sequence=seq,
                structure=struc,
            )
        )
    return out


def build_training_release(
    sequence_release_dir: str | Path,
    structure_release_dir: str | Path,
    out_dir: str | Path,
    *,
    release_id: str,
    split_assignments: Optional[Dict[str, str]] = None,
    crop_metadata: Optional[Dict[str, tuple[int, int]]] = None,
    weights: Optional[Dict[str, float]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    export_tensors: bool = True,
    row_group_size: int = 512,
    overwrite: bool = False,
) -> Path:
    """Build a training release by joining sequence and structure releases.

    With `export_tensors=True` (default), writes a streaming
    `training.parquet` artifact under the release root — one row per
    example, ragged residue axis, chunked into row groups of
    `row_group_size` examples so writer memory is bounded. The file
    is SHA-256'd into the manifest's `parquet_sha256` field. Set to
    False to keep the release pointer-only (manifest + JSONL index
    only, no tensor artifact).
    """
    # Sequence side is narrow (aatype + residue_index + seq_mask per
    # chain) and small enough to keep in memory; structure side streams
    # from its Parquet artifact so peak memory stays O(row_group_size)
    # regardless of corpus size.
    sequence_examples = load_sequence_examples(Path(sequence_release_dir) / "examples")
    seq_by_id = {ex.record_id: ex for ex in sequence_examples}

    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    split_counts: Dict[str, int] = {}
    count_examples = 0
    parquet_file: Optional[str] = None
    parquet_sha256: Optional[str] = None
    parquet_fields: List[str] = []
    row_group_size_used = row_group_size

    jsonl_path = root / "training_examples.jsonl"
    parquet_path = root / "training.parquet"
    schema = _build_training_schema() if export_tensors else None

    writer: Optional["pq.ParquetWriter"] = None
    chunk_buffer: List[TrainingExample] = []
    any_structure = False

    def _flush_chunk():
        nonlocal writer
        if not chunk_buffer:
            return
        if export_tensors:
            if writer is None:
                # Lazy-open so zero-example releases stay pointer-only
                # (no training.parquet file written if nothing joined).
                writer = pq.ParquetWriter(
                    parquet_path,
                    schema,
                    compression="zstd",
                    compression_level=3,
                )
            batch = _training_examples_to_record_batch(chunk_buffer, schema)
            writer.write_batch(batch, row_group_size=len(chunk_buffer))
        chunk_buffer.clear()

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
            for struc in iter_structure_supervision_examples(
                Path(structure_release_dir) / "examples"
            ):
                seq = seq_by_id.get(struc.record_id)
                if seq is None:
                    # Structure with no sequence counterpart — skip. Same
                    # semantics as the prior intersection-by-record_id
                    # join inside `join_training_examples`.
                    continue
                any_structure = True
                split = (split_assignments or {}).get(struc.record_id, "train")
                crop = (crop_metadata or {}).get(struc.record_id)
                weight = float((weights or {}).get(struc.record_id, 1.0))
                tex = TrainingExample(
                    record_id=struc.record_id,
                    source_id=seq.source_id or struc.source_id,
                    chain_id=seq.chain_id,
                    split=split,
                    crop_start=None if crop is None else int(crop[0]),
                    crop_stop=None if crop is None else int(crop[1]),
                    weight=weight,
                    sequence=seq,
                    structure=struc,
                )
                jsonl_handle.write(json.dumps(
                    {
                        "record_id": tex.record_id,
                        "source_id": tex.source_id,
                        "chain_id": tex.chain_id,
                        "split": tex.split,
                        "crop_start": tex.crop_start,
                        "crop_stop": tex.crop_stop,
                        "weight": tex.weight,
                    },
                    separators=(",", ":"),
                ))
                jsonl_handle.write("\n")
                split_counts[tex.split] = split_counts.get(tex.split, 0) + 1
                count_examples += 1
                chunk_buffer.append(tex)
                if len(chunk_buffer) >= row_group_size:
                    _flush_chunk()
            _flush_chunk()
    finally:
        if writer is not None:
            writer.close()

    if export_tensors and any_structure:
        parquet_file = "training.parquet"
        parquet_sha256 = sha256_file(parquet_path)
        parquet_fields = [f.name for f in schema]

    manifest = TrainingReleaseManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        sequence_release=str(Path(sequence_release_dir)),
        structure_release=str(Path(structure_release_dir)),
        count_examples=count_examples,
        split_counts=split_counts,
        parquet_file=parquet_file,
        parquet_sha256=parquet_sha256,
        parquet_fields=parquet_fields,
        row_group_size=row_group_size_used,
        provenance=dict(provenance or {}),
    )
    (root / "release_manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return root


def load_training_examples(
    release_dir: str | Path,
    *,
    split: Optional[str] = None,
    verify_checksum: bool = True,
) -> List[TrainingExample]:
    """Load a training release back into `TrainingExample` objects.

    Materializes the whole Parquet artifact. For large releases prefer
    `iter_training_examples` so you never hold more than one
    row-group in memory.

    `split` applies predicate pushdown at the row-group level so
    non-matching rows are skipped by the Parquet reader.

    Pointer-only releases (`parquet_file=None`) are reconstructed by
    re-joining the child sequence/structure releases.
    """
    return list(iter_training_examples(
        release_dir,
        split=split,
        verify_checksum=verify_checksum,
        batch_size=None,
    ))


def iter_training_examples(
    release_dir: str | Path,
    *,
    split: Optional[str] = None,
    batch_size: Optional[int] = None,
    verify_checksum: bool = True,
) -> Iterator:
    """Stream `TrainingExample` rows from a training release.

    Reads one Parquet row-group at a time so peak memory is bounded.
    `split`, when set, uses pyarrow dataset filtering with predicate
    pushdown; matching row-groups are read, others are skipped.

    `batch_size=None` (default) yields one `TrainingExample` per
    `__next__`. Setting a positive integer yields `list[TrainingExample]`
    chunks of length ≤ `batch_size` — contiguous in release order and
    respecting `split` filtering. The trailing chunk may be shorter
    than `batch_size` if the filtered row count isn't a multiple.
    """
    _require_pyarrow()
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"batch_size must be positive or None, got {batch_size}")
    root = Path(release_dir)
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != TRAINING_EXPORT_FORMAT:
        raise ValueError(f"unsupported training export format: {manifest.get('format')!r}")

    parquet_file = manifest.get("parquet_file")
    if parquet_file is None:
        # Pointer-only release — reconstruct by re-joining children.
        rows = [
            json.loads(line)
            for line in (root / manifest["examples_file"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seq_examples = load_sequence_examples(Path(manifest["sequence_release"]) / "examples")
        struc_examples = load_structure_supervision_examples(
            Path(manifest["structure_release"]) / "examples"
        )
        rejoined = join_training_examples(
            seq_examples,
            struc_examples,
            split_assignments={row["record_id"]: row["split"] for row in rows},
            weights={row["record_id"]: row["weight"] for row in rows},
            crop_metadata={
                row["record_id"]: (row["crop_start"], row["crop_stop"])
                for row in rows
                if row.get("crop_start") is not None
            },
        )
        # Pointer-only path — same per-item/batched contract as the
        # parquet path below.
        if batch_size is None:
            for ex in rejoined:
                if split is None or ex.split == split:
                    yield ex
        else:
            chunk: List[TrainingExample] = []
            for ex in rejoined:
                if split is not None and ex.split != split:
                    continue
                chunk.append(ex)
                if len(chunk) >= batch_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
        return

    parquet_path = root / parquet_file
    if verify_checksum:
        expected = manifest.get("parquet_sha256")
        if expected:
            verify_sha256(parquet_path, expected)

    pf = pq.ParquetFile(parquet_path)
    filters = None
    if split is not None:
        filters = [("split", "=", split)]

    if filters is not None:
        # Predicate pushdown via dataset API — skips row-groups whose
        # statistics prove no row matches the filter.
        import pyarrow.dataset as pads
        dataset = pads.dataset(parquet_path, format="parquet")
        scanner = dataset.scanner(filter=pads.field("split") == split)
        batches_iter: Iterable["pa.RecordBatch"] = scanner.to_batches()
    else:
        batches_iter = pf.iter_batches(batch_size=pf.metadata.row_group(0).num_rows) \
            if pf.metadata.num_row_groups > 0 else iter(())

    if batch_size is None:
        for rb in batches_iter:
            rows_tbl = rb.to_pydict()
            for i in range(rb.num_rows):
                yield _parquet_row_to_training_example(rows_tbl, i)
    else:
        chunk: List[TrainingExample] = []
        for rb in batches_iter:
            rows_tbl = rb.to_pydict()
            for i in range(rb.num_rows):
                chunk.append(_parquet_row_to_training_example(rows_tbl, i))
                if len(chunk) >= batch_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk


def _parquet_row_to_training_example(cols: Mapping[str, list], i: int) -> TrainingExample:
    """Decode one row of a Parquet RecordBatch into a TrainingExample."""
    length = int(cols["length"][i])
    # Scalars that were added to the schema for round-trip fidelity. The
    # .get() fallbacks preserve back-compat with older v0-era parquets
    # that predate these columns; those will still load but with the
    # previous zeroed-field semantics.
    sequence_str = cols.get("sequence", [None] * (i + 1))[i] or ""
    code_rev = cols.get("code_rev", [None] * (i + 1))[i]
    config_rev = cols.get("config_rev", [None] * (i + 1))[i]
    prep_run_id = cols.get("prep_run_id", [None] * (i + 1))[i]
    quality_json = cols.get("quality_json", [None] * (i + 1))[i]

    seq_kwargs: Dict[str, object] = {
        "record_id": cols["record_id"][i],
        "source_id": cols["source_id"][i],
        "chain_id": cols["chain_id"][i],
        "sequence": sequence_str,
        "length": length,
        "code_rev": code_rev,
        "config_rev": config_rev,
    }
    struc_kwargs: Dict[str, object] = {
        "record_id": cols["record_id"][i],
        "source_id": cols["source_id"][i],
        "prep_run_id": prep_run_id,
        "chain_id": cols["chain_id"][i],
        "sequence": sequence_str,
        "length": length,
        "code_rev": code_rev,
        "config_rev": config_rev,
        "quality": (
            StructureQualityMetadata(**json.loads(quality_json)) if quality_json else None
        ),
    }
    for name, inner_shape, dtype, attr in _SEQUENCE_FIELDS:
        arr = np.asarray(cols[name][i], dtype=dtype)
        reshaped = arr.reshape((length,) + inner_shape) if inner_shape else arr.reshape((length,))
        # aatype/residue_index/seq_mask live on both SequenceExample and
        # StructureSupervisionExample; one parquet column feeds both so
        # the on-disk bytes aren't duplicated.
        seq_kwargs[attr] = reshaped
        struc_kwargs[attr] = reshaped
    for name, inner_shape, dtype, attr in _STRUCTURE_FIELDS:
        arr = np.asarray(cols[name][i], dtype=dtype)
        struc_kwargs[attr] = arr.reshape((length,) + inner_shape) if inner_shape else arr.reshape((length,))
    # SequenceExample has optional MSA/template fields that the
    # v0 parquet schema doesn't carry yet — default them to None.
    seq_kwargs.setdefault("msa", None)
    seq_kwargs.setdefault("deletion_matrix", None)
    seq_kwargs.setdefault("msa_mask", None)
    seq_kwargs.setdefault("template_mask", None)
    seq = SequenceExample(**seq_kwargs)
    struc = StructureSupervisionExample(**struc_kwargs)
    return TrainingExample(
        record_id=cols["record_id"][i],
        source_id=cols["source_id"][i],
        chain_id=cols["chain_id"][i],
        split=cols["split"][i],
        crop_start=cols["crop_start"][i],
        crop_stop=cols["crop_stop"][i],
        weight=float(cols["weight"][i]),
        sequence=seq,
        structure=struc,
    )
