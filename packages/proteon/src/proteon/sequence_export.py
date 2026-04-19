"""Export and import for sequence examples.

Format v0 (Parquet):
- `manifest.json`: dataset-level metadata (format tag, count, row_group_size,
  tensor file name + sha256, schema field list)
- `examples.jsonl`: per-example scalar metadata (cheap streaming reads for
  consumers that only need ids / chain / sequence strings)
- `tensors.parquet`: per-example ragged rows, zstd-compressed, row-group chunked

One row per `SequenceExample`. Residue-axis arrays are stored as Arrow
`list<...>` so examples never pad to a batch-wide `max_length`. MSA arrays
are `list<list<...>>` — both the depth and residue axes stay ragged. Peak
writer and reader memory is bounded by `row_group_size`, not corpus size,
which is the pay-off vs. the prior `tensors.npz` sink where MSA blocks
turned the whole corpus into a `(n, max_depth, max_L)` allocation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAS_PYARROW = False

from ._artifact_checksum import sha256_file, verify_sha256
from .sequence_example import SequenceExample

SEQUENCE_EXPORT_FORMAT = "proteon.sequence_example.parquet.v0"
SEQUENCE_PARQUET_SCHEMA_VERSION = 1


# Residue-axis fields: (column_name, numpy_dtype, dataclass_attr).
# Each is stored as Arrow list<T> of length L_i per row.
RESIDUE_FIELDS: Tuple[Tuple[str, type, str], ...] = (
    ("aatype", np.int32, "aatype"),
    ("residue_index", np.int32, "residue_index"),
    ("seq_mask", np.float32, "seq_mask"),
)

# MSA fields: (column_name, numpy_dtype, dataclass_attr).
# Stored as Arrow list<list<T>>: outer = depth_i, inner = L_i.
# Null when the example has no MSA.
MSA_FIELDS: Tuple[Tuple[str, type, str], ...] = (
    ("msa", np.int32, "msa"),
    ("deletion_matrix", np.float32, "deletion_matrix"),
    ("msa_mask", np.float32, "msa_mask"),
)


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for the sequence release Parquet path. "
            "Install with `pip install pyarrow` (>=14 recommended)."
        )


def build_sequence_schema() -> "pa.Schema":
    """Arrow schema for the sequence Parquet artifact."""
    _require_pyarrow()
    fields = [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("chain_id", pa.string()),
        ("sequence", pa.string()),
        ("length", pa.int32()),
        ("code_rev", pa.string()),
        ("config_rev", pa.string()),
    ]
    for name, dtype, _attr in RESIDUE_FIELDS:
        fields.append((name, pa.list_(pa.from_numpy_dtype(np.dtype(dtype)))))
    for name, dtype, _attr in MSA_FIELDS:
        fields.append((name, pa.list_(pa.list_(pa.from_numpy_dtype(np.dtype(dtype))))))
    fields.append(("template_mask", pa.list_(pa.from_numpy_dtype(np.dtype(np.float32)))))
    return pa.schema(fields)


def _make_residue_column(arrays: List[Optional[np.ndarray]], dtype: type) -> "pa.Array":
    """Build a `list<T>` column from per-example 1D residue-axis arrays."""
    _require_pyarrow()
    lengths = [0 if a is None else int(a.shape[0]) for a in arrays]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    if lengths:
        np.cumsum(lengths, out=offsets[1:])
    if arrays:
        chunks = [
            np.zeros(0, dtype=dtype) if a is None else np.ascontiguousarray(a, dtype=dtype).reshape(-1)
            for a in arrays
        ]
        flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)
    else:
        flat = np.zeros(0, dtype=dtype)
    values = pa.array(flat, type=pa.from_numpy_dtype(np.dtype(dtype)))
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)


def _make_msa_column(arrays: List[Optional[np.ndarray]], dtype: type) -> "pa.Array":
    """Build a `list<list<T>>` column from per-example 2D MSA arrays.

    `arrays[i]` has shape `(depth_i, L_i)` or is None. Outer list is
    per-example (nullable); inner list is per-MSA-row; values are the
    per-residue tokens/weights.
    """
    _require_pyarrow()
    pa_dtype = pa.from_numpy_dtype(np.dtype(dtype))
    inner_offsets: List[int] = [0]
    outer_offsets: List[int] = [0]
    flat_chunks: List[np.ndarray] = []
    validity: List[bool] = []
    cum_rows = 0
    cum_values = 0
    for a in arrays:
        if a is None:
            validity.append(False)
            outer_offsets.append(cum_rows)
            continue
        validity.append(True)
        depth_i, L_i = a.shape
        row_view = np.ascontiguousarray(a, dtype=dtype)
        flat_chunks.append(row_view.reshape(-1))
        for _ in range(depth_i):
            cum_values += L_i
            inner_offsets.append(cum_values)
        cum_rows += depth_i
        outer_offsets.append(cum_rows)
    flat = np.concatenate(flat_chunks) if flat_chunks else np.zeros(0, dtype=dtype)
    values = pa.array(flat, type=pa_dtype)
    inner = pa.ListArray.from_arrays(pa.array(inner_offsets, type=pa.int32()), values)
    mask = pa.array([not v for v in validity], type=pa.bool_())
    return pa.ListArray.from_arrays(
        pa.array(outer_offsets, type=pa.int32()),
        inner,
        mask=mask,
    )


def _sequence_examples_to_record_batch(
    batch: List[SequenceExample],
    schema: "pa.Schema",
) -> "pa.RecordBatch":
    """Convert a buffered chunk of sequence examples to a RecordBatch."""
    _require_pyarrow()
    columns: List["pa.Array"] = [
        pa.array([ex.record_id for ex in batch], type=pa.string()),
        pa.array([ex.source_id for ex in batch], type=pa.string()),
        pa.array([ex.chain_id for ex in batch], type=pa.string()),
        pa.array([ex.sequence for ex in batch], type=pa.string()),
        pa.array([int(ex.length) for ex in batch], type=pa.int32()),
        pa.array([ex.code_rev for ex in batch], type=pa.string()),
        pa.array([ex.config_rev for ex in batch], type=pa.string()),
    ]
    for name, dtype, attr in RESIDUE_FIELDS:
        columns.append(_make_residue_column([getattr(ex, attr) for ex in batch], dtype))
    for name, dtype, attr in MSA_FIELDS:
        columns.append(_make_msa_column([getattr(ex, attr) for ex in batch], dtype))
    template_arrays: List[Optional[np.ndarray]] = [
        None if ex.template_mask is None else np.ascontiguousarray(ex.template_mask, dtype=np.float32)
        for ex in batch
    ]
    columns.append(_make_residue_column(template_arrays, np.float32))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _example_metadata(example: SequenceExample) -> Dict[str, object]:
    return {
        "record_id": example.record_id,
        "source_id": example.source_id,
        "chain_id": example.chain_id,
        "sequence": example.sequence,
        "length": example.length,
        "code_rev": example.code_rev,
        "config_rev": example.config_rev,
    }


class SequenceParquetWriter:
    """Streaming writer for the sequence Parquet artifact.

    Buffers up to `row_group_size` examples in memory, then flushes them
    as one Parquet row group. Peak memory is `O(row_group_size)`, not
    `O(n_examples)`.

    Default row_group_size is 64 — smaller than supervision's 512 because
    MSA rows dominate the byte budget (depth × length × 3 arrays).

    Usage:

        with SequenceParquetWriter(out_dir) as w:
            for ex in examples_iter:
                w.append(ex)
        # Manifest + checksum are written on __exit__.
    """

    def __init__(self, out_dir, *, row_group_size: int = 64):
        _require_pyarrow()
        self.out_dir = Path(out_dir)
        self.row_group_size = row_group_size
        self._schema: Optional["pa.Schema"] = None
        self._writer: Optional["pq.ParquetWriter"] = None
        self._jsonl = None
        self._buffer: List[SequenceExample] = []
        self.count = 0
        self._parquet_path = self.out_dir / "tensors.parquet"
        self._jsonl_path = self.out_dir / "examples.jsonl"

    def __enter__(self):
        # Parquet writer creation is deferred until the first append() so
        # empty releases don't leave a zero-row tensors.parquet on disk
        # that the manifest claims isn't there. The jsonl file opens
        # eagerly since an empty jsonl is still a valid artifact.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._schema = build_sequence_schema()
        self._jsonl = self._jsonl_path.open("w", encoding="utf-8")
        return self

    def _ensure_writer_open(self) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._parquet_path,
                self._schema,
                compression="zstd",
                compression_level=3,
            )

    def append(self, example: SequenceExample) -> None:
        self._ensure_writer_open()
        self._jsonl.write(json.dumps(_example_metadata(example), separators=(",", ":")))
        self._jsonl.write("\n")
        self._buffer.append(example)
        self.count += 1
        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = _sequence_examples_to_record_batch(self._buffer, self._schema)
        self._writer.write_batch(batch, row_group_size=len(self._buffer))
        self._buffer.clear()

    def __exit__(self, exc_type, exc, tb):
        try:
            self._flush()
        finally:
            if self._writer is not None:
                self._writer.close()
            if self._jsonl is not None:
                self._jsonl.close()
        if exc_type is None:
            manifest: Dict[str, object] = {
                "format": SEQUENCE_EXPORT_FORMAT,
                "schema_version": SEQUENCE_PARQUET_SCHEMA_VERSION,
                "count": self.count,
                "examples_file": "examples.jsonl",
                "row_group_size": self.row_group_size,
            }
            if self.count > 0:
                manifest["tensor_file"] = self._parquet_path.name
                manifest["tensor_sha256"] = sha256_file(self._parquet_path)
                manifest["tensor_fields"] = [f.name for f in self._schema]
            (self.out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        return False


def export_sequence_examples(
    examples: Iterable[SequenceExample],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    row_group_size: int = 64,
) -> Path:
    """Export sequence examples as JSONL metadata + streaming Parquet tensors.

    `examples` may be an iterator; it is consumed once without materializing
    the full list. Peak memory is `O(row_group_size)`.
    """
    out_path = Path(out_dir)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    with SequenceParquetWriter(out_path, row_group_size=row_group_size) as writer:
        for ex in examples:
            writer.append(ex)
    return out_path


def iter_sequence_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> Iterator[SequenceExample]:
    """Stream sequence examples from a release directory.

    Reads one Parquet row group at a time so peak memory is bounded,
    regardless of how many examples the release contains.
    """
    _require_pyarrow()
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SEQUENCE_EXPORT_FORMAT:
        raise ValueError(f"unsupported sequence export format: {manifest.get('format')!r}")
    if int(manifest.get("count", 0)) == 0:
        return

    parquet_path = root / manifest["tensor_file"]
    if verify_checksum:
        expected = manifest.get("tensor_sha256")
        if expected:
            verify_sha256(parquet_path, expected)

    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        rb = pf.read_row_group(rg_idx)
        cols = rb.to_pydict()
        for i in range(rb.num_rows):
            yield _parquet_row_to_sequence_example(cols, i)


def load_sequence_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> List[SequenceExample]:
    """Materialize the whole sequence release.

    Peer of `iter_sequence_examples` for small releases. Prefer the
    iterator for archive-scale corpora where MSA blocks dominate.
    """
    return list(iter_sequence_examples(path, verify_checksum=verify_checksum))


def _parquet_row_to_sequence_example(
    cols: Mapping[str, list], i: int
) -> SequenceExample:
    length = int(cols["length"][i])
    kwargs: Dict[str, object] = {
        "record_id": cols["record_id"][i],
        "source_id": cols["source_id"][i],
        "chain_id": cols["chain_id"][i],
        "sequence": cols["sequence"][i] or "",
        "length": length,
        "code_rev": cols["code_rev"][i],
        "config_rev": cols["config_rev"][i],
    }
    for name, dtype, attr in RESIDUE_FIELDS:
        kwargs[attr] = np.asarray(cols[name][i], dtype=dtype)
    for name, dtype, attr in MSA_FIELDS:
        raw = cols[name][i]
        if raw is None:
            kwargs[attr] = None
            continue
        depth = len(raw)
        if depth == 0:
            kwargs[attr] = np.zeros((0, length), dtype=dtype)
        else:
            kwargs[attr] = np.asarray(raw, dtype=dtype).reshape(depth, length)
    tmpl = cols["template_mask"][i]
    if tmpl is None or len(tmpl) == 0:
        kwargs["template_mask"] = None
    else:
        kwargs["template_mask"] = np.asarray(tmpl, dtype=np.float32)
    return SequenceExample(**kwargs)
