"""Export and import for structure supervision examples.

Format v0:
- `manifest.json`: dataset-level metadata and schema inventory
- `examples.jsonl`: per-example metadata and quality records (JSONL for cheap streaming reads)
- `tensors.parquet`: per-example ragged rows, zstd-compressed, row-group chunked

The Parquet artifact has one row per supervision example. Residue-axis length
is stored as Arrow `list<...>` so examples never get padded to a batch-wide
`max_length`; peak writer and reader memory is bounded by the row-group size
rather than the full corpus. This scales cleanly to archive-scale inputs
where a padded NPZ would run the process out of memory well before write.
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
from .supervision import StructureQualityMetadata, StructureSupervisionExample

SUPERVISION_EXPORT_FORMAT = "ferritin.structure_supervision.parquet.v0"
SUPERVISION_PARQUET_SCHEMA_VERSION = 1

# Field descriptors: (column_name, inner_shape, numpy_dtype, dataclass_attr)
# inner_shape = () means per-row shape (L,); (37,) means (L, 37); etc.
SEQUENCE_FIELDS: Tuple[Tuple[str, Tuple[int, ...], type, str], ...] = (
    ("aatype", (), np.int32, "aatype"),
    ("residue_index", (), np.int32, "residue_index"),
    ("seq_mask", (), np.float32, "seq_mask"),
)

STRUCTURE_FIELDS: Tuple[Tuple[str, Tuple[int, ...], type, str], ...] = (
    ("all_atom_positions", (37, 3), np.float32, "all_atom_positions"),
    ("all_atom_mask", (37,), np.float32, "all_atom_mask"),
    ("atom37_atom_exists", (37,), np.float32, "atom37_atom_exists"),
    ("atom14_gt_positions", (14, 3), np.float32, "atom14_gt_positions"),
    ("atom14_gt_exists", (14,), np.float32, "atom14_gt_exists"),
    ("atom14_atom_exists", (14,), np.float32, "atom14_atom_exists"),
    ("atom14_atom_is_ambiguous", (14,), np.float32, "atom14_atom_is_ambiguous"),
    ("residx_atom14_to_atom37", (14,), np.int32, "residx_atom14_to_atom37"),
    ("residx_atom37_to_atom14", (37,), np.int32, "residx_atom37_to_atom14"),
    ("pseudo_beta", (3,), np.float32, "pseudo_beta"),
    ("pseudo_beta_mask", (), np.float32, "pseudo_beta_mask"),
    ("phi", (), np.float32, "phi"),
    ("psi", (), np.float32, "psi"),
    ("omega", (), np.float32, "omega"),
    ("phi_mask", (), np.float32, "phi_mask"),
    ("psi_mask", (), np.float32, "psi_mask"),
    ("omega_mask", (), np.float32, "omega_mask"),
    ("chi_angles", (4,), np.float32, "chi_angles"),
    ("chi_mask", (4,), np.float32, "chi_mask"),
    ("rigidgroups_gt_frames", (8, 4, 4), np.float32, "rigidgroups_gt_frames"),
    ("rigidgroups_gt_exists", (8,), np.float32, "rigidgroups_gt_exists"),
    ("rigidgroups_group_exists", (8,), np.float32, "rigidgroups_group_exists"),
    ("rigidgroups_group_is_ambiguous", (8,), np.float32, "rigidgroups_group_is_ambiguous"),
)

TENSOR_FIELDS = SEQUENCE_FIELDS + STRUCTURE_FIELDS


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for the supervision release Parquet path. "
            "Install with `pip install pyarrow` (>=14 recommended)."
        )


def _ragged_arrow_type(inner_shape: Tuple[int, ...], dtype: type) -> "pa.DataType":
    pa_dtype = pa.from_numpy_dtype(np.dtype(dtype))
    current = pa_dtype
    for dim in reversed(inner_shape):
        current = pa.list_(current, dim)
    return pa.list_(current)


def build_supervision_schema() -> "pa.Schema":
    """Arrow schema for the supervision Parquet artifact."""
    _require_pyarrow()
    fields = [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("prep_run_id", pa.string()),
        ("chain_id", pa.string()),
        ("sequence", pa.string()),
        ("length", pa.int32()),
        ("code_rev", pa.string()),
        ("config_rev", pa.string()),
        ("quality_json", pa.string()),
    ]
    for name, inner_shape, dtype, _attr in TENSOR_FIELDS:
        fields.append((name, _ragged_arrow_type(inner_shape, dtype)))
    return pa.schema(fields)


def _make_ragged_column(
    arrays,
    inner_shape: Tuple[int, ...],
    dtype: type,
) -> "pa.Array":
    """Build a `list<FixedSizeList<...>>` column from per-example arrays.

    Each `arrays[i]` has shape `(L_i,) + inner_shape`. Output is a
    pyarrow ListArray whose elements are per-example slabs — no outer
    padding. Memory is `sum(L_i) * prod(inner_shape)` (the lower bound).
    """
    _require_pyarrow()
    lengths = [int(a.shape[0]) for a in arrays]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    if lengths:
        np.cumsum(lengths, out=offsets[1:])
    if arrays:
        flat = np.concatenate(
            [np.ascontiguousarray(a, dtype=dtype).reshape(-1) for a in arrays]
        )
    else:
        flat = np.zeros(0, dtype=dtype)
    current = pa.array(flat, type=pa.from_numpy_dtype(np.dtype(dtype)))
    for dim in reversed(inner_shape):
        current = pa.FixedSizeListArray.from_arrays(current, dim)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), current)


def _supervision_examples_to_record_batch(
    batch,
    schema: "pa.Schema",
) -> "pa.RecordBatch":
    """Convert a buffered chunk of supervision examples to a RecordBatch."""
    _require_pyarrow()
    columns: List["pa.Array"] = [
        pa.array([ex.record_id for ex in batch], type=pa.string()),
        pa.array([ex.source_id for ex in batch], type=pa.string()),
        pa.array([ex.prep_run_id for ex in batch], type=pa.string()),
        pa.array([ex.chain_id for ex in batch], type=pa.string()),
        pa.array([ex.sequence for ex in batch], type=pa.string()),
        pa.array([int(ex.length) for ex in batch], type=pa.int32()),
        pa.array([ex.code_rev for ex in batch], type=pa.string()),
        pa.array([ex.config_rev for ex in batch], type=pa.string()),
        pa.array(
            [
                json.dumps(asdict(ex.quality), separators=(",", ":")) if ex.quality is not None else None
                for ex in batch
            ],
            type=pa.string(),
        ),
    ]
    for name, inner_shape, dtype, attr in TENSOR_FIELDS:
        arrays = [np.ascontiguousarray(getattr(ex, attr)) for ex in batch]
        columns.append(_make_ragged_column(arrays, inner_shape, dtype))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _example_metadata(example: StructureSupervisionExample) -> Dict[str, object]:
    return {
        "record_id": example.record_id,
        "source_id": example.source_id,
        "prep_run_id": example.prep_run_id,
        "chain_id": example.chain_id,
        "sequence": example.sequence,
        "length": example.length,
        "code_rev": example.code_rev,
        "config_rev": example.config_rev,
        "quality": asdict(example.quality) if example.quality is not None else None,
    }


class SupervisionParquetWriter:
    """Streaming writer for the supervision Parquet artifact.

    Buffers up to `row_group_size` examples in memory, then flushes them
    as one Parquet row group. Peak memory is `O(row_group_size)`, not
    `O(n_examples)`.

    Usage:

        with SupervisionParquetWriter(out_dir) as w:
            for ex in examples_iter:
                w.append(ex)
        # Manifest + checksum are written on __exit__.
    """

    def __init__(self, out_dir, *, row_group_size: int = 512):
        _require_pyarrow()
        self.out_dir = Path(out_dir)
        self.row_group_size = row_group_size
        self._schema: Optional["pa.Schema"] = None
        self._writer: Optional["pq.ParquetWriter"] = None
        self._jsonl = None
        self._buffer: List[StructureSupervisionExample] = []
        self.count = 0
        self._lengths: List[int] = []
        self._parquet_path = self.out_dir / "tensors.parquet"
        self._jsonl_path = self.out_dir / "examples.jsonl"

    def __enter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._schema = build_supervision_schema()
        self._writer = pq.ParquetWriter(
            self._parquet_path,
            self._schema,
            compression="zstd",
            compression_level=3,
        )
        self._jsonl = self._jsonl_path.open("w", encoding="utf-8")
        return self

    def append(self, example: StructureSupervisionExample) -> None:
        # Stream the metadata row to jsonl immediately — no need to keep
        # it in memory; the parquet row holds the same data.
        self._jsonl.write(json.dumps(_example_metadata(example), separators=(",", ":")))
        self._jsonl.write("\n")
        self._lengths.append(int(example.length))
        self._buffer.append(example)
        self.count += 1
        if len(self._buffer) >= self.row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = _supervision_examples_to_record_batch(self._buffer, self._schema)
        self._writer.write_batch(batch, row_group_size=len(self._buffer))
        # Drop strong refs so the numpy arrays inside each example can be
        # garbage-collected before the next row group starts buffering.
        self._buffer.clear()

    def __exit__(self, exc_type, exc, tb):
        try:
            self._flush()
        finally:
            if self._writer is not None:
                self._writer.close()
            if self._jsonl is not None:
                self._jsonl.close()
        # Manifest only on clean exit; a partially-written release is
        # signaled by the missing manifest.json.
        if exc_type is None:
            manifest = {
                "format": SUPERVISION_EXPORT_FORMAT,
                "schema_version": SUPERVISION_PARQUET_SCHEMA_VERSION,
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


def export_structure_supervision_examples(
    examples: Iterable[StructureSupervisionExample],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    row_group_size: int = 512,
) -> Path:
    """Export supervision examples as JSONL metadata + streaming Parquet tensors.

    `examples` may be an iterator; it is consumed once without
    materializing the full list. Peak memory is `O(row_group_size)`.
    """
    out_path = Path(out_dir)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    with SupervisionParquetWriter(out_path, row_group_size=row_group_size) as writer:
        for ex in examples:
            writer.append(ex)
    return out_path


def iter_structure_supervision_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    batch_size: Optional[int] = None,
) -> Iterator[StructureSupervisionExample]:
    """Stream supervision examples from a release directory.

    Reads one Parquet row group at a time so peak memory is bounded,
    regardless of how many examples the release contains.
    """
    _require_pyarrow()
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SUPERVISION_EXPORT_FORMAT:
        raise ValueError(f"unsupported supervision export format: {manifest.get('format')!r}")
    if int(manifest.get("count", 0)) == 0:
        return

    parquet_path = root / manifest["tensor_file"]
    if verify_checksum:
        expected = manifest.get("tensor_sha256")
        if expected:
            verify_sha256(parquet_path, expected)

    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.metadata.num_row_groups
    for rg_idx in range(n_row_groups):
        rb = pf.read_row_group(rg_idx)
        cols = rb.to_pydict()
        n = rb.num_rows
        for i in range(n):
            yield _parquet_row_to_supervision_example(cols, i)


def load_structure_supervision_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> List[StructureSupervisionExample]:
    """Materialize the whole supervision release.

    Peer of `iter_structure_supervision_examples` for small releases.
    Prefer the iterator for archive-scale corpora.
    """
    return list(iter_structure_supervision_examples(path, verify_checksum=verify_checksum))


def _parquet_row_to_supervision_example(
    cols: Mapping[str, list], i: int
) -> StructureSupervisionExample:
    length = int(cols["length"][i])
    kwargs: Dict[str, object] = {
        "record_id": cols["record_id"][i],
        "source_id": cols["source_id"][i],
        "prep_run_id": cols["prep_run_id"][i],
        "chain_id": cols["chain_id"][i],
        "sequence": cols["sequence"][i] or "",
        "length": length,
        "code_rev": cols["code_rev"][i],
        "config_rev": cols["config_rev"][i],
    }
    for name, inner_shape, dtype, attr in TENSOR_FIELDS:
        arr = np.asarray(cols[name][i], dtype=dtype)
        kwargs[attr] = arr.reshape((length,) + inner_shape) if inner_shape else arr.reshape((length,))
    quality_json = cols["quality_json"][i]
    if quality_json:
        kwargs["quality"] = StructureQualityMetadata(**json.loads(quality_json))
    else:
        kwargs["quality"] = None
    return StructureSupervisionExample(**kwargs)
