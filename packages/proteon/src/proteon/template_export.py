"""Export/import for AlphaFold template features (`TemplateFeatures`).

Templates are a *separate* release artifact, not columns on the sequence or
structure parquet. They are large, sparse, optional, and independently
regenerable, so they live in their own `templates.parquet` keyed by `record_id`
and are left-joined onto a training example at load time. This keeps the big
sequence/structure artifacts unchanged, lets template-free jobs skip the columns
entirely, and lets template retrieval version independently.

Encoding (doubly-ragged):
- Each `(N_templates, L, *inner)` tensor is flattened to `(N, L*prod(inner))`
  and stored as Arrow `list<list<scalar>>` — outer = templates (nullable), inner
  = the flattened residue row. `N_templates` and `L` are both ragged across
  examples; the fixed inner dims (37x3, 7x2, ...) are folded into the inner row
  and reshaped on read. The model collator pads `N` to `max_templates` and builds
  `template_mask` (OpenFold `make_fixed_size`), so nothing is padded on disk.
- `None` (templates not generated) vs `N=0` (retrieval ran, no usable hits) are
  distinct and both preserved: `None` stores null columns + null
  `n_templates`/`query_len`; `N=0` stores `n_templates=0`, a real `query_len`,
  and present-but-empty list columns.
- Derived geometry (pseudo_beta, torsions) is Optional per the `TemplateFeatures`
  contract — the sequence/CIGAR path leaves it `None` while the structure path
  populates it; each derived column is independently nullable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from .templates import TemplateFeatures

TEMPLATE_EXPORT_FORMAT = "proteon.template_features.parquet.v0"
TEMPLATE_PARQUET_SCHEMA_VERSION = 1

# Per-template-tensor fields: (column, inner_shape, dtype, attr, mandatory).
# Mandatory tensors (always present when a bundle exists) define N and L;
# derived tensors are Optional (null for the sequence/CIGAR path).
TEMPLATE_TENSOR_FIELDS: Tuple[Tuple[str, Tuple[int, ...], type, str, bool], ...] = (
    ("template_aatype", (), np.int32, "template_aatype", True),
    ("template_all_atom_positions", (37, 3), np.float32, "template_all_atom_positions", True),
    ("template_all_atom_masks", (37,), np.float32, "template_all_atom_masks", True),
    ("template_pseudo_beta", (3,), np.float32, "template_pseudo_beta", False),
    ("template_pseudo_beta_mask", (), np.float32, "template_pseudo_beta_mask", False),
    ("template_torsion_angles_sin_cos", (7, 2), np.float32, "template_torsion_angles_sin_cos", False),
    ("template_alt_torsion_angles_sin_cos", (7, 2), np.float32, "template_alt_torsion_angles_sin_cos", False),
    ("template_torsion_angles_mask", (7,), np.float32, "template_torsion_angles_mask", False),
)

# The mandatory tensor whose null-ness encodes "bundle is None" vs "bundle exists".
_BUNDLE_PRESENCE_FIELD = "template_aatype"


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for the template release Parquet path. "
            "Install with `pip install pyarrow` (>=14 recommended)."
        )


def validate_template_features(tf: TemplateFeatures) -> None:
    """Reject a malformed bundle before serialization: mandatory tensors must
    agree on `(N, L)`, `template_sum_probs` must be `(N,)`, every non-None derived
    tensor must agree on `(N, L)`, and scores must be finite. `N=0` is valid."""
    n, length = int(tf.n_templates), int(tf.query_len)
    if n < 0 or length < 0:
        raise ValueError(f"template bundle has negative n_templates/query_len: {n}, {length}")
    expect = {
        "template_aatype": (n, length),
        "template_all_atom_positions": (n, length, 37, 3),
        "template_all_atom_masks": (n, length, 37),
        "template_pseudo_beta": (n, length, 3),
        "template_pseudo_beta_mask": (n, length),
        "template_torsion_angles_sin_cos": (n, length, 7, 2),
        "template_alt_torsion_angles_sin_cos": (n, length, 7, 2),
        "template_torsion_angles_mask": (n, length, 7),
    }
    for _col, _inner, _dtype, attr, mandatory in TEMPLATE_TENSOR_FIELDS:
        arr = getattr(tf, attr)
        if arr is None:
            if mandatory:
                raise ValueError(f"mandatory template tensor {attr!r} is None")
            continue
        if tuple(np.shape(arr)) != expect[attr]:
            raise ValueError(
                f"template tensor {attr!r} has shape {tuple(np.shape(arr))}, expected {expect[attr]}"
            )
    sp = np.asarray(tf.template_sum_probs, dtype=np.float32)
    if sp.shape != (n,):
        raise ValueError(f"template_sum_probs has shape {sp.shape}, expected {(n,)}")
    if sp.size and not np.all(np.isfinite(sp)):
        raise ValueError("template_sum_probs contains non-finite values")


def build_template_schema() -> "pa.Schema":
    """Arrow schema for `templates.parquet`."""
    _require_pyarrow()
    f32 = pa.from_numpy_dtype(np.dtype(np.float32))
    fields = [
        ("record_id", pa.string()),
        ("n_templates", pa.int32()),
        ("query_len", pa.int32()),
        ("template_sum_probs", pa.list_(f32)),
    ]
    for name, _inner, dtype, _attr, _mand in TEMPLATE_TENSOR_FIELDS:
        fields.append((name, pa.list_(pa.list_(pa.from_numpy_dtype(np.dtype(dtype))))))
    return pa.schema(fields)


def _make_doubly_ragged_column(
    arrays: List[Optional[np.ndarray]], dtype: type
) -> "pa.Array":
    """`list<list<scalar>>` from per-example `(N_i, L_i, *inner)` arrays.

    Each present array is flattened so one template = one inner row of length
    `L_i*prod(inner)`. Outer list = templates (nullable: `None` → null); the inner
    shape is recovered on read. An `N=0` array is present-but-empty (validity
    True, no rows) — distinct from `None` (validity False)."""
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
        n_i = int(a.shape[0])
        row_len = int(np.prod(a.shape[1:]))  # == L_i * prod(inner)
        flat_chunks.append(np.ascontiguousarray(a, dtype=dtype).reshape(-1))
        for _ in range(n_i):
            cum_values += row_len
            inner_offsets.append(cum_values)
        cum_rows += n_i
        outer_offsets.append(cum_rows)
    flat = np.concatenate(flat_chunks) if flat_chunks else np.zeros(0, dtype=dtype)
    values = pa.array(flat, type=pa_dtype)
    inner = pa.ListArray.from_arrays(pa.array(inner_offsets, type=pa.int32()), values)
    mask = pa.array([not v for v in validity], type=pa.bool_())
    return pa.ListArray.from_arrays(pa.array(outer_offsets, type=pa.int32()), inner, mask=mask)


def _make_sum_probs_column(features: List[Optional[TemplateFeatures]]) -> "pa.Array":
    """`list<float32>` of per-template scores; null when the bundle is `None`."""
    _require_pyarrow()
    arrays: List[Optional[np.ndarray]] = [
        None if tf is None else np.ascontiguousarray(tf.template_sum_probs, dtype=np.float32)
        for tf in features
    ]
    offsets: List[int] = [0]
    validity: List[bool] = []
    chunks: List[np.ndarray] = []
    cum = 0
    for a in arrays:
        if a is None:
            validity.append(False)
            offsets.append(cum)
            continue
        validity.append(True)
        chunks.append(a.reshape(-1))
        cum += int(a.shape[0])
        offsets.append(cum)
    flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    values = pa.array(flat, type=pa.from_numpy_dtype(np.dtype(np.float32)))
    mask = pa.array([not v for v in validity], type=pa.bool_())
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values, mask=mask)


def _items_to_record_batch(
    batch: List[Tuple[str, Optional[TemplateFeatures]]], schema: "pa.Schema"
) -> "pa.RecordBatch":
    _require_pyarrow()
    ids = [rid for rid, _ in batch]
    feats = [tf for _, tf in batch]
    columns: List["pa.Array"] = [
        pa.array(ids, type=pa.string()),
        pa.array([None if tf is None else int(tf.n_templates) for tf in feats], type=pa.int32()),
        pa.array([None if tf is None else int(tf.query_len) for tf in feats], type=pa.int32()),
        _make_sum_probs_column(feats),
    ]
    for name, _inner, dtype, attr, _mand in TEMPLATE_TENSOR_FIELDS:
        arrays = [None if tf is None else getattr(tf, attr) for tf in feats]
        columns.append(_make_doubly_ragged_column(arrays, dtype))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


@dataclass
class TemplateReleaseManifest:
    release_id: str
    format: str = TEMPLATE_EXPORT_FORMAT
    schema_version: int = TEMPLATE_PARQUET_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    count: int = 0
    count_with_templates: int = 0  # N >= 1
    count_zero_templates: int = 0  # N == 0 (retrieval ran, no hits)
    count_none: int = 0            # bundle absent (no retrieval)
    tensor_file: Optional[str] = None
    tensor_sha256: Optional[str] = None
    row_group_size: int = 64


class TemplateParquetWriter:
    """Streaming writer for `templates.parquet`. Buffers up to `row_group_size`
    items, flushes one row group at a time — peak memory `O(row_group_size)`."""

    def __init__(self, out_dir, *, release_id: str = "templates", row_group_size: int = 64):
        _require_pyarrow()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.release_id = release_id
        self.row_group_size = row_group_size
        self.schema = build_template_schema()
        self._buf: List[Tuple[str, Optional[TemplateFeatures]]] = []
        self._writer: Optional["pq.ParquetWriter"] = None
        self._path = self.out_dir / "templates.parquet"
        self.count = 0
        self.count_with_templates = 0
        self.count_zero_templates = 0
        self.count_none = 0

    def append(self, record_id: str, features: Optional[TemplateFeatures]) -> None:
        if features is not None:
            validate_template_features(features)
            if int(features.n_templates) == 0:
                self.count_zero_templates += 1
            else:
                self.count_with_templates += 1
        else:
            self.count_none += 1
        self.count += 1
        self._buf.append((record_id, features))
        if len(self._buf) >= self.row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        batch = _items_to_record_batch(self._buf, self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, self.schema, compression="zstd")
        self._writer.write_table(pa.Table.from_batches([batch], self.schema))
        self._buf.clear()

    def __enter__(self) -> "TemplateParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            # A failure mid-stream: release the file handle but publish NO
            # manifest — a partial parquet must not look like a complete release.
            # A reader then fails cleanly on the missing manifest (codex catch).
            if self._writer is not None:
                self._writer.close()
            return False
        self._flush()
        manifest = TemplateReleaseManifest(
            release_id=self.release_id,
            count=self.count,
            count_with_templates=self.count_with_templates,
            count_zero_templates=self.count_zero_templates,
            count_none=self.count_none,
            row_group_size=self.row_group_size,
        )
        if self._writer is not None:
            self._writer.close()
            manifest.tensor_file = "templates.parquet"
            manifest.tensor_sha256 = sha256_file(self._path)
        elif self._path.exists():  # nothing written but a stale file existed
            self._path.unlink()
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest.__dict__, indent=2))
        return False


def write_template_artifact(
    items: Iterable[Tuple[str, Optional[TemplateFeatures]]],
    out_dir,
    *,
    release_id: str = "templates",
    row_group_size: int = 64,
) -> Path:
    """Write `(record_id, TemplateFeatures | None)` pairs to a template artifact."""
    out = Path(out_dir)
    with TemplateParquetWriter(out, release_id=release_id, row_group_size=row_group_size) as w:
        for record_id, features in items:
            w.append(record_id, features)
    return out


def _row_to_features(cols: Mapping[str, list], i: int) -> Optional[TemplateFeatures]:
    presence = cols.get(_BUNDLE_PRESENCE_FIELD)
    if presence is None or presence[i] is None:
        return None  # bundle was None (not generated)
    n = int(cols["n_templates"][i])
    length = int(cols["query_len"][i])
    kwargs: Dict[str, object] = {}
    for name, inner, dtype, attr, mandatory in TEMPLATE_TENSOR_FIELDS:
        raw = cols[name][i] if name in cols else None
        if raw is None:
            if mandatory:
                raise ValueError(f"mandatory template column {name!r} is null for a present bundle")
            kwargs[attr] = None
            continue
        shape = (n, length) + inner
        if n == 0:
            kwargs[attr] = np.zeros(shape, dtype=dtype)
        else:
            kwargs[attr] = np.asarray(raw, dtype=dtype).reshape(shape)
    sp = cols["template_sum_probs"][i]
    sum_probs = np.zeros((0,), dtype=np.float32) if sp is None else np.asarray(sp, dtype=np.float32)
    return TemplateFeatures(
        template_aatype=kwargs["template_aatype"],
        template_all_atom_positions=kwargs["template_all_atom_positions"],
        template_all_atom_masks=kwargs["template_all_atom_masks"],
        template_sum_probs=sum_probs,
        n_templates=n,
        query_len=length,
        template_pseudo_beta=kwargs["template_pseudo_beta"],
        template_pseudo_beta_mask=kwargs["template_pseudo_beta_mask"],
        template_torsion_angles_sin_cos=kwargs["template_torsion_angles_sin_cos"],
        template_alt_torsion_angles_sin_cos=kwargs["template_alt_torsion_angles_sin_cos"],
        template_torsion_angles_mask=kwargs["template_torsion_angles_mask"],
    )


def _load_manifest(out_dir: Path) -> TemplateReleaseManifest:
    data = json.loads((out_dir / "manifest.json").read_text())
    fmt = data.get("format")
    if fmt != TEMPLATE_EXPORT_FORMAT:
        raise ValueError(
            f"unexpected template artifact format {fmt!r}, expected {TEMPLATE_EXPORT_FORMAT!r}"
        )
    ver = data.get("schema_version")
    if ver is None or int(ver) > TEMPLATE_PARQUET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported template schema_version {ver!r} "
            f"(reader supports <= {TEMPLATE_PARQUET_SCHEMA_VERSION})"
        )
    return TemplateReleaseManifest(
        **{k: v for k, v in data.items() if k in TemplateReleaseManifest.__dataclass_fields__}
    )


def iter_template_artifact(
    out_dir, *, verify_checksum: bool = True
) -> Iterator[Tuple[str, Optional[TemplateFeatures]]]:
    """Stream `(record_id, TemplateFeatures | None)` from a template artifact, one
    row group at a time. Peak memory is `O(row_group_size)`."""
    _require_pyarrow()
    out = Path(out_dir)
    manifest = _load_manifest(out)
    if manifest.tensor_file is None:
        return
    path = out / manifest.tensor_file
    if verify_checksum and manifest.tensor_sha256:
        verify_sha256(path, manifest.tensor_sha256)
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        cols = pf.read_row_group(rg).to_pydict()
        for i in range(len(cols["record_id"])):
            yield cols["record_id"][i], _row_to_features(cols, i)


def load_template_artifact(
    out_dir, *, verify_checksum: bool = True
) -> Dict[str, Optional[TemplateFeatures]]:
    """Load a template artifact into a `record_id -> TemplateFeatures | None` map.

    Convenience for small/medium corpora and the join in PR-C. For very large
    corpora prefer `iter_template_artifact` (streaming) or an indexed reader.

    Raises on a duplicate `record_id` — the artifact is keyed by `record_id`, so
    a duplicate is malformed and silently keeping the last row would lose template
    records and desync the count from the manifest (codex catch)."""
    out: Dict[str, Optional[TemplateFeatures]] = {}
    for rid, tf in iter_template_artifact(out_dir, verify_checksum=verify_checksum):
        if rid in out:
            raise ValueError(f"duplicate record_id {rid!r} in template artifact")
        out[rid] = tf
    return out
