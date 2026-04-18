"""Structural alphabet encoding for search and retrieval workflows."""

from __future__ import annotations

import os
import json
import hashlib
import shutil
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.feather as paf
import pyarrow.parquet as pq

try:
    import ferritin_connector

    _search = ferritin_connector.py_search
except ImportError:
    _search = None

from .io import batch_load_tolerant


SEARCH_DB_VERSION = 4
POSTINGS_BUCKET_COUNT = 64
COMPILED_LAYOUT_VERSION = 1

_BLOSUM62_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
_DEFAULT_SA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _require_search_backend():
    if _search is None:
        raise ImportError(
            "ferritin search encoding requires the native ferritin-connector "
            "extension. Install `ferritin` from PyPI or build "
            "`ferritin-connector` with `maturin develop --release`."
        )
    return _search


def _candidate_foldseek_3di_matrix_paths() -> List[Path]:
    candidates: List[Path] = []
    env_path = os.environ.get("FERRITIN_FOLDSEEK_3DI_MATRIX")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    module_path = Path(__file__).resolve()
    if len(module_path.parents) >= 5:
        repo_root = module_path.parents[4]
        candidates.append(repo_root.parent / "foldseek-src" / "data" / "mat3di.out")
        candidates.append(repo_root / "foldseek-src" / "data" / "mat3di.out")

    return candidates


def _normalize_k_values(k: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(k, int):
        values = [k]
    else:
        values = list(k)
    if not values:
        raise ValueError("k must contain at least one value")
    normalized = sorted(set(int(value) for value in values))
    bad = [value for value in normalized if value < 1]
    if bad:
        raise ValueError(f"k values must be >= 1, got {bad}")
    return normalized


def _posting_key(k: int, kmer: str) -> str:
    return f"{k}:{kmer}"


def _split_posting_key(key: str, default_k: int) -> tuple[int, str]:
    if ":" not in key:
        return default_k, key
    left, right = key.split(":", 1)
    try:
        return int(left), right
    except ValueError:
        return default_k, key


@dataclass
class SearchEntry:
    entry_index: int
    id: str
    source_path: str
    source_index: int
    residue_count: int
    valid_residue_count: int
    aa_sequence: str
    valid_aa_sequence: str
    alphabet: str
    valid_alphabet: str
    aa_kmer_count: int
    kmer_count: int


@dataclass
class SearchHit:
    id: str
    source_path: str
    source_index: int
    score: float
    prefilter_score: float
    alphabet_score: float
    aa_score: float
    shared_kmers: int
    shared_aa_kmers: int
    query_kmers: int
    target_kmers: int
    query_aa_kmers: int
    target_aa_kmers: int
    residue_count: int
    valid_residue_count: int
    diagonal_vote_score: Optional[float] = None
    diagonal_score: Optional[float] = None
    tm_score: Optional[float] = None
    rmsd: Optional[float] = None
    n_aligned: Optional[int] = None
    seq_identity: Optional[float] = None


@dataclass
class SearchDB:
    version: int
    k: int
    entries: Optional[List[SearchEntry]]
    postings: Optional[Dict[str, List[int]]]
    aa_postings: Optional[Dict[str, List[int]]]
    positional_postings: Optional[Dict[str, Dict[int, List[int]]]] = None
    aa_positional_postings: Optional[Dict[str, Dict[int, List[int]]]] = None
    k_values: List[int] = field(default_factory=list)
    posting_keys_include_k: bool = field(default=True, repr=False, compare=False)
    root_path: Optional[str] = field(default=None, repr=False, compare=False)
    n_entries: Optional[int] = field(default=None, repr=False, compare=False)
    structure_cache: "OrderedDict[str, Any]" = field(default_factory=OrderedDict, repr=False, compare=False)
    entry_cache: "OrderedDict[int, SearchEntry]" = field(default_factory=OrderedDict, repr=False, compare=False)
    posting_bucket_cache: "OrderedDict[tuple[str, int], Dict[str, List[int]]]" = field(default_factory=OrderedDict, repr=False, compare=False)
    positional_posting_bucket_cache: "OrderedDict[tuple[str, int], Dict[str, Dict[int, List[int]]]]" = field(default_factory=OrderedDict, repr=False, compare=False)
    cache_max_size: int = field(default=128, repr=False, compare=False)
    entry_cache_max_size: int = field(default=512, repr=False, compare=False)
    posting_cache_max_size: int = field(default=64, repr=False, compare=False)

    def __len__(self) -> int:
        if self.entries is not None:
            return len(self.entries)
        return int(self.n_entries or 0)

    def __post_init__(self) -> None:
        if not self.k_values:
            self.k_values = [self.k]


def _get_ptr(structure):
    if hasattr(structure, "get_py_ptr"):
        return structure.get_py_ptr()
    return structure


def encode_alphabet(structure) -> Dict[str, Any]:
    """Encode a protein structure into ferritin's 20-state structural alphabet.

    Returns a dict with residue-level outputs:
        states: uint8 state indices
        alphabet: string of structural alphabet letters
        valid_mask: boolean validity mask
        partners: nearest-neighbor residue indices
        features: Nx10 geometric feature matrix
        chain_ids, residue_names, residue_numbers, insertion_codes metadata

    Agent Notes:
        INVARIANT: Residue encoding itself is Rust-backed; downstream DB/query orchestration is separate.
        WATCH: Output is residue-level over amino-acid residues only, not atoms.
        PREFER: Use valid_mask to filter termini, chain breaks, and incomplete residues.
    """
    backend = _require_search_backend()
    result = backend.encode_alphabet(_get_ptr(structure))
    result["states"] = np.asarray(result["states"])
    result["valid_mask"] = np.asarray(result["valid_mask"], dtype=bool)
    result["partners"] = np.asarray(result["partners"])
    result["features"] = np.asarray(result["features"])
    result["residue_numbers"] = np.asarray(result["residue_numbers"])
    return result


def batch_encode_alphabet(
    structures: Sequence,
    *,
    n_threads: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Encode many structures into the structural alphabet in parallel.

    Returns a list of dicts with the same schema as `encode_alphabet()`.

    Agent Notes:
        INVARIANT: This batching goes through the Rust encoding kernel; it is not a Python loop wrapper.
        PREFER: Use this for corpus-scale encoding instead of Python loops.
        WATCH: Output order matches the input structure order.
    """
    backend = _require_search_backend()
    ptrs = [_get_ptr(s) for s in structures]
    results = backend.batch_encode_alphabet(ptrs, n_threads)
    normalized = []
    for result in results:
        result["states"] = np.asarray(result["states"])
        result["valid_mask"] = np.asarray(result["valid_mask"], dtype=bool)
        result["partners"] = np.asarray(result["partners"])
        result["features"] = np.asarray(result["features"])
        result["residue_numbers"] = np.asarray(result["residue_numbers"])
        normalized.append(result)
    return normalized


def _valid_alphabet(result: Dict[str, Any]) -> str:
    valid_mask = np.asarray(result["valid_mask"], dtype=bool)
    alphabet = result["alphabet"]
    return "".join(ch for ch, keep in zip(alphabet, valid_mask) if keep)


def _aa_sequence(result: Dict[str, Any]) -> str:
    return "".join(_AA3_TO_1.get(str(name).upper(), "X") for name in result["residue_names"])


def _valid_aa_sequence(result: Dict[str, Any]) -> str:
    valid_mask = np.asarray(result["valid_mask"], dtype=bool)
    aa_sequence = _aa_sequence(result)
    return "".join(ch for ch, keep in zip(aa_sequence, valid_mask) if keep)


def _kmer_set(sequence: str, k: int) -> set[str]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not sequence:
        return set()
    if len(sequence) < k:
        return {sequence}
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def _entry_id(path: Union[str, Path], structure) -> str:
    identifier = getattr(structure, "identifier", None)
    if identifier:
        return str(identifier)
    return Path(path).stem


def _build_entry(
    path: Union[str, Path],
    source_index: int,
    structure,
    encoded: Dict[str, Any],
    k: Union[int, Sequence[int]],
    *,
    entry_index: int,
) -> SearchEntry:
    aa_sequence = _aa_sequence(encoded)
    valid_aa_sequence = _valid_aa_sequence(encoded)
    valid_alphabet = _valid_alphabet(encoded)
    k_values = _normalize_k_values(k)
    kmers = set().union(*(_kmer_set(valid_alphabet, value) for value in k_values))
    aa_kmers = set().union(*(_kmer_set(valid_aa_sequence, value) for value in k_values))
    return SearchEntry(
        entry_index=entry_index,
        id=_entry_id(path, structure),
        source_path=str(path),
        source_index=source_index,
        residue_count=len(encoded["alphabet"]),
        valid_residue_count=int(np.asarray(encoded["valid_mask"], dtype=bool).sum()),
        aa_sequence=aa_sequence,
        valid_aa_sequence=valid_aa_sequence,
        alphabet=encoded["alphabet"],
        valid_alphabet=valid_alphabet,
        aa_kmer_count=len(aa_kmers),
        kmer_count=len(kmers),
    )


def _build_postings(entries: Iterable[SearchEntry], k: Union[int, Sequence[int]], *, attr: str) -> Dict[str, List[int]]:
    k_values = _normalize_k_values(k)
    postings: Dict[str, List[int]] = {}
    for entry in entries:
        sequence = getattr(entry, attr)
        for value in k_values:
            for kmer in _kmer_set(sequence, value):
                postings.setdefault(_posting_key(value, kmer), []).append(entry.entry_index)
    return postings


def _build_positional_postings(entries: Iterable[SearchEntry], k: Union[int, Sequence[int]], *, attr: str) -> Dict[str, Dict[int, List[int]]]:
    k_values = _normalize_k_values(k)
    postings: Dict[str, Dict[int, List[int]]] = {}
    for entry in entries:
        sequence = getattr(entry, attr)
        for value in k_values:
            positions_by_key = _query_posting_key_positions_for_k(sequence, value)
            for kmer, positions in positions_by_key.items():
                postings.setdefault(_posting_key(value, kmer), {})[entry.entry_index] = positions
    return postings


def _query_posting_key_positions_for_k(sequence: str, k: int) -> Dict[str, List[int]]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    positions: Dict[str, List[int]] = {}
    if not sequence:
        return positions
    if len(sequence) < k:
        positions.setdefault(sequence, []).append(0)
        return positions
    for i in range(len(sequence) - k + 1):
        positions.setdefault(sequence[i:i + k], []).append(i)
    return positions


def _query_posting_keys(sequence: str, db: SearchDB) -> set[str]:
    if db.posting_keys_include_k:
        keys: set[str] = set()
        for value in db.k_values:
            keys.update(_posting_key(value, kmer) for kmer in _kmer_set(sequence, value))
        return keys
    return _kmer_set(sequence, db.k)


def _query_posting_key_positions(sequence: str, db: SearchDB) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = {}
    if db.posting_keys_include_k:
        for value in db.k_values:
            for kmer, kmer_positions in _query_posting_key_positions_for_k(sequence, value).items():
                positions[_posting_key(value, kmer)] = kmer_positions
    else:
        positions.update(_query_posting_key_positions_for_k(sequence, db.k))
    return positions


def _target_kmer_positions(sequence: str, posting_key: str, db: SearchDB) -> List[int]:
    k, kmer = _split_posting_key(posting_key, db.k)
    if not sequence:
        return []
    if len(sequence) < k:
        return [0] if sequence == kmer else []
    positions = []
    start = sequence.find(kmer)
    while start != -1:
        positions.append(start)
        start = sequence.find(kmer, start + 1)
    return positions


def _posting_bucket(kmer: str, bucket_count: int = POSTINGS_BUCKET_COUNT) -> int:
    digest = hashlib.blake2b(kmer.encode("ascii"), digest_size=2).digest()
    return int.from_bytes(digest, byteorder="big") % bucket_count


def _empty_entries_table() -> pa.Table:
    return pa.table({
        "id": pa.array([], type=pa.string()),
        "entry_index": pa.array([], type=pa.int64()),
        "source_path": pa.array([], type=pa.string()),
        "source_index": pa.array([], type=pa.int64()),
        "residue_count": pa.array([], type=pa.int64()),
        "valid_residue_count": pa.array([], type=pa.int64()),
        "aa_sequence": pa.array([], type=pa.string()),
        "valid_aa_sequence": pa.array([], type=pa.string()),
        "alphabet": pa.array([], type=pa.string()),
        "valid_alphabet": pa.array([], type=pa.string()),
        "aa_kmer_count": pa.array([], type=pa.int64()),
        "kmer_count": pa.array([], type=pa.int64()),
    })


def _empty_postings_table() -> pa.Table:
    return pa.table({
        "kmer": pa.array([], type=pa.string()),
        "entry_index": pa.array([], type=pa.int64()),
    })


def _empty_positional_postings_table() -> pa.Table:
    return pa.table({
        "kmer": pa.array([], type=pa.string()),
        "entry_index": pa.array([], type=pa.int64()),
        "positions": pa.array([], type=pa.list_(pa.int64())),
    })


def _compiled_root(root: Path) -> Path:
    return root / "compiled"


def _compiled_manifest_path(root: Path) -> Path:
    return _compiled_root(root) / "manifest.json"


def _compiled_entries_path(root: Path) -> Path:
    return _compiled_root(root) / "entries.arrow"


def _compiled_postings_path(root: Path, *, kind: str) -> Path:
    return _compiled_root(root) / f"postings_{kind}.arrow"


def _compiled_positional_postings_path(root: Path, *, kind: str) -> Path:
    return _compiled_root(root) / f"positional_postings_{kind}.arrow"


def _warn_missing_compiled_search_layout(root: Path, version: int) -> None:
    warnings.warn(
        "Compiled search layout not found for "
        f"{root}. Loading lazy Parquet-backed search DB version {version}; "
        "query-time serving will stay more Python/PyArrow-heavy until you run "
        "`compile_search_db(path)` or resave with the default "
        "`write_compiled=True`.",
        UserWarning,
        stacklevel=3,
    )


def _load_compiled_search_db(root: Path) -> SearchDB:
    payload = json.loads(_compiled_manifest_path(root).read_text(encoding="utf-8"))
    if int(payload["layout_version"]) != COMPILED_LAYOUT_VERSION:
        raise ValueError(
            f"Unsupported compiled search layout version {payload['layout_version']}. "
            f"Expected {COMPILED_LAYOUT_VERSION}."
        )
    entries = [_entry_row_to_obj(row, idx) for idx, row in enumerate(paf.read_table(_compiled_entries_path(root)).to_pylist())]
    postings = _load_compiled_postings(_compiled_postings_path(root, kind="sa"))
    aa_postings = _load_compiled_postings(_compiled_postings_path(root, kind="aa"))
    positional_postings = _load_compiled_positional_postings(_compiled_positional_postings_path(root, kind="sa"))
    aa_positional_postings = _load_compiled_positional_postings(_compiled_positional_postings_path(root, kind="aa"))
    return SearchDB(
        version=SEARCH_DB_VERSION,
        k=int(payload["k"]),
        entries=entries,
        postings=postings,
        aa_postings=aa_postings,
        positional_postings=positional_postings,
        aa_positional_postings=aa_positional_postings,
        k_values=[int(value) for value in payload.get("k_values", [int(payload["k"])])],
        posting_keys_include_k=bool(payload.get("posting_keys_include_k", False)),
        root_path=str(root),
        n_entries=len(entries),
    )


def _write_compiled_search_layout(db: SearchDB, root: Path) -> None:
    compiled_root = _compiled_root(root)
    compiled_root.mkdir(parents=True, exist_ok=True)
    compiled_manifest = {
        "layout_version": COMPILED_LAYOUT_VERSION,
        "k": db.k,
        "k_values": list(db.k_values),
        "posting_keys_include_k": db.posting_keys_include_k,
        "n_entries": len(db),
        "entries_file": "entries.arrow",
        "sa_postings_file": "postings_sa.arrow",
        "aa_postings_file": "postings_aa.arrow",
        "sa_positional_postings_file": "positional_postings_sa.arrow",
        "aa_positional_postings_file": "positional_postings_aa.arrow",
    }
    _compiled_manifest_path(root).write_text(json.dumps(compiled_manifest, indent=2), encoding="utf-8")
    entry_rows = [asdict(entry) for entry in (db.entries or [])]
    paf.write_feather(
        pa.Table.from_pylist(entry_rows) if entry_rows else _empty_entries_table(),
        _compiled_entries_path(root),
    )
    paf.write_feather(_postings_grouped_table(db.postings or {}), _compiled_postings_path(root, kind="sa"))
    paf.write_feather(_postings_grouped_table(db.aa_postings or {}), _compiled_postings_path(root, kind="aa"))
    paf.write_feather(
        _positional_postings_table(db.positional_postings or {}),
        _compiled_positional_postings_path(root, kind="sa"),
    )
    paf.write_feather(
        _positional_postings_table(db.aa_positional_postings or {}),
        _compiled_positional_postings_path(root, kind="aa"),
    )


def _bucket_file(root: Path, *, kind: str, bucket: int) -> Path:
    return root / "postings" / f"kind={kind}" / f"bucket={bucket:02d}.parquet"


def _positional_bucket_file(root: Path, *, kind: str, bucket: int) -> Path:
    return root / "positional_postings" / f"kind={kind}" / f"bucket={bucket:02d}.parquet"


def _write_bucketed_postings(root: Path, *, kind: str, postings: Dict[str, List[int]]) -> None:
    if not postings:
        return
    rows_by_bucket: Dict[int, List[Dict[str, Any]]] = {}
    for kmer, entry_indices in postings.items():
        bucket = _posting_bucket(kmer)
        bucket_rows = rows_by_bucket.setdefault(bucket, [])
        for entry_index in entry_indices:
            bucket_rows.append({"kmer": kmer, "entry_index": int(entry_index)})

    for bucket, rows in rows_by_bucket.items():
        bucket_path = _bucket_file(root, kind=kind, bucket=bucket)
        bucket_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows) if rows else _empty_postings_table()
        pq.write_table(table, bucket_path, compression="zstd")


def _write_bucketed_positional_postings(root: Path, *, kind: str, postings: Dict[str, Dict[int, List[int]]]) -> None:
    if not postings:
        return
    rows_by_bucket: Dict[int, List[Dict[str, Any]]] = {}
    for kmer, entries in postings.items():
        bucket = _posting_bucket(kmer)
        bucket_rows = rows_by_bucket.setdefault(bucket, [])
        for entry_index, positions in entries.items():
            bucket_rows.append({
                "kmer": kmer,
                "entry_index": int(entry_index),
                "positions": [int(position) for position in positions],
            })

    for bucket, rows in rows_by_bucket.items():
        bucket_path = _positional_bucket_file(root, kind=kind, bucket=bucket)
        bucket_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows) if rows else _empty_positional_postings_table()
        pq.write_table(table, bucket_path, compression="zstd")


def _group_postings_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["kmer"]), []).append(int(row["entry_index"]))
    return grouped


def _group_positional_postings_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[int, List[int]]]:
    grouped: Dict[str, Dict[int, List[int]]] = {}
    for row in rows:
        grouped.setdefault(str(row["kmer"]), {})[int(row["entry_index"])] = [
            int(position)
            for position in row["positions"]
        ]
    return grouped


def _postings_grouped_table(postings: Dict[str, List[int]]) -> pa.Table:
    rows = [
        {"kmer": kmer, "entry_indices": [int(idx) for idx in entry_indices]}
        for kmer, entry_indices in sorted(postings.items())
    ]
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table({
        "kmer": pa.array([], type=pa.string()),
        "entry_indices": pa.array([], type=pa.list_(pa.int64())),
    })


def _positional_postings_table(postings: Dict[str, Dict[int, List[int]]]) -> pa.Table:
    rows = [
        {
            "kmer": kmer,
            "entry_index": int(entry_index),
            "positions": [int(position) for position in positions],
        }
        for kmer, entries in sorted(postings.items())
        for entry_index, positions in sorted(entries.items())
    ]
    if rows:
        return pa.Table.from_pylist(rows)
    return _empty_positional_postings_table()


def _jaccard(shared: int, left: int, right: int) -> float:
    union = left + right - shared
    if union == 0:
        return 0.0
    return shared / union


def _parse_substitution_matrix(path: Path) -> tuple[str, Dict[str, Dict[str, int]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [line for line in lines if line and not line.startswith("#")]
    if not rows:
        raise ValueError(f"substitution matrix file is empty: {path}")
    if len(rows) < 2:
        raise ValueError(f"substitution matrix file has no score rows: {path}")
    alphabet = rows[0].split()
    if not alphabet:
        raise ValueError(f"substitution matrix header is empty: {path}")
    matrix: Dict[str, Dict[str, int]] = {}
    for row in rows[1:]:
        parts = row.split()
        if len(parts) != len(alphabet) + 1:
            raise ValueError(
                f"substitution matrix row has {len(parts) - 1} scores, "
                f"expected {len(alphabet)}: {path}"
            )
        matrix[parts[0]] = {alphabet[i]: int(parts[i + 1]) for i in range(len(alphabet))}
    return "".join(alphabet), matrix


_AA_MATRIX = {
    aa: {_BLOSUM62_ALPHABET[j]: score for j, score in enumerate(row)}
    for aa, row in zip(_BLOSUM62_ALPHABET, _BLOSUM62_ROWS)
}


def _load_sa_matrix() -> tuple[str, Dict[str, Dict[str, int]], str]:
    env_path = os.environ.get("FERRITIN_FOLDSEEK_3DI_MATRIX")
    env_override = Path(env_path).expanduser() if env_path else None
    for path in _candidate_foldseek_3di_matrix_paths():
        try:
            alphabet, matrix = _parse_substitution_matrix(path)
            return alphabet, matrix, str(path)
        except OSError:
            continue
        except ValueError:
            if env_override is not None and path == env_override:
                raise
            continue

    matrix = {
        aa: {bb: (6 if aa == bb else -2) for bb in _DEFAULT_SA_ALPHABET}
        for aa in _DEFAULT_SA_ALPHABET
    }
    return _DEFAULT_SA_ALPHABET, matrix, "fallback"


_SA_ALPHABET, _SA_MATRIX, _SA_MATRIX_SOURCE = _load_sa_matrix()


def _matrix_score(matrix: Dict[str, Dict[str, int]], left: str, right: str) -> int:
    return matrix.get(left, matrix.get("X", {})).get(right, matrix.get("X", {}).get("X", 0))


def _ungapped_local_diagonal_score(
    query_sa: str,
    query_aa: str,
    target_sa: str,
    target_aa: str,
    *,
    aa_scale: float = 1.0,
    sa_scale: float = 1.0,
) -> float:
    q_len = min(len(query_sa), len(query_aa))
    t_len = min(len(target_sa), len(target_aa))
    if q_len == 0 or t_len == 0:
        return 0.0
    best = 0.0
    for diagonal in range(-(t_len - 1), q_len):
        q_start = max(diagonal, 0)
        t_start = max(-diagonal, 0)
        length = min(q_len - q_start, t_len - t_start)
        score = 0.0
        for offset in range(length):
            q_idx = q_start + offset
            t_idx = t_start + offset
            curr = (
                sa_scale * _matrix_score(_SA_MATRIX, query_sa[q_idx], target_sa[t_idx])
                + aa_scale * _matrix_score(_AA_MATRIX, query_aa[q_idx], target_aa[t_idx])
            )
            score = max(0.0, score + curr)
            best = max(best, score)
    return best / max(q_len, t_len)


def _idf_weight(df: int, n_entries: int) -> float:
    return float(np.log1p((n_entries + 1) / (df + 1)))


def _query_kmer_weights(
    db: SearchDB,
    *,
    kind: str,
    kmers: set[str],
) -> Dict[str, float]:
    if not kmers:
        return {}
    n_entries = max(len(db), 1)
    if db.entries is not None:
        source = db.postings if kind == "sa" else db.aa_postings
        return {
            kmer: _idf_weight(len((source or {}).get(kmer, [])), n_entries)
            for kmer in kmers
        }

    if db.root_path is None:
        return {kmer: 1.0 for kmer in kmers}

    root = Path(db.root_path)
    if db.version == 3:
        dataset = pads.dataset(root / "postings.parquet", format="parquet")
        table = dataset.to_table(
            columns=["kmer", "entry_index"],
            filter=(pads.field("kind") == kind) & pads.field("kmer").isin(sorted(kmers)),
        )
        dfs: Dict[str, int] = {}
        for row in table.to_pylist():
            kmer = str(row["kmer"])
            dfs[kmer] = dfs.get(kmer, 0) + 1
        return {kmer: _idf_weight(dfs.get(kmer, 0), n_entries) for kmer in kmers}

    grouped_kmers: Dict[int, List[str]] = {}
    for kmer in kmers:
        grouped_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)
    weights: Dict[str, float] = {}
    for bucket, bucket_kmers in grouped_kmers.items():
        bucket_postings = _get_cached_posting_bucket(db, root, kind=kind, bucket=bucket)
        for kmer in bucket_kmers:
            weights[kmer] = _idf_weight(len(bucket_postings.get(kmer, [])), n_entries)
    return weights


def _weighted_query_coverage(shared_weight: float, query_weights: Dict[str, float]) -> float:
    total = float(sum(query_weights.values()))
    if total <= 0.0:
        return 0.0
    return shared_weight / total


def _diagonal_vote_scores(
    db: SearchDB,
    entry_lookup: Dict[int, SearchEntry],
    query_positions: Dict[str, List[int]],
    query_aa_positions: Dict[str, List[int]],
    candidate_indices: Sequence[int],
) -> Dict[int, float]:
    candidate_set = set(candidate_indices)
    votes: Dict[int, Dict[int, float]] = {idx: {} for idx in candidate_indices}

    def add_positional_votes(
        positional_postings: Dict[str, Dict[int, List[int]]],
        positions_by_key: Dict[str, List[int]],
        weight: float,
    ) -> None:
        for key, query_positions_for_key in positions_by_key.items():
            for idx, target_positions in positional_postings.get(key, {}).items():
                if idx not in candidate_set:
                    continue
                if not target_positions:
                    continue
                diagonal_votes = votes[idx]
                for q_pos in query_positions_for_key:
                    for t_pos in target_positions:
                        diagonal = q_pos - t_pos
                        diagonal_votes[diagonal] = diagonal_votes.get(diagonal, 0.0) + weight

    def add_scan_votes(
        postings: Dict[str, List[int]],
        positions_by_key: Dict[str, List[int]],
        attr: str,
        weight: float,
    ) -> None:
        for key, query_positions_for_key in positions_by_key.items():
            for idx in postings.get(key, []):
                if idx not in candidate_set:
                    continue
                entry = entry_lookup.get(idx)
                if entry is None:
                    continue
                target_positions = _target_kmer_positions(getattr(entry, attr), key, db)
                if not target_positions:
                    continue
                diagonal_votes = votes[idx]
                for q_pos in query_positions_for_key:
                    for t_pos in target_positions:
                        diagonal = q_pos - t_pos
                        diagonal_votes[diagonal] = diagonal_votes.get(diagonal, 0.0) + weight

    if db.positional_postings is not None or db.aa_positional_postings is not None:
        add_positional_votes(db.positional_postings or {}, query_positions, 1.0)
        add_positional_votes(db.aa_positional_postings or {}, query_aa_positions, 0.5)
    elif db.root_path is not None and db.entries is None and db.version >= 4:
        root = Path(db.root_path)
        add_positional_votes(_fetch_positional_postings_bucketed_cached(db, root, kind="sa", kmers=set(query_positions)), query_positions, 1.0)
        add_positional_votes(_fetch_positional_postings_bucketed_cached(db, root, kind="aa", kmers=set(query_aa_positions)), query_aa_positions, 0.5)
    elif db.entries is not None:
        add_scan_votes(db.postings or {}, query_positions, "valid_alphabet", 1.0)
        add_scan_votes(db.aa_postings or {}, query_aa_positions, "valid_aa_sequence", 0.5)
    else:
        return {}

    return {
        idx: max(diagonal_counts.values()) if diagonal_counts else 0.0
        for idx, diagonal_counts in votes.items()
    }


def _apply_diagonal_prefilter(
    candidate_indices: Sequence[int],
    diagonal_vote_scores: Dict[int, float],
    shared_counts: Dict[int, int],
    shared_aa_counts: Dict[int, int],
    *,
    enabled: bool,
    min_support: float,
    keep_count: int,
    min_backfill: int,
) -> List[int]:
    if not enabled or not diagonal_vote_scores:
        return list(candidate_indices)

    supported = [
        idx
        for idx in candidate_indices
        if diagonal_vote_scores.get(idx, 0.0) >= min_support
    ]
    if not supported:
        return list(candidate_indices)

    def rank_key(idx: int) -> tuple[float, float, int]:
        lexical_support = float(shared_counts.get(idx, 0)) + 0.5 * float(shared_aa_counts.get(idx, 0))
        return (-diagonal_vote_scores.get(idx, 0.0), -lexical_support, idx)

    ranked = sorted(supported, key=rank_key)
    if len(ranked) >= min_backfill:
        return ranked[:keep_count]

    supported_set = set(ranked)
    backfill = [
        idx
        for idx in candidate_indices
        if idx not in supported_set
    ]
    backfill.sort(
        key=lambda idx: (
            -float(shared_counts.get(idx, 0)) - 0.5 * float(shared_aa_counts.get(idx, 0)),
            idx,
        )
    )
    return (ranked + backfill)[:keep_count]


def _load_structure(path: str):
    from .io import load

    return load(path)


def _get_cached_structure(db: SearchDB, path: str):
    cached = db.structure_cache.get(path)
    if cached is not None:
        db.structure_cache.move_to_end(path)
        return cached
    structure = _load_structure(path)
    db.structure_cache[path] = structure
    db.structure_cache.move_to_end(path)
    while len(db.structure_cache) > db.cache_max_size:
        db.structure_cache.popitem(last=False)
    return structure


def _get_cached_posting_bucket(db: SearchDB, root: Path, *, kind: str, bucket: int) -> Dict[str, List[int]]:
    key = (kind, bucket)
    cached = db.posting_bucket_cache.get(key)
    if cached is not None:
        db.posting_bucket_cache.move_to_end(key)
        return cached

    bucket_path = _bucket_file(root, kind=kind, bucket=bucket)
    postings: Dict[str, List[int]] = {}
    if bucket_path.exists():
        table = pq.read_table(bucket_path)
        for row in table.to_pylist():
            postings.setdefault(str(row["kmer"]), []).append(int(row["entry_index"]))

    db.posting_bucket_cache[key] = postings
    db.posting_bucket_cache.move_to_end(key)
    while len(db.posting_bucket_cache) > db.posting_cache_max_size:
        db.posting_bucket_cache.popitem(last=False)
    return postings


def _get_cached_positional_posting_bucket(db: SearchDB, root: Path, *, kind: str, bucket: int) -> Dict[str, Dict[int, List[int]]]:
    key = (kind, bucket)
    cached = db.positional_posting_bucket_cache.get(key)
    if cached is not None:
        db.positional_posting_bucket_cache.move_to_end(key)
        return cached

    bucket_path = _positional_bucket_file(root, kind=kind, bucket=bucket)
    postings: Dict[str, Dict[int, List[int]]] = {}
    if bucket_path.exists():
        table = pq.read_table(bucket_path)
        postings = _group_positional_postings_rows(table.to_pylist())

    db.positional_posting_bucket_cache[key] = postings
    db.positional_posting_bucket_cache.move_to_end(key)
    while len(db.positional_posting_bucket_cache) > db.posting_cache_max_size:
        db.positional_posting_bucket_cache.popitem(last=False)
    return postings


def _entry_row_to_obj(row: Dict[str, Any], fallback_index: Optional[int] = None) -> SearchEntry:
    data = dict(row)
    if "entry_index" not in data:
        if fallback_index is None:
            raise ValueError("entry_index missing from search DB row")
        data["entry_index"] = fallback_index
    return SearchEntry(**data)


def _load_entries_eager(root: Path, entries_file: str) -> List[SearchEntry]:
    rows = pq.read_table(root / entries_file).to_pylist()
    return [_entry_row_to_obj(row, idx) for idx, row in enumerate(rows)]


def _load_postings_eager(root: Path, postings_file: str) -> tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    postings_table = pq.read_table(root / postings_file)
    postings: Dict[str, List[int]] = {}
    aa_postings: Dict[str, List[int]] = {}
    for row in postings_table.to_pylist():
        target = postings if row["kind"] == "sa" else aa_postings
        target.setdefault(str(row["kmer"]), []).append(int(row["entry_index"]))
    return postings, aa_postings


def _load_compiled_postings(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    rows = paf.read_table(path).to_pylist()
    return {str(row["kmer"]): [int(idx) for idx in row["entry_indices"]] for row in rows}


def _load_compiled_positional_postings(path: Path) -> Optional[Dict[str, Dict[int, List[int]]]]:
    if not path.exists():
        return None
    return _group_positional_postings_rows(paf.read_table(path).to_pylist())


def _materialize_bucketed_postings(
    db: Optional[SearchDB],
    root: Path,
    *,
    kind: str,
) -> Dict[str, List[int]]:
    postings: Dict[str, List[int]] = {}
    for bucket in range(POSTINGS_BUCKET_COUNT):
        bucket_rows: Dict[str, List[int]]
        if db is not None and (kind, bucket) in db.posting_bucket_cache:
            bucket_rows = db.posting_bucket_cache[(kind, bucket)]
        else:
            bucket_rows = _get_cached_posting_bucket(db, root, kind=kind, bucket=bucket) if db is not None else {}
            if db is None:
                bucket_path = _bucket_file(root, kind=kind, bucket=bucket)
                if bucket_path.exists():
                    bucket_rows = _group_postings_rows(pq.read_table(bucket_path).to_pylist())
        for kmer, entry_indices in bucket_rows.items():
            postings.setdefault(kmer, []).extend(int(idx) for idx in entry_indices)
    return postings


def _materialize_bucketed_positional_postings(
    db: Optional[SearchDB],
    root: Path,
    *,
    kind: str,
) -> Dict[str, Dict[int, List[int]]]:
    postings: Dict[str, Dict[int, List[int]]] = {}
    for bucket in range(POSTINGS_BUCKET_COUNT):
        bucket_rows: Dict[str, Dict[int, List[int]]]
        if db is not None and (kind, bucket) in db.positional_posting_bucket_cache:
            bucket_rows = db.positional_posting_bucket_cache[(kind, bucket)]
        else:
            bucket_rows = _get_cached_positional_posting_bucket(db, root, kind=kind, bucket=bucket) if db is not None else {}
            if db is None:
                bucket_path = _positional_bucket_file(root, kind=kind, bucket=bucket)
                if bucket_path.exists():
                    bucket_rows = _group_positional_postings_rows(pq.read_table(bucket_path).to_pylist())
        for kmer, entries in bucket_rows.items():
            postings.setdefault(kmer, {}).update(entries)
    return postings


def _fetch_positional_postings_bucketed_cached(
    db: SearchDB,
    root: Path,
    *,
    kind: str,
    kmers: set[str],
) -> Dict[str, Dict[int, List[int]]]:
    if not kmers:
        return {}
    grouped_kmers: Dict[int, List[str]] = {}
    for kmer in kmers:
        grouped_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)

    postings: Dict[str, Dict[int, List[int]]] = {}
    for bucket, bucket_kmers in grouped_kmers.items():
        bucket_postings = _get_cached_positional_posting_bucket(db, root, kind=kind, bucket=bucket)
        for kmer in bucket_kmers:
            if kmer in bucket_postings:
                postings[kmer] = bucket_postings[kmer]
    return postings


def _fetch_posting_counts_v3(root: Path, *, kind: str, kmers: set[str]) -> Dict[int, int]:
    if not kmers:
        return {}
    dataset = pads.dataset(root / "postings.parquet", format="parquet")
    table = dataset.to_table(
        columns=["entry_index"],
        filter=(pads.field("kind") == kind) & pads.field("kmer").isin(sorted(kmers)),
    )
    counts: Dict[int, int] = {}
    for entry_index in table.column("entry_index").to_pylist():
        idx = int(entry_index)
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def _fetch_posting_counts_bucketed(root: Path, *, kind: str, kmers: set[str]) -> Dict[int, int]:
    if not kmers:
        return {}
    grouped_kmers: Dict[int, List[str]] = {}
    for kmer in kmers:
        grouped_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)

    counts: Dict[int, int] = {}
    for bucket, bucket_kmers in grouped_kmers.items():
        bucket_path = _bucket_file(root, kind=kind, bucket=bucket)
        if not bucket_path.exists():
            continue
        table = pq.read_table(
            bucket_path,
            columns=["entry_index"],
            filters=[("kmer", "in", sorted(bucket_kmers))],
        )
        for entry_index in table.column("entry_index").to_pylist():
            idx = int(entry_index)
            counts[idx] = counts.get(idx, 0) + 1
    return counts


def _fetch_posting_counts_bucketed_cached(
    db: SearchDB,
    root: Path,
    *,
    kind: str,
    kmers: set[str],
) -> Dict[int, int]:
    if not kmers:
        return {}
    grouped_kmers: Dict[int, List[str]] = {}
    for kmer in kmers:
        grouped_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)

    counts: Dict[int, int] = {}
    for bucket, bucket_kmers in grouped_kmers.items():
        bucket_postings = _get_cached_posting_bucket(db, root, kind=kind, bucket=bucket)
        for kmer in bucket_kmers:
            for entry_index in bucket_postings.get(kmer, []):
                counts[entry_index] = counts.get(entry_index, 0) + 1
    return counts


def _fetch_entries_by_index(
    root: Path,
    indices: Sequence[int],
    *,
    db: Optional[SearchDB] = None,
) -> Dict[int, SearchEntry]:
    if not indices:
        return {}
    ordered_indices = [int(i) for i in indices]
    entries_by_index: Dict[int, SearchEntry] = {}
    missing_indices = ordered_indices

    if db is not None:
        missing_indices = []
        for idx in ordered_indices:
            cached = db.entry_cache.get(idx)
            if cached is None:
                missing_indices.append(idx)
                continue
            db.entry_cache.move_to_end(idx)
            entries_by_index[idx] = cached

    if missing_indices:
        table = pq.read_table(
            root / "entries.parquet",
            filters=[("entry_index", "in", sorted(set(missing_indices)))],
        )
        fetched_entries = [_entry_row_to_obj(row) for row in table.to_pylist()]
        for entry in fetched_entries:
            entries_by_index[entry.entry_index] = entry
            if db is not None:
                db.entry_cache[entry.entry_index] = entry
                db.entry_cache.move_to_end(entry.entry_index)
        if db is not None:
            while len(db.entry_cache) > db.entry_cache_max_size:
                db.entry_cache.popitem(last=False)

    return {
        idx: entries_by_index[idx]
        for idx in ordered_indices
        if idx in entries_by_index
    }


def _entries_by_index(entries: Sequence[SearchEntry]) -> Dict[int, SearchEntry]:
    return {entry.entry_index: entry for entry in entries}


def build_search_db(
    paths: Sequence[Union[str, Path]],
    out: Optional[Union[str, Path]] = None,
    *,
    k: Union[int, Sequence[int]] = 6,
    n_threads: Optional[int] = None,
) -> SearchDB:
    """Build a structural-alphabet search database from structure files.

    Files that fail to load are skipped using ferritin's tolerant batch loader.

    Agent Notes:
        INVARIANT: Residue encoding is Rust-backed; DB materialization and postings persistence are Python/PyArrow today.
        PREFER: Build from many paths at once so load + encoding stay batched.
        PREFER: If you pass out=, the persisted DB also writes the eager compiled serving layout by default.
        WATCH: Only successfully loaded inputs are indexed; source_index preserves the original path position.
    """
    k_values = _normalize_k_values(k)

    loaded = batch_load_tolerant(paths, n_threads=n_threads)
    if loaded:
        source_indices, structures = zip(*loaded)
        encoded = batch_encode_alphabet(structures, n_threads=n_threads)
        entries = [
            _build_entry(
                paths[source_index],
                source_index,
                structure,
                result,
                k,
                entry_index=entry_index,
            )
            for entry_index, (source_index, structure, result) in enumerate(zip(source_indices, structures, encoded))
        ]
    else:
        entries = []

    db = SearchDB(
        version=SEARCH_DB_VERSION,
        k=k_values[-1],
        entries=entries,
        postings=_build_postings(entries, k, attr="valid_alphabet"),
        aa_postings=_build_postings(entries, k, attr="valid_aa_sequence"),
        positional_postings=_build_positional_postings(entries, k, attr="valid_alphabet"),
        aa_positional_postings=_build_positional_postings(entries, k, attr="valid_aa_sequence"),
        k_values=k_values,
        posting_keys_include_k=True,
    )
    if out is not None:
        save_search_db(db, out)
    return db


def compile_search_db(
    db: Union[SearchDB, str, Path],
    out: Optional[Union[str, Path]] = None,
) -> SearchDB:
    """Compile a Parquet-backed search DB into an eager query layout.

    Agent Notes:
        PREFER: Compile once when you need repeated low-latency queries against the same corpus.
        COST: Compilation materializes postings into an eager serving layout in addition to the Parquet corpus files.
    """
    if isinstance(db, (str, Path)):
        db = load_search_db(db, prefer_compiled=False)
    root = Path(out) if out is not None else (Path(db.root_path) if db.root_path else None)
    if root is None:
        raise ValueError("compile_search_db() requires a persisted DB path or out= destination")
    if not root.exists():
        raise ValueError(f"Search DB path does not exist: {root}")

    if db.entries is None:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        version = int(payload["version"])
        positional_postings = None
        aa_positional_postings = None
        if version == 2:
            entries = _load_entries_eager(root, payload["entries_file"])
            postings, aa_postings = _load_postings_eager(root, payload["postings_file"])
        elif version == 3:
            entries = _load_entries_eager(root, payload["entries_file"])
            postings = _group_postings_rows(
                pads.dataset(root / "postings.parquet", format="parquet").to_table(
                    filter=pads.field("kind") == "sa"
                ).to_pylist()
            )
            aa_postings = _group_postings_rows(
                pads.dataset(root / "postings.parquet", format="parquet").to_table(
                    filter=pads.field("kind") == "aa"
                ).to_pylist()
            )
        else:
            entries = _load_entries_eager(root, payload["entries_file"])
            postings = _materialize_bucketed_postings(db if isinstance(db, SearchDB) else None, root, kind="sa")
            aa_postings = _materialize_bucketed_postings(db if isinstance(db, SearchDB) else None, root, kind="aa")
            positional_postings = _materialize_bucketed_positional_postings(db if isinstance(db, SearchDB) else None, root, kind="sa")
            aa_positional_postings = _materialize_bucketed_positional_postings(db if isinstance(db, SearchDB) else None, root, kind="aa")
        db = SearchDB(
            version=SEARCH_DB_VERSION,
            k=db.k,
            entries=entries,
            postings=postings,
            aa_postings=aa_postings,
            positional_postings=positional_postings,
            aa_positional_postings=aa_positional_postings,
            k_values=list(db.k_values),
            posting_keys_include_k=db.posting_keys_include_k,
            root_path=str(root),
            n_entries=len(entries),
        )

    _write_compiled_search_layout(db, root)
    return load_search_db(root)


def save_search_db(
    db: SearchDB,
    path: Union[str, Path],
    *,
    write_compiled: bool = True,
) -> None:
    """Persist a search database as a versioned manifest + Parquet directory.

    Agent Notes:
        PREFER: Keep write_compiled=True for the common "build once, query many times" path.
        COST: The compiled layout duplicates the serving index on disk to reduce Python/PyArrow work at query time.
        WATCH: Set write_compiled=False only when you explicitly want Parquet-only storage.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for generated_dir in ("postings", "positional_postings", "compiled"):
        generated_path = root / generated_dir
        if generated_path.exists():
            shutil.rmtree(generated_path)
    db.root_path = str(root)
    db.n_entries = len(db)

    payload = {
        "version": db.version,
        "k": db.k,
        "k_values": list(db.k_values),
        "posting_keys_include_k": db.posting_keys_include_k,
        "n_entries": len(db),
        "entries_file": "entries.parquet",
        "postings_dir": "postings",
        "positional_postings_dir": "positional_postings",
        "postings_bucket_count": POSTINGS_BUCKET_COUNT,
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    entry_rows = [asdict(entry) for entry in db.entries]
    entries_table = pa.Table.from_pylist(entry_rows) if entry_rows else _empty_entries_table()
    pq.write_table(entries_table, root / "entries.parquet", compression="zstd")
    _write_bucketed_postings(root, kind="sa", postings=db.postings or {})
    _write_bucketed_postings(root, kind="aa", postings=db.aa_postings or {})
    _write_bucketed_positional_postings(root, kind="sa", postings=db.positional_postings or {})
    _write_bucketed_positional_postings(root, kind="aa", postings=db.aa_positional_postings or {})
    if write_compiled:
        _write_compiled_search_layout(db, root)


def load_search_db(
    path: Union[str, Path],
    *,
    prefer_compiled: bool = True,
    auto_compile_missing: bool = False,
) -> SearchDB:
    """Load a search database written by save_search_db().

    Agent Notes:
        PREFER: Keep prefer_compiled=True for repeated query workloads; DBs saved with default settings already include the compiled layout.
        PREFER: Set auto_compile_missing=True when you want older Parquet-only DBs upgraded in place during load.
        WATCH: With prefer_compiled=True, lazy older/Parquet-only DBs warn before falling back to Python/PyArrow-backed serving.
    """
    root = Path(path)
    if prefer_compiled and _compiled_manifest_path(root).exists():
        return _load_compiled_search_db(root)

    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    version = int(payload["version"])
    if version not in {2, 3, SEARCH_DB_VERSION}:
        raise ValueError(
            f"Unsupported search DB version {version}. Expected one of 2, 3 or {SEARCH_DB_VERSION}."
        )
    if version == 2:
        entries = _load_entries_eager(root, payload["entries_file"])
        postings, aa_postings = _load_postings_eager(root, payload["postings_file"])
        return SearchDB(
            version=version,
            k=int(payload["k"]),
            entries=entries,
            postings=postings,
            aa_postings=aa_postings,
            k_values=[int(payload["k"])],
            posting_keys_include_k=False,
            root_path=str(root),
            n_entries=len(entries),
        )

    if prefer_compiled and auto_compile_missing and version in {3, SEARCH_DB_VERSION}:
        return compile_search_db(root)

    if version == 3:
        if prefer_compiled:
            _warn_missing_compiled_search_layout(root, version)
        return SearchDB(
            version=version,
            k=int(payload["k"]),
            entries=None,
            postings=None,
            aa_postings=None,
            k_values=[int(payload["k"])],
            posting_keys_include_k=False,
            root_path=str(root),
            n_entries=int(payload["n_entries"]),
        )

    if prefer_compiled:
        _warn_missing_compiled_search_layout(root, version)
    return SearchDB(
        version=version,
        k=int(payload["k"]),
        entries=None,
        postings=None,
        aa_postings=None,
        k_values=[int(value) for value in payload.get("k_values", [int(payload["k"])])],
        posting_keys_include_k=bool(payload.get("posting_keys_include_k", False)),
        root_path=str(root),
        n_entries=int(payload["n_entries"]),
    )


def warm_search_db(
    db: Union[SearchDB, str, Path],
    *,
    kinds: Sequence[str] = ("sa", "aa"),
    posting_cache_max_size: Optional[int] = None,
    auto_compile_missing: bool = False,
) -> SearchDB:
    """Warm the postings cache for a lazy Parquet-backed search DB.

    Agent Notes:
        PREFER: Set auto_compile_missing=True when warming a persisted Parquet-only DB that you want upgraded in place first.
        PREFER: Call this once before latency-sensitive query batches on large lazy DBs.
        COST: Warmup reads postings shards into memory and trades startup time for steadier query latency.
    """
    if isinstance(db, (str, Path)):
        db = load_search_db(db, auto_compile_missing=auto_compile_missing)
    if posting_cache_max_size is not None:
        if posting_cache_max_size < 1:
            raise ValueError(f"posting_cache_max_size must be >= 1, got {posting_cache_max_size}")
        db.posting_cache_max_size = posting_cache_max_size
    if db.root_path is None or db.entries is not None or db.version < 4:
        return db

    root = Path(db.root_path)
    allowed_kinds = {"sa", "aa"}
    requested_kinds = []
    for kind in kinds:
        if kind not in allowed_kinds:
            raise ValueError(f"kinds must only contain 'sa' or 'aa', got {kind!r}")
        requested_kinds.append(kind)

    bucket_roots = [root / "postings" / f"kind={kind}" for kind in requested_kinds]
    bucket_ids = set()
    for bucket_root in bucket_roots:
        if not bucket_root.exists():
            continue
        for bucket_path in bucket_root.glob("bucket=*.parquet"):
            name = bucket_path.stem
            if not name.startswith("bucket="):
                continue
            bucket_ids.add(int(name.split("=", 1)[1]))

    required_size = max(db.posting_cache_max_size, len(bucket_ids) * len(requested_kinds))
    db.posting_cache_max_size = required_size
    for kind in requested_kinds:
        for bucket in sorted(bucket_ids):
            _get_cached_posting_bucket(db, root, kind=kind, bucket=bucket)
            _get_cached_positional_posting_bucket(db, root, kind=kind, bucket=bucket)
    return db


def search(
    query,
    db: Union[SearchDB, str, Path],
    *,
    top_k: int = 10,
    min_score: float = 0.0,
    alphabet_weight: float = 0.7,
    aa_weight: float = 0.3,
    rerank: bool = True,
    rerank_top_k: int = 5,
    rerank_fast: bool = True,
    diagonal_rescore: bool = True,
    diagonal_top_k: int = 200,
    diagonal_prefilter: bool = True,
    diagonal_min_support: float = 1.0,
    diagonal_prefilter_top_k: int = 1000,
    cache_max_size: Optional[int] = None,
    posting_cache_max_size: Optional[int] = None,
    auto_compile_missing: bool = False,
) -> List[SearchHit]:
    """Search a structural-alphabet DB using a structure or encoded query.

    Agent Notes:
        INVARIANT: Query encoding is Rust-backed; DB serving/caching stays in Python/PyArrow; diagonal rescoring uses Rust when available and otherwise falls back to Python.
        PREFER: Persisted DBs saved with default settings already include the compiled layout; use compile_search_db() only when upgrading an older Parquet-only DB.
        PREFER: Set auto_compile_missing=True to upgrade a Parquet-only DB in place instead of warning and staying lazy.
        WATCH: Passing an encoded query dict skips TM-align reranking because no original structure is available.
        COST: Lazy DB search avoids eager materialization but pays extra Parquet reads on first access; path-based loads warn when they fall back there.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if rerank_top_k < 1:
        raise ValueError(f"rerank_top_k must be >= 1, got {rerank_top_k}")
    if diagonal_top_k < 1:
        raise ValueError(f"diagonal_top_k must be >= 1, got {diagonal_top_k}")
    if diagonal_min_support <= 0.0:
        raise ValueError(f"diagonal_min_support must be > 0, got {diagonal_min_support}")
    if diagonal_prefilter_top_k < 1:
        raise ValueError(f"diagonal_prefilter_top_k must be >= 1, got {diagonal_prefilter_top_k}")
    if isinstance(db, (str, Path)):
        db = load_search_db(db, auto_compile_missing=auto_compile_missing)
    if cache_max_size is not None:
        if cache_max_size < 1:
            raise ValueError(f"cache_max_size must be >= 1, got {cache_max_size}")
        db.cache_max_size = cache_max_size
    if posting_cache_max_size is not None:
        if posting_cache_max_size < 1:
            raise ValueError(f"posting_cache_max_size must be >= 1, got {posting_cache_max_size}")
        db.posting_cache_max_size = posting_cache_max_size

    query_result = query if isinstance(query, dict) else encode_alphabet(query)
    query_valid = _valid_alphabet(query_result)
    query_valid_aa = _valid_aa_sequence(query_result)
    query_kmers = _query_posting_keys(query_valid, db)
    query_aa_kmers = _query_posting_keys(query_valid_aa, db)
    query_kmer_positions = _query_posting_key_positions(query_valid, db)
    query_aa_kmer_positions = _query_posting_key_positions(query_valid_aa, db)
    query_kmer_weights = _query_kmer_weights(db, kind="sa", kmers=query_kmers)
    query_aa_kmer_weights = _query_kmer_weights(db, kind="aa", kmers=query_aa_kmers)

    if db.root_path is not None and db.entries is None:
        root = Path(db.root_path)
        if db.version == 3:
            shared_counts = _fetch_posting_counts_v3(root, kind="sa", kmers=query_kmers)
            shared_aa_counts = _fetch_posting_counts_v3(root, kind="aa", kmers=query_aa_kmers)
        else:
            shared_counts = _fetch_posting_counts_bucketed_cached(db, root, kind="sa", kmers=query_kmers)
            shared_aa_counts = _fetch_posting_counts_bucketed_cached(db, root, kind="aa", kmers=query_aa_kmers)
        entry_lookup = _fetch_entries_by_index(
            root,
            sorted(set(shared_counts) | set(shared_aa_counts)),
            db=db if db.version >= 4 else None,
        )
    else:
        shared_counts = {}
        for kmer in query_kmers:
            for idx in (db.postings or {}).get(kmer, []):
                shared_counts[idx] = shared_counts.get(idx, 0) + 1

        shared_aa_counts = {}
        for kmer in query_aa_kmers:
            for idx in (db.aa_postings or {}).get(kmer, []):
                shared_aa_counts[idx] = shared_aa_counts.get(idx, 0) + 1
        entry_lookup = _entries_by_index(db.entries or [])

    candidate_indices = sorted(set(shared_counts) | set(shared_aa_counts))
    diagonal_vote_scores = _diagonal_vote_scores(
        db,
        entry_lookup,
        query_kmer_positions,
        query_aa_kmer_positions,
        candidate_indices,
    )
    candidate_indices = _apply_diagonal_prefilter(
        candidate_indices,
        diagonal_vote_scores,
        shared_counts,
        shared_aa_counts,
        enabled=diagonal_prefilter,
        min_support=diagonal_min_support,
        keep_count=max(top_k, rerank_top_k, diagonal_top_k, diagonal_prefilter_top_k),
        min_backfill=max(top_k, rerank_top_k),
    )
    weighted_shared = {idx: 0.0 for idx in candidate_indices}
    weighted_shared_aa = {idx: 0.0 for idx in candidate_indices}

    if db.root_path is not None and db.entries is None and db.version >= 4:
        root = Path(db.root_path)
        grouped_kmers: Dict[int, List[str]] = {}
        for kmer in query_kmers:
            grouped_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)
        for bucket, bucket_kmers in grouped_kmers.items():
            bucket_postings = _get_cached_posting_bucket(db, root, kind="sa", bucket=bucket)
            for kmer in bucket_kmers:
                weight = query_kmer_weights.get(kmer, 0.0)
                for idx in bucket_postings.get(kmer, []):
                    if idx in weighted_shared:
                        weighted_shared[idx] += weight

        grouped_aa_kmers: Dict[int, List[str]] = {}
        for kmer in query_aa_kmers:
            grouped_aa_kmers.setdefault(_posting_bucket(kmer), []).append(kmer)
        for bucket, bucket_kmers in grouped_aa_kmers.items():
            bucket_postings = _get_cached_posting_bucket(db, root, kind="aa", bucket=bucket)
            for kmer in bucket_kmers:
                weight = query_aa_kmer_weights.get(kmer, 0.0)
                for idx in bucket_postings.get(kmer, []):
                    if idx in weighted_shared_aa:
                        weighted_shared_aa[idx] += weight
    else:
        sa_postings = db.postings or {}
        for kmer in query_kmers:
            weight = query_kmer_weights.get(kmer, 0.0)
            for idx in sa_postings.get(kmer, []):
                if idx in weighted_shared:
                    weighted_shared[idx] += weight
        aa_postings = db.aa_postings or {}
        for kmer in query_aa_kmers:
            weight = query_aa_kmer_weights.get(kmer, 0.0)
            for idx in aa_postings.get(kmer, []):
                if idx in weighted_shared_aa:
                    weighted_shared_aa[idx] += weight

    hits: List[SearchHit] = []
    for idx in candidate_indices:
        entry = entry_lookup[idx]
        shared = shared_counts.get(idx, 0)
        shared_aa = shared_aa_counts.get(idx, 0)
        alphabet_score = 1.0 if query_valid == entry.valid_alphabet and query_valid else _weighted_query_coverage(weighted_shared.get(idx, 0.0), query_kmer_weights)
        aa_score = 1.0 if query_valid_aa == entry.valid_aa_sequence and query_valid_aa else _weighted_query_coverage(weighted_shared_aa.get(idx, 0.0), query_aa_kmer_weights)
        prefilter_score = alphabet_weight * alphabet_score + aa_weight * aa_score
        if prefilter_score < min_score:
            continue
        hits.append(
            SearchHit(
                id=entry.id,
                source_path=entry.source_path,
                source_index=entry.source_index,
                score=prefilter_score,
                prefilter_score=prefilter_score,
                alphabet_score=alphabet_score,
                aa_score=aa_score,
                shared_kmers=shared,
                shared_aa_kmers=shared_aa,
                query_kmers=len(query_kmers),
                target_kmers=entry.kmer_count,
                query_aa_kmers=len(query_aa_kmers),
                target_aa_kmers=entry.aa_kmer_count,
                residue_count=entry.residue_count,
                valid_residue_count=entry.valid_residue_count,
                diagonal_vote_score=diagonal_vote_scores.get(idx),
            )
        )

    hits.sort(key=lambda hit: (-hit.score, -hit.shared_kmers, hit.source_index, hit.id))

    if diagonal_rescore and hits:
        diagonal_count = min(len(hits), max(top_k, rerank_top_k, diagonal_top_k))
        entries_by_path = {entry.source_path: entry for entry in entry_lookup.values()}
        diagonal_hits = []
        diagonal_entries = []
        for hit in hits[:diagonal_count]:
            entry = entries_by_path.get(hit.source_path)
            if entry is None:
                continue
            diagonal_hits.append(hit)
            diagonal_entries.append(entry)
        if diagonal_hits and _search is not None and hasattr(_search, "diagonal_rescore_batch"):
            scores = _search.diagonal_rescore_batch(
                query_valid,
                query_valid_aa,
                [entry.valid_alphabet for entry in diagonal_entries],
                [entry.valid_aa_sequence for entry in diagonal_entries],
                None,
            )
        else:
            scores = [
                _ungapped_local_diagonal_score(
                    query_valid,
                    query_valid_aa,
                    entry.valid_alphabet,
                    entry.valid_aa_sequence,
                )
                for entry in diagonal_entries
            ]
        for hit, score in zip(diagonal_hits, scores):
            hit.diagonal_score = float(score)
            hit.score = hit.diagonal_score
        hits.sort(
            key=lambda hit: (
                -(hit.diagonal_score if hit.diagonal_score is not None else -1.0),
                -hit.prefilter_score,
                hit.source_index,
                hit.id,
            )
        )

    if rerank and hits and not isinstance(query, dict):
        from .align import tm_align

        rerank_count = min(len(hits), max(top_k, rerank_top_k))
        for hit in hits[:rerank_count]:
            target = _get_cached_structure(db, hit.source_path)
            result = tm_align(query, target, fast=rerank_fast)
            hit.tm_score = max(result.tm_score_chain1, result.tm_score_chain2)
            hit.rmsd = result.rmsd
            hit.n_aligned = result.n_aligned
            hit.seq_identity = result.seq_identity
            hit.score = hit.tm_score

        hits.sort(
            key=lambda hit: (
                -(hit.tm_score if hit.tm_score is not None else -1.0),
                -hit.prefilter_score,
                hit.source_index,
                hit.id,
            )
        )

    return hits[:top_k]
