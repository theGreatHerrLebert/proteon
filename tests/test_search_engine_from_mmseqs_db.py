"""Python-side smoke for PySearchEngine.from_mmseqs_db.

The authoritative parity test is
`ferritin-search::search::tests::build_from_mmseqs_db_round_trips_against_in_memory_build`
(Rust-side). This file exercises the Python binding + error-path surface
+ round-trips a tiny DB written byte-compatibly with `mmseqs createdb`.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

import ferritin


def _rust_msa_available() -> bool:
    try:
        return ferritin.rust_msa_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rust_msa_available(),
    reason="requires ferritin_connector.py_msa (build with MSA bindings)",
)


# Byte-level DB writer. Keeps the test self-contained: no standalone
# `mmseqs` binary dependency and no Python-exposed DBWriter needed.
# Format comes from MMseqs2/docs/DB_FORMAT_SPEC.md (bits 15..0 of dbtype
# are the base type; AMINO_ACIDS = 0). Each entry's payload is appended
# with a trailing `\n\0` upstream-style; the index records the full
# length including those two bytes.
_DBTYPE_AMINO_ACIDS = 0


def _write_db(prefix: Path, entries: list[tuple[int, bytes]]) -> None:
    data_chunks: list[bytes] = []
    index_lines: list[str] = []
    offset = 0
    for key, payload in entries:
        framed = payload + b"\n\0"
        data_chunks.append(framed)
        index_lines.append(f"{key}\t{offset}\t{len(framed)}\n")
        offset += len(framed)
    prefix.write_bytes(b"".join(data_chunks))
    Path(str(prefix) + ".index").write_text("".join(index_lines))
    Path(str(prefix) + ".dbtype").write_bytes(struct.pack("<i", _DBTYPE_AMINO_ACIDS))


def test_from_mmseqs_db_parity_against_in_memory_build(tmp_path: Path):
    entries: list[tuple[int, bytes]] = [
        (1, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ"),
        (2, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP"),
        (3, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR"),
    ]
    prefix = tmp_path / "tiny"
    _write_db(prefix, entries)

    db_engine = ferritin.build_search_engine_from_mmseqs_db(str(prefix))
    mem_engine = ferritin.build_search_engine(
        [(k, p.decode("ascii")) for k, p in entries]
    )
    assert db_engine.target_count() == len(entries) == mem_engine.target_count()

    for key, payload in entries:
        query = payload.decode("ascii")
        db_hits = db_engine.search(query)
        mem_hits = mem_engine.search(query)
        assert [h["target_id"] for h in db_hits] == [h["target_id"] for h in mem_hits]
        assert [h["score"] for h in db_hits] == [h["score"] for h in mem_hits]
        assert db_hits, f"DB engine: query {key} returned no hits"
        assert db_hits[0]["target_id"] == key, (
            f"DB engine: query {key} top hit was {db_hits[0]['target_id']} (expected self)"
        )


def test_from_mmseqs_db_builds_msa_end_to_end(tmp_path: Path):
    """from_mmseqs_db → .search_and_build_msa works through the Python API."""
    entries: list[tuple[int, bytes]] = [
        (10, b"MKLVRQPSTNLKACDFGHIY"),
        (11, b"MKLVRKPSTALKACDFGHIV"),  # near-copy with a few substitutions
        (12, b"WWWWWWWWWWWWWWWWWWWW"),  # decoy
    ]
    prefix = tmp_path / "msa-db"
    _write_db(prefix, entries)

    engine = ferritin.build_search_engine_from_mmseqs_db(str(prefix))
    msa = engine.search_and_build_msa("MKLVRQPSTNLKACDFGHIY", max_seqs=16, gap_idx=21)
    assert int(msa["query_len"]) == 20
    assert int(msa["n_seqs"]) >= 2  # query + at least the near-copy
    assert msa["msa"].shape == (int(msa["n_seqs"]), 20)


def test_from_mmseqs_db_missing_path_raises_value_error():
    with pytest.raises(Exception) as excinfo:
        ferritin.build_search_engine_from_mmseqs_db("/nonexistent/path/to/db")
    assert "mmseqs DB build failed" in str(excinfo.value)
