"""Unit tests for the a3m parser and msa_dir loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from proteon.msa_io import (
    load_msas_from_dir,
    parse_a3m_file,
    parse_a3m_text,
)


def test_parse_a3m_query_only():
    text = dedent(
        """\
        >query
        ABCDE
        """
    )
    rows, delmat, query = parse_a3m_text(text)
    assert query == "ABCDE"
    assert rows == ["ABCDE"]
    assert delmat == [[0, 0, 0, 0, 0]]


def test_parse_a3m_homolog_with_gap():
    text = dedent(
        """\
        >query
        ABCDE
        >hit1
        AB-DE
        """
    )
    rows, delmat, query = parse_a3m_text(text)
    assert query == "ABCDE"
    assert rows == ["ABCDE", "AB-DE"]
    # No lowercase insertions anywhere → all zeros.
    assert delmat == [[0] * 5, [0] * 5]


def test_parse_a3m_lowercase_counts_into_next_aligned_column():
    """AF2 convention: lowercase before col j counts into deletion_matrix[..,j]."""
    text = dedent(
        """\
        >query
        ABCDE
        >hit_with_inserts
        AbCxxDE
        """
    )
    # hit_with_inserts walk:
    #   A (upper)  → aligned[0]='A', pending=0
    #   b (lower)  → pending=1
    #   C (upper)  → aligned[1]='C', deletion[1]=1, pending=0
    #   x (lower)  → pending=1
    #   x (lower)  → pending=2
    #   D (upper)  → aligned[2]='D', deletion[2]=2, pending=0
    #   E (upper)  → aligned[3]='E', deletion[3]=0
    # But query has 5 columns so aligned length mismatch → ValueError.
    with pytest.raises(ValueError, match="aligned length"):
        parse_a3m_text(text)


def test_parse_a3m_trailing_lowercase_is_dropped():
    """AF2 convention: lowercase after the last aligned column has no place
    to go and is silently dropped. The row's aligned length must still
    match the query's."""
    text = dedent(
        """\
        >query
        ABCDE
        >hit
        AbCDEfgh
        """
    )
    # Aligned walk: A, [b], C, D, E, [f,g,h trailing].
    # Aligned chars A,C,D,E → length 4, mismatches query's 5 → raises.
    with pytest.raises(ValueError, match="aligned length"):
        parse_a3m_text(text)


def test_parse_a3m_trailing_lowercase_dropped_when_aligned_length_matches():
    text = dedent(
        """\
        >query
        ABCDE
        >hit
        ABCDEfgh
        """
    )
    rows, delmat, _ = parse_a3m_text(text)
    assert rows == ["ABCDE", "ABCDE"]
    # Trailing fgh has no aligned column to attribute to → dropped.
    assert delmat[1] == [0, 0, 0, 0, 0]


def test_parse_a3m_query_with_gap_defines_columns():
    text = dedent(
        """\
        >query
        A-BCDE
        >hit
        A-BCDE
        """
    )
    rows, delmat, query = parse_a3m_text(text)
    assert query == "A-BCDE"
    assert rows == ["A-BCDE", "A-BCDE"]
    assert delmat == [[0] * 6, [0] * 6]


def test_parse_a3m_query_cannot_have_lowercase():
    text = dedent(
        """\
        >query
        AbCDE
        """
    )
    with pytest.raises(ValueError, match="lowercase insertions"):
        parse_a3m_text(text)


def test_parse_a3m_insertions_deletion_matrix_values():
    """Two homologs: one aligned cleanly, one with a 2-residue insertion."""
    text = dedent(
        """\
        >query
        ABCDE
        >clean
        ABCDE
        >with_ins
        ABxxCDE
        """
    )
    rows, delmat, _ = parse_a3m_text(text)
    assert rows == ["ABCDE", "ABCDE", "ABCDE"]
    assert delmat[0] == [0, 0, 0, 0, 0]
    assert delmat[1] == [0, 0, 0, 0, 0]
    # hit_with_ins walk:
    #   A → aligned[0]='A', deletion[0]=0
    #   B → aligned[1]='B', deletion[1]=0
    #   x → pending=1
    #   x → pending=2
    #   C → aligned[2]='C', deletion[2]=2
    #   D → aligned[3]='D', deletion[3]=0
    #   E → aligned[4]='E', deletion[4]=0
    assert delmat[2] == [0, 0, 2, 0, 0]


def test_parse_a3m_empty_raises():
    with pytest.raises(ValueError, match="empty a3m"):
        parse_a3m_text("")


def test_parse_a3m_file_reads_from_disk(tmp_path: Path):
    p = tmp_path / "q.a3m"
    p.write_text(">q\nABC\n>h\nA-C\n", encoding="utf-8")
    rows, delmat, query = parse_a3m_file(p)
    assert query == "ABC"
    assert rows == ["ABC", "A-C"]
    assert delmat == [[0, 0, 0], [0, 0, 0]]


def test_load_msas_from_dir_missing_files_are_none(tmp_path: Path):
    (tmp_path / "alpha.a3m").write_text(">alpha\nABC\n>h\nABC\n")
    # beta.a3m intentionally missing.
    msas, delmats = load_msas_from_dir(tmp_path, ["alpha", "beta", "alpha"])
    assert msas[0] == ["ABC", "ABC"]
    assert delmats[0] == [[0, 0, 0], [0, 0, 0]]
    assert msas[1] is None
    assert delmats[1] is None
    assert msas[2] == ["ABC", "ABC"]  # same record_id resolves twice


def test_load_msas_from_dir_strict_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no MSA file"):
        load_msas_from_dir(tmp_path, ["missing"], strict=True)


def test_load_msas_from_dir_custom_suffix(tmp_path: Path):
    (tmp_path / "q.sto").write_text(">q\nABC\n")  # custom suffix
    msas, _ = load_msas_from_dir(tmp_path, ["q"], suffix=".sto")
    assert msas[0] == ["ABC"]
