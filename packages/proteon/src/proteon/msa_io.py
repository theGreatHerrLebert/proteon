"""Parsers for pre-computed MSA files (a3m) and directory helpers.

AF2 / ColabFold pipelines emit MSAs as a3m files, one per query. This
module reads that format without needing the upstream search tools
installed. The output shape matches `build_sequence_example`'s
`msa` / `deletion_matrix` kwargs so external MSAs can feed the
sequence release alongside — or instead of — an in-process search
engine.

a3m convention (AF2-compatible):
  - Each FASTA-style record is one MSA row; the first record is the query.
  - Within a row, uppercase letters and `-` map to aligned columns;
    aligned length must equal the query's aligned length across all rows.
  - Lowercase letters are insertions — they don't occupy a column.
    The count of lowercase residues between two aligned columns is
    attributed to the `deletion_matrix` cell of the *following* aligned
    column. Trailing lowercase residues (after the last aligned column)
    are dropped, matching the AF2 MSA-featurization reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_a3m_text(text: str) -> Tuple[List[str], List[List[int]], str]:
    """Parse a3m text into `(aligned_rows, deletion_matrix, query_aligned)`.

    - `aligned_rows[i]` is row `i`'s aligned sequence: uppercase letters
      plus `-`. Every row has length `len(query_aligned)`.
    - `deletion_matrix[i][j]` is the number of lowercase insertion
      residues in row `i` that occurred immediately before aligned
      column `j`.
    - `query_aligned` is the first record's aligned sequence and serves
      as the column contract for every subsequent row.

    Raises `ValueError` if the query contains lowercase characters
    (invalid — the query defines the columns) or if any homolog row's
    aligned length doesn't match the query's.
    """
    records = _split_fasta(text)
    if not records:
        raise ValueError("empty a3m input")

    query_name, query_raw = records[0]
    if any(ch.islower() for ch in query_raw):
        raise ValueError(
            f"a3m query {query_name!r} contains lowercase insertions — "
            "the query row defines the column layout and must be all-uppercase"
        )
    query_aligned = "".join(ch for ch in query_raw if ch.isupper() or ch == "-")
    query_len = len(query_aligned)

    aligned_rows: List[str] = []
    deletion_matrix: List[List[int]] = []
    for name, raw in records:
        aligned, deletions = _split_row(raw, query_len)
        if len(aligned) != query_len:
            raise ValueError(
                f"a3m row {name!r}: aligned length {len(aligned)} != "
                f"query aligned length {query_len}"
            )
        aligned_rows.append(aligned)
        deletion_matrix.append(deletions)

    return aligned_rows, deletion_matrix, query_aligned


def parse_a3m_file(path: str | Path) -> Tuple[List[str], List[List[int]], str]:
    """Read an a3m file from disk and return `parse_a3m_text`'s triple."""
    return parse_a3m_text(Path(path).read_text(encoding="utf-8"))


def load_msas_from_dir(
    msa_dir: str | Path,
    record_ids: Sequence[str],
    *,
    suffix: str = ".a3m",
    strict: bool = False,
) -> Tuple[List[Optional[List[str]]], List[Optional[List[List[int]]]]]:
    """Resolve `{msa_dir}/{record_id}{suffix}` for each record_id.

    Returns `(msas, deletion_matrices)` with one entry per record_id,
    suitable for passing straight into `build_sequence_dataset`. Records
    without an MSA file get `None` entries so downstream logic can fall
    back to the `msa_engine` or to a no-MSA `SequenceExample`.

    When `strict=True`, any missing file raises `FileNotFoundError`.
    """
    root = Path(msa_dir)
    msas: List[Optional[List[str]]] = []
    deletions: List[Optional[List[List[int]]]] = []
    for rid in record_ids:
        path = root / f"{rid}{suffix}"
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"no MSA file for record {rid!r}: {path}")
            msas.append(None)
            deletions.append(None)
            continue
        rows, delmat, _query = parse_a3m_file(path)
        msas.append(rows)
        deletions.append(delmat)
    return msas, deletions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_fasta(text: str) -> List[Tuple[str, str]]:
    """Return `[(name, raw_sequence), ...]` preserving record order."""
    records: List[Tuple[str, str]] = []
    name: Optional[str] = None
    body: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            if name is not None:
                records.append((name, "".join(body)))
            name = stripped[1:].strip()
            body = []
        elif stripped:
            body.append(stripped)
    if name is not None:
        records.append((name, "".join(body)))
    return records


def _split_row(raw: str, query_len: int) -> Tuple[str, List[int]]:
    """Split one a3m row into (aligned_string, deletion_counts).

    Walks the row once. Lowercase residues accumulate into a pending
    insertion count; the next aligned character (uppercase or `-`)
    flushes that count into deletion_matrix[j] and advances the column.
    """
    aligned_chars: List[str] = []
    deletions: List[int] = []
    pending = 0
    for ch in raw:
        if ch.islower():
            pending += 1
        elif ch.isupper() or ch == "-":
            aligned_chars.append(ch.upper())
            deletions.append(pending)
            pending = 0
        # Everything else (digits, whitespace inside a body line, etc.)
        # is ignored; a3m body lines are usually clean but callers
        # occasionally paste numbered variants.
    return "".join(aligned_chars), deletions
