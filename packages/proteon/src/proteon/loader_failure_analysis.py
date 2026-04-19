"""Utilities for analyzing structure-loader failures.

The release taxonomy's `parse_error` class is intentionally broad. For
rescue work we need a second layer: concrete loader buckets that map to
deterministic repair strategies, so we can rank "worth rescuing" cases
from validation artifacts instead of reading giant exception blobs by eye.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class LoaderFailureBucket:
    """One concrete loader-failure bucket with rescue guidance."""

    code: str
    rescueable: bool
    summary: str
    rescue_strategy: Optional[str] = None


ATOM_CHARGE_SUFFIX = LoaderFailureBucket(
    code="atom_charge_suffix",
    rescueable=True,
    summary="PDB atom element/charge suffix is malformed (e.g. terminal `N0`).",
    rescue_strategy="rewrite malformed element/charge columns and retry parse",
)
SEQADV_INVALID_FIELD = LoaderFailureBucket(
    code="seqadv_invalid_field",
    rescueable=True,
    summary="SEQADV records contain integer-field formatting the reader rejects.",
    rescue_strategy="strip SEQADV records or parse them permissively before retry",
)
SOLITARY_DBREF1 = LoaderFailureBucket(
    code="solitary_dbref1_definition",
    rescueable=True,
    summary="DBREF1 is present without matching DBREF2.",
    rescue_strategy="drop incomplete DBREF metadata records before retry",
)
SEQRES_MULTIPLE_RESIDUES = LoaderFailureBucket(
    code="seqres_multiple_residues",
    rescueable=True,
    summary="SEQRES validation trips on alternate/multiple residues at one index.",
    rescue_strategy="disable strict SEQRES validation or normalize alternate residues first",
)
SEQRES_TOTAL_INVALID = LoaderFailureBucket(
    code="seqres_total_invalid",
    rescueable=True,
    summary="SEQRES/DBREF residue totals disagree with the parsed chain.",
    rescue_strategy="ignore SEQRES/DBREF count mismatches during rescue load",
)
MODEL_ATOM_MISMATCH = LoaderFailureBucket(
    code="model_atom_correspondence",
    rescueable=False,
    summary="Multi-model atom lists differ across models.",
    rescue_strategy="prefer first-model-only rescue only if downstream semantics allow it",
)
SSBOND_MISSING_PARTNER = LoaderFailureBucket(
    code="ssbond_missing_partner",
    rescueable=True,
    summary="SSBOND references atoms/residues not present in coordinates.",
    rescue_strategy="drop invalid SSBOND annotations and retry parse",
)
INVALID_DATA_FIELD_OTHER = LoaderFailureBucket(
    code="invalid_data_field_other",
    rescueable=False,
    summary="Reader rejected a fixed-width field, but no known rescue bucket matched.",
)
OTHER_LOAD_ERROR = LoaderFailureBucket(
    code="other_load_error",
    rescueable=False,
    summary="Loader failure did not match any known rescue bucket.",
)


KNOWN_BUCKETS: Dict[str, LoaderFailureBucket] = {
    bucket.code: bucket
    for bucket in (
        ATOM_CHARGE_SUFFIX,
        SEQADV_INVALID_FIELD,
        SOLITARY_DBREF1,
        SEQRES_MULTIPLE_RESIDUES,
        SEQRES_TOTAL_INVALID,
        MODEL_ATOM_MISMATCH,
        SSBOND_MISSING_PARTNER,
        INVALID_DATA_FIELD_OTHER,
        OTHER_LOAD_ERROR,
    )
}


@dataclass
class LoaderFailureSummary:
    """Aggregate counts for one concrete loader-failure bucket."""

    bucket: LoaderFailureBucket
    count: int
    pdb_examples: List[str]


def load_failure_rows(path: str | Path) -> List[dict]:
    """Load loader-failure JSONL rows from a validation artifact."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "load_error" and "exception" in row:
            rows.append(row)
    return rows


def bucket_loader_failure(exception_text: str) -> LoaderFailureBucket:
    """Classify one raw loader exception into a rescue-oriented bucket."""
    message = exception_text.lower()

    if "atom charge is not correct" in message:
        return ATOM_CHARGE_SUFFIX
    if "solitary dbref1 definition" in message:
        return SOLITARY_DBREF1
    if "atoms in models not corresponding" in message:
        return MODEL_ATOM_MISMATCH
    if "could not find a bond partner" in message and "ssbond" in message:
        return SSBOND_MISSING_PARTNER
    if "multiple residues in seqres validation" in message:
        return SEQRES_MULTIPLE_RESIDUES
    if "seqres residue total invalid" in message:
        return SEQRES_TOTAL_INVALID
    if "invalid data in field" in message:
        if "seqadv" in message or "the text presented is not of the right kind (isize)" in message:
            return SEQADV_INVALID_FIELD
        return INVALID_DATA_FIELD_OTHER
    return OTHER_LOAD_ERROR


def summarize_loader_failures(rows: Sequence[dict], *, max_examples: int = 5) -> List[LoaderFailureSummary]:
    """Aggregate loader-failure rows by concrete rescue bucket."""
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}

    for row in rows:
        bucket = bucket_loader_failure(str(row.get("exception", "")))
        counts[bucket.code] = counts.get(bucket.code, 0) + 1
        examples.setdefault(bucket.code, [])
        pdb = str(row.get("pdb", row.get("path", "unknown")))
        if pdb not in examples[bucket.code] and len(examples[bucket.code]) < max_examples:
            examples[bucket.code].append(pdb)

    summaries = [
        LoaderFailureSummary(
            bucket=KNOWN_BUCKETS[code],
            count=count,
            pdb_examples=examples.get(code, []),
        )
        for code, count in counts.items()
    ]
    summaries.sort(key=lambda s: (-s.count, s.bucket.code))
    return summaries


def summaries_to_markdown(summaries: Iterable[LoaderFailureSummary]) -> str:
    """Render a loader-failure summary as a compact markdown table."""
    lines = [
        "| bucket | count | rescueable | strategy | example pdbs |",
        "|---|---:|:---:|---|---|",
    ]
    for item in summaries:
        strategy = item.bucket.rescue_strategy or ""
        examples = ", ".join(item.pdb_examples)
        lines.append(
            f"| `{item.bucket.code}` | {item.count} | "
            f"{'yes' if item.bucket.rescueable else 'no'} | {strategy} | {examples} |"
        )
    return "\n".join(lines)
