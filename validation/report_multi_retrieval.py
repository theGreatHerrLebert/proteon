#!/usr/bin/env python3
"""Compare structural and sequence retrieval results per query."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_query(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["query"]: row for row in data.get("per_query", [])}


def _recall(row: dict[str, Any] | None, threshold: str, key: str) -> float | None:
    if row is None:
        return None
    value = row.get("thresholds", {}).get(threshold, {}).get(key)
    return None if value is None else float(value)


def _hit(value: float | None) -> bool:
    return value is not None and value > 0.0


def _source(row: dict[str, Any] | None, field: str) -> str | None:
    if row is None:
        return None
    hit = row.get(field)
    if not hit:
        return None
    source = hit.get("source_path")
    return None if source is None else str(source)


def _truth(row: dict[str, Any] | None) -> tuple[str | None, float | None]:
    if row is None:
        return None, None
    hit = row.get("best_truth_nonself")
    if not hit:
        return None, None
    return hit.get("source_path"), hit.get("tm_score")


def classify_query(struct_row: dict[str, Any] | None, seq_row: dict[str, Any] | None, threshold: str) -> dict[str, Any]:
    foldseek = _recall(struct_row, threshold, "recall_at_k")
    proteon = _recall(struct_row, threshold, "proteon_recall_at_k")
    sequence = _recall(seq_row, threshold, "sequence_recall_at_k")
    foldseek_hit = _hit(foldseek)
    proteon_hit = _hit(proteon)
    sequence_hit = _hit(sequence)

    truth_source, truth_tm = _truth(struct_row or seq_row)
    if truth_source is None:
        bucket = "no_truth"
    elif proteon_hit and sequence_hit:
        bucket = "proteon_and_sequence"
    elif proteon_hit:
        bucket = "proteon_only_vs_sequence"
    elif sequence_hit:
        bucket = "sequence_only_vs_proteon"
    elif foldseek_hit:
        bucket = "foldseek_only"
    else:
        bucket = "all_miss"

    return {
        "query": (struct_row or seq_row or {}).get("query"),
        "bucket": bucket,
        "truth": truth_source,
        "truth_tm_score": truth_tm,
        "foldseek_recall": foldseek,
        "proteon_recall": proteon,
        "sequence_recall": sequence,
        "foldseek_top": _source(struct_row, "foldseek_top_nonself"),
        "proteon_top": _source(struct_row, "proteon_top_nonself"),
        "sequence_top": _source(seq_row, "sequence_top_nonself"),
    }


def summarize(
    structural: dict[str, Any],
    sequence: dict[str, Any],
    *,
    threshold: str,
) -> dict[str, Any]:
    structural_rows = _rows_by_query(structural)
    sequence_rows = _rows_by_query(sequence)
    queries = sorted(set(structural_rows) | set(sequence_rows))
    per_query = [
        classify_query(structural_rows.get(query), sequence_rows.get(query), threshold)
        for query in queries
    ]
    counts = Counter(row["bucket"] for row in per_query)

    def values(key: str) -> list[float]:
        return [row[key] for row in per_query if row[key] is not None]

    return {
        "threshold": threshold,
        "n_queries": len(per_query),
        "counts": dict(counts),
        "mean_recall": {
            "foldseek": round(mean(values("foldseek_recall")), 4) if values("foldseek_recall") else None,
            "proteon": round(mean(values("proteon_recall")), 4) if values("proteon_recall") else None,
            "sequence": round(mean(values("sequence_recall")), 4) if values("sequence_recall") else None,
        },
        "per_query": per_query,
        "by_bucket": {
            bucket: [row for row in per_query if row["bucket"] == bucket]
            for bucket in [
                "proteon_and_sequence",
                "proteon_only_vs_sequence",
                "sequence_only_vs_proteon",
                "foldseek_only",
                "all_miss",
                "no_truth",
            ]
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-Retrieval Report",
        "",
        "## Headline",
        "",
        f"- Queries: {summary['n_queries']}",
        f"- Primary threshold: TM >= {summary['threshold']}",
        f"- Mean Foldseek recall: {summary['mean_recall']['foldseek']}",
        f"- Mean proteon recall: {summary['mean_recall']['proteon']}",
        f"- Mean sequence recall: {summary['mean_recall']['sequence']}",
        "",
        "## Buckets",
        "",
    ]
    for bucket in [
        "proteon_and_sequence",
        "proteon_only_vs_sequence",
        "sequence_only_vs_proteon",
        "foldseek_only",
        "all_miss",
        "no_truth",
    ]:
        lines.append(f"- {bucket}: {summary['counts'].get(bucket, 0)}")

    for bucket in [
        "proteon_only_vs_sequence",
        "sequence_only_vs_proteon",
        "foldseek_only",
        "all_miss",
    ]:
        lines.extend([
            "",
            f"## {bucket}",
            "",
            "| Query | Truth | TM | Foldseek | proteon | sequence | Foldseek top | proteon top | sequence top |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ])
        for row in summary["by_bucket"][bucket]:
            lines.append(
                "| {query} | {truth} | {truth_tm_score} | {foldseek_recall} | {proteon_recall} | {sequence_recall} | {foldseek_top} | {proteon_top} | {sequence_top} |".format(
                    **{key: "" if value is None else value for key, value in row.items()}
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", required=True, type=Path, help="Foldseek/proteon benchmark JSON")
    parser.add_argument("--sequence", required=True, type=Path, help="Sequence benchmark JSON")
    parser.add_argument("--threshold", default="0.7")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize(_load(args.structural), _load(args.sequence), threshold=str(args.threshold))
    markdown = render_markdown(summary)
    print(markdown, end="")
    if args.json_output:
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
