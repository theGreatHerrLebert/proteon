"""Summarize loader failures from a validation JSONL artifact."""

from __future__ import annotations

import argparse

from proteon.loader_failure_analysis import (
    load_failure_rows,
    summarize_loader_failures,
    summaries_to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="validation JSONL with load_error rows")
    args = parser.parse_args()

    rows = load_failure_rows(args.jsonl_path)
    summaries = summarize_loader_failures(rows)
    print(summaries_to_markdown(summaries))


if __name__ == "__main__":
    main()
