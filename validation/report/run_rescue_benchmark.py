#!/usr/bin/env python3
"""Run a small realized-rescue benchmark and write a JSON summary artifact."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE_SRC = ROOT / "packages" / "proteon" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

import proteon


STAGE2_FAILURES = ROOT / "validation" / "stage2_1k_postfix.jsonl"
OUT_JSON = HERE / "rescue_benchmark_summary.json"
TMP_SMOKE = HERE / "rescue_benchmark_smoke"


def _candidate_failure_names(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "load_error" and row.get("pdb"):
            rows.append(str(row["pdb"]))
    return rows


def _find_available_paths(names: list[str]) -> list[Path]:
    wanted = set(names)
    found = []
    for path in (ROOT / "validation").rglob("*.pdb"):
        if path.name in wanted:
            found.append(path)
    found.sort()
    return found


def main() -> None:
    candidate_names = _candidate_failure_names(STAGE2_FAILURES)
    available_paths = _find_available_paths(candidate_names)

    raw_results = proteon.batch_load_tolerant_with_rescue(available_paths)
    rescued_bucket_counts: dict[str, int] = {}
    rescued_paths: list[str] = []
    for index, result in raw_results:
        if not result.rescued or result.rescue_bucket is None:
            continue
        rescued_paths.append(str(available_paths[index]))
        code = result.rescue_bucket.code
        rescued_bucket_counts[code] = rescued_bucket_counts.get(code, 0) + 1

    pipeline_ok = True
    pipeline_error = None
    if TMP_SMOKE.exists():
        shutil.rmtree(TMP_SMOKE)
    try:
        proteon.build_local_corpus_smoke_release(
            available_paths,
            TMP_SMOKE,
            release_id="rescue-benchmark-v0",
            rescue_load=True,
            overwrite=True,
        )
    except Exception as exc:
        pipeline_ok = False
        pipeline_error = f"{type(exc).__name__}: {exc}"

    summary = {
        "candidate_failure_count": len(candidate_names),
        "available_slice_count": len(available_paths),
        "rescued_count": len(rescued_paths),
        "unrescued_count": max(len(available_paths) - len(rescued_paths), 0),
        "rescued_bucket_counts": rescued_bucket_counts,
        "rescued_paths": rescued_paths,
        "pipeline_ok": pipeline_ok,
        "pipeline_error": pipeline_error,
        "smoke_output_dir": str(TMP_SMOKE),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
