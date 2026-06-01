#!/usr/bin/env python3
"""Score an EVIDENT claim's tolerances against its evidence artifact(s).

Where ``replay_claim.py`` re-runs a claim's ``evidence.command`` and only
checks the subprocess exit code, this script opens the recorded artifact
and checks that each tolerance with a ``scoring:`` block still meets its
recorded band. A claim that has drifted out of its own tolerance exits
non-zero.

Usage::

    python evident/scripts/score_claim.py <claim-id>
    python evident/scripts/score_claim.py --all
    python evident/scripts/score_claim.py --all --release-only

Exit codes:
  0  every scored tolerance passed (claims with no scoring: blocks count
     as a clean skip)
  1  at least one scored tolerance failed its band
  2  a scoring spec was malformed or its artifact was missing/unreadable
  64 usage / argument error
  65 claim id not found
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import claim_scoring as cs  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "evident" / "claims"


def _all_claim_files() -> list[pathlib.Path]:
    return sorted(CLAIMS_DIR.glob("*.yaml")) if CLAIMS_DIR.is_dir() else []


def _iter_claims():
    for path in _all_claim_files():
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            print(f"warning: skipping unparseable {path.name}: "
                  f"{str(e).splitlines()[0]}", file=sys.stderr)
            continue
        for claim in doc.get("claims") or []:
            if isinstance(claim, dict):
                yield path, claim


def _find_claim(claim_id: str):
    for path, claim in _iter_claims():
        if claim.get("id") == claim_id:
            return path, claim
    return None


def _has_scoring(claim: dict) -> bool:
    return any(
        isinstance(t, dict) and t.get("scoring") is not None
        for t in claim.get("tolerances") or []
    )


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.5g}"


def _print_claim(score: cs.ClaimScore) -> None:
    print(f"\nclaim: {score.claim_id}  [{score.status.upper()}]")
    for t in score.tolerances:
        if not t.scored:
            print(f"  · {t.metric:<22} {t.op} {_fmt_num(t.threshold)}"
                  f"{(' ('+t.output+')') if t.output else ''}  — unscored ({t.reason})")
            continue
        mark = "PASS" if t.passed else "FAIL"
        out = f" ({t.output})" if t.output else ""
        print(f"  [{mark}] {t.metric}{out}: observed {_fmt_num(t.observed)} "
              f"{t.op} {_fmt_num(t.threshold)}  [{t.reason}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score EVIDENT claim tolerances.")
    parser.add_argument("claim_id", nargs="?", help="Claim id to score.")
    parser.add_argument("--all", action="store_true", help="Score every claim.")
    parser.add_argument("--release-only", action="store_true",
                        help="With --all, score only tier=release claims.")
    args = parser.parse_args()

    if not args.all and not args.claim_id:
        parser.error("give a claim id or --all")

    if args.all:
        targets = [
            (p, c) for p, c in _iter_claims()
            if (not args.release_only or c.get("tier") == "release")
        ]
    else:
        found = _find_claim(args.claim_id)
        if found is None:
            print(f"claim not found: {args.claim_id}", file=sys.stderr)
            return 65
        targets = [found]

    any_fail = False
    n_scored_claims = 0
    try:
        for _path, claim in targets:
            if args.all and not _has_scoring(claim):
                continue  # keep the --all sweep focused on scorable claims
            score = cs.score_claim(claim, REPO_ROOT)
            if not score.scored:
                continue
            n_scored_claims += 1
            _print_claim(score)
            any_fail = any_fail or score.any_failed
    except cs.ScoringError as e:
        print(f"\nscoring error: {e}", file=sys.stderr)
        return 2

    print(f"\n{'─' * 50}")
    if not args.all and not _has_scoring(targets[0][1]):
        print(f"{args.claim_id}: no scoring: blocks — nothing to enforce")
        return 0
    print(f"scored {n_scored_claims} claim(s): "
          f"{'FAIL — at least one band not met' if any_fail else 'all bands met'}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
