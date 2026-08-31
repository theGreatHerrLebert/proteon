#!/usr/bin/env python3
"""Build the one-page EVIDENT claim viewer: ``evident/reports/site.html``.

Runs the upstream engine, ``typed-trust --format site``, over
``evident/evident.yaml`` and writes a self-contained HTML page: every
claim with kind / tier / status / checks, a subsystem × tier coverage
matrix, a claim–oracle–capability graph, and the per-claim trust report
as a drill-down. The page has a "Start here" tab that explains the
vocabulary, so it is the entry point to hand to reviewers.

The generated file is **committed** (unlike ``index.html``) because the
engine lives in the private EVIDENT repository and cannot be built in
this repository's CI. Regenerate it whenever claims change::

    python evident/scripts/build_site.py                # uses inline last_verified
    python evident/scripts/build_site.py --sidecar evident/last_verified.json

``typed-trust`` is located via ``$TYPED_TRUST``, then ``PATH``, then a
sibling checkout of the framework (``../evident/typed-trust/target/``).
The Pages workflow mirrors ``evident/reports/`` wholesale, so the page is
published at ``<pages-url>/evident/reports/site.html`` and linked from
the release index.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "evident" / "evident.yaml"
OUT = REPO_ROOT / "evident" / "reports" / "site.html"


def _find_typed_trust() -> pathlib.Path | None:
    env = os.environ.get("TYPED_TRUST")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    on_path = shutil.which("typed-trust")
    if on_path:
        return pathlib.Path(on_path)
    # Sibling checkout: prefer the most recently built binary, whichever
    # profile it came from, so a stale release build never shadows a fresh
    # debug one (or vice versa).
    candidates = [
        base / "typed-trust" / "target" / rel / "typed-trust"
        for base in (REPO_ROOT.parent / "evident", REPO_ROOT.parent.parent)
        for rel in ("release", "debug")
    ]
    candidates = [c for c in candidates if c.is_file()]
    if candidates:
        return max(candidates, key=lambda c: c.stat().st_mtime)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--sidecar", type=pathlib.Path, default=None,
                    help="last_verified.json written by `evident-agent replay`; "
                         "fills the checks with observed values")
    ap.add_argument("--review-events", type=pathlib.Path, default=None,
                    help="review_events.json sidecar (endorse / dissent / challenge)")
    args = ap.parse_args()

    tt = _find_typed_trust()
    if tt is None:
        print("typed-trust not found: set $TYPED_TRUST, put it on PATH, or build it in a "
              "sibling checkout of the EVIDENT framework (cargo build --release in typed-trust/)",
              file=sys.stderr)
        return 2
    cmd = [str(tt), "--format", "site"]
    if args.sidecar:
        cmd += ["--last-verified-sidecar", str(args.sidecar)]
    if args.review_events:
        cmd += ["--review-events-sidecar", str(args.review_events)]
    # Run from the repo root so the manifest path embedded in the page is
    # the repo-relative one, not an absolute path from someone's machine.
    cmd.append(os.path.relpath(args.manifest, REPO_ROOT))
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        return res.returncode
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(res.stdout, encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO_ROOT)} ({len(res.stdout):,} bytes) with {tt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
