"""Build the top-level EVIDENT release index.

Walks ``evident/reports/`` and writes ``evident/reports/index.html`` —
a single page that links to every release bundle (one directory per
``v*`` tag), with the lock date, claim count, and the release-level
verdict pulled from each bundle's ``manifest.json``.

This is the entry point a reviewer hits at
``https://thegreatherrlebert.github.io/proteon/evident/reports/`` once
GH Pages publishes the site. It's deliberately static — no JS, no
build step beyond reading the locked manifests.

Usage::

    python evident/scripts/build_index.py [--out-dir DIR]

By default writes to the same ``evident/reports/`` directory the
manifests live in. ``lock_release_replays.py`` calls this at the end
of every lock, so the index stays fresh without a separate step.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "evident" / "reports"


def _load_manifest(path: pathlib.Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _release_dirs(reports_dir: pathlib.Path) -> list[pathlib.Path]:
    """All subdirectories of reports/ that contain a manifest.json.

    Sorted by directory name in reverse, so newer releases (v0.2.0)
    sort above older ones (v0.1.0). For ``dev*`` directories the
    sort still puts them at the bottom because lowercase 'd' < 'v'.
    """
    if not reports_dir.is_dir():
        return []
    out = []
    for child in reports_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "manifest.json").is_file():
            out.append(child)
    out.sort(key=lambda p: p.name, reverse=True)
    return out


def _release_summary(manifest: dict) -> dict:
    claims = manifest.get("claims", []) or []
    n_total = len(claims)
    n_locked = sum(1 for c in claims if c.get("status") == "locked")
    n_missing = sum(1 for c in claims if c.get("status") == "missing")
    n_ci_only = sum(1 for c in claims if c.get("status") == "ci-only")
    n_no_artifact = sum(1 for c in claims if c.get("status") == "no-artifact")

    n_ok_total = n_records_total = 0
    for c in claims:
        s = c.get("summary") or {}
        n_ok_total += int(s.get("n_ok", 0))
        n_records_total += int(s.get("n_total", 0))

    return {
        "n_total": n_total,
        "n_locked": n_locked,
        "n_missing": n_missing,
        "n_ci_only": n_ci_only,
        "n_no_artifact": n_no_artifact,
        "n_ok_records": n_ok_total,
        "n_records": n_records_total,
    }


def _render(reports_dir: pathlib.Path, releases: list[tuple[str, dict, dict]]) -> str:
    # The one-page claim viewer (evident/scripts/build_site.py) is the
    # entry point for anyone who does not yet know the claims: it explains
    # the vocabulary and lets a reviewer filter, browse coverage, and
    # drill into every claim. Link it first when it has been generated.
    site_link = (
        '<p class="site"><a href="site.html"><strong>Browse all current claims →</strong></a> '
        '— filterable table, coverage by subsystem and tier, claim–oracle graph, and a '
        '"Start here" introduction for readers new to EVIDENT.</p>'
        if (reports_dir / "site.html").is_file() else ""
    )
    rows = []
    for tag, manifest, summary in releases:
        # Coverage = locked artifacts / claims that should produce one
        # (i.e. n_total minus the ci-only bucket, which has no artifact
        # by design). Avoids inflating the denominator with claims that
        # correctly don't have a file to pin.
        denom = summary["n_total"] - summary["n_ci_only"]
        if summary["n_total"] == 0:
            coverage = "no claims"
        elif denom == 0:
            coverage = f"all {summary['n_ci_only']} claims CI-only"
        else:
            coverage = (
                f"{summary['n_locked']}/{denom} locked "
                f"+ {summary['n_ci_only']} CI-only"
            )
        verdict = (
            f"{summary['n_ok_records']}/{summary['n_records']} records ok"
            if summary["n_records"]
            else "—"
        )
        rows.append(
            "<tr>"
            f'<td><a href="{tag}/"><code>{tag}</code></a></td>'
            f"<td>{manifest.get('locked_at_utc', '')}</td>"
            f"<td><code>{(manifest.get('commit', '') or '')[:12]}</code></td>"
            f"<td>{coverage}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    table = (
        "<p><em>No release bundles yet — run "
        "<code>python evident/scripts/lock_release_replays.py --release vX.Y.Z</code> "
        "to produce the first one.</em></p>"
        if not rows
        else f"""<table>
  <thead>
    <tr>
      <th>Release</th>
      <th>Locked at (UTC)</th>
      <th>Commit</th>
      <th>Coverage</th>
      <th>Headline</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>"""
    )

    return f"""<!doctype html>
<meta charset="utf-8">
<title>Proteon EVIDENT — Releases</title>
<style>
  body {{ font: 14px system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: #666; margin-bottom: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }}
  th {{ background: #fafafa; font-weight: 600; border-bottom: 2px solid #ddd; }}
  tr:hover td {{ background: #f7f9fc; }}
  td code {{ font-size: 12px; }}
  .footer {{ margin-top: 3em; color: #999; font-size: 12px; }}
  .site {{ background: #eef3f8; border-left: 4px solid #0b4f9c; padding: 0.6em 1em; border-radius: 4px; }}
</style>
<h1>Proteon EVIDENT — Releases</h1>
{site_link}
<p class="meta">
  Each row is a frozen release bundle. Click into a release for the per-claim manifest,
  rendered HTML reports, and sha256-pinned artifacts. Bundles are immutable —
  re-locking against the same release tag overwrites in place; cutting a new
  release tag adds a new row here.
</p>
{table}
<p class="footer">
  Generated by <code>evident/scripts/build_index.py</code>. Bundles produced by
  <code>evident/scripts/lock_release_replays.py</code>.
</p>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate evident/reports/index.html from per-release manifests."
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help="Directory containing per-release subdirs and where index.html is written.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_dir():
        print(f"reports dir does not exist: {out_dir}", file=sys.stderr)
        return 2

    releases: list[tuple[str, dict, dict]] = []
    for path in _release_dirs(out_dir):
        manifest = _load_manifest(path / "manifest.json")
        if manifest is None:
            continue
        releases.append((path.name, manifest, _release_summary(manifest)))

    html = _render(out_dir, releases)
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    if releases:
        print(
            f"wrote {out_dir.relative_to(REPO_ROOT)}/index.html "
            f"({len(releases)} release{'s' if len(releases) != 1 else ''})"
        )
    else:
        print(
            f"wrote {out_dir.relative_to(REPO_ROOT)}/index.html "
            f"(no release bundles yet)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
