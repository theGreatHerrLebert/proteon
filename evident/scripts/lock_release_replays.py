"""Freeze the EVIDENT claim bundle at release-tag time.

Walks every claim YAML under ``evident/claims/`` and produces an
immutable bundle at ``evident/reports/<release-tag>/``:

* ``manifest.json`` — for each claim: id, tier, oracle, command,
  artifact path, sha256 of artifact bytes, byte size, and (if the
  artifact is a JSONL with `n_total`/`n_ok`/`n_skipped`/`n_failed`
  records) a coarse run summary. This is the audit trail.
* ``<claim-id>.html`` — per-claim report, when a renderer is wired
  up for the claim's subsystem. Today only the corpus-oracle
  renderer is registered; other claims are listed in the index but
  do not yet emit a per-claim HTML.
* ``index.html`` — aggregator. One row per claim with tier, oracle,
  artifact size + sha256 prefix, a verdict, and (when present) a
  link to the per-claim HTML.

Design constraints:

* **Does not run heavy compute.** Replays produce artifacts; this
  script only freezes them. The 50K oracle run on monster3 emits a
  JSONL; this script reads it and pins it. Re-running the script
  with the same artifact tree produces a byte-identical manifest.
* **Does not mutate claim YAMLs.** ``last_verified`` in the claim
  files stays head-of-tree. The release bundle lives in its own
  directory keyed by ``--release``. Tag-time archives are immutable
  by directory, not by YAML diff.
* **Tolerant of missing artifacts.** If a claim's
  ``evidence.artifact`` does not exist at lock time, it is recorded
  in the manifest with ``status: missing`` and surfaced in the
  index. The lock does not fail the release — the index is the
  reviewer's signal of coverage.

Usage::

    python evident/scripts/lock_release_replays.py --release v0.2.0
    python evident/scripts/lock_release_replays.py --release dev --commit HEAD

The ``--commit`` flag is optional; when present, it is recorded in
the manifest verbatim. Otherwise the script reads ``git rev-parse
HEAD`` and falls back to ``unknown`` if not in a git checkout.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import claim_scoring as cs  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "evident" / "claims"
REPORTS_DIR = REPO_ROOT / "evident" / "reports"
RENDER_CORPUS = REPO_ROOT / "validation" / "report" / "render_corpus_oracle.py"
RENDER_SASA = REPO_ROOT / "validation" / "report" / "render_sasa_release.py"
RENDER_BATTLE_50K = REPO_ROOT / "validation" / "report" / "render_50k_battle_test.py"
RENDER_FOLD_PRES = REPO_ROOT / "validation" / "report" / "render_fold_preservation.py"

CHUNK = 1 << 16


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _git_head_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def _utc_today() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()


def _load_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _all_claims() -> list[tuple[pathlib.Path, dict]]:
    """Walk every YAML file under claims/ and yield each declared claim.

    A claim file's top-level shape is ``{claims: [{...}, {...}]}``.
    Some claim files declare multiple claims; each gets its own row in
    the manifest, but they share the source path.
    """
    out: list[tuple[pathlib.Path, dict]] = []
    if not CLAIMS_DIR.is_dir():
        return out
    for path in sorted(CLAIMS_DIR.glob("*.yaml")):
        doc = _load_yaml(path)
        for claim in doc.get("claims") or []:
            out.append((path, claim))
    return out


def _resolve_artifact(claim: dict) -> pathlib.Path | None:
    artifact = (claim.get("evidence") or {}).get("artifact")
    if not artifact:
        return None
    return (REPO_ROOT / str(artifact)).resolve()


# Literal-string markers that some claims use in `evidence.artifact`
# to declare "this claim has no persisted artifact by design" — most
# CI-tier oracle claims fall into this bucket because their command
# is a pytest invocation whose verdict is the exit code, not a file.
# Treating these as `missing` would inflate the index's coverage-gap
# column with claims that *correctly* don't produce files; mark them
# `ci-only` instead so the index can show real gaps separately.
_NON_PERSISTED_PREFIXES = (
    "pytest console output",
    "stdout summary",
)


def _is_persisted_artifact(claim: dict) -> bool:
    """True if the claim is supposed to produce a file we can sha256-pin."""
    artifact = (claim.get("evidence") or {}).get("artifact")
    if not artifact:
        return False
    s = str(artifact).strip().lower()
    return not any(s.startswith(p) for p in _NON_PERSISTED_PREFIXES)


def _capture_environment() -> dict[str, Any]:
    """Snapshot the lock-time runtime + dependency state.

    Embedded under ``manifest["environment"]`` so a reviewer downloading
    the bundle can pin not just the source commit and artifact sha256,
    but the exact wheels that produced the artifact. Best-effort:
    failures degrade individual fields to ``null`` rather than failing
    the lock.
    """
    import platform

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pip_freeze": None,
        "cargo_metadata_sha256": None,
    }
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if out.returncode == 0:
            lines = [l for l in out.stdout.splitlines() if l.strip()]
            env["pip_freeze"] = sorted(lines)
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        out = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if out.returncode == 0:
            env["cargo_metadata_sha256"] = (
                "sha256:" + hashlib.sha256(out.stdout.encode()).hexdigest()
            )
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return env


def _classify_record(rec: dict) -> str:
    """Map a JSONL record to ok|skipped|failed|unknown.

    Handles two schemas in proteon today:

    * 50K battle test: ``status``: ok|skipped|error|failed.
    * Corpus oracle: ``skipped``: true → skipped; otherwise both
      ``proteon`` and ``ball`` keys are present with energy dicts on
      success, or an ``error`` key on failure.
    """
    status = rec.get("status")
    if status in ("ok",):
        return "ok"
    if status in ("skipped", "skip"):
        return "skipped"
    if status in ("error", "failed", "fail"):
        return "failed"
    if rec.get("skipped"):
        return "skipped"
    if rec.get("error"):
        return "failed"
    if rec.get("proteon") is not None and rec.get("ball") is not None:
        return "ok"
    return "unknown"


def _summarise_jsonl(path: pathlib.Path) -> dict[str, int] | None:
    """Coarse summary across known JSONL record shapes."""
    if path.suffix != ".jsonl":
        return None
    n_total = n_ok = n_skipped = n_failed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_total += 1
                kind = _classify_record(rec)
                if kind == "ok":
                    n_ok += 1
                elif kind == "skipped":
                    n_skipped += 1
                elif kind == "failed":
                    n_failed += 1
    except OSError:
        return None
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
    }


# Renderer registry. Maps (subsystem, artifact_extension) → renderer
# callable. Renderers receive (claim_yaml_path, artifact_path,
# output_html_path) and are expected to write the HTML in place.
def _render_corpus_oracle(
    claim_yaml: pathlib.Path,
    artifact: pathlib.Path,
    out_html: pathlib.Path,
) -> None:
    cmd = [
        sys.executable,
        str(RENDER_CORPUS),
        "--claim",
        str(claim_yaml),
        "--artifact",
        str(artifact),
        "--output",
        str(out_html),
    ]
    subprocess.run(cmd, check=True)


def _render_sasa_release(
    claim_yaml: pathlib.Path,
    artifact: pathlib.Path,
    out_html: pathlib.Path,
) -> None:
    cmd = [
        sys.executable,
        str(RENDER_SASA),
        "--claim",
        str(claim_yaml),
        "--artifact",
        str(artifact),
        "--output",
        str(out_html),
    ]
    subprocess.run(cmd, check=True)


def _render_50k_battle_test(
    claim_yaml: pathlib.Path,
    artifact: pathlib.Path,
    out_html: pathlib.Path,
) -> None:
    cmd = [
        sys.executable,
        str(RENDER_BATTLE_50K),
        "--claim",
        str(claim_yaml),
        "--artifact",
        str(artifact),
        "--output",
        str(out_html),
    ]
    subprocess.run(cmd, check=True)


def _render_fold_preservation(
    claim_yaml: pathlib.Path,
    artifact: pathlib.Path,
    out_html: pathlib.Path,
) -> None:
    cmd = [
        sys.executable,
        str(RENDER_FOLD_PRES),
        "--claim",
        str(claim_yaml),
        "--artifact",
        str(artifact),
        "--output",
        str(out_html),
    ]
    subprocess.run(cmd, check=True)


RENDERERS: dict[tuple[str, str], Any] = {
    ("forcefield.charmm19", ".jsonl"): _render_corpus_oracle,
    ("sasa", ".json"): _render_sasa_release,
    ("pipeline.batch", ".jsonl"): _render_50k_battle_test,
    ("minimize", ".jsonl"): _render_fold_preservation,
}


def _renderer_for(claim: dict, artifact: pathlib.Path | None):
    if artifact is None:
        return None
    subsystem = claim.get("subsystem", "")
    return RENDERERS.get((subsystem, artifact.suffix))


def _verdict_from_summary(summary: dict[str, int] | None) -> str:
    if summary is None:
        return "—"
    n_ok = summary.get("n_ok", 0)
    n_total = summary.get("n_total", 0)
    if n_total == 0:
        return "empty"
    pct = 100.0 * n_ok / n_total
    return f"{n_ok}/{n_total} ok ({pct:.1f}%)"


def _score_bands(claim: dict) -> dict | None:
    """Score a claim's tolerance bands against their scoring: artifacts.

    Returns None if the claim declares no scoring: blocks. Otherwise a dict
    with a coarse ``status`` (pass|fail|error), a one-line ``summary``, and
    the per-tolerance ``detail`` for the manifest. Unlike the JSONL
    coverage verdict (``n_ok/n_total``), this asserts the recorded
    tolerance is actually met — closing the gap where the band was only
    ever checked by a human reading a number off a report.
    """
    has_scoring = any(
        isinstance(t, dict) and t.get("scoring") is not None
        for t in claim.get("tolerances") or []
    )
    if not has_scoring:
        return None
    try:
        score = cs.score_claim(claim, REPO_ROOT)
    except cs.ScoringError as e:
        return {"status": "error", "summary": str(e), "detail": []}
    detail = [
        {
            "metric": t.metric,
            "output": t.output,
            "op": t.op,
            "threshold": t.threshold,
            "observed": t.observed,
            "passed": t.passed,
            "n_in_scope": t.n_in_scope,
            "n_records": t.n_records,
        }
        for t in score.tolerances
        if t.scored
    ]
    n_pass = sum(1 for d in detail if d["passed"])
    n = len(detail)
    return {
        "status": "fail" if score.any_failed else "pass",
        "summary": f"{n_pass}/{n} bands met",
        "detail": detail,
    }


def _row_for(
    claim_yaml: pathlib.Path,
    claim: dict,
    artifact: pathlib.Path | None,
    out_dir: pathlib.Path,
    rendered_html: pathlib.Path | None,
) -> dict:
    evidence = claim.get("evidence") or {}
    row: dict[str, Any] = {
        "id": claim.get("id"),
        "title": claim.get("title"),
        "tier": claim.get("tier"),
        "subsystem": claim.get("subsystem"),
        "oracles": list(evidence.get("oracle") or []),
        "command": evidence.get("command"),
        "claim_yaml": str(claim_yaml.relative_to(REPO_ROOT)),
        "artifact": evidence.get("artifact"),
    }
    # Score tolerance bands first — the scoring: blocks carry their own clean
    # artifact paths, so a claim can be band-scored even when evidence.artifact
    # is prose (e.g. "... archived in the bundle") that doesn't resolve to a
    # file. Attached before the status branches below so it survives their
    # early returns.
    bands = _score_bands(claim)
    if bands is not None:
        row["bands"] = bands
    # CI-tier and other claims that declare "pytest console output" /
    # "stdout summary" as their artifact don't produce a file we could
    # sha256-pin, by design. Mark them `ci-only` so the index can
    # report real coverage gaps without inflating the count with
    # claims that correctly don't have a persisted artifact.
    if not _is_persisted_artifact(claim):
        row["status"] = "ci-only"
        return row
    if artifact is None:
        row["status"] = "no-artifact"
        return row
    if not artifact.exists():
        row["status"] = "missing"
        return row
    row["status"] = "locked"
    row["artifact_size"] = artifact.stat().st_size
    row["artifact_sha256"] = _sha256_file(artifact)
    summary = _summarise_jsonl(artifact)
    if summary is not None:
        row["summary"] = summary
        row["verdict"] = _verdict_from_summary(summary)
    if rendered_html is not None:
        row["report_html"] = str(rendered_html.relative_to(out_dir))
    return row


def _render_index(release_tag: str, manifest: dict, out_dir: pathlib.Path) -> None:
    """Single-file aggregator. No CSS framework, no JS — one table."""
    rows: list[str] = []
    for r in manifest["claims"]:
        report_link = ""
        if r.get("report_html"):
            report_link = (
                f'<a href="{r["report_html"]}">report</a>'
            )
        sha = r.get("artifact_sha256", "")
        sha_short = sha[:12] + "…" if sha else ""
        size = r.get("artifact_size")
        size_s = f"{size / 1024:.0f} KB" if size else ""
        verdict = r.get("verdict", "")
        oracles = ", ".join(r.get("oracles") or [])
        bands = r.get("bands")
        if bands is None:
            bands_cell = ""
        else:
            status = bands.get("status", "")
            bands_cell = (
                f'<span class="pill pill-band-{status}">{bands.get("summary", status)}</span>'
            )
        rows.append(
            "<tr>"
            f'<td><code>{r["id"]}</code></td>'
            f'<td>{r.get("tier", "")}</td>'
            f'<td>{r.get("subsystem", "")}</td>'
            f"<td>{oracles}</td>"
            f'<td>{r.get("status", "")}</td>'
            f"<td>{verdict}</td>"
            f"<td>{bands_cell}</td>"
            f"<td>{size_s}</td>"
            f"<td><code>{sha_short}</code></td>"
            f"<td>{report_link}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Proteon EVIDENT — {release_tag}</title>
<style>
  body {{ font: 14px system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: #666; margin-bottom: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }}
  th {{ background: #fafafa; font-weight: 600; border-bottom: 2px solid #ddd; }}
  td code {{ font-size: 12px; }}
  tr:hover td {{ background: #f7f9fc; }}
  .footer {{ margin-top: 3em; color: #999; font-size: 12px; }}
  .pill {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; }}
  .pill-locked {{ background: #d4f4dd; color: #1d6b32; }}
  .pill-missing {{ background: #fde2e2; color: #6b1d1d; }}
  .pill-noartifact {{ background: #eee; color: #555; }}
  .pill-band-pass {{ background: #d4f4dd; color: #1d6b32; }}
  .pill-band-fail {{ background: #fde2e2; color: #6b1d1d; font-weight: 600; }}
  .pill-band-error {{ background: #fff3cd; color: #7a5b00; }}
</style>
<h1>Proteon EVIDENT — {release_tag}</h1>
<p class="meta">
  Locked at <code>{manifest["locked_at_utc"]}</code> from commit <code>{manifest["commit"]}</code>.<br>
  {len(manifest["claims"])} claims surveyed:
  {sum(1 for r in manifest["claims"] if r.get("status") == "locked")} locked,
  {sum(1 for r in manifest["claims"] if r.get("status") == "ci-only")} CI-only (no persisted artifact),
  {sum(1 for r in manifest["claims"] if r.get("status") == "missing")} missing artifact.
</p>
<table>
  <thead>
    <tr>
      <th>Claim</th>
      <th>Tier</th>
      <th>Subsystem</th>
      <th>Oracle</th>
      <th>Status</th>
      <th>Verdict</th>
      <th>Bands</th>
      <th>Size</th>
      <th>sha256</th>
      <th>Report</th>
    </tr>
  </thead>
  <tbody>
    {chr(10).join(rows)}
  </tbody>
</table>
<p class="footer">
  Generated by <code>evident/scripts/lock_release_replays.py</code>.
  This bundle is the immutable record for release <code>{release_tag}</code>;
  edit by re-locking against a new release tag, not in place.
</p>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the EVIDENT claim bundle at release time."
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Release tag (e.g. v0.2.0) — names the output bundle directory.",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Override the commit recorded in the manifest. Default: git HEAD.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Override the output directory (default: evident/reports/<release>).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing (use for re-locks).",
    )
    parser.add_argument(
        "--fail-on-band-regression",
        action="store_true",
        help=(
            "Exit non-zero if any claim with scoring: blocks fails a recorded "
            "tolerance band. Off by default — the lock stays tolerant and the "
            "index is the reviewer's signal; turn on for a hard release gate."
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (REPORTS_DIR / args.release)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = args.commit or _git_head_commit() or "unknown"
    locked_at = _utc_today()

    claims = _all_claims()
    if not claims:
        print(f"no claims found under {CLAIMS_DIR}", file=sys.stderr)
        return 2

    rendered_paths: list[pathlib.Path] = []
    rows: list[dict] = []
    for claim_yaml, claim in claims:
        artifact = _resolve_artifact(claim)
        renderer = _renderer_for(claim, artifact)
        rendered_html: pathlib.Path | None = None
        if (
            renderer is not None
            and artifact is not None
            and artifact.exists()
        ):
            target = out_dir / f"{claim['id']}.html"
            try:
                renderer(claim_yaml, artifact, target)
                rendered_html = target
                rendered_paths.append(target)
            except subprocess.CalledProcessError as e:
                print(
                    f"renderer failed for {claim.get('id')}: {e}",
                    file=sys.stderr,
                )
        row = _row_for(claim_yaml, claim, artifact, out_dir, rendered_html)
        rows.append(row)

    manifest = {
        "release": args.release,
        "commit": commit,
        "locked_at_utc": locked_at,
        "environment": _capture_environment(),
        "claim_count": len(rows),
        "claims": rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _render_index(args.release, manifest, out_dir)

    # Refresh the top-level evident/reports/index.html so the new
    # release shows up alongside any existing ones. Best-effort: a
    # build_index.py failure should not block the lock.
    build_index = REPO_ROOT / "evident" / "scripts" / "build_index.py"
    if build_index.is_file():
        try:
            subprocess.run(
                [sys.executable, str(build_index), "--out-dir", str(out_dir.parent)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"warning: build_index.py exited {e.returncode}", file=sys.stderr)

    n_locked = sum(1 for r in rows if r.get("status") == "locked")
    n_rendered = len(rendered_paths)
    print(
        f"locked {n_locked}/{len(rows)} claims; rendered {n_rendered} HTML report(s) "
        f"into {out_dir.relative_to(REPO_ROOT)}",
    )

    scored = [r for r in rows if r.get("bands")]
    failed = [r for r in scored if r["bands"]["status"] == "fail"]
    errored = [r for r in scored if r["bands"]["status"] == "error"]
    n_band_pass = len(scored) - len(failed) - len(errored)
    print(
        f"tolerance bands: {n_band_pass}/{len(scored)} claims pass, "
        f"{len(failed)} fail, {len(errored)} unscorable"
    )
    for r in failed:
        offenders = [
            f"{d['metric']}{('('+d['output']+')') if d.get('output') else ''}="
            f"{d['observed']:.4g}{d['op']}{d['threshold']:.4g}"
            for d in r["bands"]["detail"] if not d["passed"]
        ]
        print(f"  FAIL {r['id']}: {'; '.join(offenders)}", file=sys.stderr)
    for r in errored:
        print(f"  ERROR {r['id']}: {r['bands']['summary']}", file=sys.stderr)

    if args.fail_on_band_regression and (failed or errored):
        print(
            "release gate: tolerance band regression(s) present "
            "(--fail-on-band-regression)", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
