"""Stand-alone HTML report for the SASA-vs-Biopython release claim.

Reads the SASA claim YAML + its JSON artifact (from
``validation/run_validation.py --output validation/results.json``),
renders four plots, and emits a self-contained HTML with figures
embedded as base64.

The artifact is a single JSON file with a top-level ``results`` array;
each entry has ``file`` and ``tests``, where ``tests`` is a list of
per-component test records. The SASA-specific record has::

    {"test": "sasa", "status": "pass" | "warn" | "fail" | "error",
     "details": {"biopython_total": ..., "ferritin_total": ...,
                 "relative_diff": 0.0058, "biopython_time_ms": ...,
                 "ferritin_time_ms": ..., "speedup": ..., "n_atoms": ...}}

We extract those, compute the headline (median rel-diff, p95, p99,
pass-rate, median speedup), and produce four plots: rel-diff
histogram, rel-diff vs n_atoms scatter, speedup distribution, status
breakdown.

Usage::

    python validation/report/render_sasa_release.py \\
        --claim evident/claims/sasa.yaml \\
        --artifact validation/results.json \\
        --output evident/reports/v0.1.3/proteon-sasa-vs-biopython-release-1k-pdbs.html
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import html
import json
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML required: pip install pyyaml") from e


# Tolerance bands cited in the claim YAML.
MEDIAN_BAND = 0.005   # 0.5%
HARD_BAND = 0.02      # 2% (the per-PDB CI tolerance from the sister claim)
# FreeSASA-side band: looser than the Biopython gate because Biopython
# and FreeSASA themselves disagree by ~0.5–1% on identical inputs from
# atom-radius and probe-discretisation conventions.
FREESASA_BAND = 0.02  # 2%


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _embed_png(path: pathlib.Path, alt: str) -> str:
    if not path.exists():
        return f'<div class="missing">[{alt} — figure missing]</div>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


# --------------------------------------------------------------------------
# Data extraction
# --------------------------------------------------------------------------
def _load_sasa_records(artifact_path: pathlib.Path) -> list[dict]:
    """Return one record per PDB with the SASA test's details flattened.

    Records that don't contain a SASA test (very rare, only when the
    pipeline aborted before SASA ran) are dropped. The status field
    is preserved so the renderer can show the pass/warn/fail breakdown.
    """
    with artifact_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    out: list[dict] = []
    for entry in doc.get("results", []) or []:
        sasa_t = next(
            (t for t in entry.get("tests", []) if t.get("test") == "sasa"),
            None,
        )
        if sasa_t is None:
            continue
        details = sasa_t.get("details") or {}
        out.append(
            {
                "file": entry.get("file"),
                "status": sasa_t.get("status", "unknown"),
                "biopython_total": details.get("biopython_total"),
                "proteon_total": details.get("ferritin_total")
                                  or details.get("proteon_total"),
                "relative_diff": details.get("relative_diff"),
                "biopython_time_ms": details.get("biopython_time_ms"),
                "proteon_time_ms": details.get("ferritin_time_ms")
                                    or details.get("proteon_time_ms"),
                "speedup": details.get("speedup"),
                "n_atoms": details.get("n_atoms"),
                "freesasa_total": details.get("freesasa_total"),
                "freesasa_time_ms": details.get("freesasa_time_ms"),
                "freesasa_relative_diff": details.get("freesasa_relative_diff"),
            }
        )
    return out


def _summary(records: list[dict]) -> dict[str, Any]:
    """Compute the headline numbers cited in the claim."""
    statuses = [r["status"] for r in records]
    n_total = len(records)
    n_pass = statuses.count("pass")
    n_warn = statuses.count("warn")
    n_fail = statuses.count("fail")
    n_error = statuses.count("error")

    # Successful records carry numeric rel-diffs; ignore the rest.
    diffs = np.array(
        [r["relative_diff"] for r in records if r.get("relative_diff") is not None],
        dtype=float,
    )
    speedups = np.array(
        [r["speedup"] for r in records if r.get("speedup") is not None],
        dtype=float,
    )

    fs_diffs = np.array(
        [r["freesasa_relative_diff"]
         for r in records if r.get("freesasa_relative_diff") is not None],
        dtype=float,
    )

    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_error": n_error,
        "n_with_diff": int(diffs.size),
        "median_diff": float(np.median(diffs)) if diffs.size else None,
        "p95_diff": float(np.percentile(diffs, 95)) if diffs.size else None,
        "p99_diff": float(np.percentile(diffs, 99)) if diffs.size else None,
        "max_diff": float(diffs.max()) if diffs.size else None,
        "n_under_median_band": int((diffs < MEDIAN_BAND).sum()) if diffs.size else 0,
        "n_under_hard_band": int((diffs < HARD_BAND).sum()) if diffs.size else 0,
        "median_speedup": float(np.median(speedups)) if speedups.size else None,
        # FreeSASA side: same shape, separate band.
        "n_with_freesasa_diff": int(fs_diffs.size),
        "median_freesasa_diff": float(np.median(fs_diffs)) if fs_diffs.size else None,
        "p95_freesasa_diff": float(np.percentile(fs_diffs, 95)) if fs_diffs.size else None,
        "p99_freesasa_diff": float(np.percentile(fs_diffs, 99)) if fs_diffs.size else None,
        "max_freesasa_diff": float(fs_diffs.max()) if fs_diffs.size else None,
        "n_under_freesasa_band": int((fs_diffs < FREESASA_BAND).sum()) if fs_diffs.size else 0,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _save(fig: matplotlib.figure.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_freesasa_distribution(records: list[dict], path: pathlib.Path) -> None:
    """Histogram of |proteon - FreeSASA| / FreeSASA with the 2% band marker."""
    diffs = np.array(
        [r["freesasa_relative_diff"]
         for r in records if r.get("freesasa_relative_diff") is not None],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if diffs.size == 0:
        ax.text(0.5, 0.5, "no FreeSASA data", ha="center", va="center",
                transform=ax.transAxes)
        _save(fig, path)
        return
    upper = max(FREESASA_BAND * 1.5, float(np.percentile(diffs, 99)) * 1.1)
    bins = np.linspace(0, upper, 60)
    ax.hist(np.clip(diffs, 0, upper), bins=bins, color="#9c27b0", alpha=0.8,
            edgecolor="white")
    ax.axvline(FREESASA_BAND, color="#cb2431", linestyle="--", linewidth=1.5,
               label=f"FreeSASA band <{FREESASA_BAND*100:.0f}%")
    median = float(np.median(diffs))
    ax.axvline(median, color="#0a7e8c", linestyle="-", linewidth=1.5,
               label=f"median {median*100:.2f}%")
    ax.set_xlabel("|proteon − FreeSASA| / |FreeSASA|")
    ax.set_ylabel(f"# structures (n = {diffs.size})")
    ax.set_title("Per-structure relative difference vs FreeSASA")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_rel_diff_distribution(records: list[dict], path: pathlib.Path) -> None:
    """Histogram of relative_diff with median + tolerance band markers."""
    diffs = np.array(
        [r["relative_diff"] for r in records if r.get("relative_diff") is not None],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if diffs.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return

    # Linear-scale histogram, clipped to a sensible upper bound.
    upper = max(HARD_BAND * 1.5, float(np.percentile(diffs, 99)) * 1.1)
    bins = np.linspace(0, upper, 60)
    ax.hist(np.clip(diffs, 0, upper), bins=bins, color="#0a7e8c", alpha=0.85)
    ax.axvline(MEDIAN_BAND, color="#28a745", linestyle="--", linewidth=1.5,
               label=f"median band <{MEDIAN_BAND*100:.1f}%")
    ax.axvline(HARD_BAND, color="#cb2431", linestyle="--", linewidth=1.5,
               label=f"per-PDB band <{HARD_BAND*100:.1f}%")
    median = float(np.median(diffs))
    ax.axvline(median, color="black", linewidth=1.0,
               label=f"observed median ({median*100:.3f}%)")
    ax.set_xlabel("|proteon − Biopython| / |Biopython|  (clipped)")
    ax.set_ylabel(f"# structures (n = {diffs.size})")
    ax.set_title("Per-structure relative SASA difference vs Biopython Shrake-Rupley")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_rel_diff_vs_size(records: list[dict], path: pathlib.Path) -> None:
    """Scatter of rel-diff against atom count — does scale matter?"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pts = [
        (r["n_atoms"], r["relative_diff"])
        for r in records
        if r.get("n_atoms") is not None and r.get("relative_diff") is not None
    ]
    if not pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=10, alpha=0.5, color="#0a7e8c")
    ax.axhline(MEDIAN_BAND, color="#28a745", linestyle="--", linewidth=1.0,
               label=f"median band <{MEDIAN_BAND*100:.1f}%")
    ax.axhline(HARD_BAND, color="#cb2431", linestyle="--", linewidth=1.0,
               label=f"per-PDB band <{HARD_BAND*100:.1f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_atoms (log)")
    ax.set_ylabel("relative diff (log)")
    ax.set_title("SASA rel-diff vs structure size (log-log)")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    _save(fig, path)


def plot_speedup_distribution(records: list[dict], path: pathlib.Path) -> None:
    """Distribution of proteon speedup over Biopython."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    speedups = np.array(
        [r["speedup"] for r in records if r.get("speedup") is not None],
        dtype=float,
    )
    if speedups.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return
    upper = float(np.percentile(speedups, 99)) * 1.05
    bins = np.linspace(0, upper, 50)
    ax.hist(np.clip(speedups, 0, upper), bins=bins, color="#0a7e8c", alpha=0.85)
    median = float(np.median(speedups))
    ax.axvline(median, color="black", linewidth=1.5,
               label=f"median {median:.1f}×")
    ax.axvline(1.0, color="#cb2431", linestyle="--", linewidth=1.0,
               label="parity (1×)")
    ax.set_xlabel("proteon speedup over Biopython (×)")
    ax.set_ylabel(f"# structures (n = {speedups.size})")
    ax.set_title("proteon SASA wall-time speedup vs Biopython")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_status_breakdown(records: list[dict], path: pathlib.Path) -> None:
    """Bar chart of status distribution (pass / warn / fail / error)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    counts = {k: 0 for k in ("pass", "warn", "fail", "error")}
    for r in records:
        s = r.get("status", "?")
        if s in counts:
            counts[s] += 1
    other = len(records) - sum(counts.values())
    if other:
        counts["other"] = other
    labels = list(counts.keys())
    values = list(counts.values())
    colors = {"pass": "#28a745", "warn": "#d4a017", "fail": "#cb2431",
              "error": "#7e1818", "other": "#888"}
    ax.barh(labels, values, color=[colors.get(k, "#888") for k in labels])
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v}", va="center", fontsize=10)
    ax.set_xlabel(f"# structures (n = {len(records)})")
    ax.set_title("Per-structure SASA test verdict")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, path)


# --------------------------------------------------------------------------
# Outlier table
# --------------------------------------------------------------------------
def _outlier_table(records: list[dict], top_k: int = 20) -> str:
    rows_with_diff = [r for r in records if r.get("relative_diff") is not None]
    rows_with_diff.sort(key=lambda r: -float(r["relative_diff"]))
    head = rows_with_diff[:top_k]
    if not head:
        return "<p><em>No outliers.</em></p>"
    rows_html = []
    for r in head:
        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(str(r.get('file', '')))}</code></td>"
            f"<td>{r.get('n_atoms', '')}</td>"
            f"<td>{(r.get('biopython_total') or 0):.1f}</td>"
            f"<td>{(r.get('proteon_total') or 0):.1f}</td>"
            f"<td>{(r.get('relative_diff') or 0)*100:.3f}%</td>"
            f"<td>{html.escape(str(r.get('status', '')))}</td>"
            "</tr>"
        )
    return (
        "<table class='summary'>\n"
        "<thead><tr>"
        "<th>file</th><th>n_atoms</th>"
        "<th>Biopython total (Å²)</th><th>proteon total (Å²)</th>"
        "<th>rel diff</th><th>status</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows_html) + "\n</tbody>\n"
        "</table>\n"
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def _claim_metadata(claim_doc: dict, prefer_id_substring: str) -> dict[str, Any]:
    """Extract the release-tier SASA claim from a YAML that may host both
    CI-tier and release-tier variants in the same `claims:` list."""
    claims = claim_doc.get("claims") or []
    chosen = None
    for c in claims:
        if prefer_id_substring in c.get("id", ""):
            chosen = c
            break
    if chosen is None and claims:
        chosen = claims[0]
    if chosen is None:
        raise ValueError("claim YAML contains no claims")
    return {
        "id": chosen.get("id", "(missing id)"),
        "title": chosen.get("title", "(missing title)"),
        "tier": chosen.get("tier", "?"),
        "subsystem": chosen.get("subsystem", "?"),
        "claim": (chosen.get("claim") or "").strip(),
        "command": (chosen.get("evidence") or {}).get("command", ""),
        "oracle": ", ".join((chosen.get("evidence") or {}).get("oracle") or []),
        "pinned": chosen.get("pinned_versions") or {},
        "last_verified": chosen.get("last_verified") or {},
    }


def render(
    claim_path: pathlib.Path,
    artifact_path: pathlib.Path,
    output_path: pathlib.Path,
    fig_dir: pathlib.Path | None = None,
) -> None:
    claim_doc = yaml.safe_load(claim_path.read_text(encoding="utf-8"))
    meta = _claim_metadata(claim_doc, "release-1k-pdbs")

    # Source the two release-claim median bands from the claim's own
    # tolerances so the figures and table draw the same numbers the scorer
    # enforces. All SASA tolerances share output=total_sasa, so they're keyed
    # by the scoring.value field that distinguishes the Biopython arm from the
    # FreeSASA arm. HARD_BAND stays the sister CI claim's per-PDB band.
    #
    # Reassigning module globals is safe here because the lock gate invokes
    # this renderer as a fresh subprocess per claim (one render() call per
    # process). If that ever changes to render multiple SASA claims in-process,
    # thread the bands through as parameters instead.
    global MEDIAN_BAND, FREESASA_BAND
    for _c in claim_doc.get("claims") or []:
        if "release-1k-pdbs" not in _c.get("id", ""):
            continue
        for _t in _c.get("tolerances") or []:
            if _t.get("metric") != "median_relative_error":
                continue
            _vf = (_t.get("scoring") or {}).get("value")
            _val = _t.get("value")
            if not isinstance(_val, (int, float)) or isinstance(_val, bool):
                continue
            if _vf == "relative_diff":
                MEDIAN_BAND = float(_val)
            elif _vf == "freesasa_relative_diff":
                FREESASA_BAND = float(_val)

    records = _load_sasa_records(artifact_path)
    if not records:
        raise SystemExit(f"no SASA records in artifact: {artifact_path}")
    summary = _summary(records)

    if fig_dir is None:
        fig_dir = output_path.parent / "_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_dist = fig_dir / "20_sasa_rel_diff.png"
    fig_size = fig_dir / "21_sasa_size_scatter.png"
    fig_spd = fig_dir / "22_sasa_speedup.png"
    fig_stat = fig_dir / "23_sasa_status.png"
    fig_fs = fig_dir / "24_sasa_freesasa_diff.png"
    plot_rel_diff_distribution(records, fig_dist)
    plot_rel_diff_vs_size(records, fig_size)
    plot_speedup_distribution(records, fig_spd)
    plot_status_breakdown(records, fig_stat)
    plot_freesasa_distribution(records, fig_fs)

    artifact_sha = _sha256_file(artifact_path)
    today = datetime.date.today().isoformat()

    last_verified = meta["last_verified"] or {}
    lv_lines = []
    for k in ("commit", "date", "value", "corpus_sha"):
        v = last_verified.get(k)
        lv_lines.append(
            f"<tr><th>{k}</th><td><code>"
            f"{html.escape(str(v) if v is not None else '—')}</code></td></tr>"
        )
    lv_table = "<table class='kv'>" + "".join(lv_lines) + "</table>"

    pinned_lines = "".join(
        f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in meta["pinned"].items()
    ) or "<tr><td colspan='2'><em>not pinned</em></td></tr>"

    title = html.escape(meta["title"])
    claim_text = html.escape(meta["claim"])
    cmd = html.escape(meta["command"])
    oracle = html.escape(meta["oracle"])

    def _pct(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:.3f}%"

    def _opt(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EVIDENT — {title}</title>
<style>
:root {{
  --proteon: #0a7e8c;
  --ink:     #222;
  --muted:   #666;
  --bg:      #fafafa;
  --card:    #ffffff;
  --rule:    #e4e4e4;
  --pass:    #28a745;
  --fail:    #cb2431;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 15px; line-height: 1.55; color: var(--ink); background: var(--bg);
}}
main {{ max-width: 1180px; margin: 0 auto; padding: 40px 32px 80px; }}
header {{ border-bottom: 3px solid var(--proteon); margin-bottom: 28px; padding-bottom: 16px; }}
h1 {{ font-size: 28px; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }}
.subtitle {{ color: var(--muted); font-size: 14px; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: 11px; font-weight: 600; color: white;
  background: var(--proteon); margin-left: 6px; vertical-align: 2px;
}}
section {{ background: var(--card); border: 1px solid var(--rule);
          border-radius: 8px; padding: 24px 28px; margin: 18px 0; }}
h2 {{ font-size: 20px; font-weight: 700; margin: 0 0 14px; color: var(--proteon); }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
th, td {{ border: 1px solid var(--rule); padding: 6px 10px; text-align: left;
         vertical-align: top; }}
th {{ background: #f6f8fa; font-weight: 600; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
table.kv th {{ width: 130px; background: #fafafa; }}
table.summary {{ font-variant-numeric: tabular-nums; }}
img {{ max-width: 100%; height: auto; border: 1px solid var(--rule);
      border-radius: 4px; margin: 12px 0; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
       font-size: 12.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
.muted {{ color: var(--muted); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
               gap: 12px; margin: 16px 0; }}
.summary-card {{ background: #f6f8fa; border-radius: 6px; padding: 14px;
                text-align: center; }}
.summary-card .num {{ font-size: 24px; font-weight: 700; color: var(--proteon); }}
.summary-card .lbl {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
.missing {{ color: var(--fail); font-style: italic; padding: 8px; }}
footer {{ margin-top: 40px; color: var(--muted); font-size: 12px;
         border-top: 1px solid var(--rule); padding-top: 12px; }}
</style>
</head>
<body>
<main>
<header>
  <h1>{title}<span class="badge">{html.escape(meta['tier'])}</span></h1>
  <div class="subtitle">
    Claim id <code>{html.escape(meta['id'])}</code> ·
    Subsystem <code>{html.escape(meta['subsystem'])}</code> ·
    Oracle <code>{oracle}</code> ·
    Generated {today}
  </div>
</header>

<section>
  <h2>Claim</h2>
  <p>{claim_text}</p>
  <p class="muted"><strong>Replay command:</strong> <code>{cmd}</code></p>
</section>

<section>
  <h2>Headline</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="num">{summary['n_total']}</div>
      <div class="lbl">structures evaluated</div>
    </div>
    <div class="summary-card">
      <div class="num">{_pct(summary['median_diff'])}</div>
      <div class="lbl">median rel diff (band &lt;{MEDIAN_BAND*100:.1f}%)</div>
    </div>
    <div class="summary-card">
      <div class="num">{_pct(summary['p95_diff'])}</div>
      <div class="lbl">p95 rel diff</div>
    </div>
    <div class="summary-card">
      <div class="num">{_opt(summary['median_speedup'], '.1f')}×</div>
      <div class="lbl">median proteon speedup</div>
    </div>
  </div>
  <table class="kv">
    <tr><th>pass / warn / fail / error</th>
        <td><code>{summary['n_pass']} / {summary['n_warn']} / {summary['n_fail']} / {summary['n_error']}</code></td></tr>
    <tr><th>vs Biopython: median &lt; {MEDIAN_BAND*100:.1f}% band</th>
        <td><code>{summary['n_under_median_band']} / {summary['n_with_diff']}</code></td></tr>
    <tr><th>vs Biopython: per-PDB &lt; {HARD_BAND*100:.1f}% band</th>
        <td><code>{summary['n_under_hard_band']} / {summary['n_with_diff']}</code></td></tr>
    <tr><th>vs Biopython: p99 / max rel diff</th>
        <td><code>{_pct(summary['p99_diff'])} / {_pct(summary['max_diff'])}</code></td></tr>
    <tr><th>vs FreeSASA: median rel diff</th>
        <td><code>{_pct(summary['median_freesasa_diff'])} (band &lt;{FREESASA_BAND*100:.0f}%)</code></td></tr>
    <tr><th>vs FreeSASA: p95 / p99 / max</th>
        <td><code>{_pct(summary['p95_freesasa_diff'])} / {_pct(summary['p99_freesasa_diff'])} / {_pct(summary['max_freesasa_diff'])}</code></td></tr>
    <tr><th>vs FreeSASA: per-PDB &lt; {FREESASA_BAND*100:.0f}% band</th>
        <td><code>{summary['n_under_freesasa_band']} / {summary['n_with_freesasa_diff']}</code></td></tr>
  </table>
</section>

<section>
  <h2>Per-structure relative-difference vs FreeSASA</h2>
  <p class="muted">Histogram of |proteon − FreeSASA| / |FreeSASA| across the
  {summary['n_with_freesasa_diff']} structures with a usable FreeSASA result.
  Red dashed line = the {FREESASA_BAND*100:.0f}% release band; the band is looser
  than the Biopython gate because the two reference Shrake-Rupley
  implementations disagree by ~0.5–1% on each other from atom-radius
  conventions. Two-oracle agreement at this level is much stronger evidence
  of correctness than agreement against either alone.</p>
  {_embed_png(fig_fs, "Per-structure rel-diff vs FreeSASA")}
</section>

<section>
  <h2>Per-structure relative-difference distribution</h2>
  <p class="muted">Histogram of |proteon − Biopython| / |Biopython| across the
  {summary['n_with_diff']} structures with a usable SASA test record. Green dashed
  line = the {MEDIAN_BAND*100:.1f}% release median band; red dashed line = the
  {HARD_BAND*100:.1f}% per-PDB CI band. Counts are clipped at the 99th-percentile
  rel-diff for visibility; outliers are listed below.</p>
  {_embed_png(fig_dist, "Per-structure rel-diff distribution")}
</section>

<section>
  <h2>Rel-diff vs structure size</h2>
  <p class="muted">Log-log scatter of rel-diff against atom count.
  A flat cloud = no scale dependence (proteon's Shrake-Rupley converges
  uniformly across small to large structures); a sloping cloud would
  flag a sphere-discretisation or boundary-effect issue.</p>
  {_embed_png(fig_size, "Rel-diff vs n_atoms")}
</section>

<section>
  <h2>Wall-time speedup distribution</h2>
  <p class="muted">Per-structure proteon-vs-Biopython wall-time ratio.
  Median speedup is the headline number; the right tail comes from
  larger structures where the Rust kernel's vectorised inner loop
  amortises better against Biopython's Python overhead.</p>
  {_embed_png(fig_spd, "Speedup distribution")}
</section>

<section>
  <h2>Verdict breakdown</h2>
  <p class="muted">Per-structure status from
  <code>validation/run_validation.py::test_sasa</code>: pass = within band,
  warn = within hard band but above median band, fail = outside hard band,
  error = SASA stage didn't complete. Loading-stage failures cascade into
  fail; the claim's pass-rate metric excludes those.</p>
  {_embed_png(fig_stat, "Status breakdown")}
</section>

<section>
  <h2>Top-{min(20, summary['n_with_diff'])} outlier structures</h2>
  <p class="muted">Structures with the largest |proteon − Biopython|
  rel-diffs. These are the diagnostic targets — if a single residue
  type or chain topology shows up repeatedly, that's an upstream typing
  or sphere-discretisation gap.</p>
  {_outlier_table(records, top_k=20)}
</section>

<section>
  <h2>Pinned versions</h2>
  <table class="kv">{pinned_lines}</table>
</section>

<section>
  <h2>last_verified</h2>
  {lv_table}
  <p class="muted">Populated by <code>lock_release_replays.py</code> at
  release-tag time; ``value`` cites the latest median rel-diff.</p>
</section>

<footer>
  EVIDENT SASA-vs-Biopython release report · proteon ·
  artifact <code>{html.escape(artifact_path.name)}</code>
  (sha256 <code>{artifact_sha}</code>)
</footer>
</main>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")
    print(
        f"Rendered {output_path} "
        f"(n_total={summary['n_total']}, "
        f"pass={summary['n_pass']}, warn={summary['n_warn']}, "
        f"fail={summary['n_fail']}, error={summary['n_error']}, "
        f"median={_pct(summary['median_diff'])})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--claim", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--fig-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    render(args.claim, args.artifact, args.output, args.fig_dir)


if __name__ == "__main__":
    main()
