"""Stand-alone HTML report for the CHARMM corpus oracle.

Reads a claim YAML + its artifact JSONL, renders the 4 corpus-oracle
plots, and emits a self-contained HTML (figures embedded as base64).
This is the Phase-1 validation that the renderer pipeline works
end-to-end on the existing 1k JSONL.

Phase 2 will fold this into a generic claim-walk renderer in
`build_report.py`. For now it's deliberately minimal so the EVIDENT
report shape can land in the repo and be iterated on.

Usage:

    python validation/report/render_corpus_oracle.py \\
        --claim evident/claims/forcefield_charmm19_ball_corpus.yaml \\
        --artifact validation/charmm19_eef1_ball_oracle.jsonl \\
        --output evident/reports/dev-snapshot.html

The output is committed to the repo at the named path so reviewers
can read it without re-running anything. The artifact JSONL is gitignored
(too large for in-repo storage long-term — see the EVIDENT design doc
for the GH-Release-attachments plan).
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

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

import numpy as np

from corpus_oracle_plots import (
    BANDS,
    LABELS,
    COMPONENT_ORDER,
    load_jsonl,
    plot_correlations,
    plot_error_distribution,
    plot_runtime,
    render_outlier_table,
    successful_records,
)


def _embed_png(path: pathlib.Path, alt: str) -> str:
    if not path.exists():
        return f'<div class="missing">[{alt} — figure missing]</div>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _summary_table(records: list[dict]) -> str:
    """Per-component summary (median / p95 / p99 / max / pass-rate)."""
    ok = successful_records(records)
    if not ok:
        return "<p><em>No successful records.</em></p>"

    rows = []
    for comp in COMPONENT_ORDER:
        vals = np.array([
            r["rel_diff"].get(comp, np.nan) for r in ok
        ])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        band = BANDS.get(comp, 0.025)
        n_pass = (vals < band).sum()
        rate = n_pass / vals.size * 100
        rows.append(
            "<tr>"
            f"<td>{LABELS.get(comp, comp)}</td>"
            f"<td>{np.median(vals)*100:.3f}%</td>"
            f"<td>{np.percentile(vals, 95)*100:.3f}%</td>"
            f"<td>{np.percentile(vals, 99)*100:.3f}%</td>"
            f"<td>{vals.max()*100:.3f}%</td>"
            f"<td>{n_pass}/{vals.size}</td>"
            f'<td>{rate:.1f}%<div class="bar"><div class="bar-fill" '
            f'style="width:{rate:.1f}%"></div></div></td>'
            f"<td><code>&lt;&nbsp;{band*100:.1f}%</code></td>"
            "</tr>"
        )

    return (
        "<table class='summary'>\n"
        "<thead><tr>"
        "<th>Component</th><th>Median</th><th>P95</th><th>P99</th><th>Max</th>"
        "<th>Pass</th><th>Rate</th><th>Band</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n"
        "</table>\n"
    )


def _claim_metadata(claim_doc: dict) -> dict[str, Any]:
    """Extract the bits we cite at the top of the report."""
    if "claims" in claim_doc and isinstance(claim_doc["claims"], list):
        if not claim_doc["claims"]:
            raise ValueError("claims list is empty")
        c = claim_doc["claims"][0]
    else:
        c = claim_doc
    return {
        "id": c.get("id", "(missing id)"),
        "title": c.get("title", "(missing title)"),
        "tier": c.get("tier", "?"),
        "subsystem": c.get("subsystem", "?"),
        "claim": c.get("claim", "").strip(),
        "command": c.get("evidence", {}).get("command", ""),
        "oracle": ", ".join(c.get("evidence", {}).get("oracle", [])),
        "pinned": c.get("pinned_versions", {}),
        "last_verified": c.get("last_verified", {}),
    }


def render(
    claim_path: pathlib.Path,
    artifact_path: pathlib.Path,
    output_path: pathlib.Path,
    fig_dir: pathlib.Path | None = None,
) -> None:
    claim_doc = yaml.safe_load(claim_path.read_text())
    meta = _claim_metadata(claim_doc)

    records = load_jsonl(artifact_path)
    if not records:
        raise SystemExit(f"empty artifact: {artifact_path}")

    if fig_dir is None:
        fig_dir = output_path.parent / "_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_err = fig_dir / "10_corpus_error_distribution.png"
    fig_run = fig_dir / "11_corpus_runtime.png"
    fig_cor = fig_dir / "12_corpus_correlations.png"
    plot_error_distribution(records, fig_err)
    plot_runtime(records, fig_run)
    plot_correlations(records, fig_cor)

    n_total = len(records)
    n_ok = len(successful_records(records))
    n_skip = sum(1 for r in records if "skipped" in r)
    n_fail = n_total - n_ok - n_skip

    artifact_sha = _sha256_file(artifact_path)
    today = datetime.date.today().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_verified = meta["last_verified"] or {}
    lv_lines = []
    for k in ("commit", "date", "value", "corpus_sha"):
        v = last_verified.get(k)
        lv_lines.append(f"<tr><th>{k}</th><td><code>{html.escape(str(v) if v else '—')}</code></td></tr>")
    lv_table = "<table class='kv'>" + "".join(lv_lines) + "</table>"

    pinned_lines = "".join(
        f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in meta["pinned"].items()
    ) or "<tr><td colspan='2'><em>not pinned</em></td></tr>"

    title = html.escape(meta["title"])
    claim_text = html.escape(meta["claim"])
    cmd = html.escape(meta["command"])

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
table.summary .bar {{ height: 6px; background: #e9ecef; border-radius: 3px;
                     margin-top: 3px; overflow: hidden; }}
table.summary .bar-fill {{ height: 100%; background: var(--pass); }}
img {{ max-width: 100%; height: auto; border: 1px solid var(--rule);
      border-radius: 4px; margin: 12px 0; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
       font-size: 12.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
.muted {{ color: var(--muted); }}
.kv {{ font-size: 13px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
               gap: 12px; margin: 16px 0; }}
.summary-card {{ background: #f6f8fa; border-radius: 6px; padding: 14px;
                text-align: center; }}
.summary-card .num {{ font-size: 24px; font-weight: 700; color: var(--proteon); }}
.summary-card .lbl {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
.missing {{ color: var(--fail); font-style: italic; padding: 8px; }}
footer {{ margin-top: 40px; color: var(--muted); font-size: 12px;
         border-top: 1px solid var(--rule); padding-top: 12px; }}
.note {{ background: #fff8dc; border-left: 3px solid #d4a017;
        padding: 10px 14px; margin: 12px 0; font-size: 13px; }}
</style>
</head>
<body>
<main>
<header>
  <h1>{title}<span class="badge">{html.escape(meta['tier'])}</span></h1>
  <div class="subtitle">
    Claim id <code>{html.escape(meta['id'])}</code> ·
    Subsystem <code>{html.escape(meta['subsystem'])}</code> ·
    Oracle <code>{html.escape(meta['oracle'])}</code> ·
    Generated {today}
  </div>
</header>

<section>
  <h2>Claim</h2>
  <p>{claim_text}</p>
  <p class="muted"><strong>Replay command:</strong> <code>{cmd}</code></p>
  <div class="note">
    <strong>Phase-1 dev snapshot</strong> — this report is rendered
    against an in-repo JSONL artifact (<code>{html.escape(artifact_path.name)}</code>,
    sha256 <code>{artifact_sha[:16]}…</code>). Phase 2 wires this into the
    release-tag flow so future reports cite a frozen image digest.
  </div>
</section>

<section>
  <h2>Run summary</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="num">{n_total}</div>
      <div class="lbl">PDB records</div>
    </div>
    <div class="summary-card">
      <div class="num">{n_ok}</div>
      <div class="lbl">successful comparisons</div>
    </div>
    <div class="summary-card">
      <div class="num">{n_skip}</div>
      <div class="lbl">skipped (BALL typing / non-AA)</div>
    </div>
    <div class="summary-card">
      <div class="num">{n_fail}</div>
      <div class="lbl">errors / worker crashes</div>
    </div>
  </div>
</section>

<section>
  <h2>Per-component summary</h2>
  {_summary_table(records)}
</section>

<section>
  <h2>Per-component error distribution</h2>
  <p class="muted">Histogram of |proteon − BALL| / |BALL| per component
  across the {n_ok} successful structures. Red dashed line marks the
  per-component oracle band (the same number the unit-test asserts).
  Tall left bar = "almost all structures within band".</p>
  {_embed_png(fig_err, "Per-component error distribution")}
</section>

<section>
  <h2>Per-PDB wall time</h2>
  <p class="muted">Wall time per PDB and how it scales with structure size.
  Dominated by BALL's CharmmFF setup and the 1-4 / non-bonded list build;
  scales roughly linearly with atom count.</p>
  {_embed_png(fig_run, "Wall time distribution and scaling")}
</section>

<section>
  <h2>Per-component error correlation</h2>
  <p class="muted">Pearson r of rel-diff between components across the
  corpus. Strong off-diagonal correlation suggests a shared upstream
  cause (e.g. atom-typing failure cascading to multiple terms);
  near-zero correlation means the per-component errors are independent.</p>
  {_embed_png(fig_cor, "Per-component rel-diff correlation matrix")}
</section>

<section>
  <h2>Top-20 outlier structures</h2>
  <p class="muted">Structures with the largest sum-of-rel-diffs across
  components. Each row's "worst component" / "2nd-worst" diagnostic
  points at the failure mode — outliers usually have one dominant
  off-band component, not uniform drift.</p>
  {render_outlier_table(records, top_k=20)}
</section>

<section>
  <h2>Pinned versions</h2>
  <table class="kv">{pinned_lines}</table>
</section>

<section>
  <h2>last_verified</h2>
  {lv_table}
  <p class="muted">Phase 2's <code>lock_release_replays.py</code> populates
  this block at release tag time. Currently empty until that orchestrator
  lands.</p>
</section>

<footer>
  EVIDENT corpus-oracle dev snapshot · proteon ·
  artifact <code>{html.escape(artifact_path.name)}</code>
  (sha256 <code>{artifact_sha}</code>)
</footer>
</main>
</body>
</html>
"""

    output_path.write_text(html_doc)
    print(
        f"Rendered {output_path} "
        f"(n_total={n_total}, ok={n_ok}, skip={n_skip}, fail={n_fail})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--claim", type=pathlib.Path, required=True,
                        help="Path to the claim YAML.")
    parser.add_argument("--artifact", type=pathlib.Path, required=True,
                        help="Path to the JSONL artifact produced by the replay.")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="Output HTML path.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=None,
                        help="Directory for intermediate PNGs (default: <output>/_figures).")
    args = parser.parse_args()
    render(args.claim, args.artifact, args.output, args.fig_dir)


if __name__ == "__main__":
    main()
