"""Stand-alone HTML report for the fold-preservation release claims.

Reads a joined JSONL artifact (one record per PDB, with both proteon
and openmm fields plus a pre-computed `tm_diff`) produced by
`validation/fold_preservation/join_fold_preservation.py`. Same general
shape as render_corpus_oracle.py / render_sasa_release.py:

* Headline cards: n attempted, n ok, proteon median TM, OpenMM median
  TM, median tm_diff
* TM-score distribution (per side, overlaid)
* tm_diff histogram (clipped to ± p99 percentile)
* TM scatter: proteon vs openmm, log-y on the tail
* Top-20 outlier table (largest |tm_diff|)

Usage::

    python validation/report/render_fold_preservation.py \\
        --claim evident/claims/fold_preservation_charmm.yaml \\
        --artifact validation/fold_preservation/charmm_pair_1k.jsonl \\
        --output evident/reports/v0.1.4/proteon-charmm19-fold-preservation-vs-openmm-release-1k-pdbs.html

The renderer is registered in evident/scripts/lock_release_replays.py
under (subsystem="minimize", suffix=".jsonl") so the locker
auto-dispatches it for both the CHARMM and AMBER fold-preservation
claims.
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


def _load_records(artifact_path: pathlib.Path) -> list[dict]:
    out: list[dict] = []
    with artifact_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _summary(records: list[dict]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("skipped") and "proteon" in r and "openmm" in r]
    n_total = len(records)
    n_ok = len(ok)
    n_skip = n_total - n_ok

    proteon_tm = np.array([r["proteon"]["tm_score"] for r in ok])
    openmm_tm = np.array([r["openmm"]["tm_score"] for r in ok])
    proteon_rmsd = np.array([r["proteon"]["rmsd"] for r in ok])
    openmm_rmsd = np.array([r["openmm"]["rmsd"] for r in ok])
    tm_diffs = np.array([r["tm_diff"] for r in ok])

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "pass_rate": n_ok / n_total if n_total else 0.0,
        "proteon_tm_median": float(np.median(proteon_tm)) if n_ok else None,
        "proteon_tm_mean": float(np.mean(proteon_tm)) if n_ok else None,
        "openmm_tm_median": float(np.median(openmm_tm)) if n_ok else None,
        "openmm_tm_mean": float(np.mean(openmm_tm)) if n_ok else None,
        "proteon_rmsd_median": float(np.median(proteon_rmsd)) if n_ok else None,
        "openmm_rmsd_median": float(np.median(openmm_rmsd)) if n_ok else None,
        "tm_diff_median": float(np.median(tm_diffs)) if n_ok else None,
        "tm_diff_p95": float(np.percentile(np.abs(tm_diffs), 95)) if n_ok else None,
        "tm_diff_p99": float(np.percentile(np.abs(tm_diffs), 99)) if n_ok else None,
        "tm_diff_max_abs": float(np.max(np.abs(tm_diffs))) if n_ok else None,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _save(fig: matplotlib.figure.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _ok_records(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("skipped") and "proteon" in r and "openmm" in r]


def plot_tm_distributions(records: list[dict], path: pathlib.Path,
                          proteon_label: str, openmm_label: str) -> None:
    """Per-side TM-score distribution, overlaid."""
    ok = _ok_records(records)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not ok:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return
    p = np.array([r["proteon"]["tm_score"] for r in ok])
    o = np.array([r["openmm"]["tm_score"] for r in ok])
    bins = np.linspace(0.5, 1.0, 60)
    ax.hist(p, bins=bins, alpha=0.55, label=proteon_label, color="#0a7e8c")
    ax.hist(o, bins=bins, alpha=0.55, label=openmm_label, color="#cb2431")
    ax.axvline(np.median(p), color="#0a7e8c", linestyle="--", linewidth=1.2,
               label=f"proteon median {np.median(p):.4f}")
    ax.axvline(np.median(o), color="#cb2431", linestyle="--", linewidth=1.2,
               label=f"openmm median {np.median(o):.4f}")
    ax.set_xlabel("TM-score (input vs minimized CA trace)")
    ax.set_ylabel(f"# structures (n_ok = {len(ok)})")
    ax.set_title("Per-side TM-score distribution after 100 minimizer steps")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_tm_diff_histogram(
    records: list[dict], path: pathlib.Path, band: float = 0.01
) -> None:
    """Distribution of (openmm − proteon) per-PDB TM-score diff.

    ``band`` is the claim's drift tolerance (drawn as the ±band line), passed
    from the claim YAML so the figure matches the enforced number.
    """
    ok = _ok_records(records)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not ok:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return
    diffs = np.array([r["tm_diff"] for r in ok])
    p99 = float(np.percentile(np.abs(diffs), 99))
    upper = max(p99 * 1.1, 0.02)
    bins = np.linspace(-upper, upper, 60)
    ax.hist(np.clip(diffs, -upper, upper), bins=bins, color="#0a7e8c", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0, label="no diff")
    median = float(np.median(diffs))
    ax.axvline(median, color="#cb2431", linewidth=1.5,
               label=f"median {median:+.4f}")
    ax.axvline(band, color="#d4a017", linestyle="--", linewidth=1.0,
               label=f"claim band ±{band:g}")
    ax.axvline(-band, color="#d4a017", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"openmm.tm_score − proteon.tm_score (clipped to ±p99 = ±{p99:.4f})")
    ax.set_ylabel(f"# structures (n_ok = {len(ok)})")
    ax.set_title("Per-PDB TM-score diff distribution")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_tm_scatter(records: list[dict], path: pathlib.Path,
                    proteon_label: str, openmm_label: str) -> None:
    """Scatter of proteon TM vs openmm TM."""
    ok = _ok_records(records)
    fig, ax = plt.subplots(figsize=(7, 7))
    if not ok:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, path)
        return
    p = np.array([r["proteon"]["tm_score"] for r in ok])
    o = np.array([r["openmm"]["tm_score"] for r in ok])
    ax.scatter(p, o, s=10, alpha=0.45, color="#0a7e8c")
    lo = float(min(p.min(), o.min()))
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], color="#cb2431", linestyle="--",
            linewidth=1.0, label="proteon = openmm")
    ax.set_xlim(lo - 0.02, 1.0)
    ax.set_ylim(lo - 0.02, 1.0)
    ax.set_xlabel(f"{proteon_label} — TM-score")
    ax.set_ylabel(f"{openmm_label} — TM-score")
    ax.set_title("Per-PDB TM-score: proteon vs OpenMM")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    _save(fig, path)


# --------------------------------------------------------------------------
# Outlier table
# --------------------------------------------------------------------------
def _outlier_table(records: list[dict], top_k: int = 20) -> str:
    ok = _ok_records(records)
    ok.sort(key=lambda r: -abs(float(r["tm_diff"])))
    head = ok[:top_k]
    if not head:
        return "<p><em>No outliers (no ok records).</em></p>"
    rows_html = []
    for r in head:
        p = r["proteon"]
        o = r["openmm"]
        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(str(r.get('pdb', '')))}</code></td>"
            f"<td>{r.get('n_ca', '')}</td>"
            f"<td>{p['tm_score']:.4f}</td>"
            f"<td>{o['tm_score']:.4f}</td>"
            f"<td>{r['tm_diff']:+.4f}</td>"
            f"<td>{p['rmsd']:.2f}</td>"
            f"<td>{o['rmsd']:.2f}</td>"
            "</tr>"
        )
    return (
        "<table class='summary'>\n"
        "<thead><tr>"
        "<th>file</th><th>n_ca</th>"
        "<th>proteon TM</th><th>openmm TM</th><th>tm_diff</th>"
        "<th>proteon RMSD (Å)</th><th>openmm RMSD (Å)</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows_html) + "\n</tbody>\n"
        "</table>\n"
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def _claim_metadata(claim_doc: dict) -> dict[str, Any]:
    claims = claim_doc.get("claims") or []
    if not claims:
        raise ValueError("claim YAML contains no claims")
    c = claims[0]
    pinned = c.get("pinned_versions") or {}
    return {
        "id": c.get("id", "(missing id)"),
        "title": c.get("title", "(missing title)"),
        "tier": c.get("tier", "?"),
        "subsystem": c.get("subsystem", "?"),
        "claim": (c.get("claim") or "").strip(),
        "command": (c.get("evidence") or {}).get("command", ""),
        "oracle": ", ".join((c.get("evidence") or {}).get("oracle") or []),
        "pinned": pinned,
        "last_verified": c.get("last_verified") or {},
        "proteon_label": pinned.get("runner_proteon", "proteon"),
        "openmm_label": pinned.get("runner_openmm", "OpenMM"),
    }


def _label_from_id(claim_id: str) -> tuple[str, str]:
    """Pick proteon_label / openmm_label from the claim id."""
    cid = claim_id.lower()
    if "charmm" in cid:
        return "proteon CHARMM19+EEF1", "OpenMM CHARMM36+OBC2"
    if "amber" in cid:
        return "proteon AMBER96", "OpenMM AMBER96"
    return "proteon", "OpenMM"


def render(
    claim_path: pathlib.Path,
    artifact_path: pathlib.Path,
    output_path: pathlib.Path,
    fig_dir: pathlib.Path | None = None,
) -> None:
    claim_doc = yaml.safe_load(claim_path.read_text(encoding="utf-8"))
    meta = _claim_metadata(claim_doc)
    proteon_label, openmm_label = _label_from_id(meta["id"])
    # Drift band from the claim's own tolerance, so the ±band line on the
    # tm_diff figure matches the number the scorer enforces.
    drift_band = 0.01
    for _c in claim_doc.get("claims") or []:
        for _t in _c.get("tolerances") or []:
            if _t.get("metric") == "drift" and isinstance(_t.get("value"), (int, float)):
                drift_band = float(_t["value"])

    records = _load_records(artifact_path)
    if not records:
        raise SystemExit(f"no records in artifact: {artifact_path}")
    summary = _summary(records)

    if fig_dir is None:
        fig_dir = output_path.parent / "_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    slug = meta["id"].replace("proteon-", "").replace("/", "_")
    fig_dist = fig_dir / f"40_fold_{slug}_tm_distributions.png"
    fig_diff = fig_dir / f"41_fold_{slug}_tm_diff.png"
    fig_scat = fig_dir / f"42_fold_{slug}_scatter.png"
    plot_tm_distributions(records, fig_dist, proteon_label, openmm_label)
    plot_tm_diff_histogram(records, fig_diff, band=drift_band)
    plot_tm_scatter(records, fig_scat, proteon_label, openmm_label)

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

    def _fmt(v: float | None, fmt: str = ".4f") -> str:
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
    Oracle <code>{html.escape(meta['oracle'])}</code> ·
    Generated {today}
  </div>
</header>

<section>
  <h2>Claim</h2>
  <p>{claim_text}</p>
  <p class="muted"><strong>Replay command (verify-from-locked-JSONL today; full re-produce blocked on #42 + #48):</strong>
  <code>{cmd}</code></p>
</section>

<section>
  <h2>Headline</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="num">{summary['n_ok']:,}</div>
      <div class="lbl">paired records ({summary['pass_rate']*100:.1f}% pass)</div>
    </div>
    <div class="summary-card">
      <div class="num">{_fmt(summary['proteon_tm_median'])}</div>
      <div class="lbl">{html.escape(proteon_label)} median TM</div>
    </div>
    <div class="summary-card">
      <div class="num">{_fmt(summary['openmm_tm_median'])}</div>
      <div class="lbl">{html.escape(openmm_label)} median TM</div>
    </div>
    <div class="summary-card">
      <div class="num">{summary['tm_diff_median']:+.4f}</div>
      <div class="lbl">median tm_diff (openmm − proteon)</div>
    </div>
  </div>
  <table class="kv">
    <tr><th>{html.escape(proteon_label)}</th>
        <td><code>median TM {_fmt(summary['proteon_tm_median'])}, mean {_fmt(summary['proteon_tm_mean'])},
        RMSD median {_fmt(summary['proteon_rmsd_median'], '.2f')} Å</code></td></tr>
    <tr><th>{html.escape(openmm_label)}</th>
        <td><code>median TM {_fmt(summary['openmm_tm_median'])}, mean {_fmt(summary['openmm_tm_mean'])},
        RMSD median {_fmt(summary['openmm_rmsd_median'], '.2f')} Å</code></td></tr>
    <tr><th>tm_diff abs (p95 / p99 / max)</th>
        <td><code>{_fmt(summary['tm_diff_p95'])} / {_fmt(summary['tm_diff_p99'])}
        / {_fmt(summary['tm_diff_max_abs'])}</code></td></tr>
    <tr><th>n_skip / n_total</th>
        <td><code>{summary['n_skip']} / {summary['n_total']}</code></td></tr>
  </table>
</section>

<section>
  <h2>Per-side TM-score distribution</h2>
  <p class="muted">Histograms of input-vs-minimized TM-score per side.
  Both implementations cluster near 1.0 (fold preserved); the small
  shift between them is what the headline median diff captures.</p>
  {_embed_png(fig_dist, "TM-score distribution per side")}
</section>

<section>
  <h2>Per-PDB TM-score diff (openmm − proteon)</h2>
  <p class="muted">Centered on the median diff; clipped to ±p99 for
  visibility. The dashed gold lines mark the claim's ±0.01 tolerance band.
  Most PDBs sit comfortably inside the band; outliers are listed below.</p>
  {_embed_png(fig_diff, "tm_diff histogram")}
</section>

<section>
  <h2>Per-PDB TM-score scatter</h2>
  <p class="muted">{html.escape(proteon_label)} TM-score (x-axis) vs
  {html.escape(openmm_label)} TM-score (y-axis). Points on the dashed
  diagonal = the two implementations agree perfectly on that PDB. Points
  above the diagonal = OpenMM held the fold tighter; below = proteon did.</p>
  {_embed_png(fig_scat, "TM-score scatter")}
</section>

<section>
  <h2>Top-{min(20, summary['n_ok'])} outlier structures</h2>
  <p class="muted">Largest |tm_diff| structures. Diagnostic targets — if a
  small set of residue compositions or chain topologies dominates the
  outliers, that's the next force-field-implementation bug to chase.</p>
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
  release-tag time. <code>value</code> cites the proteon median TM.</p>
</section>

<footer>
  EVIDENT fold-preservation report · proteon ·
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
        f"(n_total={summary['n_total']}, ok={summary['n_ok']}, "
        f"pass_rate={summary['pass_rate']*100:.1f}%, "
        f"proteon_median_tm={_fmt(summary['proteon_tm_median'])}, "
        f"tm_diff_median={summary['tm_diff_median']:+.4f})"
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
