"""Build the single-file HTML validation report for proteon.

Generates report.html with:
  * Headline numbers
  * Force-field math oracle (crambin triangulation)
  * Fold preservation — CHARMM family
  * Fold preservation — AMBER family
  * Throughput comparison
  * Input robustness analysis
  * Reproducibility appendix

Images are embedded as base64 so the HTML is fully self-contained.
"""
from __future__ import annotations

import base64
import datetime
import json
import pathlib
import sys
from collections import Counter

import numpy as np


HERE = pathlib.Path(__file__).parent
FIG_DIR = HERE / "figures"
OUT = HERE / "report.html"

PROTEON_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/proteon.jsonl")
OPENMM_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/openmm.jsonl")
GROMACS_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/gmx_fold_preservation/tm_fold_gromacs.jsonl")
PROTEON_AMBER_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/proteon_amber.jsonl")
OPENMM_AMBER_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/openmm_amber.jsonl")
PROTEON_STAGE2_JSONL = pathlib.Path("/scratch/TMAlign/proteon/validation/stage2_1k_postfix.jsonl")
RESCUE_BENCHMARK_JSON = HERE / "rescue_benchmark_summary.json"

PACKAGE_SRC = HERE.parent.parent / "packages" / "proteon" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from proteon.loader_failure_analysis import summarize_loader_failures  # noqa: E402


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def img_tag(path: pathlib.Path, alt: str, style: str = "") -> str:
    if not path.exists():
        return f'<div class="missing">[{alt} — figure not yet generated]</div>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{alt}" style="{style}">'


def summary_stats(records: list[dict]) -> dict:
    tms = [r["tm_score"] for r in records if "tm_score" in r]
    rmsds = [r["rmsd"] for r in records if "rmsd" in r]
    n_total = len(records)
    n_ok = len(tms)
    if not tms:
        return {"n_total": n_total, "n_ok": 0}
    tms = np.array(tms); rmsds = np.array(rmsds)
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "success_pct": 100 * n_ok / max(n_total, 1),
        "tm_mean": float(tms.mean()),
        "tm_median": float(np.median(tms)),
        "tm_p05": float(np.percentile(tms, 5)),
        "tm_min": float(tms.min()),
        "rmsd_mean": float(rmsds.mean()),
        "rmsd_median": float(np.median(rmsds)),
        "rmsd_max": float(rmsds.max()),
    }


def tools_table(runs: dict[str, dict]) -> str:
    """Three-column summary table."""
    keys = list(runs.keys())
    rows = [
        ("Structures sampled",      [f"{runs[k].get('n_total', '—')}" for k in keys]),
        ("Successfully minimized",  [f"{runs[k].get('n_ok', '—')} ({runs[k].get('success_pct', 0):.1f}%)" for k in keys]),
        ("TM-score mean",           [f"{runs[k].get('tm_mean', 0):.4f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("TM-score median",         [f"{runs[k].get('tm_median', 0):.4f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("TM-score p05",            [f"{runs[k].get('tm_p05', 0):.4f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("TM-score min",            [f"{runs[k].get('tm_min', 0):.4f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("RMSD median (Å)",         [f"{runs[k].get('rmsd_median', 0):.3f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("RMSD mean (Å)",           [f"{runs[k].get('rmsd_mean', 0):.3f}" if runs[k].get('n_ok') else "—" for k in keys]),
        ("RMSD max (Å)",            [f"{runs[k].get('rmsd_max', 0):.3f}" if runs[k].get('n_ok') else "—" for k in keys]),
    ]
    hdr = "<tr><th></th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    body = "".join(
        "<tr><td class='metric'>" + name + "</td>" +
        "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
        for name, vals in rows
    )
    return f"<table class='stats'>{hdr}{body}</table>"


def _load_rescue_candidate_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "load_error" and "exception" in row:
            rows.append(row)
    return rows


def rescue_summary_table(rows: list[dict]) -> str:
    if not rows:
        return (
            "<div class='missing'>[No loader-failure artifact found for rescue summary]</div>"
        )

    summaries = summarize_loader_failures(rows)
    rescueable = [item for item in summaries if item.bucket.rescueable]
    rescueable_count = sum(item.count for item in rescueable)
    total_count = sum(item.count for item in summaries)

    rows_html = "".join(
        "<tr>"
        f"<td><code>{item.bucket.code}</code></td>"
        f"<td>{item.count}</td>"
        f"<td>{', '.join(item.pdb_examples) or '—'}</td>"
        f"<td>{item.bucket.rescue_strategy or '—'}</td>"
        "</tr>"
        for item in rescueable[:5]
    )
    if not rows_html:
        rows_html = (
            "<tr><td colspan='4'>No deterministic rescue buckets matched the current "
            "loader failures.</td></tr>"
        )

    return f"""
<div class="callout">
<strong>Rescue potential inside proteon's parse failures:</strong>
{rescueable_count}/{total_count} current loader failures ({(100 * rescueable_count / max(total_count, 1)):.1f}%)
match deterministic rescue buckets in the first-pass rescue layer. These are primarily metadata cleanup
or fixed-width PDB field repairs, not ambiguous structural reconstruction.
</div>

<table class='stats'>
  <tr>
    <th>Rescue bucket</th>
    <th>Count</th>
    <th>Example PDBs</th>
    <th>Deterministic fix</th>
  </tr>
  {rows_html}
</table>
"""


def realized_rescue_summary_table(summary_path: pathlib.Path) -> str:
    if not summary_path.exists():
        return (
            "<div class='missing'>[No realized rescue benchmark artifact found]</div>"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bucket_counts = summary.get("rescued_bucket_counts", {})
    rows_html = "".join(
        "<tr>"
        f"<td><code>{bucket}</code></td>"
        f"<td>{count}</td>"
        "</tr>"
        for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    if not rows_html:
        rows_html = "<tr><td colspan='2'>No rescues succeeded on this benchmark slice.</td></tr>"

    notes = []
    if summary.get("available_slice_count", 0) < summary.get("candidate_failure_count", 0):
        notes.append(
            f"Only {summary.get('available_slice_count', 0)}/{summary.get('candidate_failure_count', 0)} "
            "original failure files are present in this workspace."
        )
    if not summary.get("pipeline_ok", False):
        notes.append(
            "The downstream smoke/release run did not finish cleanly on this slice, so the realized number "
            "measures raw-intake rescue yield rather than full release yield."
        )
    note_html = ""
    if notes:
        note_html = "<div class='callout warn'><strong>Caveat:</strong> " + " ".join(notes) + "</div>"

    return f"""
<div class="callout">
<strong>Realized rescue yield on the local benchmark slice:</strong>
{summary.get('rescued_count', 0)}/{summary.get('available_slice_count', 0)} available failed inputs
loaded successfully through the explicit rescue path.
</div>

{note_html}

<table class='stats'>
  <tr>
    <th>Realized rescue bucket</th>
    <th>Count</th>
  </tr>
  {rows_html}
</table>
"""


def main() -> None:
    # Load all the jsonls that exist.
    rec_fer_ch = load_jsonl(PROTEON_JSONL)
    rec_omm_ch = load_jsonl(OPENMM_JSONL)
    rec_gmx    = load_jsonl(GROMACS_JSONL)
    rec_fer_a  = load_jsonl(PROTEON_AMBER_JSONL)
    rec_omm_a  = load_jsonl(OPENMM_AMBER_JSONL)

    stats_ch = {
        "Proteon CHARMM19+EEF1": summary_stats(rec_fer_ch),
        "OpenMM CHARMM36+OBC2":   summary_stats(rec_omm_ch),
    }
    stats_a = {
        "Proteon AMBER96":    summary_stats(rec_fer_a),
        "OpenMM AMBER96+OBC":  summary_stats(rec_omm_a),
        "GROMACS AMBER96":     summary_stats(rec_gmx),
    }

    # Headline numbers for the hero box.
    headlines = {
        "tm_median_charmm":  stats_ch["Proteon CHARMM19+EEF1"].get("tm_median", 0),
        "n_pdbs":             stats_ch["Proteon CHARMM19+EEF1"].get("n_total", 0),
        "success_pct":        stats_ch["Proteon CHARMM19+EEF1"].get("success_pct", 0),
        "throughput_ratio":   30,   # computed: 449.7/14.9 ≈ 30
        "amber_agreement":    "0.2%",
        "omm_gmx_agreement":  "0.03%",
    }

    today = datetime.date.today().isoformat()
    rescue_rows = _load_rescue_candidate_rows(PROTEON_STAGE2_JSONL)
    rescue_html = rescue_summary_table(rescue_rows)
    realized_rescue_html = realized_rescue_summary_table(RESCUE_BENCHMARK_JSON)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Proteon validation report — {today}</title>
<style>
:root {{
  --proteon: #0a7e8c;
  --openmm:   #d94801;
  --gromacs:  #6a3d9a;
  --ink:      #222;
  --muted:    #666;
  --bg:       #fafafa;
  --card:     #ffffff;
  --rule:     #e4e4e4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  color: var(--ink);
  background: var(--bg);
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px 32px 80px;
}}
header {{
  border-bottom: 3px solid var(--proteon);
  margin-bottom: 28px;
  padding-bottom: 16px;
}}
h1 {{
  font-size: 30px; font-weight: 700; margin: 0 0 4px;
  letter-spacing: -0.01em;
}}
h2 {{
  font-size: 22px; font-weight: 700; margin: 44px 0 6px;
  color: var(--ink); letter-spacing: -0.005em;
}}
h2::before {{
  content: ""; display: inline-block; width: 4px; height: 18px;
  background: var(--proteon); margin-right: 10px; vertical-align: -3px;
}}
h3 {{
  font-size: 16px; font-weight: 600; margin: 22px 0 8px;
  color: var(--ink);
}}
.meta {{
  font-size: 13px; color: var(--muted);
}}
.hero {{
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 16px; margin: 18px 0 8px;
}}
.hero .card {{
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 18px 20px;
}}
.hero .num {{
  font-size: 32px; font-weight: 700; color: var(--proteon);
  letter-spacing: -0.02em; line-height: 1;
}}
.hero .lbl {{
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-top: 6px;
}}
.hero .sub {{
  font-size: 13px; color: var(--muted); margin-top: 8px;
}}
p.lede {{
  font-size: 17px; color: var(--ink); margin: 10px 0 22px;
  max-width: 760px;
}}
section {{
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 28px 28px 20px;
  margin-bottom: 22px;
}}
section img {{
  display: block; max-width: 100%;
  margin: 14px 0; border-radius: 4px;
}}
section p {{ max-width: 780px; }}
.callout {{
  background: #f4fbfc;
  border-left: 3px solid var(--proteon);
  padding: 12px 16px;
  margin: 18px 0;
  font-size: 14px;
  color: var(--ink);
  border-radius: 2px;
}}
.warn {{
  background: #fff6ee;
  border-left: 3px solid var(--openmm);
}}
table.stats {{
  border-collapse: collapse;
  width: 100%;
  font-size: 14px;
  margin: 14px 0;
}}
table.stats th, table.stats td {{
  padding: 8px 12px;
  text-align: right;
  border-bottom: 1px solid var(--rule);
}}
table.stats th {{
  font-weight: 600;
  text-align: right;
  color: var(--ink);
  background: #fafafa;
}}
table.stats th:first-child, table.stats td:first-child {{
  text-align: left;
}}
table.stats td.metric {{
  font-weight: 500;
  color: var(--muted);
}}
code {{
  font-family: SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: 13px;
  background: #f1f1f1;
  padding: 1px 5px;
  border-radius: 3px;
}}
pre {{
  background: #f6f6f6;
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 12px 14px;
  font-family: SFMono-Regular, Consolas, monospace;
  font-size: 13px;
  overflow-x: auto;
}}
.missing {{
  padding: 40px; background: #fcfcfc; color: var(--muted);
  text-align: center; font-style: italic;
  border: 1px dashed var(--rule); border-radius: 4px;
}}
.tag {{
  display: inline-block;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  padding: 2px 8px; border-radius: 10px;
  font-weight: 600;
  vertical-align: 2px;
}}
.tag-amber    {{ background: #fde7cd; color: #7a3d00; }}
.tag-charmm   {{ background: #dde8f5; color: #154068; }}
.tag-pending  {{ background: #f0f0f0; color: #666; }}
.toc {{
  background: #f8f8f8; border: 1px solid var(--rule); border-radius: 6px;
  padding: 10px 16px; margin: 18px 0 28px;
  font-size: 14px;
}}
.toc a {{ color: var(--proteon); text-decoration: none; margin-right: 14px; }}
.toc a:hover {{ text-decoration: underline; }}
footer {{
  margin-top: 60px; padding-top: 18px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--muted);
}}
</style>
</head>
<body>
<main>

<header>
  <h1>Proteon validation report</h1>
  <div class="meta">Structural bioinformatics compute kernel · generated {today}</div>
</header>

<p class="lede">
Proteon is a Rust-based force-field + alignment engine with Python bindings,
designed as a fast, embeddable alternative to full MD packages for structural
bioinformatics workflows. This report documents the correctness and
performance of its energy + minimization pipeline against two reference
implementations (OpenMM and GROMACS) on identical inputs.
</p>

<div class="toc">
  <strong>Contents:</strong>
  <a href="#headlines">Headlines</a>
  <a href="#oracle">Energy oracle</a>
  <a href="#fold-charmm">Fold (CHARMM)</a>
  <a href="#fold-amber">Fold (AMBER)</a>
  <a href="#throughput">Throughput</a>
  <a href="#robustness">Robustness</a>
  <a href="#appendix">Reproducibility</a>
</div>

<!-- ==================================================================== -->
<h2 id="headlines">Headlines</h2>

<div class="hero">
  <div class="card">
    <div class="num">{headlines['tm_median_charmm']:.4f}</div>
    <div class="lbl">Median TM-score, 1000 PDBs</div>
    <div class="sub">Proteon CHARMM19+EEF1 pre vs post minimization. Fold is preserved.</div>
  </div>
  <div class="card">
    <div class="num">{headlines['amber_agreement']}</div>
    <div class="lbl">AMBER96 total-energy agreement with OpenMM</div>
    <div class="sub">Per-component: bond 0.02%, angle 0.22%, torsion 0.44%, NB 0.26%.</div>
  </div>
  <div class="card">
    <div class="num">{headlines['throughput_ratio']}×</div>
    <div class="lbl">Faster than OpenMM end-to-end</div>
    <div class="sub">1000 PDBs in 14.9 min (proteon) vs 449.7 min (OpenMM) at equal parallelism.</div>
  </div>
</div>

<section>
<p>
This document quantifies three claims that a production-grade structural MM
kernel should satisfy: <strong>correctness</strong> (does the force-field math
match established references?), <strong>fold preservation</strong> (does
energy minimization leave real protein structure intact?), and
<strong>throughput</strong> (can the pipeline scale to archive-sized
datasets?). Each section presents one of those claims and the evidence
behind it on a seed-42 random sample of 1000 PDB entries drawn from a 50,000
PDB corpus.
</p>
<div class="callout">
Proteon ships two force fields: <strong>CHARMM19+EEF1</strong> (polar-H
united-atom, the production default — used by the 50K battle test and the
main fold-preservation benchmark) and <strong>AMBER96</strong> (all-atom,
validated against both OpenMM and GROMACS implementations at reference
quality). Plots in this report are tagged
<span class="tag tag-charmm">CHARMM</span> or
<span class="tag tag-amber">AMBER</span> to make the comparison group
unambiguous.
</div>
</section>

<!-- ==================================================================== -->
<h2 id="oracle">Force-field math oracle <span class="tag tag-amber">AMBER</span></h2>

<section>
<p>
Three independent AMBER96 implementations (proteon, OpenMM 8.5, GROMACS 2026.1)
compute the single-point energy of crambin (PDB 1crn, 46 residues, ≈640 atoms
after hydrogen placement). All three tools consume the same atomic
coordinates within each preprocessing path. Components — bond stretch,
angle bend, proper+improper torsions, nonbonded (LJ+Coulomb) — are compared
bar-for-bar.
</p>

{img_tag(FIG_DIR / "04_energy_oracle.png", "AMBER96 triangulation")}

<p>
The left panel uses PDBFixer's hydrogen-placement library (consumed by
proteon and OpenMM). All four components agree to below 0.5%, with bond
stretching matching to <strong>0.02%</strong> — well below the noise
introduced by parameter precision. The right panel uses GROMACS's own
<code>pdb2gmx</code> output (different hydrogen positions, which is why the
totals differ between panels). Within that path, OpenMM and GROMACS agree on
bonded terms to machine precision and on the total to
<strong>0.03%</strong> — the expected behavior for two reference-quality
AMBER96 implementations.
</p>

<div class="callout">
<strong>Interpretation:</strong> proteon's AMBER96 force-field math is
canonical. The remaining &lt;0.5% per-component residual is within the
noise floor of hydrogen-placement library differences, not a math bug. The
reproducer lives at
<code>validation/amber96_oracle_triangulate.py</code>.
</div>
</section>

<!-- ==================================================================== -->
<h2 id="fold-charmm">Fold preservation — CHARMM family <span class="tag tag-charmm">CHARMM</span></h2>

<section>
<p>
Each tool runs its production CHARMM-family pipeline on the same 1000 PDBs:
<strong>proteon</strong> with CHARMM19+EEF1 (polar-H united-atom with
Lazaridis-Karplus implicit solvent), <strong>OpenMM</strong> with
CHARMM36+OBC2 (all-atom with GB implicit solvent). TM-score and RMSD are
computed on Cα coordinates before vs after minimization.
</p>

{img_tag(FIG_DIR / "01a_tm_charmm.png", "CHARMM TM distribution")}

<p>
Both tools preserve fold at the "same-fold" level (TM &gt; 0.5) on every
successful structure. Proteon's median TM-score of
<strong>{stats_ch['Proteon CHARMM19+EEF1'].get('tm_median', 0):.4f}</strong>
vs OpenMM's
<strong>{stats_ch['OpenMM CHARMM36+OBC2'].get('tm_median', 0):.4f}</strong>
differs marginally; the p05 tails show proteon has slightly more
structures with TM in the 0.90–0.97 range, driven by CHARMM19's inflated
united-atom carbon radii (which absorb implicit hydrogens and thus relax
slightly more during minimization).
</p>

{img_tag(FIG_DIR / "02a_rmsd_charmm.png", "CHARMM RMSD")}

<p>
The RMSD gap (≈0.58 Å proteon vs ≈0.21 Å OpenMM median) is the
expected signature of a polar-H united-atom force field: the absorbed
implicit hydrogens create small initial clashes that heavy-atom-free
minimization must relax. All-atom CHARMM36 has explicit hydrogens, smaller
initial clashes, and consequently less atom movement during minimization.
Both answers are physically valid; they measure different things.
</p>

{img_tag(FIG_DIR / "03a_tm_vs_size_charmm.png", "TM vs size")}

<p>
Low-TM outliers are concentrated in short peptides (&lt;30 Cα), where
TM-score's d0 normalization breaks down. Above 50 residues, both tools
hit TM &gt; 0.98 on every sampled structure.
</p>

{tools_table(stats_ch)}

</section>

<!-- ==================================================================== -->
<h2 id="fold-amber">Fold preservation — AMBER family <span class="tag tag-amber">AMBER</span></h2>

<section>
<p>
The AMBER96 run uses the same 1000-PDB sample, but with each tool running
AMBER96 force field parameters. This is the <em>cleanest</em> comparison
because all three implementations target the same force field; disagreement
can only come from implementation choices (hydrogen placement, nonbonded
cutoff policy, minimizer flavor).
</p>

{img_tag(FIG_DIR / "01b_tm_amber.png", "AMBER TM distribution")}
{img_tag(FIG_DIR / "02b_rmsd_amber.png", "AMBER RMSD")}
{img_tag(FIG_DIR / "03b_tm_vs_size_amber.png", "AMBER TM vs size")}

{tools_table(stats_a)}

</section>

<!-- ==================================================================== -->
<h2 id="throughput">Throughput and scaling</h2>

<section>
<p>
End-to-end wall time per structure is dominated by force-field evaluation
during minimization. Proteon's LBFGS minimizer auto-dispatches to a CUDA
kernel on structures ≥ 2000 atoms (on a machine with a CUDA-capable GPU);
OpenMM and GROMACS ran here on CPU. The dashed horizontal lines mark
amortized throughput when each tool is run batch-parallel across the full
1000-PDB sample:
</p>

{img_tag(FIG_DIR / "05_throughput.png", "Throughput scaling")}

<p>
The <strong>left panel</strong> is per-structure wall time vs residue count
on log-log axes. Power-law fits (<code>y = a · N<sup>b</sup></code>, solid
lines) expose the empirical algorithmic complexity of each tool:
</p>
<ul>
  <li><strong>OpenMM CHARMM36+OBC2</strong> scales as <code>N<sup>0.95</sup></code> — effectively linear, thanks to the NonbondedForce neighbor list and the absence of per-structure context reconstruction costs.</li>
  <li><strong>OpenMM AMBER96+OBC</strong> scales as <code>N<sup>1.41</sup></code> — the OBC Born-radius all-pair term pushes this toward O(N²) at large N.</li>
  <li><strong>GROMACS AMBER96</strong> scales as <code>N<sup>1.47</sup></code> with a consistently lower prefactor, dominated here by per-structure pdb2gmx/grompp/mdrun startup.</li>
</ul>
<p>
Faint grey guides mark pure O(N) and O(N²) for visual reference. No tool in this
benchmark crosses O(N²) in its empirical fit — in practice, all three get close
to linear behavior because short-range cutoffs keep the nonbonded cost
bounded.
</p>

<p>
The <strong>right panel</strong> is the production number: total wall time to
process a 1000-PDB batch on each tool's native parallelism. Proteon at
14.9 min on CHARMM19+EEF1 is the fastest configuration — benefiting from
(a) cheap polar-H united-atom evaluation, (b) rayon-parallel
<code>batch_prepare</code> with zero Python overhead, and (c) GPU dispatch
for structures ≥ 2000 atoms. Proteon AMBER96 at 210 min pays the expected
cost of all-atom evaluation with a 15 Å nonbonded cutoff (≈14× heavier per
structure than CHARMM19+EEF1's 9 Å polar-H path). OpenMM on CHARMM36+OBC2
sits at 449.7 min despite equivalent parallelism — the per-structure context
setup cost is substantial, visible as the high prefactor of its fit line.
GROMACS looks fastest on its aggregate bar, but only because
<code>pdb2gmx</code> rejected 64% of inputs before they ever reached mdrun.
</p>

<div class="callout">
<strong>Platform notes:</strong> proteon ran 64-way rayon with GPU
auto-dispatch on monster3 (128 AMD cores, RTX 5090). OpenMM ran 64
single-thread CPU workers via ProcessPoolExecutor on monster3. GROMACS ran
16 single-thread CPU workers on a 16-core Ubuntu local box. The per-structure
scatter removes platform-parallelism effects; the aggregate bars include
them (and are what you'd get in production).
</div>
</section>

<!-- ==================================================================== -->
<h2 id="robustness">Input robustness</h2>

<section>
<p>
Archive-scale structural bioinformatics demands tolerance of real-world
PDB inputs: non-standard residues (MSE, SEP, TPO, PTR, ligands), unusual
termini, mixed protein/DNA/RNA chains, disulfide variants, and historical
hydrogen naming conventions. Each tool was presented the same 1000 raw
PDB entries with no manual curation.
</p>

{img_tag(FIG_DIR / "06_input_robustness.png", "Robustness comparison")}

<p>
<strong>Proteon accepts 94.9% of raw PDBs end-to-end</strong>, compared to
OpenMM's 92.8% (PDBFixer + CHARMM36 template strictness) and GROMACS's 36.3%
(<code>pdb2gmx</code> + AMBER96 residue database is significantly stricter).
Proteon's dominant failure mode is <em>input parse error</em> in pdbtbx on
rare malformed records; OpenMM and GROMACS fail chiefly because their
force-field residue databases don't recognize non-standard amino acids
that appear in natural PDB entries.
</p>

<div class="callout">
<strong>Practical consequence:</strong> if your pipeline feeds 50,000 PDBs
in, proteon returns ≈47,000 minimized structures, OpenMM returns ≈46,000,
GROMACS returns ≈18,000. For archive-scale workflows this is the
dominant filter on the other end of the pipeline.
</div>

<h3>What looks rescuable inside proteon's remaining parse errors?</h3>

<p>
The remaining proteon failures are not all equally hard. Most are concentrated
in a small number of malformed-record buckets, which makes them good candidates
for an explicit rescue pass rather than a broader parser rewrite.
</p>

{rescue_html}

<h3>What has actually been rescued on a real-file benchmark slice?</h3>

<p>
Potential buckets are useful, but realized rescue yield matters more. When a
local benchmark artifact is present, the report also shows the buckets that
were actually recovered by the explicit rescue path on that slice.
</p>

{realized_rescue_html}
</section>

<!-- ==================================================================== -->
<h2 id="appendix">Reproducibility appendix</h2>

<section>
<h3>Scripts</h3>
<ul>
  <li><code>validation/tm_fold_preservation.py</code> — proteon CHARMM19+EEF1 fold run</li>
  <li><code>validation/tm_fold_preservation_amber.py</code> — proteon AMBER96 fold run</li>
  <li><code>validation/tm_fold_preservation_openmm.py</code> — OpenMM CHARMM36+OBC2 run</li>
  <li><code>validation/tm_fold_preservation_openmm_amber.py</code> — OpenMM AMBER96+OBC run</li>
  <li><code>validation/tm_fold_preservation_gromacs.py</code> — GROMACS AMBER96 run</li>
  <li><code>validation/amber96_oracle_triangulate.py</code> — single-point energy oracle</li>
  <li><code>validation/report/plots.py</code> — figure generation</li>
  <li><code>validation/report/build_report.py</code> — this HTML report</li>
</ul>

<h3>Environments</h3>
<pre>Proteon:   cargo +stable (Rust 1.83) / maturin develop --release / PyO3
OpenMM:     OpenMM 8.5 + PDBFixer in sota_venv
GROMACS:    GROMACS 2026.1 (local build, CPU + OpenMP, no CUDA)
Python:     3.10.x
Sample:     seed=42, 1000 PDBs from /globalscratch/dateschn/proteon-benchmark/pdbs_50k/</pre>

<h3>Hardware</h3>
<pre>proteon GPU benchmark:   monster3 — 128 AMD cores, 1 × NVIDIA RTX 5090
OpenMM benchmark:         monster3 — 128 AMD cores, CPU only
GROMACS benchmark:        local — 16 cores, CPU only</pre>

</section>

<footer>
<p>Generated from jsonl data by <code>validation/report/build_report.py</code>
on {today}. Re-run after any benchmark update to pick up fresh numbers.</p>
</footer>

</main>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
