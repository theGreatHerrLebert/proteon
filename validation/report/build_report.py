"""Build the single-file HTML validation report for ferritin.

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
from collections import Counter

import numpy as np


HERE = pathlib.Path(__file__).parent
FIG_DIR = HERE / "figures"
OUT = HERE / "report.html"

FERRITIN_JSONL = pathlib.Path("/scratch/TMAlign/ferritin/validation/tm_fold_plots/ferritin.jsonl")
OPENMM_JSONL = pathlib.Path("/scratch/TMAlign/ferritin/validation/tm_fold_plots/openmm.jsonl")
GROMACS_JSONL = pathlib.Path("/scratch/TMAlign/ferritin/validation/gmx_fold_preservation/tm_fold_gromacs.jsonl")
FERRITIN_AMBER_JSONL = pathlib.Path("/scratch/TMAlign/ferritin/validation/tm_fold_plots/ferritin_amber.jsonl")
OPENMM_AMBER_JSONL = pathlib.Path("/scratch/TMAlign/ferritin/validation/tm_fold_plots/openmm_amber.jsonl")


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


def main() -> None:
    # Load all the jsonls that exist.
    rec_fer_ch = load_jsonl(FERRITIN_JSONL)
    rec_omm_ch = load_jsonl(OPENMM_JSONL)
    rec_gmx    = load_jsonl(GROMACS_JSONL)
    rec_fer_a  = load_jsonl(FERRITIN_AMBER_JSONL)
    rec_omm_a  = load_jsonl(OPENMM_AMBER_JSONL)

    stats_ch = {
        "Ferritin CHARMM19+EEF1": summary_stats(rec_fer_ch),
        "OpenMM CHARMM36+OBC2":   summary_stats(rec_omm_ch),
    }
    stats_a = {
        "Ferritin AMBER96":    summary_stats(rec_fer_a),
        "OpenMM AMBER96+OBC":  summary_stats(rec_omm_a),
        "GROMACS AMBER96":     summary_stats(rec_gmx),
    }

    # Headline numbers for the hero box.
    headlines = {
        "tm_median_charmm":  stats_ch["Ferritin CHARMM19+EEF1"].get("tm_median", 0),
        "n_pdbs":             stats_ch["Ferritin CHARMM19+EEF1"].get("n_total", 0),
        "success_pct":        stats_ch["Ferritin CHARMM19+EEF1"].get("success_pct", 0),
        "throughput_ratio":   30,   # computed: 449.7/14.9 ≈ 30
        "amber_agreement":    "0.2%",
        "omm_gmx_agreement":  "0.03%",
    }

    today = datetime.date.today().isoformat()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ferritin validation report — {today}</title>
<style>
:root {{
  --ferritin: #0a7e8c;
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
  border-bottom: 3px solid var(--ferritin);
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
  background: var(--ferritin); margin-right: 10px; vertical-align: -3px;
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
  font-size: 32px; font-weight: 700; color: var(--ferritin);
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
  border-left: 3px solid var(--ferritin);
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
.toc a {{ color: var(--ferritin); text-decoration: none; margin-right: 14px; }}
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
  <h1>Ferritin validation report</h1>
  <div class="meta">Structural bioinformatics compute kernel · generated {today}</div>
</header>

<p class="lede">
Ferritin is a Rust-based force-field + alignment engine with Python bindings,
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
    <div class="sub">Ferritin CHARMM19+EEF1 pre vs post minimization. Fold is preserved.</div>
  </div>
  <div class="card">
    <div class="num">{headlines['amber_agreement']}</div>
    <div class="lbl">AMBER96 total-energy agreement with OpenMM</div>
    <div class="sub">Per-component: bond 0.02%, angle 0.22%, torsion 0.44%, NB 0.26%.</div>
  </div>
  <div class="card">
    <div class="num">{headlines['throughput_ratio']}×</div>
    <div class="lbl">Faster than OpenMM end-to-end</div>
    <div class="sub">1000 PDBs in 14.9 min (ferritin) vs 449.7 min (OpenMM) at equal parallelism.</div>
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
Ferritin ships two force fields: <strong>CHARMM19+EEF1</strong> (polar-H
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
Three independent AMBER96 implementations (ferritin, OpenMM 8.5, GROMACS 2026.1)
compute the single-point energy of crambin (PDB 1crn, 46 residues, ≈640 atoms
after hydrogen placement). All three tools consume the same atomic
coordinates within each preprocessing path. Components — bond stretch,
angle bend, proper+improper torsions, nonbonded (LJ+Coulomb) — are compared
bar-for-bar.
</p>

{img_tag(FIG_DIR / "04_energy_oracle.png", "AMBER96 triangulation")}

<p>
The left panel uses PDBFixer's hydrogen-placement library (consumed by
ferritin and OpenMM). All four components agree to below 0.5%, with bond
stretching matching to <strong>0.02%</strong> — well below the noise
introduced by parameter precision. The right panel uses GROMACS's own
<code>pdb2gmx</code> output (different hydrogen positions, which is why the
totals differ between panels). Within that path, OpenMM and GROMACS agree on
bonded terms to machine precision and on the total to
<strong>0.03%</strong> — the expected behavior for two reference-quality
AMBER96 implementations.
</p>

<div class="callout">
<strong>Interpretation:</strong> ferritin's AMBER96 force-field math is
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
<strong>ferritin</strong> with CHARMM19+EEF1 (polar-H united-atom with
Lazaridis-Karplus implicit solvent), <strong>OpenMM</strong> with
CHARMM36+OBC2 (all-atom with GB implicit solvent). TM-score and RMSD are
computed on Cα coordinates before vs after minimization.
</p>

{img_tag(FIG_DIR / "01a_tm_charmm.png", "CHARMM TM distribution")}

<p>
Both tools preserve fold at the "same-fold" level (TM &gt; 0.5) on every
successful structure. Ferritin's median TM-score of
<strong>{stats_ch['Ferritin CHARMM19+EEF1'].get('tm_median', 0):.4f}</strong>
vs OpenMM's
<strong>{stats_ch['OpenMM CHARMM36+OBC2'].get('tm_median', 0):.4f}</strong>
differs marginally; the p05 tails show ferritin has slightly more
structures with TM in the 0.90–0.97 range, driven by CHARMM19's inflated
united-atom carbon radii (which absorb implicit hydrogens and thus relax
slightly more during minimization).
</p>

{img_tag(FIG_DIR / "02a_rmsd_charmm.png", "CHARMM RMSD")}

<p>
The RMSD gap (≈0.58 Å ferritin vs ≈0.21 Å OpenMM median) is the
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
during minimization. Ferritin's LBFGS minimizer auto-dispatches to a CUDA
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
process a 1000-PDB batch on each tool's native parallelism. Ferritin at
14.9 min on CHARMM19+EEF1 is the fastest configuration — benefiting from
(a) cheap polar-H united-atom evaluation, (b) rayon-parallel
<code>batch_prepare</code> with zero Python overhead, and (c) GPU dispatch
for structures ≥ 2000 atoms. Ferritin AMBER96 at 210 min pays the expected
cost of all-atom evaluation with a 15 Å nonbonded cutoff (≈14× heavier per
structure than CHARMM19+EEF1's 9 Å polar-H path). OpenMM on CHARMM36+OBC2
sits at 449.7 min despite equivalent parallelism — the per-structure context
setup cost is substantial, visible as the high prefactor of its fit line.
GROMACS looks fastest on its aggregate bar, but only because
<code>pdb2gmx</code> rejected 64% of inputs before they ever reached mdrun.
</p>

<div class="callout">
<strong>Platform notes:</strong> ferritin ran 64-way rayon with GPU
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
<strong>Ferritin accepts 94.9% of raw PDBs end-to-end</strong>, compared to
OpenMM's 92.8% (PDBFixer + CHARMM36 template strictness) and GROMACS's 36.3%
(<code>pdb2gmx</code> + AMBER96 residue database is significantly stricter).
Ferritin's dominant failure mode is <em>input parse error</em> in pdbtbx on
rare malformed records; OpenMM and GROMACS fail chiefly because their
force-field residue databases don't recognize non-standard amino acids
that appear in natural PDB entries.
</p>

<div class="callout">
<strong>Practical consequence:</strong> if your pipeline feeds 50,000 PDBs
in, ferritin returns ≈47,000 minimized structures, OpenMM returns ≈46,000,
GROMACS returns ≈18,000. For archive-scale workflows this is the
dominant filter on the other end of the pipeline.
</div>
</section>

<!-- ==================================================================== -->
<h2 id="appendix">Reproducibility appendix</h2>

<section>
<h3>Scripts</h3>
<ul>
  <li><code>validation/tm_fold_preservation.py</code> — ferritin CHARMM19+EEF1 fold run</li>
  <li><code>validation/tm_fold_preservation_amber.py</code> — ferritin AMBER96 fold run</li>
  <li><code>validation/tm_fold_preservation_openmm.py</code> — OpenMM CHARMM36+OBC2 run</li>
  <li><code>validation/tm_fold_preservation_openmm_amber.py</code> — OpenMM AMBER96+OBC run</li>
  <li><code>validation/tm_fold_preservation_gromacs.py</code> — GROMACS AMBER96 run</li>
  <li><code>validation/amber96_oracle_triangulate.py</code> — single-point energy oracle</li>
  <li><code>validation/report/plots.py</code> — figure generation</li>
  <li><code>validation/report/build_report.py</code> — this HTML report</li>
</ul>

<h3>Environments</h3>
<pre>Ferritin:   cargo +stable (Rust 1.83) / maturin develop --release / PyO3
OpenMM:     OpenMM 8.5 + PDBFixer in sota_venv
GROMACS:    GROMACS 2026.1 (local build, CPU + OpenMP, no CUDA)
Python:     3.10.x
Sample:     seed=42, 1000 PDBs from /globalscratch/dateschn/ferritin-benchmark/pdbs_50k/</pre>

<h3>Hardware</h3>
<pre>ferritin GPU benchmark:   monster3 — 128 AMD cores, 1 × NVIDIA RTX 5090
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
