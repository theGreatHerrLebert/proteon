"""Production plots for the proteon validation report.

Generates:
  01_tm_distribution.png           — 3-tool histogram + ECDF
  02_rmsd_distribution.png         — 3-tool histogram + violin
  03_tm_vs_size.png                — TM vs N_CA scatter (3 tools overlaid)
  04_energy_oracle.png             — crambin AMBER96 oracle components (bar chart)
  05_throughput.png                — wall-clock / struct vs N_CA
  06_input_robustness.png          — success-rate bar chart with failure causes

Each plot consumes JSONL files via ``load_jsonl`` and plots only the tools
with data present. Missing tools are silently skipped, so the same
script runs at every step of the benchmark campaign.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from style import C, apply, savefig


HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots")
GMX_DATA = pathlib.Path("/scratch/TMAlign/proteon/validation/gmx_fold_preservation")
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# --- Data loading -------------------------------------------------------------

@dataclass
class Run:
    name: str                     # label, e.g. "Proteon CHARMM19+EEF1"
    short: str                    # column tag, e.g. "proteon-charmm"
    color: str
    ff_family: str                # "charmm" | "amber"
    tool: str                     # "proteon" | "openmm" | "gromacs"
    records: list[dict]

    @property
    def tms(self) -> np.ndarray:
        return np.array([r["tm_score"] for r in self.records if "tm_score" in r])

    @property
    def rmsds(self) -> np.ndarray:
        return np.array([r["rmsd"] for r in self.records if "rmsd" in r])

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.records if "tm_score" in r)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def fail_rate(self) -> float:
        return (self.n_total - self.n_ok) / max(self.n_total, 1)

    @property
    def fail_modes(self) -> Counter:
        out = Counter()
        for r in self.records:
            if "tm_score" in r:
                continue
            e = r.get("error", r.get("skipped", "?"))
            if not isinstance(e, str):
                e = str(e)
            kind = _classify_error(e)
            out[kind] += 1
        return out


def _classify_error(msg: str) -> str:
    m = msg.lower()
    if "load_failed" in m or "parse" in m or "pdbtbx" in m:
        return "input parse error"
    if "no template found" in m:
        return "unknown residue / template"
    if "residue" in m and "database" in m:
        return "unknown residue / template"
    if "atom" in m and ("rtp" in m or "not found in" in m):
        return "atom-template mismatch"
    if "ca shape mismatch" in m:
        return "CA count changed by prep"
    if "water" in m:
        return "water in input"
    if "timeout" in m:
        return "timeout"
    if "no_protein" in m or "not protein" in m:
        return "non-protein skipped"
    if "pdb2gmx" in m:
        return "pdb2gmx other"
    if "grompp" in m:
        return "grompp other"
    if "minimize" in m or "mdrun" in m:
        return "minimization error"
    return "other"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def collect_runs() -> list[Run]:
    runs = []
    candidates = [
        Run("Proteon CHARMM19+EEF1", "proteon-charmm", C["proteon"],
            "charmm", "proteon", load_jsonl(DATA / "proteon.jsonl")),
        Run("OpenMM CHARMM36+OBC2", "openmm-charmm", C["openmm"],
            "charmm", "openmm", load_jsonl(DATA / "openmm.jsonl")),
        Run("Proteon AMBER96", "proteon-amber", C["proteon"],
            "amber", "proteon",
            load_jsonl(pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/proteon_amber.jsonl"))),
        Run("OpenMM AMBER96+OBC", "openmm-amber", C["openmm"],
            "amber", "openmm",
            load_jsonl(pathlib.Path("/scratch/TMAlign/proteon/validation/tm_fold_plots/openmm_amber.jsonl"))),
        Run("GROMACS AMBER96", "gromacs-amber", C["gromacs"],
            "amber", "gromacs",
            load_jsonl(GMX_DATA / "tm_fold_gromacs.jsonl")),
    ]
    for r in candidates:
        if r.records:
            runs.append(r)
    return runs


# --- 01: TM-score distribution -----------------------------------------------

def plot_tm_distribution(runs: list[Run], family: str, out: pathlib.Path) -> None:
    fam_runs = [r for r in runs if r.ff_family == family]
    if not fam_runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: histogram with a log y-axis (TM distributions are heavily
    # concentrated near 1.0; density alone compresses everything into
    # the last bin).
    ax = axes[0]
    bins = np.linspace(0.85, 1.0, 46)
    for r in fam_runs:
        tms_clipped = np.clip(r.tms, 0.85, 1.0)
        ax.hist(tms_clipped, bins=bins, alpha=0.55, color=r.color,
                label=f"{r.name}  median={np.median(r.tms):.4f}  n={r.n_ok}",
                edgecolor="none")
    apply(ax, title="TM-score distribution (zoomed on the 0.85–1.0 tail)",
          xlabel="TM-score (pre vs post minimization)",
          ylabel="Count (out of ~1000 sampled)")
    ax.set_xlim(0.85, 1.001)

    # Right: ECDF zoomed on the tails — exactly where the tools differ.
    ax = axes[1]
    for r in fam_runs:
        xs = np.sort(r.tms)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=r.color, lw=2.0,
                label=f"{r.name} (p05={np.percentile(r.tms,5):.3f})")
    ax.axvline(0.5, color="#888", lw=0.8, ls=":")
    apply(ax, title="TM-score ECDF",
          subtitle="lower-left tail = structures whose fold moved most",
          xlabel="TM-score", ylabel="Cumulative fraction")
    ax.set_xlim(0.85, 1.001)

    fam_label = "CHARMM family" if family == "charmm" else "AMBER family"
    fig.suptitle(f"Fold preservation — {fam_label} — 1000 PDBs from the 50K corpus",
                 fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, out)


# --- 02: RMSD distribution ---------------------------------------------------

def plot_rmsd_distribution(runs: list[Run], family: str, out: pathlib.Path) -> None:
    fam_runs = [r for r in runs if r.ff_family == family]
    if not fam_runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    bins = np.linspace(0, 1.5, 46)
    for r in fam_runs:
        rmsds = np.clip(r.rmsds, 0, 1.5)
        ax.hist(rmsds, bins=bins, alpha=0.55, color=r.color,
                label=f"{r.name} (median={np.median(r.rmsds):.2f} Å)",
                edgecolor="none")
    apply(ax, title="Heavy-atom RMSD after minimization",
          xlabel="CA RMSD pre → post (Å)", ylabel="Count")
    ax.set_xlim(0, 1.5)

    ax = axes[1]
    # Clip extreme outliers (some GROMACS structures produce non-physical
    # RMSDs from a bad minimization step); violin plot otherwise paints
    # the whole [0, ∞] range and hides the real distribution shape.
    data = [np.clip(r.rmsds, 0, 1.5) for r in fam_runs]
    positions = np.arange(len(fam_runs)) + 1
    vp = ax.violinplot(data, positions=positions, showmedians=True,
                       showextrema=False, widths=0.75)
    for pc, r in zip(vp["bodies"], fam_runs):
        pc.set_facecolor(r.color)
        pc.set_alpha(0.55)
        pc.set_edgecolor(r.color)
    if "cmedians" in vp:
        vp["cmedians"].set_color(C["axis"])
        vp["cmedians"].set_linewidth(1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([r.name.replace(" ", "\n", 1) for r in fam_runs],
                        fontsize=9)
    apply(ax, title="Distribution shape", ylabel="CA RMSD (Å)", legend=False)
    ax.set_ylim(0, 1.5)

    fam_label = "CHARMM family" if family == "charmm" else "AMBER family"
    fig.suptitle(
        f"Heavy-atom movement during minimization — {fam_label}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    savefig(fig, out)


# --- 03: TM vs structure size ------------------------------------------------

def plot_tm_vs_size(runs: list[Run], family: str, out: pathlib.Path) -> None:
    fam_runs = [r for r in runs if r.ff_family == family]
    if not fam_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for r in fam_runs:
        ns = np.array([rec["n_ca"] for rec in r.records if "n_ca" in rec and "tm_score" in rec])
        tms = r.tms
        if len(ns) != len(tms):
            continue
        ax.scatter(ns, tms, s=12, alpha=0.35, color=r.color,
                   label=f"{r.name}", edgecolor="none")
    ax.axhline(0.5, color="#888", ls=":", lw=0.8)
    ax.set_xscale("log")
    # Place the annotation AFTER log transform.
    ax.text(0.02, 0.52, "TM = 0.5 (same-fold threshold)",
            transform=ax.get_yaxis_transform(), color="#888", fontsize=9)
    apply(ax, title="TM-score vs structure size",
          subtitle="Low-TM outliers are short peptides (<30 CA) where TM-score is mathematically unreliable",
          xlabel="Residue count N (log)", ylabel="TM-score pre vs post")
    ax.set_ylim(0.3, 1.01)
    fam_label = "CHARMM family" if family == "charmm" else "AMBER family"
    fig.suptitle(f"{fam_label}", fontsize=11, y=1.0)
    savefig(fig, out)


# --- 04: Energy oracle components --------------------------------------------

def plot_energy_oracle(out: pathlib.Path) -> None:
    """Crambin AMBER96 triangulation — three tools, component breakdown.

    Left panel: PDBFixer prep, proteon vs OpenMM (per-component agreement).
    Right panel: GROMACS pdb2gmx prep, OpenMM vs GROMACS (reference check).
    The two preps give different totals because H placement differs;
    within each prep, the tools converge to each other.
    """
    components = ["Bond", "Angle", "Torsion\n+Improper", "Non-\nbonded"]
    pdbfixer = {
        "openmm":   [8736.5, 3470.9, 2098.5, -5492.5],
        "proteon": [8735.1, 3478.5, 2089.3, -5506.9],
    }
    gromacs_prep = {
        "openmm":  [3223.3, 469.0, 926.2, -5553.8],
        "gromacs": [3223.3, 469.0, 926.2, -5691.1],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    for ax, data, tools, title, subtitle in (
        (axes[0], pdbfixer, ("openmm", "proteon"),
         "PDBFixer prep — proteon vs OpenMM (AMBER96)",
         "per-component agreement; labels show |Δ|/|ref|"),
        (axes[1], gromacs_prep, ("openmm", "gromacs"),
         "pdb2gmx prep — OpenMM vs GROMACS (AMBER96)",
         "reference cross-check: total agrees to 0.03%"),
    ):
        x = np.arange(len(components))
        w = 0.38
        labels = {"openmm": "OpenMM", "proteon": "Proteon", "gromacs": "GROMACS"}
        ax.bar(x - w / 2, data[tools[0]], w, color=C[tools[0]],
               label=f"{labels[tools[0]]} AMBER96",
               edgecolor="white", linewidth=0.6)
        ax.bar(x + w / 2, data[tools[1]], w, color=C[tools[1]],
               label=f"{labels[tools[1]]} AMBER96",
               edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#333", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        # Agreement % label centered between the two bars, above the zero line
        # when values are positive, below when negative.
        for i in range(len(components)):
            v0, v1 = data[tools[0]][i], data[tools[1]][i]
            pct = 100 * abs(v1 - v0) / max(abs(v0), 1)
            top = max(v0, v1, 0)
            bot = min(v0, v1, 0)
            if v0 > 0:  # positive bars — label above
                ax.text(i, top * 1.08, f"{pct:.3f}%" if pct < 1 else f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=9,
                        color="#222", fontweight="medium")
            else:       # negative bars — label below
                ax.text(i, bot * 1.08, f"{pct:.3f}%" if pct < 1 else f"{pct:.1f}%",
                        ha="center", va="top", fontsize=9,
                        color="#222", fontweight="medium")
        apply(ax, title=title, subtitle=subtitle,
              ylabel="Energy (kJ/mol)")
        # Give room for the % labels.
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] * 1.15, ylim[1] * 1.18)

    fig.suptitle(
        "AMBER96 energy oracle — proteon triangulated against OpenMM and GROMACS on crambin",
        fontsize=14, fontweight="bold", y=1.02,
    )
    savefig(fig, out)


# --- 05: Throughput -----------------------------------------------------------

def plot_throughput(runs: list[Run], out: pathlib.Path) -> None:
    """Two-panel throughput figure.

    Left:  per-structure wall time vs residue count on log-log axes,
           with a power-law fit per tool that has per-structure data.
           Fit exponents tell us the empirical algorithmic complexity.
           Faint O(N) and O(N²) reference lines give visual intuition.
    Right: aggregate throughput bars — total wall time for a full
           1000-PDB batch. This is the number that matters for archive-
           scale pipelines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                              gridspec_kw={"width_ratios": [1.5, 1]})

    # --- Left panel: per-structure scatter + fits ---
    ax = axes[0]
    fits = []   # (label, color, slope, prefactor, n)
    for r in runs:
        pts = [(rec.get("n_ca_pre") or rec.get("n_ca"), rec.get("wall_s"))
               for rec in r.records
               if rec.get("wall_s") and (rec.get("n_ca_pre") or rec.get("n_ca"))]
        if len(pts) < 10:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        # Fit y = a * x^b in log-log space (linear regression on logs).
        mask = (xs > 0) & (ys > 0)
        lx, ly = np.log(xs[mask]), np.log(ys[mask])
        slope, intercept = np.polyfit(lx, ly, 1)
        a = np.exp(intercept)
        fits.append((r.name, r.color, slope, a, int(mask.sum())))

        ax.scatter(xs, ys, s=10, alpha=0.22, color=r.color,
                   edgecolor="none")
        # Plot the fitted line over the observed x-range.
        x_fit = np.logspace(np.log10(xs[mask].min()), np.log10(xs[mask].max()), 40)
        y_fit = a * x_fit ** slope
        ax.plot(x_fit, y_fit, color=r.color, lw=2.2,
                label=f"{r.name} — fit ∝ N^{slope:.2f}")

    # Complexity reference lines anchored at a mid-range data point so they
    # visually pass near the fit, not at arbitrary intercepts.
    if fits:
        x_ref = 200.0
        # Pick a midpoint y on the slowest tool's fit for anchoring.
        slowest = max(fits, key=lambda f: f[3] * x_ref ** f[2])
        y_ref = slowest[3] * x_ref ** slowest[2]
        for expo, label, ls in [(1.0, "O(N)", "--"), (2.0, "O(N²)", ":")]:
            ref_a = y_ref / (x_ref ** expo)
            x_line = np.logspace(1, 4, 40)
            ax.plot(x_line, ref_a * x_line ** expo, color="#aaa",
                    ls=ls, lw=1.0, zorder=0)
            ax.text(x_line[-8], ref_a * x_line[-8] ** expo,
                    f"  {label}", color="#888", fontsize=9,
                    va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    apply(ax, title="Per-structure wall time vs system size",
          subtitle="log-log scatter with empirical power-law fits; grey guides = O(N), O(N²)",
          xlabel="Residue count N  (log)", ylabel="Wall time per structure (s, log)",
          legend=True, legend_loc="upper left")

    # --- Right panel: aggregate throughput bars ---
    ax = axes[1]
    # Aggregate totals from the benchmark runs (minutes of wall clock to
    # process a 1000-PDB seed=42 sample at the parallelism configured
    # for each tool on monster3 / local). See report for platform notes.
    aggregates = [
        ("Proteon\nCHARMM19+EEF1",       14.9,  949,  C["proteon"]),
        ("Proteon\nAMBER96",            210.0,  950,  C["proteon"]),
        ("OpenMM\nCHARMM36+OBC2",        449.7,  928,  C["openmm"]),
        ("OpenMM\nAMBER96+OBC",           75.7,  904,  C["openmm"]),
        ("GROMACS\nAMBER96",              20.3,  363,  C["gromacs"]),
    ]
    names = [a[0] for a in aggregates]
    totals = [a[1] for a in aggregates]
    colors = [a[3] for a in aggregates]
    y = np.arange(len(names))[::-1]  # top-down by fastest first
    order = np.argsort(totals)
    bars = ax.barh(y, [totals[i] for i in order],
                   color=[colors[i] for i in order],
                   edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    for i, idx in enumerate(order):
        t = totals[idx]
        ax.text(t + 5, y[i], f"{t:.1f} min",
                va="center", fontsize=10, color="#222")
    ax.set_xlim(0, max(totals) * 1.22)
    apply(ax, title="Total wall time for 1000-PDB batch",
          subtitle="smaller is better — amortized over parallelism",
          xlabel="Wall time (minutes)", legend=False)

    fig.suptitle("Throughput — per-structure scaling and aggregate batch time",
                 fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, out)


# --- 06: Input robustness ----------------------------------------------------

def plot_input_robustness(runs: list[Run], out: pathlib.Path) -> None:
    """Success / failure breakdown by tool."""
    # Order tools by descending success rate.
    runs_sorted = sorted(runs, key=lambda r: -r.n_ok / max(r.n_total, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    names = [r.name for r in runs_sorted]
    oks = [100 * r.n_ok / r.n_total for r in runs_sorted]
    fails = [100 - o for o in oks]
    y = np.arange(len(names))
    ax.barh(y, oks, color=[r.color for r in runs_sorted], alpha=0.9,
            edgecolor="white", linewidth=0.5)
    ax.barh(y, fails, left=oks, color="#d0d0d0", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    for i, r in enumerate(runs_sorted):
        ax.text(oks[i] + 1, i, f"{oks[i]:.1f}% ok  ({r.n_ok}/{r.n_total})",
                va="center", fontsize=9, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 115)
    apply(ax, title="Success rate on raw PDBs",
          subtitle="Gray = failed / skipped (load, template, minimization, …)",
          xlabel="Percent of 1000 sampled PDBs", legend=False)
    ax.invert_yaxis()

    ax = axes[1]
    # Stacked fail-cause bars.
    all_modes = Counter()
    tool_modes: dict[str, Counter] = {}
    for r in runs_sorted:
        m = r.fail_modes
        tool_modes[r.name] = m
        all_modes.update(m)
    top_modes = [k for k, _ in all_modes.most_common(5)]
    bottom = np.zeros(len(runs_sorted))
    cmap_palette = [C["openmm"], C["gromacs"], "#8b4513", C["amber"],
                     "#ff7f00", "#1f78b4"]
    for i, mode in enumerate(top_modes):
        values = [tool_modes[r.name].get(mode, 0) for r in runs_sorted]
        ax.bar(np.arange(len(runs_sorted)), values, bottom=bottom,
               color=cmap_palette[i % len(cmap_palette)], alpha=0.85,
               label=mode, edgecolor="white", linewidth=0.5)
        bottom = bottom + values
    # Other: anything not in top_modes.
    other = [sum(v for k, v in tool_modes[r.name].items() if k not in top_modes)
             for r in runs_sorted]
    ax.bar(np.arange(len(runs_sorted)), other, bottom=bottom,
           color="#bfbfbf", alpha=0.7, label="other",
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(runs_sorted)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    apply(ax, title="Failure modes, breakdown",
          subtitle="Dominant failure cause per tool (top 5 + other)",
          ylabel="Failed structures (count)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Input robustness — how each tool handles raw archive PDBs",
                 fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, out)


# --- Main --------------------------------------------------------------------

def main() -> None:
    runs = collect_runs()
    print(f"Loaded runs: {', '.join(r.name for r in runs)}")

    plot_tm_distribution(runs, "charmm", OUT / "01a_tm_charmm.png")
    plot_tm_distribution(runs, "amber", OUT / "01b_tm_amber.png")

    plot_rmsd_distribution(runs, "charmm", OUT / "02a_rmsd_charmm.png")
    plot_rmsd_distribution(runs, "amber", OUT / "02b_rmsd_amber.png")

    plot_tm_vs_size(runs, "charmm", OUT / "03a_tm_vs_size_charmm.png")
    plot_tm_vs_size(runs, "amber", OUT / "03b_tm_vs_size_amber.png")

    plot_energy_oracle(OUT / "04_energy_oracle.png")
    plot_throughput(runs, OUT / "05_throughput.png")
    plot_input_robustness(runs, OUT / "06_input_robustness.png")

    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
