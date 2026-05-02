"""Plots for the CHARMM corpus oracle (proteon vs BALL).

Consumes a JSONL artifact from `validation/charmm19_eef1_ball_oracle.py`
and produces 4 figures + an outlier table for inclusion in the EVIDENT
release report:

  10_corpus_error_distribution.png    — per-component rel diff histograms
                                        (one panel per gated component)
  11_corpus_runtime.png               — wall-time histogram + scatter vs n_atoms
  12_corpus_correlations.png          — pairwise rel-diff correlations
                                        across components (does a structure
                                        that's bad on bond also bad on vdw?)
  13_corpus_outliers.html             — top-20 worst-fitting structures
                                        as a small HTML table fragment

Each function reads ONE JSONL file and writes one output. Designed to
slot into `build_report.py` as a new "Force-field component oracle"
section.

Per-component bands (mirrors the unit-oracle test bands):
    bond_stretch:  1%
    angle_bend:    1%
    vdw:           2.5%
    electrostatic: 1%
    solvation:     5%
    proper torsion (xfail-historical / now-passing): 2.5%
    improper torsion (xfail-historical / now-passing): 2.5%
"""
from __future__ import annotations

import html
import json
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from style import C, apply, savefig


# Per-component oracle band (same numbers the unit test asserts).
BANDS = {
    "bond_stretch":  0.01,
    "angle_bend":    0.01,
    "vdw":           0.025,
    "electrostatic": 0.01,
    "solvation":     0.05,
    "torsion":       0.025,    # proper torsion (called "torsion" in the schema)
    "improper_torsion": 0.025,
}

# Display labels for component plots.
LABELS = {
    "bond_stretch":     "Bond stretch",
    "angle_bend":       "Angle bend",
    "torsion":          "Proper torsion",
    "improper_torsion": "Improper torsion",
    "vdw":              "van der Waals",
    "electrostatic":    "Electrostatic",
    "solvation":        "EEF1 solvation",
}

# Order components in the layout left→right, top→bottom.
COMPONENT_ORDER = (
    "bond_stretch", "angle_bend", "torsion",
    "improper_torsion", "vdw", "electrostatic", "solvation",
)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """Read all JSON records from a JSONL file. Skips malformed lines."""
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def successful_records(records: Iterable[dict]) -> list[dict]:
    """Return only records that have a complete `rel_diff` block."""
    return [r for r in records if "rel_diff" in r]


# --- 10: per-component rel-diff distribution ----------------------------------

def plot_error_distribution(records: list[dict], out: pathlib.Path) -> None:
    """7-panel grid (one per component) of rel-diff histograms.

    Each panel:
      * histogram of rel_diff[component] across all successful PDBs
      * red dashed line at the per-component band
      * pass-rate annotation in the corner
    """
    ok = successful_records(records)
    if not ok:
        return

    ncols = 4
    nrows = 2  # 7 components fit in 2x4 with one empty cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), squeeze=False)

    for i, comp in enumerate(COMPONENT_ORDER):
        ax = axes[i // ncols][i % ncols]
        vals = np.array([
            r["rel_diff"].get(comp, np.nan) for r in ok
        ])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=C["axis"], fontsize=11)
            ax.set_axis_off()
            continue

        # Plot in PERCENT units. Band is per-component (1% to 5%); pick a
        # tail-aware xlim so a single outlier doesn't crush the bulk.
        pct = vals * 100
        # Clip the visualization at the 99th percentile so the histogram
        # readable when one PDB is 1000% off; the outlier is still listed
        # in the outliers table.
        vis_max = max(np.percentile(pct, 99) * 1.5, BANDS[comp] * 100 * 3)
        bins = np.linspace(0, vis_max, 41)
        n_above = (pct > vis_max).sum()
        ax.hist(pct, bins=bins, color=C["proteon"], alpha=0.85, edgecolor="white")

        band_pct = BANDS[comp] * 100
        ax.axvline(band_pct, color="#cb2431", lw=1.2, ls="--", alpha=0.85)
        ax.text(
            band_pct, ax.get_ylim()[1] * 0.92,
            f" {band_pct:.1f}% band",
            color="#cb2431", fontsize=9, va="top",
        )

        # Headline numbers in the panel.
        median = np.median(pct)
        p95 = np.percentile(pct, 95)
        n_pass = (vals < BANDS[comp]).sum()
        n_total = vals.size
        rate = n_pass / n_total * 100
        annotation = (
            f"n={n_total}  pass={n_pass} ({rate:.1f}%)\n"
            f"median={median:.3f}%  p95={p95:.3f}%"
        )
        if n_above > 0:
            annotation += f"\n({n_above} >= {vis_max:.1f}% off-axis)"
        ax.text(
            0.97, 0.97, annotation,
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8.5, color=C["axis"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.9, edgecolor=C["grid"]),
        )

        apply(
            ax,
            title=LABELS[comp],
            xlabel="|proteon − BALL| / |BALL|  (%)",
            ylabel="# structures",
        )
        ax.set_xlim(0, vis_max)

    # Hide unused panels.
    for i in range(len(COMPONENT_ORDER), nrows * ncols):
        axes[i // ncols][i % ncols].set_axis_off()

    fig.suptitle(
        f"CHARMM19+EEF1 vs BALL — per-component rel-diff distribution  (n={len(ok)})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, out)


# --- 11: runtime distribution + scatter ---------------------------------------

def plot_runtime(records: list[dict], out: pathlib.Path) -> None:
    """Two-panel: wall-time histogram (left) + wall-time vs n_atoms scatter (right)."""
    ok = [r for r in records if "wall_s" in r]
    if not ok:
        return

    walls = np.array([r["wall_s"] for r in ok])
    has_n = [r for r in ok if "n_atoms_polh" in r]
    ns = np.array([r["n_atoms_polh"] for r in has_n])
    walls_with_n = np.array([r["wall_s"] for r in has_n])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: histogram with log-y for the long tail.
    bins = np.linspace(0, np.percentile(walls, 99) * 1.5, 41)
    ax1.hist(walls, bins=bins, color=C["proteon"], alpha=0.85, edgecolor="white")
    ax1.set_yscale("log")
    median = np.median(walls)
    p95 = np.percentile(walls, 95)
    ax1.axvline(median, color=C["accent"], lw=1.2, ls="--", alpha=0.85,
                label=f"median {median:.2f}s")
    ax1.axvline(p95, color="#cb2431", lw=1.2, ls=":", alpha=0.85,
                label=f"p95 {p95:.2f}s")
    ax1.legend(loc="upper right", fontsize=9)
    apply(
        ax1, title=f"Per-PDB wall time  (n={len(walls)})",
        xlabel="seconds", ylabel="# structures (log)",
    )

    # Right: scatter wall vs n_atoms.
    if ns.size:
        ax2.scatter(ns, walls_with_n, s=12, alpha=0.4, color=C["proteon"])
        # Linear fit for trend visibility.
        if len(ns) >= 3:
            coef = np.polyfit(ns, walls_with_n, 1)
            xs_fit = np.linspace(ns.min(), ns.max(), 50)
            ax2.plot(xs_fit, np.polyval(coef, xs_fit), color=C["accent"],
                     lw=1.6, alpha=0.85,
                     label=f"linear fit: {coef[0]*1000:.2f}ms/atom + {coef[1]:.2f}s")
            ax2.legend(loc="upper left", fontsize=9)
        apply(
            ax2,
            title="Wall time vs structure size",
            xlabel="n_atoms (polar-H)", ylabel="seconds",
        )
    else:
        ax2.text(0.5, 0.5, "n_atoms not recorded", ha="center", va="center",
                 transform=ax2.transAxes, color=C["axis"])
        ax2.set_axis_off()

    fig.tight_layout()
    savefig(fig, out)


# --- 12: per-component correlation matrix -------------------------------------

def plot_correlations(records: list[dict], out: pathlib.Path) -> None:
    """Pearson correlation matrix of rel_diff across components.

    Diagonal is 1.0 (trivial). Off-diagonal answers: when one component
    is off-band, are others systematically off too? A high correlation
    suggests a shared upstream cause (e.g. atom-typing); zero
    correlation means component-level errors are independent.
    """
    ok = successful_records(records)
    if len(ok) < 5:
        return

    comps = [c for c in COMPONENT_ORDER]
    matrix = np.full((len(comps), len(comps)), np.nan)
    for i, ci in enumerate(comps):
        for j, cj in enumerate(comps):
            xs = np.array([r["rel_diff"].get(ci, np.nan) for r in ok])
            ys = np.array([r["rel_diff"].get(cj, np.nan) for r in ok])
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() < 3:
                continue
            x_v = xs[valid]
            y_v = ys[valid]
            if x_v.std() < 1e-12 or y_v.std() < 1e-12:
                # All identical → correlation is undefined but we draw
                # 1.0 on the diagonal (trivial).
                matrix[i, j] = 1.0 if i == j else np.nan
                continue
            matrix[i, j] = np.corrcoef(x_v, y_v)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(comps)))
    ax.set_yticks(range(len(comps)))
    ax.set_xticklabels([LABELS[c] for c in comps], rotation=30, ha="right")
    ax.set_yticklabels([LABELS[c] for c in comps])
    for i in range(len(comps)):
        for j in range(len(comps)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(v) > 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10)
    ax.set_title(
        f"Per-component rel-diff correlation  (n={len(ok)})",
        fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    savefig(fig, out)


# --- 13: outlier table (HTML fragment) ----------------------------------------

def render_outlier_table(records: list[dict], top_k: int = 20) -> str:
    """Top-K worst-fitting structures by total rel-diff (sum across components).

    Returns an HTML <table> fragment for embedding in the release report.
    Each row: PDB name, n_atoms, the worst component + its rel_diff,
    and the second-worst component + its rel_diff (since outliers
    typically have ONE dominant failure mode).
    """
    ok = successful_records(records)
    if not ok:
        return "<p><em>No successful records to extract outliers from.</em></p>"

    scored: list[tuple[float, dict]] = []
    for r in ok:
        # Score by SUM of rel diffs across the 7 components, weighted
        # equally — gives a scalar "how bad overall" while keeping the
        # per-component breakdown visible in the row.
        comps = r["rel_diff"]
        total = sum(
            v for v in comps.values()
            if isinstance(v, (int, float)) and np.isfinite(v)
        )
        scored.append((total, r))

    scored.sort(key=lambda t: -t[0])
    top = scored[:top_k]

    rows = []
    for total, r in top:
        comps = r["rel_diff"]
        # Sort components by their rel_diff descending.
        comp_pairs = sorted(
            ((c, v) for c, v in comps.items()
             if isinstance(v, (int, float)) and np.isfinite(v)),
            key=lambda kv: -kv[1],
        )
        worst1 = (
            f"{LABELS.get(comp_pairs[0][0], comp_pairs[0][0])}"
            f"&nbsp;<code>{comp_pairs[0][1]*100:.2f}%</code>"
            if comp_pairs else "—"
        )
        worst2 = (
            f"{LABELS.get(comp_pairs[1][0], comp_pairs[1][0])}"
            f"&nbsp;<code>{comp_pairs[1][1]*100:.2f}%</code>"
            if len(comp_pairs) > 1 else "—"
        )
        n_atoms = r.get("n_atoms_polh", "—")
        wall = r.get("wall_s", None)
        wall_s = f"{wall:.2f}s" if isinstance(wall, (int, float)) else "—"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(r['pdb'])}</code></td>"
            f"<td>{n_atoms}</td>"
            f"<td>{wall_s}</td>"
            f"<td>{worst1}</td>"
            f"<td>{worst2}</td>"
            f"<td>{total*100:.2f}%</td>"
            "</tr>"
        )

    return (
        "<table>\n"
        "<thead><tr>"
        "<th>PDB</th><th>n_atoms</th><th>wall</th>"
        "<th>worst component</th><th>2nd-worst</th>"
        "<th>sum rel-diff</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n"
        "</table>\n"
    )


# --- standalone CLI for local rendering ---------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build corpus-oracle plots.")
    parser.add_argument(
        "jsonl",
        type=pathlib.Path,
        help="Path to the JSONL artifact from charmm19_eef1_ball_oracle.py",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "figures",
        help="Output directory for PNG figures + outlier HTML.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_error_distribution(records, args.out_dir / "10_corpus_error_distribution.png")
    plot_runtime(records, args.out_dir / "11_corpus_runtime.png")
    plot_correlations(records, args.out_dir / "12_corpus_correlations.png")
    table_path = args.out_dir / "13_corpus_outliers.html"
    table_path.write_text(render_outlier_table(records))

    n_ok = len(successful_records(records))
    print(f"Rendered corpus-oracle plots from n={len(records)} records "
          f"({n_ok} successful) → {args.out_dir}")


if __name__ == "__main__":
    main()
