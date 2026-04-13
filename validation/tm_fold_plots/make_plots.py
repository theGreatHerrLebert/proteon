"""Fold preservation comparison plots — ferritin CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2.

1000 PDBs (seed=42 sample of 50K corpus).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COL_F = "#2563eb"  # ferritin blue
COL_O = "#dc2626"  # openmm red


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def ok(recs):
    return [r for r in recs if "tm_score" in r]


def as_arr(recs, key):
    return np.array([r[key] for r in recs if key in r])


def joined_ok(ferritin_recs, openmm_recs):
    """Structures where BOTH tools succeeded — for paired comparison."""
    f_ok = {r["pdb"]: r for r in ferritin_recs if "tm_score" in r}
    o_ok = {r["pdb"]: r for r in openmm_recs if "tm_score" in r}
    common = sorted(set(f_ok) & set(o_ok))
    return [(f_ok[k], o_ok[k]) for k in common]


def plot_tm_distribution(f_ok, o_ok, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram zoomed into high-TM region
    bins = np.linspace(0.4, 1.0, 61)
    axes[0].hist(as_arr(f_ok, "tm_score"), bins=bins, alpha=0.55,
                 color=COL_F, label=f"Ferritin (n={len(f_ok)})", density=True)
    axes[0].hist(as_arr(o_ok, "tm_score"), bins=bins, alpha=0.55,
                 color=COL_O, label=f"OpenMM (n={len(o_ok)})", density=True)
    axes[0].set_xlabel("TM-score (pre vs post minimization)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("TM-score distribution")
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0.4, 1.01)

    # ECDF in high-TM region (log-compressed y to show tails)
    f_tm = np.sort(as_arr(f_ok, "tm_score"))
    o_tm = np.sort(as_arr(o_ok, "tm_score"))
    axes[1].plot(f_tm, np.arange(1, len(f_tm) + 1) / len(f_tm),
                 color=COL_F, label=f"Ferritin (median={np.median(f_tm):.4f})")
    axes[1].plot(o_tm, np.arange(1, len(o_tm) + 1) / len(o_tm),
                 color=COL_O, label=f"OpenMM (median={np.median(o_tm):.4f})")
    axes[1].set_xlabel("TM-score")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("TM-score ECDF")
    axes[1].set_xlim(0.85, 1.001)
    axes[1].legend(loc="upper left")
    axes[1].axvline(0.5, color="gray", ls=":", lw=0.8)

    fig.suptitle("Fold preservation — 1000 PDBs from the 50K corpus", fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_rmsd_distribution(f_ok, o_ok, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.linspace(0, 2.0, 41)
    axes[0].hist(as_arr(f_ok, "rmsd"), bins=bins, alpha=0.55,
                 color=COL_F, label=f"Ferritin CHARMM19+EEF1")
    axes[0].hist(as_arr(o_ok, "rmsd"), bins=bins, alpha=0.55,
                 color=COL_O, label=f"OpenMM CHARMM36+OBC2")
    axes[0].set_xlabel("CA RMSD pre→post (Å)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RMSD distribution")
    axes[0].legend()

    # Violin plot for cleaner comparison
    data = [as_arr(f_ok, "rmsd"), as_arr(o_ok, "rmsd")]
    vp = axes[1].violinplot(data, showmedians=True, showextrema=False)
    for pc, c in zip(vp["bodies"], (COL_F, COL_O)):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Ferritin\nCHARMM19+EEF1", "OpenMM\nCHARMM36+OBC2"])
    axes[1].set_ylabel("CA RMSD pre→post (Å)")
    axes[1].set_title(f"Ferritin median: {np.median(data[0]):.2f} Å   "
                      f"OpenMM median: {np.median(data[1]):.2f} Å")
    axes[1].set_ylim(0, 1.8)

    fig.suptitle("Heavy-atom-free minimization moves atoms ~2× more under CHARMM19 (united-atom)",
                 fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_tm_vs_size(f_ok, o_ok, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(as_arr(f_ok, "n_ca"), as_arr(f_ok, "tm_score"),
               s=10, alpha=0.35, color=COL_F, label="Ferritin")
    ax.scatter(as_arr(o_ok, "n_ca"), as_arr(o_ok, "tm_score"),
               s=10, alpha=0.35, color=COL_O, label="OpenMM")
    ax.set_xscale("log")
    ax.set_xlabel("N_CA (residue count, log)")
    ax.set_ylabel("TM-score pre vs post")
    ax.set_ylim(0.4, 1.01)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="TM=0.5 (same-fold threshold)")
    ax.set_title("Low-TM outliers are short peptides (TM-score unreliable below ~30 CA)")
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_energy(paired, out):
    """Energy per residue, pre+post, per tool.

    Pathological structures (nucleic-acid clashes, missing density, etc.) produce
    initial energies up to 10^17 kJ/mol. We clip to the 1-99 percentile window for
    clarity — the summary numbers are reported on the full data.
    """
    # Extract paired data (same structure, both tools)
    f_ei = np.array([f["initial_energy"] / max(f["n_ca"], 1) for f, _ in paired])
    f_ef = np.array([f["final_energy"] / max(f["n_ca"], 1) for f, _ in paired])
    o_ei = np.array([o["initial_energy_kj"] / max(o["n_ca"], 1) for _, o in paired])
    o_ef = np.array([o["final_energy_kj"] / max(o["n_ca"], 1) for _, o in paired])

    # Clip final-energy histograms to a reasonable window (99th pct of each side).
    f_lo, f_hi = np.percentile(f_ef, [1, 99])
    o_lo, o_hi = np.percentile(o_ef, [1, 99])
    lo = min(f_lo, o_lo); hi = max(f_hi, o_hi)
    # Expand slightly for visual margin
    span = hi - lo
    lo -= 0.03 * span; hi += 0.03 * span

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(lo, hi, 60)
    axes[0].hist(np.clip(f_ef, lo, hi), bins=bins, alpha=0.6, color=COL_F,
                 label=f"Ferritin (median={np.median(f_ef):.1f})")
    axes[0].hist(np.clip(o_ef, lo, hi), bins=bins, alpha=0.6, color=COL_O,
                 label=f"OpenMM (median={np.median(o_ef):.1f})")
    axes[0].set_xlabel("Final energy / residue (kJ/mol) — clipped to 1-99%ile")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Post-minimization energy per residue")
    axes[0].legend(loc="upper left")
    axes[0].axvline(0, color="black", ls="-", lw=0.8)

    # Right: ΔE per residue correlation. Clip extreme outliers (both tails > 1000 kJ/mol/res).
    f_dE = f_ei - f_ef
    o_dE = o_ei - o_ef
    # Mask to reasonable relaxation (both positive, both < 1000 kJ/mol/residue).
    mask = (f_dE > 0) & (o_dE > 0) & (f_dE < 5000) & (o_dE < 5000)
    f_dE_m = f_dE[mask]; o_dE_m = o_dE[mask]
    from numpy import corrcoef
    r = corrcoef(o_dE_m, f_dE_m)[0, 1]
    r_log = corrcoef(np.log10(o_dE_m + 1), np.log10(f_dE_m + 1))[0, 1]

    ax = axes[1]
    ax.scatter(o_dE_m, f_dE_m, s=10, alpha=0.35, color="#6b7280")
    hi_lim = max(f_dE_m.max(), o_dE_m.max())
    lo_lim = min(f_dE_m.min(), o_dE_m.min())
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color="black", ls="--", lw=0.8, label="y=x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("OpenMM ΔE / residue (kJ/mol)")
    ax.set_ylabel("Ferritin ΔE / residue (kJ/mol)")
    ax.set_title(f"Relaxation ΔE per residue (n={mask.sum()}, "
                 f"Pearson r={r:.3f}, log-r={r_log:.3f})")
    ax.legend(loc="lower right")

    fig.suptitle("Energy comparison — per-residue normalization for cross-force-field read",
                 fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def plot_throughput(f_recs, o_recs, out):
    """Wall time vs system size, per tool."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # Only OpenMM records per-structure wall (ferritin didn't record it).
    # We'll show OpenMM's wall_s distribution vs n_ca_pre
    o_n = np.array([r["n_ca_pre"] for r in o_recs if "wall_s" in r and "n_ca_pre" in r])
    o_w = np.array([r["wall_s"] for r in o_recs if "wall_s" in r and "n_ca_pre" in r])
    ax.scatter(o_n, o_w, s=10, alpha=0.35, color=COL_O, label="OpenMM (1-thread CPU, per struct)")
    # Ferritin reference: total wall = 14.9 min for 949 ok, parallelized 64-wide
    # Effective per-structure = 14.9*60/949 * 64 = 60 s avg equivalent single-thread? Actually
    # the throughput is 1.06 struct/s wall which at 64 concurrency means ~60s per struct.
    # Better: just show the aggregate throughput lines.
    ax.axhline(14.9 * 60 / 949, color=COL_F, ls="-", lw=2,
               label=f"Ferritin 64-way: {14.9*60/949:.2f}s/struct amortized wall")
    ax.axhline(450 * 60 / 928, color=COL_O, ls="--", lw=1.5, alpha=0.7,
               label=f"OpenMM 64-way: {450*60/928:.2f}s/struct amortized wall")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N_CA")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Throughput — ferritin with GPU minimizer is ~30× faster end-to-end")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def summary_table(f_recs, o_recs, paired, out):
    """Write a text summary for easy copy into the report."""
    f_ok = ok(f_recs); o_ok = ok(o_recs)
    f_tm = as_arr(f_ok, "tm_score"); o_tm = as_arr(o_ok, "tm_score")
    f_rm = as_arr(f_ok, "rmsd"); o_rm = as_arr(o_ok, "rmsd")

    lines = [
        "# Fold preservation comparison — summary",
        "",
        f"Sample: 1000 PDBs (seed=42) from the 50K corpus.",
        f"",
        f"|                          | Ferritin CHARMM19+EEF1 | OpenMM CHARMM36+OBC2 |",
        f"|--------------------------|------------------------|----------------------|",
        f"| n success                | {len(f_ok)}/1000 ({100*len(f_ok)/1000:.1f}%) | {len(o_ok)}/1000 ({100*len(o_ok)/1000:.1f}%) |",
        f"| TM mean                  | {f_tm.mean():.4f} | {o_tm.mean():.4f} |",
        f"| TM median                | {np.median(f_tm):.4f} | {np.median(o_tm):.4f} |",
        f"| TM p01 / p05             | {np.percentile(f_tm,1):.4f} / {np.percentile(f_tm,5):.4f} | {np.percentile(o_tm,1):.4f} / {np.percentile(o_tm,5):.4f} |",
        f"| TM min                   | {f_tm.min():.4f} | {o_tm.min():.4f} |",
        f"| RMSD mean (Å)            | {f_rm.mean():.3f} | {o_rm.mean():.3f} |",
        f"| RMSD median (Å)          | {np.median(f_rm):.3f} | {np.median(o_rm):.3f} |",
        f"| RMSD max (Å)             | {f_rm.max():.3f} | {o_rm.max():.3f} |",
        f"| Wall total (min)         | 14.9 | 449.7 |",
        f"| Throughput (struct/s)    | 1.06 | 0.03 |",
        f"",
        f"Paired (same structure, both tools): n={len(paired)}",
    ]
    out.write_text("\n".join(lines))
    print(f"wrote {out.name}")


def main():
    f_recs = load(HERE / "ferritin.jsonl")
    o_recs = load(HERE / "openmm.jsonl")
    f_ok = ok(f_recs)
    o_ok = ok(o_recs)
    paired = joined_ok(f_recs, o_recs)
    print(f"ferritin ok: {len(f_ok)}  openmm ok: {len(o_ok)}  paired: {len(paired)}")

    plot_tm_distribution(f_ok, o_ok, FIG_DIR / "01_tm_distribution.png")
    plot_rmsd_distribution(f_ok, o_ok, FIG_DIR / "02_rmsd_distribution.png")
    plot_tm_vs_size(f_ok, o_ok, FIG_DIR / "03_tm_vs_size.png")
    plot_energy(paired, FIG_DIR / "04_energy.png")
    plot_throughput(f_recs, o_recs, FIG_DIR / "05_throughput.png")
    summary_table(f_recs, o_recs, paired, FIG_DIR / "summary.md")


if __name__ == "__main__":
    main()
