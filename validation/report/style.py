"""Shared matplotlib style for the proteon validation report.

Design principles:
  * Sans-serif DejaVu / Helvetica-style, 11 pt body, 13 pt labels, 14 pt titles.
  * Each tool has a distinct, printable color. Colors chosen for adjacent
    contrast in bar/line plots and for viridis-adjacent perception:
      proteon = deep teal (brand, calm, technical)
      openmm   = warm orange-red (matches openmm.org branding)
      gromacs  = purple (matches gromacs.org)
  * Lines are 1.6 pt. Markers are 30 px^2 (size=5 in scatter).
  * Minor gridlines off; major gridlines at alpha 0.25.
  * Tight layout + bbox_inches='tight' on save.
  * 200 dpi PNG + 2x-density via plt.rcParams['figure.dpi'].

Import this module once at the top of any plot script:

    from style import apply, C, savefig

    fig, ax = plt.subplots()
    ax.plot(xs, ys, color=C["proteon"], lw=1.6, label="Proteon")
    apply(ax, title="Fold preservation", xlabel="TM-score")
    savefig(fig, "01_tm_overlay.png")
"""
from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Color palette ------------------------------------------------------------
#
# Colorblind-safe, distinct under printing. Verified in Chromatic Vision
# Simulator (protanopia/deuteranopia/tritanopia).
C = {
    "proteon": "#0a7e8c",    # deep teal
    "openmm":   "#d94801",    # OpenMM orange-red
    "gromacs":  "#6a3d9a",    # GROMACS purple
    "ball":     "#4a4a4a",    # slate
    "amber":    "#b87333",    # warm amber tag (for FF-family shading)
    "charmm":   "#1f78b4",    # cool blue (for FF-family shading)
    "grid":     "#cccccc",
    "axis":     "#333333",
    "accent":   "#ffb400",    # highlight
}

# Neutral palette for 3+ category plots that aren't about specific tools.
PALETTE = [C["proteon"], C["openmm"], C["gromacs"], C["ball"], "#e31a1c", "#33a02c"]

# --- RC params ----------------------------------------------------------------
_RC = {
    # Fonts: stay on DejaVu for universal availability; report-grade but
    # works without system font installs.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 12,
    "axes.labelweight": "normal",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "legend.frameon": False,
    # Axis / ticks / spines.
    "axes.edgecolor": C["axis"],
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.color": C["axis"],
    "ytick.color": C["axis"],
    # Grid.
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    # Lines / markers.
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "patch.linewidth": 0.6,
    # Figure.
    "figure.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.transparent": False,
    # Style overrides for report context.
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
}


def apply_rc() -> None:
    """Apply the report-wide rcParams."""
    plt.rcParams.update(_RC)


def apply(ax, *, title: str | None = None, xlabel: str | None = None,
          ylabel: str | None = None, legend: bool = True,
          legend_loc: str = "best", subtitle: str | None = None) -> None:
    """Uniform axis finalization — call before savefig on each plot.

    Title / subtitle: the subtitle is rendered as a smaller grey line ABOVE
    the axis using `ax.text` in axes coords, anchored at y=1.015. The main
    title sits above it at y=1.08 via the standard `set_title` mechanism.
    """
    if title is not None:
        if subtitle:
            ax.set_title(title, loc="left", pad=22)
            ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                    fontsize=9.5, color="#666", ha="left", va="bottom",
                    style="italic")
        else:
            ax.set_title(title, loc="left")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)
    ax.tick_params(which="both", length=4)


def savefig(fig, path: str | pathlib.Path) -> None:
    """Save and close. Always writes PNG at savefig.dpi from rcParams."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)


# Apply once at import — modules that use this file inherit the style.
apply_rc()
