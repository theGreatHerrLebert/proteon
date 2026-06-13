#!/usr/bin/env python3
"""Example 12: Virtual Screening with AutoDock-Vina Docking.

Screens a small ligand library against one receptor pocket and ranks the
hits by docked affinity — the core virtual-screening loop:

    1. Define the search box once, at the receptor's binding site.
    2. ``proteon.dock`` each library ligand into that box (Monte-Carlo
       global search places the ligand from scratch, so a ligand's input
       coordinate frame doesn't matter — only its topology).
    3. Rank by the best docked affinity.

Inputs are PDBQT (receptor + ligands), the same format AutoDock Vina uses;
real screens prepare ligands with Meeko / AutoDockTools. Here we reuse
proteon-vina's bundled fixtures as a tiny stand-in library so the example
runs with no extra data. We also show ``proteon.score_only`` for the case
where you already have a pose and just want to re-score it.

Usage:
    python examples/12_virtual_screening.py
"""

from pathlib import Path

import proteon

FIX = Path(__file__).resolve().parent.parent / "proteon-vina" / "tests" / "fixtures" / "pairs"
RECEPTOR = FIX / "1iep" / "receptor.pdbqt"
COGNATE = FIX / "1iep" / "ligand.pdbqt"          # imatinib — defines the pocket
LIBRARY = {
    "imatinib": FIX / "1iep" / "ligand.pdbqt",   # the cognate ligand
    "1fpu_lig": FIX / "1fpu" / "ligand.pdbqt",
    "1s63_lig": FIX / "1s63" / "ligand.pdbqt",
    "bace1_lig": FIX / "bace1" / "ligand.pdbqt",
}


def pdbqt_box(pdbqt_text: str, padding: float = 5.0):
    """Search box (center, size) enclosing a PDBQT's atoms, padded."""
    xs, ys, zs = [], [], []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    if not xs:
        raise ValueError("PDBQT has no ATOM/HETATM records — cannot build a box")
    lo = [min(xs), min(ys), min(zs)]
    hi = [max(xs), max(ys), max(zs)]
    center = tuple((l + h) / 2 for l, h in zip(lo, hi))
    size = tuple((h - l) + 2 * padding for l, h in zip(lo, hi))
    return center, size


receptor = RECEPTOR.read_text()

# Re-scoring an existing pose (when you already have one): the cognate pose.
cognate_score = proteon.score_only(receptor, COGNATE.read_text())
print(f"score_only on the cognate pose (imatinib): {cognate_score.total:.2f} kcal/mol\n")

# Define the pocket box once, from the cognate ligand, and screen the library
# into it. Small budget so the example finishes quickly; real screens use the
# defaults (exhaustiveness=8, global_steps=2500).
center, size = pdbqt_box(COGNATE.read_text())
print(f"Docking {len(LIBRARY)} ligands into the 1iep pocket "
      f"(center {tuple(round(c, 1) for c in center)}, grid path):\n")

results = []
for name, path in LIBRARY.items():
    modes = proteon.dock(
        receptor,
        path.read_text(),
        center=center,
        size=size,
        exhaustiveness=4,
        n_poses=5,
        global_steps=150,
        seed=0,
        use_grid=True,
        n_threads=-1,
    )
    best = min((m.total for m in modes), default=float("nan"))
    results.append((name, best, len(modes)))

results.sort(key=lambda r: r[1])
print(f"  {'rank':>4}  {'ligand':10} {'affinity':>9}  modes")
for rank, (name, best, n) in enumerate(results, 1):
    print(f"  {rank:>4}  {name:10} {best:9.2f}  {n}")

print("\nThe cognate ligand (imatinib) should top the ranking. Swap in your own "
      "prepared PDBQT library and a real budget to screen for real.")
