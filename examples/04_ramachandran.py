#!/usr/bin/env python3
"""Example 04: Ramachandran Analysis (Backbone Dihedrals).

Demonstrates:
    - Computing phi/psi/omega backbone dihedral angles
    - Ramachandran statistics and region classification
    - Secondary structure correlation
    - Comparing Ramachandran profiles across structures

Usage:
    python examples/04_ramachandran.py

If matplotlib is available, generates a Ramachandran plot saved as
ramachandran.png.
"""

import proteon
import numpy as np

# ---------------------------------------------------------------------------
# 1. Compute backbone dihedrals
# ---------------------------------------------------------------------------
structure = proteon.load("test-pdbs/1crn.pdb")
phi, psi, omega = proteon.backbone_dihedrals(structure)

print("=== Backbone Dihedrals: Crambin (1CRN) ===")
print(f"  Residues: {len(phi)}")

valid_phi = phi[~np.isnan(phi)]
valid_psi = psi[~np.isnan(psi)]
valid_omega = omega[~np.isnan(omega)]

print(f"  Phi:   {len(valid_phi)} defined, range [{valid_phi.min():.1f}, {valid_phi.max():.1f}]")
print(f"  Psi:   {len(valid_psi)} defined, range [{valid_psi.min():.1f}, {valid_psi.max():.1f}]")
print(f"  Omega: {len(valid_omega)} defined, mean={valid_omega.mean():.1f} "
      f"(expect ~180 for trans peptide bonds)")
print()

# ---------------------------------------------------------------------------
# 2. Ramachandran region classification
# ---------------------------------------------------------------------------
print("=== Ramachandran Regions ===")

# Define broad regions (simplified)
# Alpha-helix:  phi ~ -60, psi ~ -45
# Beta-sheet:   phi ~ -120, psi ~ 130
# Left-handed:  phi > 0


def classify_ramachandran(phi_val, psi_val):
    """Classify a (phi, psi) pair into Ramachandran regions."""
    if np.isnan(phi_val) or np.isnan(psi_val):
        return "undefined"
    if -160 < phi_val < -20 and -80 < psi_val < 0:
        return "alpha_R"  # right-handed alpha helix
    if -180 <= phi_val < -20 and 50 < psi_val <= 180:
        return "beta"     # beta sheet region
    if -180 <= phi_val < -20 and -180 <= psi_val < -100:
        return "beta"     # beta sheet (wrapped region)
    if 0 < phi_val < 120 and -60 < psi_val < 60:
        return "alpha_L"  # left-handed alpha helix
    return "other"


regions = [classify_ramachandran(p, s) for p, s in zip(phi, psi)]
from collections import Counter
counts = Counter(regions)

for region in ["alpha_R", "beta", "alpha_L", "other", "undefined"]:
    n = counts.get(region, 0)
    pct = n / len(phi) * 100
    print(f"  {region:12s}: {n:3d} ({pct:5.1f}%)")
print()

# ---------------------------------------------------------------------------
# 3. Correlate with secondary structure
# ---------------------------------------------------------------------------
print("=== Secondary Structure Correlation ===")

ca_coords = proteon.extract_ca_coords(structure)
ss = proteon.assign_secondary_structure(ca_coords)
print(f"  SS assignment: {ss}")
print()

# Compare SS with Ramachandran region
print(f"  {'Res#':>4s}  {'SS':>2s}  {'Rama':>8s}  {'Phi':>7s}  {'Psi':>7s}")
print(f"  {'----':>4s}  {'--':>2s}  {'--------':>8s}  {'---':>7s}  {'---':>7s}")

df = proteon.to_dataframe(structure)
ca_df = df[df.atom_name.str.strip() == "CA"].reset_index(drop=True)

for i in range(min(20, len(phi))):
    resname = ca_df.iloc[i]["residue_name"] if i < len(ca_df) else "?"
    phi_s = f"{phi[i]:7.1f}" if not np.isnan(phi[i]) else "    NaN"
    psi_s = f"{psi[i]:7.1f}" if not np.isnan(psi[i]) else "    NaN"
    print(f"  {i+1:4d}  {ss[i]:>2s}  {regions[i]:>8s}  {phi_s}  {psi_s}  {resname}")
print()

# ---------------------------------------------------------------------------
# 4. Omega angle analysis (cis/trans peptide bonds)
# ---------------------------------------------------------------------------
print("=== Peptide Bond Geometry ===")

cis_bonds = np.sum(np.abs(valid_omega) < 30)
trans_bonds = np.sum(np.abs(valid_omega) > 150)
other_bonds = len(valid_omega) - cis_bonds - trans_bonds

print(f"  Trans (|omega| > 150):  {trans_bonds}")
print(f"  Cis   (|omega| < 30):   {cis_bonds}")
print(f"  Other:                  {other_bonds}")
if cis_bonds > 0:
    cis_idx = np.where(np.abs(omega) < 30)[0]
    for idx in cis_idx:
        if idx < len(ca_df):
            print(f"    Cis bond before residue {idx+1} ({ca_df.iloc[idx]['residue_name']})")
print()

# ---------------------------------------------------------------------------
# 5. Compare across structures
# ---------------------------------------------------------------------------
print("=== Comparison Across Structures ===")

files = [
    ("test-pdbs/1crn.pdb", "Crambin"),
    ("test-pdbs/1bpi.pdb", "BPTI"),
    ("test-pdbs/1ubq.pdb", "Ubiquitin"),
]

print(f"  {'Structure':>20s}  {'Res':>4s}  {'Alpha%':>6s}  {'Beta%':>6s}  {'Other%':>6s}")
print(f"  {'--------':>20s}  {'---':>4s}  {'------':>6s}  {'------':>6s}  {'------':>6s}")

for path, name in files:
    s = proteon.load(path)
    p, ps, _ = proteon.backbone_dihedrals(s)
    regs = [classify_ramachandran(a, b) for a, b in zip(p, ps)]
    c = Counter(regs)
    n = len(p)
    alpha_pct = c.get("alpha_R", 0) / n * 100
    beta_pct = c.get("beta", 0) / n * 100
    other_pct = 100 - alpha_pct - beta_pct - c.get("undefined", 0) / n * 100
    print(f"  {name:>20s}  {n:4d}  {alpha_pct:5.1f}%  {beta_pct:5.1f}%  {other_pct:5.1f}%")
print()

# ---------------------------------------------------------------------------
# 6. Optional: Ramachandran plot (if matplotlib available)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Color by SS
    colors = {"H": "#e41a1c", "E": "#377eb8", "T": "#4daf4a", "C": "#999999"}
    for i in range(len(valid_phi)):
        # phi[0] is NaN, valid indices are 1..N-1 for phi, 0..N-2 for psi
        pass

    # Plot all valid phi/psi pairs
    mask = ~np.isnan(phi) & ~np.isnan(psi)
    phi_valid = phi[mask]
    psi_valid = psi[mask]
    ss_valid = [ss[i] for i in range(len(ss)) if mask[i]]

    for ss_type, color in colors.items():
        idx = [i for i, s in enumerate(ss_valid) if s == ss_type]
        if idx:
            ax.scatter(phi_valid[idx], psi_valid[idx],
                      c=color, s=30, label=f"{ss_type}", alpha=0.8, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Phi (degrees)", fontsize=12)
    ax.set_ylabel("Psi (degrees)", fontsize=12)
    ax.set_title(f"Ramachandran Plot — {structure.identifier}", fontsize=14)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(title="SS Type")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("ramachandran.png", dpi=150)
    print("  Ramachandran plot saved to ramachandran.png")
except ImportError:
    print("  (matplotlib not installed — skipping plot)")

print("\nDone!")
