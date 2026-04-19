#!/usr/bin/env python3
"""Example 03: Contact Maps and Distance Analysis.

Demonstrates:
    - Computing CA-CA distance matrices
    - Creating contact maps at various cutoffs
    - Cross-structure distance comparison
    - Contact statistics useful for geometric deep learning

Usage:
    python examples/03_contact_map.py
"""

import proteon
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load structure and extract CA coordinates
# ---------------------------------------------------------------------------
structure = proteon.load("test-pdbs/1crn.pdb")
ca = proteon.extract_ca_coords(structure)

print("=== Crambin (1CRN) ===")
print(f"  Residues: {len(ca)}")
print(f"  All atoms: {structure.atom_count}")
print()

# ---------------------------------------------------------------------------
# 2. Distance matrix
# ---------------------------------------------------------------------------
dm = proteon.distance_matrix(ca)

print("=== CA-CA Distance Matrix ===")
print(f"  Shape: {dm.shape}")
print(f"  Min neighbor distance: {dm[dm > 0.1].min():.2f} A")
print(f"  Max distance: {dm.max():.2f} A")
print(f"  Mean distance: {dm.mean():.2f} A")
print()

# Show a 10x10 corner of the matrix
print("  First 10x10 corner (Angstroms):")
print("       " + "".join(f"{i:6d}" for i in range(10)))
for i in range(10):
    row = "".join(f"{dm[i,j]:6.1f}" for j in range(10))
    print(f"  {i:4d} {row}")
print()

# ---------------------------------------------------------------------------
# 3. Contact maps at different cutoffs
# ---------------------------------------------------------------------------
print("=== Contact Maps ===")

for cutoff in [6.0, 8.0, 10.0, 12.0]:
    cm = proteon.contact_map(ca, cutoff=cutoff)
    n_contacts = cm.sum() - len(ca)  # subtract self-contacts (diagonal)
    n_pairs = n_contacts // 2  # each contact counted twice
    density = n_contacts / (len(ca) * (len(ca) - 1))
    print(f"  Cutoff {cutoff:5.1f} A: {n_pairs:4d} contacts, density={density:.3f}")
print()

# ---------------------------------------------------------------------------
# 4. Long-range vs short-range contacts
# ---------------------------------------------------------------------------
print("=== Contact Classification ===")

cm8 = proteon.contact_map(ca, cutoff=8.0)
n = len(ca)

short_range = 0   # |i-j| <= 4  (within a helix turn)
medium_range = 0  # 4 < |i-j| <= 12
long_range = 0    # |i-j| > 12

for i in range(n):
    for j in range(i + 1, n):
        if cm8[i, j]:
            sep = j - i
            if sep <= 4:
                short_range += 1
            elif sep <= 12:
                medium_range += 1
            else:
                long_range += 1

total = short_range + medium_range + long_range
print(f"  Short-range  (|i-j| <= 4):  {short_range:4d} ({short_range/total:.1%})")
print(f"  Medium-range (4 < |i-j| <= 12): {medium_range:4d} ({medium_range/total:.1%})")
print(f"  Long-range   (|i-j| > 12): {long_range:4d} ({long_range/total:.1%})")
print()

# ---------------------------------------------------------------------------
# 5. Cross-structure distance comparison
# ---------------------------------------------------------------------------
print("=== Cross-Structure Distances ===")

s2 = proteon.load("test-pdbs/1bpi.pdb")
ca2 = proteon.extract_ca_coords(s2)

# NxM cross-distance matrix
cross_dm = proteon.distance_matrix(ca, ca2)
print(f"  Structure 1: {len(ca)} residues (1CRN)")
print(f"  Structure 2: {len(ca2)} residues ({s2.identifier})")
print(f"  Cross-distance matrix: {cross_dm.shape}")
print(f"  Min cross-distance: {cross_dm.min():.2f} A")
print()

# ---------------------------------------------------------------------------
# 6. Contact number per residue (coordination number)
# ---------------------------------------------------------------------------
print("=== Per-Residue Contact Count (8A cutoff) ===")

contact_count = cm8.sum(axis=1) - 1  # subtract self
df = proteon.to_dataframe(structure)
ca_df = df[df.atom_name.str.strip() == "CA"].reset_index(drop=True)

print(f"  {'Residue':>10s}  {'Contacts':>8s}")
print(f"  {'-------':>10s}  {'--------':>8s}")
# Show top 10 most connected residues
top_idx = np.argsort(contact_count)[::-1][:10]
for idx in top_idx:
    resname = ca_df.iloc[idx]["residue_name"]
    resnum = ca_df.iloc[idx]["residue_number"]
    print(f"  {resname}{resnum:>4d}       {contact_count[idx]:4d}")
print()

print("Done!")
