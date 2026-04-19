#!/usr/bin/env python3
"""Example 02: Structural Alignment with TM-align.

Demonstrates:
    - Pairwise TM-align between two structures
    - Interpreting alignment results (TM-score, RMSD, sequence identity)
    - Applying the superposition transform
    - One-to-many batch alignment (parallel)
    - Many-to-many all-pairs comparison (parallel)
    - Tabular output of batch results

Usage:
    python examples/02_structural_alignment.py
"""

import proteon
import numpy as np

# ---------------------------------------------------------------------------
# 1. Pairwise alignment
# ---------------------------------------------------------------------------
print("=== Pairwise TM-align ===")

s1 = proteon.load("test-pdbs/1crn.pdb")
s2 = proteon.load("test-pdbs/1bpi.pdb")

result = proteon.tm_align(s1, s2)

print(f"  Structure 1:    {s1.identifier} ({s1.residue_count} residues)")
print(f"  Structure 2:    {s2.identifier} ({s2.residue_count} residues)")
print(f"  TM-score (s1):  {result.tm_score_chain1:.4f}")
print(f"  TM-score (s2):  {result.tm_score_chain2:.4f}")
print(f"  RMSD:           {result.rmsd:.2f} A")
print(f"  Aligned:        {result.n_aligned} residues")
print(f"  Seq identity:   {result.seq_identity:.1%}")
print()

# Aligned sequences with gaps
print("  Alignment:")
seq_x = result.aligned_seq_x
seq_y = result.aligned_seq_y
match_line = "".join(
    "|" if a == b and a != "-" else " " for a, b in zip(seq_x, seq_y)
)
# Show first 70 characters
print(f"    {seq_x[:70]}")
print(f"    {match_line[:70]}")
print(f"    {seq_y[:70]}")
print()

# ---------------------------------------------------------------------------
# 2. Apply superposition to coordinates
# ---------------------------------------------------------------------------
print("=== Superposition ===")

# The rotation/translation from alignment maps s1 onto s2's frame
R = result.rotation_matrix
t = result.translation

# Superpose all atoms of s1
original_coords = s1.coords
superposed = proteon.apply_transform(original_coords, R, t)

print(f"  Original centroid:   ({proteon.centroid(original_coords)[0]:.2f}, "
      f"{proteon.centroid(original_coords)[1]:.2f}, "
      f"{proteon.centroid(original_coords)[2]:.2f})")
print(f"  Superposed centroid: ({proteon.centroid(superposed)[0]:.2f}, "
      f"{proteon.centroid(superposed)[1]:.2f}, "
      f"{proteon.centroid(superposed)[2]:.2f})")
print(f"  Target centroid:     ({proteon.centroid(s2.coords)[0]:.2f}, "
      f"{proteon.centroid(s2.coords)[1]:.2f}, "
      f"{proteon.centroid(s2.coords)[2]:.2f})")
print()

# ---------------------------------------------------------------------------
# 3. One-to-many alignment (parallel)
# ---------------------------------------------------------------------------
print("=== One-to-Many (Parallel) ===")

# Load several target structures
target_files = ["test-pdbs/1crn.pdb", "test-pdbs/1bpi.pdb",
                "test-pdbs/1aaj.pdb", "test-pdbs/1ubq.pdb"]
targets = [proteon.load(f) for f in target_files]

# Align query against all targets using all available CPU cores
results = proteon.tm_align_one_to_many(s1, targets, n_threads=4)

print(f"  Query: {s1.identifier} ({s1.residue_count} residues)")
print(f"  {'Target':>10s}  {'Res':>4s}  {'TM1':>6s}  {'TM2':>6s}  {'RMSD':>6s}  {'Aligned':>7s}  {'SeqId':>6s}")
print(f"  {'------':>10s}  {'---':>4s}  {'---':>6s}  {'---':>6s}  {'----':>6s}  {'-------':>7s}  {'-----':>6s}")
for i, r in enumerate(results):
    t = targets[i]
    tid = t.identifier or f"struct{i}"
    print(f"  {tid:>10s}  {t.residue_count:4d}  "
          f"{r.tm_score_chain1:6.4f}  {r.tm_score_chain2:6.4f}  "
          f"{r.rmsd:6.2f}  {r.n_aligned:7d}  {r.seq_identity:6.1%}")
print()

# ---------------------------------------------------------------------------
# 4. Many-to-many (all pairs, parallel)
# ---------------------------------------------------------------------------
print("=== Many-to-Many (All Pairs) ===")

# Use the same set as both queries and targets for a symmetric comparison
all_results = proteon.tm_align_many_to_many(targets, targets, n_threads=4)

# Build a TM-score matrix
n = len(targets)
tm_matrix = np.zeros((n, n))
for qi, ti, r in all_results:
    # Use the TM-score normalized by the shorter structure (max of both)
    tm_matrix[qi, ti] = max(r.tm_score_chain1, r.tm_score_chain2)

names = [t.identifier or f"struct{i}" for i, t in enumerate(targets)]
print(f"  TM-score matrix (higher = more similar):")
print(f"  {'':>10s}  " + "  ".join(f"{n:>6s}" for n in names))
for i in range(n):
    row = "  ".join(f"{tm_matrix[i, j]:6.3f}" for j in range(n))
    print(f"  {names[i]:>10s}  {row}")
print()

# ---------------------------------------------------------------------------
# 5. Fast mode for large-scale screening
# ---------------------------------------------------------------------------
print("=== Fast Mode ===")
fast_result = proteon.tm_align(s1, s2, fast=True)
print(f"  Normal: TM={result.tm_score_chain1:.4f}, RMSD={result.rmsd:.2f}")
print(f"  Fast:   TM={fast_result.tm_score_chain1:.4f}, RMSD={fast_result.rmsd:.2f}")
print(f"  (fast mode trades ~1% accuracy for ~3x speed)")
print()

print("Done!")
