#!/usr/bin/env python3
"""Example 05: Solvent Accessible Surface Area (SASA).

Demonstrates:
    - Per-atom and per-residue SASA computation
    - Relative solvent accessibility (RSA) for burial classification
    - Batch SASA across many structures (parallel)
    - Load + SASA in one call (zero GIL pipeline)

Usage:
    python examples/05_sasa_analysis.py
"""

import proteon
import numpy as np

# ---------------------------------------------------------------------------
# 1. Per-atom SASA
# ---------------------------------------------------------------------------
structure = proteon.load("test-pdbs/1crn.pdb")

print("=== Crambin SASA ===")
atom_sasa = proteon.atom_sasa(structure)
print(f"  Total SASA:    {atom_sasa.sum():.0f} A²")
print(f"  Exposed atoms: {(atom_sasa > 0).sum()}/{len(atom_sasa)}")
print(f"  Buried atoms:  {(atom_sasa == 0).sum()}")
print()

# ---------------------------------------------------------------------------
# 2. Per-residue SASA + burial classification
# ---------------------------------------------------------------------------
print("=== Per-Residue SASA ===")
res_sasa = proteon.residue_sasa(structure)
rsa = proteon.relative_sasa(structure)

df = proteon.to_dataframe(structure)
ca_df = df[df.atom_name.str.strip() == "CA"].reset_index(drop=True)

print(f"  {'Res':>6s}  {'SASA':>7s}  {'RSA':>5s}  {'Class':>7s}")
print(f"  {'---':>6s}  {'----':>7s}  {'---':>5s}  {'-----':>7s}")
for i in range(min(20, len(res_sasa))):
    if i >= len(ca_df):
        break
    name = ca_df.iloc[i]["residue_name"]
    num = ca_df.iloc[i]["residue_number"]
    r = rsa[i] if i < len(rsa) and not np.isnan(rsa[i]) else 0
    cls = "exposed" if r >= 0.25 else "buried"
    print(f"  {name}{num:3d}  {res_sasa[i]:7.1f}  {r:5.2f}  {cls:>7s}")
print()

# Summary
valid_rsa = rsa[~np.isnan(rsa)]
print(f"  Buried (RSA < 0.25):   {(valid_rsa < 0.25).sum()} residues")
print(f"  Exposed (RSA >= 0.25): {(valid_rsa >= 0.25).sum()} residues")
print()

# ---------------------------------------------------------------------------
# 3. Batch SASA (parallel)
# ---------------------------------------------------------------------------
print("=== Batch SASA ===")
import glob
files = sorted(glob.glob("test-pdbs/*.pdb"))
structures = []
for f in files:
    try:
        structures.append(proteon.load(f))
    except:
        pass

totals = proteon.batch_total_sasa(structures, n_threads=-1)
print(f"  {len(structures)} structures analyzed in parallel")
for i, s in enumerate(structures):
    name = s.identifier or f"struct{i}"
    print(f"    {name:>6s}: {totals[i]:8.0f} A²  ({s.atom_count:5d} atoms)")
print()

# ---------------------------------------------------------------------------
# 4. Load + SASA (zero GIL pipeline)
# ---------------------------------------------------------------------------
print("=== Load + SASA Pipeline ===")
import os
all_files = sorted(glob.glob("test-pdbs/*.pdb"))[:10]
results = proteon.load_and_sasa(all_files, n_threads=-1)
print(f"  {len(results)}/{len(all_files)} structures loaded + analyzed")
for idx, total_sasa in results[:5]:
    basename = os.path.basename(all_files[idx])
    print(f"    {basename:>15s}: {total_sasa:8.0f} A²")
print()

print("Done!")
