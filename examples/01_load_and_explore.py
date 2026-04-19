#!/usr/bin/env python3
"""Example 01: Load and Explore a Protein Structure.

Demonstrates:
    - Loading PDB and mmCIF files
    - Navigating the hierarchy (Model > Chain > Residue > Atom)
    - Extracting coordinates as numpy arrays
    - Exporting to a pandas DataFrame for interactive exploration

Usage:
    python examples/01_load_and_explore.py
"""

import proteon
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load a structure
# ---------------------------------------------------------------------------
# proteon auto-detects format from the extension (.pdb, .cif, .mmcif)
structure = proteon.load("test-pdbs/1crn.pdb")

print("=== Structure Overview ===")
print(f"  Identifier:  {structure.identifier}")
print(f"  Models:      {structure.model_count}")
print(f"  Chains:      {structure.chain_count}")
print(f"  Residues:    {structure.residue_count}")
print(f"  Atoms:       {structure.atom_count}")
print()

# ---------------------------------------------------------------------------
# 2. Navigate the hierarchy
# ---------------------------------------------------------------------------
print("=== Chain Summary ===")
for chain in structure.chains:
    print(f"  Chain {chain.id}: {chain.residue_count} residues, {chain.atom_count} atoms")
print()

print("=== First 5 Residues ===")
for res in structure.residues[:5]:
    aa_flag = " (amino acid)" if res.is_amino_acid else ""
    print(f"  {res.chain_id}:{res.name} {res.serial_number} — {len(res)} atoms{aa_flag}")
print()

print("=== First 5 Atoms ===")
for atom in structure.atoms[:5]:
    print(f"  {atom.name:4s}  {atom.element:2s}  "
          f"{atom.residue_name:3s} {atom.chain_id}{atom.residue_serial_number:4d}  "
          f"({atom.x:7.3f}, {atom.y:7.3f}, {atom.z:7.3f})  "
          f"B={atom.b_factor:.2f}")
print()

# ---------------------------------------------------------------------------
# 3. Numpy arrays — zero-copy access to bulk data
# ---------------------------------------------------------------------------
print("=== Numpy Arrays ===")
coords = structure.coords       # Nx3 float64
bfactors = structure.b_factors  # N float64

print(f"  Coordinates:  {coords.shape} {coords.dtype}")
print(f"  B-factors:    {bfactors.shape}, mean={bfactors.mean():.1f}, max={bfactors.max():.1f}")
print(f"  Coord range:  x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]"
      f"  y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]"
      f"  z=[{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")
print()

# ---------------------------------------------------------------------------
# 4. Quick geometric properties
# ---------------------------------------------------------------------------
print("=== Geometry ===")
center = proteon.centroid(coords)
rg = proteon.radius_of_gyration(coords)
print(f"  Centroid:     ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print(f"  Rg:           {rg:.2f} A")

# CA coordinates only
ca_coords = proteon.extract_ca_coords(structure)
print(f"  CA atoms:     {len(ca_coords)}")
print(f"  CA Rg:        {proteon.radius_of_gyration(ca_coords):.2f} A")
print()

# ---------------------------------------------------------------------------
# 5. DataFrame export — bridge to pandas/polars
# ---------------------------------------------------------------------------
print("=== DataFrame Export ===")
df = proteon.to_dataframe(structure)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print()

# Filter to CA atoms
ca_df = df[df.atom_name.str.strip() == "CA"]
print(f"  CA atoms ({len(ca_df)} rows):")
print(ca_df[["residue_name", "residue_number", "chain_id", "x", "y", "z"]].head(10).to_string(index=False))
print()

# Per-residue B-factor summary
print("  Mean B-factor by residue type (top 5):")
by_restype = df.groupby("residue_name")["b_factor"].mean().sort_values(ascending=False)
for name, val in by_restype.head(5).items():
    print(f"    {name}: {val:.1f}")
print()

# ---------------------------------------------------------------------------
# 6. Loading mmCIF format
# ---------------------------------------------------------------------------
# Same API — just change the file extension
# cif_structure = proteon.load("structure.cif")

print("Done! Crambin (1CRN) has 46 residues and 327 atoms.")
