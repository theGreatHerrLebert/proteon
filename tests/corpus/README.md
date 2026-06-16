# Regression Corpus

Curated PDB fixtures for edge-case testing. Each fixture exercises a specific
parsing or analysis failure mode that has been observed in production.

## Structure

Each subdirectory contains minimal PDB files that trigger a specific edge case:

- `insertion_codes/` — Residues with insertion codes (e.g., prosegment interleaving)
- `multimodel/` — NMR-style multi-model structures
- `altloc/` — Alternate conformations (A/B occupancy)
- `missing_atoms/` — Incomplete residues:
  - `missing_cb.pdb` — a missing **sidechain** atom (CB); reconstruct should add it
  - `missing_backbone_c.pdb` — a missing **backbone** atom (carbonyl C of res 2);
    the affected dihedrals must NaN, not crash
- `ligands/` — Structures with HETATM ligands (`protein_with_ligand.pdb`)
- `waters/` — Crystallographic HOH solvent (`protein_with_waters.pdb`); must load,
  stay distinguishable from protein, and not break energy/prepare
- `chain_breaks/` — Chains with sequence gaps (missing residues)

## Policy

- Every parsing or analysis bug gets a regression fixture before closing
- Fixtures should be as small as possible while reproducing the issue
- Each fixture has a comment explaining why it exists
