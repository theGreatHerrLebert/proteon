# Regression Corpus

Curated PDB fixtures for edge-case testing. Each fixture exercises a specific
parsing or analysis failure mode that has been observed in production.

## Structure

Each subdirectory contains minimal PDB files that trigger a specific edge case:

- `insertion_codes/` — Residues with insertion codes (e.g., prosegment interleaving)
- `multimodel/` — NMR-style multi-model structures
- `altloc/` — Alternate conformations (A/B occupancy)
- `missing_atoms/` — Incomplete residues (missing backbone or sidechain atoms)
- `ligands/` — Structures with HETATM ligands/waters
- `chain_breaks/` — Chains with sequence gaps (missing residues)

## Policy

- Every parsing or analysis bug gets a regression fixture before closing
- Fixtures should be as small as possible while reproducing the issue
- Each fixture has a comment explaining why it exists
