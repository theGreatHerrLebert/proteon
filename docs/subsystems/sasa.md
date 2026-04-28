# SASA

Solvent-accessible surface area. Per-atom, per-residue, and total.

Validated against Biopython on 1,000 PDBs: **0.17% median deviation**
(see [Validation](../validation.md)).

## Example

```python
import proteon

s = proteon.load("1crn.pdb")

per_atom = proteon.atom_sasa(s)          # array, len == n_atoms
per_res  = proteon.residue_sasa(s)       # per-residue sums
rsa      = proteon.relative_sasa(s)      # residue SASA / max for residue type
total    = proteon.total_sasa(s)         # scalar

# Batch (total only — fastest path)
structures = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
totals = proteon.batch_total_sasa(structures, n_threads=-1)
```

## API reference

::: proteon.sasa
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
