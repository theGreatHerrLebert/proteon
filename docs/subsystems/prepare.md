# Preparation

End-to-end "make this structure usable" pipeline: add missing hydrogens,
optionally minimize, return a structure ready for MD or geometric-DL work.

`batch_prepare` is the most common entry point. The pieces are also exposed
individually in `proteon.hydrogens` and `proteon.forcefield` if you want to
mix and match.

## Example

```python
import proteon

paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = proteon.batch_load(paths, n_threads=-1)

prep = proteon.batch_prepare(
    structures,
    hydrogens="backbone",   # "none" | "backbone" | "all"
    minimize=True,
    n_threads=-1,
)
```

## Validation

50K random PDB battle test on RTX 5090: **99.1% correct in 3.5 hours**
(CHARMM19 + EEF1 + SASA on CUDA). Fold preservation on 1000 PDBs:
proteon CHARMM19 + EEF1 has median TM = 0.9945, **30× faster** than
OpenMM CHARMM36 + OBC2.

## API reference

### `proteon.prepare`

::: proteon.prepare
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### `proteon.hydrogens`

::: proteon.hydrogens
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
