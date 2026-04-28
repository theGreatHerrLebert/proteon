# DSSP

Secondary-structure assignment using the DSSP algorithm.

## Example

```python
import proteon

s = proteon.load("1crn.pdb")
codes = proteon.dssp(s)              # one code per residue: H, E, T, S, ...

# Batch
structures = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
batch = proteon.batch_dssp(structures, n_threads=-1)
```

## API reference

::: proteon.dssp
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
