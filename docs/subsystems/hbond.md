# H-bonds

Hydrogen-bond detection between donor / acceptor atoms.

## Example

```python
import proteon

s = proteon.load("1crn.pdb")
bonds = proteon.hbonds(s)

# Batch
structures = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
batch = proteon.batch_hbonds(structures, n_threads=-1)
```

## API reference

::: proteon.hbond
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
