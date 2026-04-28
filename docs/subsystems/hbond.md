# H-bonds

Hydrogen-bond detection between donor / acceptor atoms.

Three top-level entry points are exposed, all returning the same shape:

- `proteon.backbone_hbonds(s)` — restrict to backbone N-H ⋯ O=C donors / acceptors.
- `proteon.geometric_hbonds(s)` — geometric criterion over all donor / acceptor pairs.
- `proteon.hbond_count(s)` — convenience: just the integer count.

## Example

```python
import proteon

s = proteon.load("1crn.pdb")
bonds = proteon.backbone_hbonds(s)
n     = proteon.hbond_count(s)

# Batch
structures = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
batch = proteon.batch_backbone_hbonds(structures, n_threads=-1)
```

## API reference

::: proteon.hbond
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
