# I/O

Loading and saving PDB / mmCIF structures. Batch-first. Backed by `pdbtbx`
on the Rust side.

## Example

```python
import proteon

# Single
s = proteon.load("1crn.pdb")
print(s.coords.shape)

# Batch — the default shape
paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = proteon.batch_load(paths, n_threads=-1)
```

`batch_load` returns a list of `proteon.Structure`. Failed loads are reported
in the loader-failure analysis path (see `proteon.loader_failure_analysis`)
rather than raising on the first error.

## API reference

::: proteon.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
