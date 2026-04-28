# Supervision

"Layer 5" — geometric-DL data export. Turns prepared structures into
training-ready supervision targets (atom37 indexing, masks, frames).

The Rust supervision pipeline is parity-tested against the Python
implementation; see `devdocs/RUST_BATCH_SUPERVISION_CONTRACT.md` and
`devdocs/STRUCTURE_SUPERVISION_SCHEMA.md` (in the repo) for the schema and
the contract between the two paths.

## Example

```python
import proteon

structures = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
prepared   = proteon.batch_prepare(structures, n_threads=-1)
batch      = proteon.batch_supervision(prepared, n_threads=-1)
```

## API reference

::: proteon.supervision
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
