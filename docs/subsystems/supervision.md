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

structures   = proteon.batch_load(["1crn.pdb", "1ubq.pdb"], n_threads=-1)
prep_reports = proteon.batch_prepare(structures, n_threads=-1)  # mutates in place

examples = proteon.batch_build_structure_supervision_examples(
    structures,
    prep_reports=prep_reports,   # optional but recommended
)
```

`batch_prepare` returns a list of `PrepReport` (energies, convergence flags,
hydrogen counts) and mutates the input structures in place. Pass both the
structures and the reports into the supervision builder so quality metadata
flows through.

`examples` is a list of `StructureSupervisionExample`, ready to be exported
to Parquet / Arrow for geometric-DL training pipelines. Single-structure
entry point: `proteon.build_structure_supervision_example(structure)`.

## API reference

::: proteon.supervision
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
