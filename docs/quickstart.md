# Quickstart

Proteon is **batch-first**. Single-structure helpers exist, but the default
shape is "load many, compute many, prepare many". The examples below assume
`pip install proteon` has already succeeded.

## Load structures

```python
import proteon

paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = proteon.batch_load(paths, n_threads=-1)
```

`structures` is a list of `proteon.Structure` objects. Each wraps a Rust-side
`PyPDB` and exposes coordinates, residues, chains, and metadata.

!!! note "Threading"
    `n_threads=-1` uses all logical cores. `n_threads=None` does the same.
    `n_threads=0` runs **serially** — almost certainly not what you want.

## Analyze

```python
sasa  = proteon.batch_total_sasa(structures, n_threads=-1)
dssp  = proteon.batch_dssp(structures, n_threads=-1)
hbnd  = proteon.batch_backbone_hbonds(structures, n_threads=-1)
```

## Align

```python
# Single pair
result = proteon.tm_align(structures[0], structures[1])
print(result.tm_score_chain1, result.tm_score_chain2, result.rmsd)

# One query against many
hits = proteon.tm_align_one_to_many(structures[0], structures[1:], n_threads=-1)

# Many-to-many (full pairwise)
matrix = proteon.tm_align_many_to_many(structures, structures, n_threads=-1)
```

The same one-pair / one-to-many / many-to-many shape is available for
`soi_align`, `flex_align`, and `mm_align`.

## Prepare for downstream work

```python
prep = proteon.batch_prepare(
    structures,
    hydrogens="backbone",   # "none" | "backbone" | "all"
    minimize=True,
    n_threads=-1,
)
```

`batch_prepare` adds hydrogens, runs energy minimization with the chosen
forcefield (CHARMM19+EEF1 by default), and returns prepared structures
ready for MD or geometric-DL pipelines.

## Export

`proteon.to_parquet` writes one structure per file (Rust-side, no Python deps):

```python
proteon.to_parquet(structures[0], "1crn.parquet")
restored = proteon.from_parquet("1crn.parquet")
```

For columnar manipulation, `proteon.to_arrow(structure)` returns a
`pyarrow.Table`. `proteon.to_dataframe(structure)` returns a pandas
DataFrame but requires the optional pandas extra:

```bash
pip install 'proteon[pandas]'
```

## Where to go next

- Per-subsystem API reference: [Subsystems](subsystems/index.md).
- How the wrappers map onto the Rust workspace: [Architecture](architecture.md).
- Validation against OpenMM / Biopython / USAlign: [Validation](validation.md).
