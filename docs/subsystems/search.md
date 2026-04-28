# Search

!!! warning "Experimental"
    The search stack is the newest part of proteon and the most likely to
    change. Treat the public API as unstable until 1.0.

MMseqs2-compatible structural / sequence search. Includes:

- DB I/O for the `.ffindex` / `.ffdata` / `.dbtype` triplet.
- Reduced-alphabet k-mer prefilter (`.kmi v2` sorted-keys array format).
- Ungapped + gapped Smith-Waterman, both CPU and CUDA.
- PSSM and MSA feature generation.
- GPU kernels: diagonal SW, warp-collab PSSM SW (RTX 5090: up to 124× on
  short queries).

## Example

```python
import proteon

# Build a search DB from a FASTA
proteon.fasta_to_mmseqs_db("uniref50.fa", "out_db")

# Run a query
hits = proteon.search(
    queries=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSV..."],
    target_db="out_db",
    sensitivity=7.5,
)
```

## API reference

### `proteon.search`

::: proteon.search
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### `proteon.msa`

::: proteon.msa
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
