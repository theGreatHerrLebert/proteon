# Ferritin

Structural bioinformatics toolkit in Rust with Python bindings.

Ferritin ports battle-tested structural alignment algorithms (TM-align, US-align) to Rust, organized as a modular workspace with a PyO3 Python bridge. Named after the iron-storage protein — because all the best structural biology tools should be forged in Rust.

## What's Inside

| Crate | Purpose |
|-------|---------|
| **ferritin-align** | Core alignment algorithms — Kabsch rotation, TM-score, Needleman-Wunsch DP, secondary structure assignment, plus US-align extensions (SOI-align, MM-align, flex-align) |
| **ferritin-io** | Structure I/O for PDB and mmCIF formats via [pdbtbx](https://github.com/douweschulte/pdbtbx) |
| **ferritin-bin** | CLI binaries: `tmalign` and `usalign` with rayon-parallel batch modes |
| **ferritin-connector** | PyO3 Python bindings (scaffold) |
| **packages/ferritin** | Pythonic wrapper package |

## Quick Start

```bash
# Build everything
cargo build --release

# Align two structures
cargo run --release --bin usalign -- structure1.pdb structure2.pdb

# Tabular output
cargo run --release --bin usalign -- structure1.pdb structure2.pdb --outfmt 2

# Circular permutation
cargo run --release --bin usalign -- structure1.pdb structure2.pdb --cp

# Sequence-order-independent alignment
cargo run --release --bin usalign -- structure1.pdb structure2.pdb --mm 5

# Flexible alignment with hinge detection
cargo run --release --bin usalign -- structure1.pdb structure2.pdb --mm 7
```

## Parallel Batch Alignment

Ferritin uses [rayon](https://github.com/rayon-rs/rayon) to parallelize batch alignments across all available cores — something the original C++ tools can't do.

```bash
# All-against-all: N structures → N*(N-1)/2 pairs, computed in parallel
usalign chain_list.txt --dir /path/to/pdbs/ --outfmt 2

# One-against-many: query against a database
usalign chain_list.txt query.pdb --dir1 /path/to/pdbs/ --outfmt 2

# Many-against-one: database against a query
usalign query.pdb chain_list.txt --dir2 /path/to/pdbs/ --outfmt 2
```

100 structures (4,950 pairs) complete in ~15 seconds on a modern machine — roughly 16x faster than serial C++ USalign.

## Architecture

```
ferritin-align/src/
├── core/                  # TM-align port (~4,500 lines)
│   ├── kabsch.rs          # Kabsch optimal rotation (SVD)
│   ├── tmscore.rs         # TM-score computation
│   ├── nwdp.rs            # Needleman-Wunsch DP (4 variants)
│   ├── secondary_structure.rs
│   ├── residue_map.rs
│   ├── types.rs           # Coord3D, Transform, AlignResult, TMParams
│   └── align/             # tmalign, cpalign, dp_iter, 5 initialization strategies
└── ext/                   # US-align extensions (~6,500 lines)
    ├── blosum.rs           # BLOSUM62 + BLASTN scoring matrix
    ├── nwalign.rs          # Gotoh affine-gap alignment
    ├── se.rs               # Structure Extension refinement
    ├── hwrmsd.rs           # Iterative HwRMSD refinement
    ├── flexalign.rs        # Flexible hinge-based alignment
    ├── soialign/           # Sequence Order Independent alignment
    └── mmalign/            # Multi-chain complex alignment
```

The Python bridge follows the [rustims/imspy](https://github.com/theGreatHerrLebert/rustims) pattern:

```
Pure Rust library (ferritin-align, ferritin-io)
    → PyO3 connector (ferritin-connector, cdylib)
        → Python package (packages/ferritin)
```

## Numerical Fidelity

TM-scores match the original C++ implementations to ~4-5 decimal places. The C++ versions compile with `-ffast-math` which reorders float operations; Rust preserves strict IEEE 754 ordering. On rare small-protein edge cases, the two may find different local optima — this is expected and inherent to the heuristic search.

## References

- Y Zhang, J Skolnick. "TM-align: a protein structure alignment algorithm based on the TM-score." *Nucleic Acids Research* 33, 2302-9 (2005)
- C Zhang, M Shine, AM Pyle, Y Zhang. "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." *Nature Methods* 19(9), 1109-1115 (2022)

## License

MIT — see [LICENSE](LICENSE).
