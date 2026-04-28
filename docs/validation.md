# Validation

Proteon is validated against established oracles wherever the algorithm has
a canonical reference implementation. The numbers below are reproducible
from `tests/oracle/` if the corresponding fixtures are installed (see
[Oracle setup](ORACLE_SETUP.md)).

## Headline numbers

| Component | Oracle | Result |
|-----------|--------|--------|
| TM-align | USAlign reference | 0.003 median TM drift on 4,656 pairs |
| SASA | Biopython | 0.17% median deviation on 1,000 PDBs |
| AMBER96 (NoCutoff) | OpenMM | ≤0.5% on all components — 218/218 invariants pass |
| OBC GB (Phase B) | OpenMM | ≤5% GB / ≤1% total on crambin |
| OBC GB GPU vs CPU | self | matches to 1e-11 |
| CHARMM19 + EEF1 prep | self (50K random PDBs) | 99.1% correct in 3.5h on RTX 5090 |
| Fold preservation | OpenMM CHARMM36 + OBC2 | proteon median TM = 0.9945, **30× faster** |
| End-to-end ingest | self | 45,100 PDBs in 17.9 min on 120 cores |

## Numerical-precision policy

The C++ TMalign / USalign references are compiled with `-ffast-math`, which
Rust does not permit. As a result:

- **TM-scores match to ~4–5 decimal places.** This is the precision floor
  imposed by `-ffast-math` reordering, not a bug in the Rust port.
- **Kabsch / SVD constants are preserved exactly:** `epsilon=1e-8`,
  `tol=0.01`, `sqrt3=1.73205080756888`.
- The `d0` formula uses `.powf(1.0/3.0)`, not `.cbrt()`, to match the C++
  `pow(x, 1.0/3)` rounding.

## Oracle list

Full oracle inventory and rationale lives in `devdocs/ORACLE.md` (in the
repo). The currently wired oracles:

- **USAlign** — TM-align / MM-align / SOI / FlexAlign reference.
- **OpenMM** — AMBER96 + OBC GB reference for energy components.
- **Biopython** — SASA reference (Shrake-Rupley).
- **BALL / BALLJL** — bonded geometry and topology cross-check.
- **Gemmi** — alternate PDB / mmCIF parser sanity check.
- **freesasa** — second SASA opinion when Biopython disagrees.
- **MMseqs2** — search prefilter / SW / PSSM reference.
- **foldseek** — structural-search reference (planned).

## Cross-path parity

Within proteon, every component has a "slow path" reference implementation,
and the optimized paths (NBL, SIMD, GPU, rayon) are guarded by parity tests
against it on every commit. Force-field-parameterized cross-path tests run
across CHARMM19 + EEF1, AMBER96, and OBC GB.
