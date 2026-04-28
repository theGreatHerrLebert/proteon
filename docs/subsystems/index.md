# Subsystems

Each subsystem corresponds to a module under `packages/proteon/src/proteon/`,
which is the **public Python surface** (Layer 3 in the
[architecture](../architecture.md)). API reference on these pages is
auto-generated from the docstrings of those wrappers, not from the underlying
`proteon_connector` PyO3 bindings.

| Subsystem | Module | Notes |
|-----------|--------|-------|
| [I/O](io.md) | `proteon.io` | Load and save PDB / mmCIF, with batch variants. |
| [Alignment](align.md) | `proteon.align` | TM-align, SOI-align, FlexAlign, MM-align. |
| [Geometry](geometry.md) | `proteon.geometry`, `proteon.analysis` | Transforms, RMSD, dihedrals, contact maps. |
| [DSSP](dssp.md) | `proteon.dssp` | Secondary-structure assignment. |
| [SASA](sasa.md) | `proteon.sasa` | Solvent-accessible surface area. |
| [H-bonds](hbond.md) | `proteon.hbond` | Hydrogen-bond detection. |
| [Forcefield / MD](forcefield.md) | `proteon.forcefield` | CHARMM19+EEF1, AMBER96, OBC GB. |
| [Preparation](prepare.md) | `proteon.prepare`, `proteon.hydrogens` | Add hydrogens + minimize. |
| [Search](search.md) | `proteon.search`, `proteon.msa` | MMseqs2-compatible search stack. *Experimental.* |
| [Supervision](supervision.md) | `proteon.supervision` | Geometric-DL data export (Layer 5). |

## Conventions

- **Batch-first.** Most subsystems expose `batch_*` variants alongside the
  single-structure helpers. `n_threads=-1` uses all cores; `n_threads=0` runs
  serially (almost certainly not what you want).
- **Wrappers, not bindings.** The pages here document the Pythonic wrappers.
  The PyO3 layer is implementation detail — depend on it at your own risk.
- **Search is experimental.** API may change without deprecation cycles until
  we cut a 1.0.
