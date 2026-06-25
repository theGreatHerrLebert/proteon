# Subsystems

Each subsystem corresponds to a module under `packages/proteon/src/proteon/`,
which is the **public Python surface** (Layer 3 in the
[architecture](../architecture.md)). API reference on these pages is
auto-generated from the docstrings of those wrappers, not from the underlying
`proteon_connector` PyO3 bindings.

The **Tier** column reflects the [API stability](../stability.md) split:
🟢 stable (contract-frozen) / 🧪 experimental (may change; canonical access via
`proteon.experimental.*`).

| Subsystem | Module | Tier | Notes |
|-----------|--------|------|-------|
| [I/O](io.md) | `proteon.io` | 🟢 stable | Load and save PDB / mmCIF, with batch variants. |
| [Alignment](align.md) | `proteon.align` | 🟢 stable | TM-align, SOI-align, FlexAlign, MM-align. |
| [Geometry](geometry.md) | `proteon.geometry`, `proteon.analysis` | 🟢 stable | Transforms, RMSD, dihedrals, contact maps. |
| [DSSP](dssp.md) | `proteon.dssp` | 🟢 stable | Secondary-structure assignment. |
| [SASA](sasa.md) | `proteon.sasa` | 🟢 stable | Solvent-accessible surface area. |
| [H-bonds](hbond.md) | `proteon.hbond` | 🟢 stable | Hydrogen-bond detection. |
| [Forcefield / MD](forcefield.md) | `proteon.forcefield` | 🟢/🧪 | `compute_energy`, `minimize_*`, `gpu_*` stable; `run_md` experimental. |
| [Preparation](prepare.md) | `proteon.prepare`, `proteon.hydrogens` | 🧪 experimental | Add hydrogens + minimize; heuristics still evolving. |
| [Search](search.md) | `proteon.search`, `proteon.msa` | 🧪 experimental | MMseqs2-compatible search stack. |
| [Supervision](supervision.md) | `proteon.supervision` | 🧪 experimental | Geometric-DL data export (Layer 5). |

## Conventions

- **Batch-first.** Most subsystems expose `batch_*` variants alongside the
  single-structure helpers. `n_threads=-1` uses all cores; `n_threads=0` runs
  serially (almost certainly not what you want).
- **Wrappers, not bindings.** The pages here document the Pythonic wrappers.
  The PyO3 layer is implementation detail — depend on it at your own risk.
- **Stable vs experimental.** Stable subsystems carry a back-compat promise;
  experimental ones (prepare, search, supervision, electrostatics, Vina,
  `run_md`) may change without a deprecation cycle and are reached via
  `proteon.experimental.*`. See [API stability](../stability.md).
