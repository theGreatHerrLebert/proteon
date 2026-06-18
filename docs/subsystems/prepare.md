# Preparation

End-to-end "make this structure usable" pipeline: add missing hydrogens,
optionally minimize, return a structure ready for MD or geometric-DL work.

`batch_prepare` is the most common entry point. The pieces are also exposed
individually in `proteon.hydrogens` and `proteon.forcefield` if you want to
mix and match.

## Example

```python
import proteon

paths = ["1crn.pdb", "1ubq.pdb", "1bpi.pdb"]
structures = proteon.batch_load(paths, n_threads=-1)

prep = proteon.batch_prepare(
    structures,
    hydrogens="backbone",   # "none" | "backbone" | "all"
    minimize=True,
    n_threads=-1,
)
```

## Choosing how much to minimize (`constrain_heavy`)

`prepare`, `batch_prepare`, and `load_and_prepare` share one default:
`constrain_heavy=True` — **H-only** minimization. The minimizer relaxes the
placed hydrogens but freezes heavy atoms, so the structure keeps its deposited
coordinates exactly (CA-RMSD 0). This is the right default for a structure-prep
toolkit: it never silently moves your experimental structure. But it is **not**
an equilibration — read the decision guide before trusting energies.

```text
What do you need from prepare()?

1. A faithful protonated structure for ANALYSIS
   (alignment, SASA, DSSP, contacts, ML / supervision features)
   → constrain_heavy=True            (the DEFAULT)
     Heavy atoms keep their deposited coordinates; only H are placed & relaxed.
     ⚠ report.heavy_relaxed is False: final_energy is NOT a heavy-atom minimum
        (still carries crystal strain).
     ⚠ if reconstruct=True added heavy atoms, they sit at template positions,
        unrelaxed (prepare emits a warning). Fine for global/geometric analysis;
        not fine if the rebuilt atoms are in active sites, interfaces, contacts,
        or feed energy/supervision labels — then use (2).

2. An energy-minimized / clash-reduced structure
   (compute_energy, minimize, anything that trusts final_energy)
   → CHARMM19+EEF1 : constrain_heavy=False
       The more physically appropriate proteon mode for unconstrained
       minimization — EEF1 implicit solvent screens electrostatics. Good for
       clash relief and local relaxation. Still not a substitute for full,
       system-specific equilibration; it moves the backbone ~0.5 Å.
   → AMBER96       : constrain_heavy=True (H-only)
       Full all-atom minimization in proteon's vacuum + 15 Å cutoff model is a
       model mismatch — it can distort charged/polar geometry. For production
       AMBER work, do a restrained / solvated minimization in an MD engine.
   → Unsure?       : constrain_heavy=None  (FF-aware)
       Heavy-relax for CHARMM19+EEF1, H-only for AMBER96 — the per-force-field
       right answer in one switch.
```

### Caveats where the tree can mislead

- **NMR ensembles** — H-only preserves each model; heavy minimization can
  collapse ensemble variability, making models look more alike than the
  experiment supports.
- **Membrane proteins** — EEF1 is a water-like implicit solvent, not a
  membrane. Heavy relaxation can mislead transmembrane / lipid-facing regions
  and buried charges.
- **Crystal packing / assemblies** — minimizing an isolated chain can relax
  away crystal contacts and oligomer-interface geometry; missing symmetry
  mates, ligands, ions, or cofactors are not accounted for.
- **Severe clashes / bad geometry** — H-only won't fix clashing heavy atoms,
  but unconstrained minimization may "solve" them by distorting the structure;
  restrained local relaxation is often better.
- **Ligands / metals / PTMs** — if FF typing is incomplete or approximate (see
  `report.fully_typed` / `READY_WITH_LIGANDS`), minimized energies are
  untrustworthy regardless of `constrain_heavy`.

Use `report.heavy_relaxed` to know which regime you got, and gate any trust in
`final_energy` / `components` on it.

## Validation

50K random PDB battle test on RTX 5090: **99.1% correct in 3.5 hours**
(CHARMM19 + EEF1 + SASA on CUDA). Fold preservation on 1000 PDBs:
proteon CHARMM19 + EEF1 has median TM = 0.9945, **30× faster** than
OpenMM CHARMM36 + OBC2.

## API reference

### `proteon.prepare`

::: proteon.prepare
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### `proteon.hydrogens`

::: proteon.hydrogens
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
