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

## Label-safe preparation for deep learning

`prepare` is step 0 of geometric-DL supervision: the prepared coordinates and FF
assignment *become the training labels*, so a silent corruption here poisons
every example invisibly. The report exposes that whole class of error as
structured, impossible-to-ignore signals — gate on them instead of parsing
`warnings`.

Use the preset, which loads + prepares with the conservative DL default
(`reconstruct=False` — a fabricated atom is a model-derived guess, not an
observation):

```python
for res in proteon.prepare_for_supervision(glob.glob("pdbs/*.pdb"), n_threads=-1):
    if res.label_safe:
        add_training_example(res.structure)
    else:
        log.info("skip %s: %s", res.path, res.label_hazards)
```

`label_safe` is the strict gate (safe for any label type). For a specific label
type, use a profile on `res.report` to tolerate hazards that don't affect it:

| Profile | Requires | Tolerates |
|---|---|---|
| `label_safe_heavy_coords` | observed heavy atoms, no clashes, single model/conformer | untyped cofactors, missing H |
| `label_safe_all_atom_coords` | + hydrogens placed | — |
| `label_safe_energy` | + `fully_typed` | — |
| `label_safe_sequence_indexed` | no insertion codes, single model | clashes, typing |
| `label_safe` | all of the above | — |

### The hazards (each a first-class flag, surfaced in `label_hazards`)

| Hazard | Signal | Why it poisons a label |
|---|---|---|
| Heavy clashes | `n_heavy_clashes` / `has_heavy_clashes` | H-only minimization can't relax a deposited/rebuilt clash away |
| Fabricated atoms | `has_reconstructed_atoms` | reconstructed coords are model priors, not observations |
| Missing atoms | `has_missing_atoms` (`n_missing_heavy_atoms`) | incomplete residue → a partial coordinate label (reconstruct off) |
| Incomplete FF typing | `has_untyped_atoms` / `fully_typed` | partial topology → wrong energies/forces |
| Alternate locations | `has_altlocs` | a conformer was silently chosen |
| Multiple models | `has_multiple_models` (`n_models`) | only model 0 prepared (NMR ensemble = a distribution) |
| Insertion codes | `has_insertion_codes` | `(chain, resnum)` label keys shift |
| Non-standard residues | `has_nonstandard_residues` | modified AA (MSE, SEP…) — not a canonical token, no FF typing |
| Metals | `has_metals` | coordination chemistry the protein-only FF doesn't model |
| Chain gaps | `has_chain_gaps` (`n_chain_gaps`) | broken peptide bond → false sequential edge in graph/sequence labels |
| Chirality outliers | `has_chirality_outliers` (`n_chirality_outliers`) | D-amino acid / modeling error — coordinate-geometry anomaly |

The clash count is **protein-scoped and validated**: pristine high-resolution
structures (1crn, 0.5 Å) report 0; older/lower-resolution structures report many.
Pairs touching un-templated residues (ligands / metals) are excluded — those are
expected binding contacts, not coordinate errors — and `clash_count_inferred`
flags when that exclusion happened. Ligand chemistry, protonation/tautomer
states, assembly/symmetry, and chirality are deeper hazards tracked in
`devdocs/LABEL_SAFE_PREPARATION_DESIGN.md` for later phases.

Per-atom provenance masks (observed / reconstructed) are produced by the
supervision tensor export, where atoms are indexed; `prepare_for_supervision` is
the structure-level gate that precedes it.

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
