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
| `label_safe_heavy_coords` | observed heavy atoms, no *severe* clashes, single model/conformer | untyped cofactors, missing H, mild clashes |
| `label_safe_all_atom_coords` | + hydrogens placed | — |
| `label_safe_energy` | + `fully_typed` | — |
| `label_safe_sequence_indexed` | no insertion codes, single model | clashes, typing |
| `label_safe` | all of the above | — |

### The hazards (each a first-class flag, surfaced in `label_hazards`)

| Hazard | Signal | Why it poisons a label |
|---|---|---|
| Severe clashes | `has_severe_clashes` (`clashscore`, `max_heavy_overlap`) | pervasive or catastrophically-local steric overlap — coordinates are physically wrong |
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
| Assembly mismatch | `assembly_is_asu is False` (`assembly_mismatch`) | deposited ASU is not the biological assembly → wrong oligomeric state for interface labels |

### Biological assembly (interface labels)

The asymmetric unit you load is not always the biological assembly, and that
only matters for **interface / contact / SASA / neighbor-graph** labels (per-chain
coordinates, energy, and sequence are oligomer-invariant). The path-based
`prepare_for_supervision` parses PDB `REMARK 350` and sets:

- `report.biological_assembly_copies` — operators in the first assembly (1 = no expansion);
- `report.assembly_is_asu` — three-state: `True` the deposited chains already are
  the assembly (identity transforms over exactly the present chains); `False`
  they are not (expansion needed, crystal-packing extras, or multiple separate
  assemblies — e.g. two chains that are each a monomer); `None` not determined
  (no path, or no `REMARK 350`).

Gate interface-type labels on **`report.label_safe_interface`** (sane coords AND
`assembly_is_asu is True`). It is intentionally NOT part of the strict
`label_safe` gate — interface labels are opt-in. This is **detection only** (a
conservative gate that prevents training on the wrong oligomer); it does not
build the assembly. Capability: PDB `REMARK 350` only — mmCIF
`_pdbx_struct_assembly` and applying the transforms are follow-ons.

#### Exporting a verified complex (label-safe substrate)

`build_complex_supervision_examples` turns a verified assembly into masked
per-chain examples — connecting the assembly gate to the coverage/masking path:

```python
out = proteon.build_complex_supervision_examples(
    res.structure, prep_report=res.report, min_coverage=0.8)
if isinstance(out, proteon.ComplexSupervisionExamples):
    for cid in out.chain_order:
        ex = out.chain_examples[cid]   # masked, cross-chain-correct
        ...  # the consumer computes pair labels from the chains' coordinates
else:
    log.info("dropped: %s", out)       # a distinct drop reason
```

The interface gate keeps a complex iff `assembly_is_asu is True` (which already
requires the deposited chains to equal the assembly's chain list), it has **≥ 2
protein chains**, and **every** chain clears `min_coverage` (the weakest chain
gates — both interface partners must be usable). The clash scan is whole-complex,
so a residue clashing *across* an interface is already masked in its own chain.
Drops carry a distinct reason — notably `requires_assembly_expansion` (a
monomeric ASU that BIOMT would expand: valid, just not built here — recoverable
by the assembly-builder follow-on), vs `assembly_unverified` (no `REMARK 350`).

This is a **label-safe complex *substrate*** — the trustworthy, masked coordinate
inputs — not a turnkey interface-label format: it emits no contact map / neighbor
edges and no multimer tensor packing (`devdocs/MULTI_CHAIN_COVERAGE_DESIGN.md`).
On a 293-structure diverse sample (floor 0.8): **15% kept** verified complexes
(mostly dimers), **38% `requires_assembly_expansion`** (the recoverable bucket),
27% verified monomers, 14% below coverage, 5% an unmasked hazard (e.g. chirality).

### Clash *severity*, not "any clash"

A single 0.4 Å heavy-atom overlap is not a poisoned coordinate label — and 99%
of deposited PDB structures have at least one. So the gate keys on **severity**,
not presence:

- `clashscore` = clashing heavy-atom pairs per 1000 heavy atoms (MolProbity
  convention, heavy-only) — the *pervasiveness*.
- `max_heavy_overlap` = the single worst overlap depth (Å) — one deep
  interpenetration is toxic even when the average is fine (a small badly-modelled
  region in a large complex would otherwise be diluted away).
- `has_severe_clashes` (the label hazard, surfaced as **`severe_heavy_clashes`**)
  is true when `clashscore > 20` **or** `max_heavy_overlap > 1.0 Å`.

`has_heavy_clashes` / `n_heavy_clashes` remain available as honest *observations*
— a structure can have `has_heavy_clashes is True` and still be `label_safe`.
The `20` threshold is calibrated against deposited resolution: it passes 100% of
≤ 1.5 Å and 95% of 1.5–2.0 Å structures while rejecting the clashy low-resolution
tail (`devdocs/CLASH_SEVERITY_THRESHOLD_DESIGN.md`).

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

## What a diverse PDB sample actually looks like

Run the full detect→repair pipeline over **1,000 random PDB structures** and the
honest distribution falls out (`validation/eval_prepare_diverse.py`,
`validation/prepare_diverse_results.json`). This is the number that matters for
training data: raw deposits are *overwhelmingly* hazardous, and the gate's job is
to make that visible instead of silent.

**Load + detect** (974/1,000 parsed; 26 unreadable):

| Hazard | of 974 | Notes |
|---|---|---|
| missing atoms | 832 (85%) | incomplete residues / loops (median 265 heavy atoms absent) |
| untyped atoms | 763 (78%) | cofactors, ligands, modified residues |
| altlocs | 424 (44%) | a conformer would be silently chosen |
| assembly mismatch | 411 (42%) | deposited ASU ≠ biological assembly |
| metals | 376 (39%) | coordination chemistry the protein FF doesn't model |
| **severe clashes** | **293 (30%)** | `clashscore > 20` or one overlap `> 1.0 Å` |
| nonstandard / models / gaps / ins-codes | 91 / 50 / 36 / 34 | |
| chirality outliers | 0 | gross D-chirality is genuinely rare in deposits |

The median structure trips **3–4 hazards**; only 5/974 trip none. Strict
`label_safe` is 7/974 and `label_safe_heavy_coords` 37/974 — diverse PDB is *not*
training-ready out of the box, by a wide margin.

**Why severity matters.** A *binary* "any clash" gate flags 965/974 (99%) and
rejects essentially the entire PDB. The severity gate flags 289 (30%) — the
difference is real structures with a handful of mild overlaps that do not corrupt
a coordinate label. Median deposited `clashscore` is 11.8; calibrated by
resolution:

| Resolution | n | median clashscore | frac ≤ 20 (the gate) |
|---|---|---|---|
| ≤ 1.5 Å | 80 | 7.8 | 100% |
| 1.5–2.0 Å | 295 | 9.7 | 93% |
| 2.0–2.5 Å | 248 | 12.3 | 84% |
| 2.5–3.0 Å | 157 | 16.8 | 59% |
| ≥ 3.0 Å | 142 | 18.9 | 54% |

**Detect → repair.** Apply a `heavy_coords` policy (reconstruct missing atoms,
collapse altlocs, pick model 0, drop on severe clash) and the bottleneck shifts:

```python
policy = proteon.RepairPolicy.for_profile(
    "heavy_coords",
    missing_atoms="reconstruct", reconstructed_atoms="accept",
    altlocs="select_highest_occupancy", multiple_models="select_first",
    severe_heavy_clashes="drop",
)
```

On 150 structures: **15 pass (10%)**, and the dominant drop reason is
**`reconstruct_failed:missing_atoms` (120/150, 80%)** — template reconstruction
fills missing *side-chain* atoms but cannot rebuild whole missing residues or
loops, so structures with large gaps stay incomplete. Severe clashes drop a
further 72. With clashes demoted to a severity gate, **structural incompleteness
is now the binding constraint** on complete heavy-coordinate labels, not sterics.

Swapping `drop` for `severe_heavy_clashes="relax"` (heavy minimization on just the
severe subset, median CA drift 0.66 Å) confirms it: relaxation now **clears the
severe clash on 28 of 30** structures it touches (only 2 `relax_failed`) — versus
just 1 of 39 under the old binary "zero clashes" bar, because the target is now
`clashscore ≤ 20`, not literally zero. Yet overall yield barely moves (15% vs
10%): most severe-clash structures are *also* incomplete, and relaxation cannot
add missing residues. Sterics are fixable; missing density is not.

**The takeaway for training-data builders.** ~80% of diverse PDB clears the
clash gate as-deposited; the realistic ceiling for *complete, observed*
heavy-coordinate labels is bounded by structural completeness (missing
loops/residues), not clashes. Sequence-indexed labels fare far better (768/974,
79%) — they don't depend on coordinate completeness or sterics at all.

### Coverage-based masking (keep the good residues)

Demanding a *whole* clean structure throws away most of diverse PDB: only ~15% of
structures are 100% complete, but **~87% of all residues are**. So rather than
drop a structure for a *localized* defect, keep it and mask the affected residues.
`prepare_for_supervision` takes a `min_coverage` floor:

```python
for res in proteon.prepare_for_supervision(paths, min_coverage=0.8, only_safe=True):
    # MUST pass the report + mask flag: the gate kept this structure on the
    # promise that its untrustworthy residues get masked here.
    ex = proteon.build_structure_supervision_example(
        res.structure, prep_report=res.report, mask_untrustworthy_coords=True)
    add_training_example(ex)
```

`res.coverage` is the fraction of **usable** residues; `node_valid` is the
per-residue mask (aligned to `residue_index`). Usable = **complete AND
trustworthy** — a residue is counted invalid if it is missing atoms, is an altloc
pick, **or sits in a severe clash** (`res.report.clash_residue_indices`). So the
gate now *keeps* a structure with a **localized** clash/altloc (high coverage) and
masks those residues, but a **pervasively** clashing one still drops via low
coverage. The calibrated default floor is **0.8**; `coverage_profile="backbone"`
requires only N/CA/C/O.

> ⚠ **The gate keeps clash/altloc structures on the promise you mask them.** If
> you export a coverage-gated structure *without* `mask_untrustworthy_coords=True`
> (and the `prep_report`), the kept clashing/altloc residues become labels — the
> corruption the gate was supposed to prevent. Always pair the two.

### Trustworthiness masking into the export (the last mile)

Completeness drives the *gate*, but the export's presence masks already zero
missing atoms per label. What presence **can't** see is the *trustworthiness*
hazards — a residue with every atom present can still be an arbitrary altloc
pick, a severe clash, or a chirality outlier, which corrupt even the backbone.
The supervision export combines a per-residue trustworthiness mask into the
**coordinate** label masks (opt-in, off by default so the oracle-gated tensors
keep byte-parity):

```python
ex = proteon.build_structure_supervision_example(structure, mask_untrustworthy_coords=True)
# atom37/14, pseudo-beta, φ/ψ/ω, χ, torsions, frame masks are zeroed on
# untrustworthy residues; seq_mask / aatype / residue_index are untouched.
```

The combination respects each label's dependency — **not one broadcast mask**:
`phi`/`pre_omega` zero on residue *i* **and** *i+1* (they read residue *i-1*),
the classic `psi_mask` zeros on *i* and *i-1* (reads *i+1*), residue-local masks
(atoms, χ, pseudo-beta, frames, and the AF `psi` torsion column) zero only on *i*.

Three trustworthiness hazards are localized: **altloc** (`conformer_count > 1`),
**severe clash**, and **D-chirality**. Clash and chirality attribution come from
the same Rust scans that compute `clashscore` / `n_chirality_outliers` —
`PrepReport.clash_residue_indices` and `chirality_residue_indices` list the
affected residues, and `residue_clash_mask` / `residue_chirality_mask` align them
to the export's `residue_index` (the topology `res_idx` walks the identical
`models[0].chains → residues` order, so the alignment is exact). When you pass the
`prep_report`, those residues' coordinate masks are zeroed too — so a single D /
mis-modelled CA centre masks just that residue instead of dropping the whole
structure. The remaining follow-on is per-contact interface-local masks
(`devdocs/PER_RESIDUE_MASKING_SKETCH.md`).

## Validation

50K random PDB battle test on RTX 5090: **99.1% correct in 3.5 hours**
(CHARMM19 + EEF1 + SASA on CUDA). Fold preservation on 1000 PDBs:
proteon CHARMM19 + EEF1 has median TM = 0.9945, **30× faster** than
OpenMM CHARMM36 + OBC2.

### Label-safe path battle test

The full path — `prepare → coverage gate → supervision export with
trustworthiness masking` — over **9,422 diverse real PDB structures**
(`validation/eval_archive_scale.py`):

- **0 crashes.** Every structure either prepared + exported cleanly or was
  recorded as a graceful skip (349 unparseable files, 1 with no protein chain) —
  never an exception through the pipeline.
- **6,072 label-safe masked training examples** exported (67% of the 9,073 that
  prepared) at coverage floor 0.8; the rest fell below the floor.
- **6.3%** of exported residues (120k / 1.9M) carry a zeroed coordinate-label
  mask (missing or untrustworthy); most structures mask < 20%.
- **No perf cliff** — slowest single export 4.2 s; 2.67 structures/s single-process.

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
