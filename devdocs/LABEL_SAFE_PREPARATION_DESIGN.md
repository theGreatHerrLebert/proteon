# Label-safe preparation — design

## Why this is first-class, not a flag

`prepare` is step 0 of every downstream pipeline. For geometric-DL supervision
the prepared coordinates and FF assignment *become the training labels*. A
silent corruption here — a guessed atom, a residual clash, an arbitrary altloc
pick — is garbage-in that propagates into every example and every gradient,
invisibly, and is nearly impossible to diagnose after the fact. Making that
class of error **impossible to ignore** is arguably the whole point of the
tool. So the contract we want is:

> A consumer can compute, from the report alone, whether a prepared structure
> is safe to use as a training label — as a single decision, with the specific
> hazards enumerated when it is not.

This doc designs that contract: the hazard taxonomy, the structured fields, a
correctly-defined and validated clash metric, a `label_safe` gate, and a
DL-oriented preset.

## Hazard taxonomy — every way default prep can degrade a label

| # | Hazard | Currently | Silent? | Proposed signal |
|---|--------|-----------|---------|-----------------|
| 1 | Guessed heavy atoms (reconstruct) | `atoms_reconstructed`, warning | flagged (miss-able) | `has_reconstructed_atoms` bool + keep count |
| 2 | Incomplete FF coverage (protein chain) | `incomplete_ff` / `status` | flagged | `fully_typed` (exists) |
| 3 | Untyped cofactors/ligands | `untyped_cofactors` / `READY_WITH_LIGANDS` | flagged | `has_untyped_atoms` bool |
| 4 | Not a protein | `skipped_no_protein` / `status` | flagged | `status` |
| 5 | Minimize numerical failure | `status` MINIMIZE_FAILED | flagged | `status` |
| 6 | **Heavy clashes** (deposited or from reconstruction) | — | **SILENT** | `n_heavy_clashes` count + `has_heavy_clashes` |
| 7 | **Altloc ambiguity** (a conformer is silently picked) | — | **SILENT** | `has_altlocs` bool (+ optionally count) |
| 8 | **Multi-model** (only model 0 prepared) | — | **SILENT** | `n_models` (+ `has_multiple_models`) |
| 9 | Insertion codes (residue-id shifts) | supervision schema handles | flagged in Layer 5 only | `has_insertion_codes` bool on report |
| 10 | Partial sidechain H (energy/H-bond labels) | `hydrogens_skipped` | flagged (miss-able) | keep; fold into `label_safe` only for energy labels |
| 11 | Chain breaks (peptide H skipped) | handled in placer | n/a | informational |

Items 6–9 are the gaps — 6, 7, 8 are fully SILENT today.

## The contract: `label_safe` + `label_hazards`

- `label_safe: bool` — True iff none of the label-corrupting hazards fired. The
  single gate a DL consumer uses instead of a checklist:
  `status == READY` AND `fully_typed` AND NOT `has_reconstructed_atoms` AND
  NOT `has_heavy_clashes` AND NOT `has_altlocs` AND NOT `has_multiple_models`.
  (Insertion codes: included or not — see open question.)
- `label_hazards: list[str]` — the specific hazards that fired (e.g.
  `["reconstructed_atoms", "heavy_clashes"]`), so a consumer that is willing to
  tolerate some (e.g. cofactor proteins for backbone-only labels) can decide
  per-hazard. This is the structured replacement for parsing `warnings`.

Warnings stay for humans; `label_safe` / `label_hazards` are the machine contract.

## Clash metric — definition (must be correct, will be validated)

A **heavy-atom clash** is a pair of heavy atoms (i, j) that are:
- both non-hydrogen,
- NOT 1-2 bonded and NOT 1-3 (angle) related,
- with interatomic distance `d < r_i + r_j - OVERLAP_TOL`,

using **Bondi van der Waals radii** (element-based, FF-independent —
`sasa::vdw_radius`) and `OVERLAP_TOL = 0.4 Å` (the MolProbity "clash" overlap
threshold). This is FF-independent on purpose: the question is "do heavy atoms
physically overlap," not "is the CHARMM LJ unhappy."

Exclusions (1-2, 1-3) are built from a distance-based bond graph (heavy–heavy
within `BOND_MAX ≈ 2.1 Å`, covering C–C/C–N/C–O up to S–S disulfides). 1-4 is
NOT excluded (real 1-4 clashes are meaningful). Neighbor search is a uniform
grid at cutoff `max(r_i+r_j) ≈ 4 Å` → O(N).

Knobs, defaulted from MolProbity convention, exposed for tuning:
`OVERLAP_TOL = 0.4`, `BOND_MAX = 2.1`.

### Validation plan (non-negotiable for a metric that gates training data)

1. **Synthetic unit tests** — two atoms at a known separation: clean contact
   (no clash), 0.5 Å overlap (clash), bonded pair (excluded), 1-3 pair
   (excluded). Exact expected counts.
2. **Clean vs perturbed** — a clean PDB (e.g. 1crn) reports ~0 clashes; the
   same structure with two atoms shoved together reports exactly that clash.
3. **Cross-check magnitude** — clash counts on a handful of PDBs are in the
   right ballpark vs MolProbity clashscore if `probe`/`reduce` is available
   (skip-if-absent, like the other oracles); otherwise document the comparison
   as deferred.
4. **Reconstruction interaction** — a structure with reconstructed atoms that
   are then heavy-relaxed should drop clash count vs H-only (atoms settle).

## DL preset

A dedicated entry point so DL users don't have to know the safe defaults:

```python
structures, reports = proteon.prepare_for_supervision(paths)  # or a flag
# == batch_load_and_prepare(paths, reconstruct=False, constrain_heavy=True,
#                           minimize=...) then filter on report.label_safe
```

Conservative choices vs the general default:
- `reconstruct=False` — do NOT inject guessed atoms into labels (drop/flag
  incomplete residues instead of fabricating them). This is the key change.
- keep H-only (coords preserved).
- surface `label_safe` and let the caller gate (the preset can also return only
  the safe ones with a dropped-list, mirroring `batch_load_tolerant`).

Open question: a separate function vs a `label_safe_defaults=True` flag on the
existing API.

## Phased implementation

- **P1** — clash metric in proteon-core (`count_heavy_clashes`) + the synthetic
  & clean/perturbed unit tests. Pure Rust, self-contained, the riskiest piece —
  land and validate first.
- **P2** — report fields: `n_heavy_clashes`, `has_altlocs`, `n_models`,
  `has_insertion_codes` (Rust → connector → Python), plus derived booleans and
  `label_safe` / `label_hazards` (Python).
- **P3** — `prepare_for_supervision` preset (reconstruct=False, gate on
  label_safe) + docs (DL gating recipe, the hazard table) + the
  MolProbity cross-check oracle (skip-if-absent).

## Claudex revisions (architecture — adopted)

These change the design from "a few extra fields" to a provenance-first contract:

1. **Reconstruction is PROVENANCE, not geometry.** A clash-free reconstructed
   atom is still a model-derived label (the net would learn the reconstruction
   algorithm's rotamer/loop priors). So the report carries per-atom masks —
   `observed_atom_mask`, `missing_atom_mask`, `reconstructed_atom_mask` — and
   `label_safe` means *experimentally observed*, not merely *clean*. The DL
   preset defaults `reconstruct=False` (omit, don't fabricate); an explicit
   `completion_mode="reconstruct_verified"` keeps reconstructed atoms but marks
   them and never blends them silently with deposited coordinates.

2. **Label profiles** — different labels have different safety bars:
   - `label_safe_heavy_coords` — single model, no altloc ambiguity, no
     reconstructed heavy atoms, no heavy clashes.
   - `label_safe_all_atom_coords` — the above + complete, correct H.
   - `label_safe_energy` — the above + fully typed + resolved protonation/HIS.
   - `label_safe_sequence_indexed` — no insertion-code/renumber ambiguity, no
     unresolved chain gaps (false sequence adjacency).
   `label_safe` is the strict AND; the profiles let a consumer relax per task.

3. **Policies, not silent picks** — `multi_model_policy` (default `reject` for
   label-safe; `first`/`all`/`representative` explicit), `assembly_policy`
   (asymmetric unit vs biological assembly). Silent model-0 / ASU use is unsafe.

4. **Clash metric corrections** — build 1-2/1-3 exclusions from the REAL
   topology, not a distance-inferred bond graph (distance inference can hide a
   real clash by mislabeling it a bond); fall back to distance only with
   `bond_graph_inferred=True`. Exclude mutually-exclusive altloc pairs (A/B in
   the same atom group never coexist) but still set `has_altlocs`. Element-based
   Bondi radii confirmed correct (FF-independent; FF energy is an optional
   diagnostic, never the gate).

## Expanded hazard taxonomy (claudex, ranked by how badly they poison labels)

Beyond items 1–11 above, in rough severity order:

| Hazard | Signal | Worst for |
|--------|--------|-----------|
| Wrong biological assembly / symmetry context | `assembly_policy`, `used_biological_assembly` | interface/contact/SASA/pocket labels |
| Ligand bond order / formal charge / aromaticity / metal coordination | `ligand_chemistry_inferred`, `has_metals`, `has_ambiguous_bond_orders` | protein–ligand DL, energies |
| Protonation / tautomer / HIS flip / terminal patches | `protonation_inferred`, `his_state_inferred` | H-bonds, energies, electrostatics |
| Chirality / stereochemistry violations | `n_chirality_violations` | all-atom geometry labels |
| Cis/trans peptide & planarity outliers | `n_cis_nonproline`, `n_planarity_outliers` | backbone geometry (informational unless severe) |
| Chain gaps → false spatial/sequence adjacency | `has_chain_gaps`, `gap_lengths` | graph edges / local frames |
| Occupancy < 1, disorder, high B-factor | `min_occupancy`, `high_b_factor_fraction` | noisy/partially-supported labels |
| Nonstandard residues, covalent mods, disulfides | `has_nonstandard_residues`, `disulfide_assignment_inferred` | topology/exclusions |
| Atom-naming/remapping ambiguity (symmetric Asp/Glu/Phe/Tyr/Arg) | `has_ambiguous_atom_names` | atom-indexed labels |

Most are detectors that slot into `label_hazards` incrementally; the CONTRACT
(masks, profiles, policies, `label_safe`) must be designed right once so they
can be added without breaking consumers.

## Phased plan (revised — contract first, detectors incrementally)

- **P1 (foundation)** — the contract + the fully-silent gaps:
  clash metric (topology-excluded, validated), `n_heavy_clashes`/`has_altlocs`/
  `n_models`/`has_insertion_codes`, the provenance masks, `label_safe` +
  `label_hazards` + the label profiles, and the `prepare_for_supervision`
  preset (`reconstruct=False`, `multi_model_policy="reject"`). Establishes the
  extensible API.
- **P2** — high-severity chemistry detectors: protonation/HIS-state +
  ligand chemistry/metals + nonstandard residues/disulfides.
- **P3** — assembly/symmetry, chirality, cis-peptide, chain-gap adjacency,
  occupancy/B-factor; MolProbity clashscore cross-check oracle (skip-if-absent).

Each phase ships behind the P1 contract, adding `label_hazards` entries and
profile conditions without breaking `label_safe` consumers.

## Original open questions (now answered by claudex above)

1. Is the clash definition right for label-safety — Bondi + 0.4 Å overlap,
   exclude 1-2/1-3 only? Should 1-4 be excluded or down-weighted? Should
   alternate-conformer pairs (same residue, different altloc) be excluded from
   clashes (they never coexist physically)?
2. Is element-based (FF-independent) the right basis, or should the metric use
   the FF's own LJ so it is consistent with what was minimized?
3. Should `has_insertion_codes` and partial-sidechain-H gate `label_safe`, or
   are they hazards only for specific label types (and so belong in
   `label_hazards` but not the default gate)?
4. Is `reconstruct=False` the correct DL default, or is "reconstruct + relax +
   gate on no-clash" better (gives complete structures, just verified)? i.e.
   should the preset prefer *completeness with verification* over *omission*?
5. Multi-model: is preparing model 0 and flagging `n_models>1` enough, or should
   the preset refuse multi-model inputs (NMR ensembles) outright for labels?
6. Anything missing from the hazard taxonomy that silently corrupts a label?
