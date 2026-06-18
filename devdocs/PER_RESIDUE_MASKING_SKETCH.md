# Per-residue label masking — design sketch

> Status: **sketch** (pre-claudex). The lever the diverse-PDB evaluation surfaced.

## Why

The label-safe gate is **all-or-nothing per structure**. On 1,000 random PDBs,
that caps the complete-heavy-coords yield at ~10–15%: a structure with a pristine
300-residue core and one missing surface loop is *dropped whole*
(`reconstruct_failed:missing_atoms`), even though 95% of its residues are perfect
coordinate labels. Geometric-DL training (AlphaFold/OpenFold) does not work this
way — it trains on incomplete structures by carrying a **mask** that excludes the
unobserved/untrustworthy positions from the loss. We should emit that mask
instead of discarding the structure.

The clash fix (#191) already moved sterics off the "binding constraint"; this
moves *completeness* off it too. After both, the gate stops rejecting good data
for local, maskable defects.

## What already exists (the substrate we build on)

The supervision tensor export (`supervision_geometry.py`) already produces, per
residue `i` (0-based `residue_index`):

- `atom37["mask"][i, :]` / `atom14["mask"][i, :]` — **presence** mask: which atom
  slots actually have coordinates (the AlphaFold `atom_mask` convention).
- per-residue backbone mask, `phi/psi/omega` torsion masks, `alt_mask`.
- the **observed vs reconstructed** provenance distinction (per the prepare
  contract) — already the per-residue idea, but only for the `reconstruct` hazard.

So the granularity, the indexing, and one hazard's per-residue mask are in place.
The gap is a **general, profile-scoped label-VALIDITY mask** over the full hazard
taxonomy, and a **structure-level gate that uses coverage instead of all-or-nothing**.

## Core idea (two-tier, per claudex)

Add ONE new per-residue signal — **node validity** — and combine it into the
export's *existing, label-specific* masks. Do **not** broadcast a single residue
mask into every loss: derived labels have their own dependency structure, and the
export already encodes the *presence* half of it (`phi/psi/omega` masks,
pseudo-beta mask, `atom_exists`). Node validity adds the *trustworthiness* half,
combined the SAME way each existing mask already handles presence.

```
node_validity[i]  — are residue i's own coordinates a TRUSTWORTHY label (profile)?  (NEW, per-residue)

# combined per LABEL, mirroring that label's existing presence dependency:
atom_loss_mask[i,a]   = atom_presence[i,a]   * node_validity[i]
pseudo_beta_mask[i]   = pb_presence[i]        * node_validity[i]              # pb_presence already encodes CB(or CA for GLY)
phi_mask[i]           = phi_presence[i]       * node_validity[i] * node_validity[i-1]   # phi needs prev residue
psi/omega_mask[i]     = …_presence[i]         * node_validity[i] * node_validity[i±1]
frame_mask[i]         = (N,CA,C present[i])   * node_validity[i]              # FAPE frame ≠ target atom: FAPE = frame_mask[...,None]*atom_mask
pair_mask[i,j]        = node_validity[i]      * node_validity[j]              # a gap elsewhere doesn't invalidate a valid i–j distance
edge_mask[i,i+1]      = adjacency_ok[i]       * node_validity[i] * node_validity[i+1]   # chain-gap masks the EDGE (see below)
```

The point: **residue validity ≠ label validity.** A residue can have a valid
backbone (frame OK) but a missing CB (pseudo-beta masked); a valid residue next
to a missing one keeps its own coords but loses its `phi`/edge. Each label's mask
is `existing_presence_mask × node_validity(dependencies)`. Crop boundaries create
artificial missing neighbours, so neighbour-dependent masks (torsions, edges)
must be **boundary-invalidated** after the crop slice.

`node_validity` is the per-residue analogue of the `label_safe_*` profiles: a
residue is valid for a profile iff **none of that profile's blocking hazards
touch it**. Same taxonomy, localized:

| Hazard | Localizable? | Masks… |
|---|---|---|
| missing atoms | yes | the incomplete residue |
| reconstructed atoms | yes | the rebuilt residue (already the observed/reconstructed mask) |
| severe clash | yes (pairwise) | **both** residues whose atoms overlap (MolProbity attributes to both) |
| altlocs | yes | the multi-conformer residue |
| insertion codes | **profile-only** | NOT coords — `residue_index` is already a stable 0-based tensor index, so an icode is invalid only for sequence-numbering-dependent labels, not for structure tensors |
| chirality outlier | yes | the D/centre residue |
| non-standard residue | yes | the residue (sequence/energy labels; coords may still be valid) |
| metals | **profile-only** | NOT coords — a metal-adjacent protein residue's coordinates are still valid; only energy/interface labels are unsafe |
| chain gaps | **edge, not node** | the broken *adjacency* (graph/sequence edge), not a residue |
| not_protein / minimize_failed / incomplete_ff | **no** (structure-wide) | drop the whole structure |
| assembly mismatch | no (whole-oligomer concept) | gates interface labels at structure level |

Two things are deliberately *not* per-residue: structure-wide failures (no usable
label anywhere) and chain gaps (a gap is a property of the *bond between* two
residues — for graph labels it masks the **edge**; the flanking residues' own
coordinates are still valid).

## What's new (the work)

1. **Rust — per-residue hazard attribution.** The detectors in `prepare.rs` /
   `clash.rs` currently aggregate to scalars (`n_heavy_clashes`,
   `n_missing_heavy_atoms`, …). Make them also return **which residue indices**
   each hazard touches:
   - clash scan: already has `residue_idx` per atom → emit the set of residues in
     any severe-clash pair (and per-residue worst overlap).
   - missing-atom scan: emit residues that are incomplete.
   - altloc / insertion / chirality / nonstandard / reconstructed: emit residue
     index sets (the scans already visit per residue).
   Shape: a compact `Vec<ResidueHazardBits>` (one bitset per residue) or parallel
   index lists. 0-based, aligned to `residue_index`.

2. **Python — `ResidueLabelMask`.** `PrepReport` gains the per-residue arrays (or
   a sidecar object). A method `validity_mask(profile) -> np.ndarray[bool]`
   builds the per-residue bool by AND-ing the profile's blockers, reusing
   `PROFILE_BLOCKERS` so structure-level and per-residue gates **cannot drift**
   (same consistency-test discipline as today).

3. **Export — loss mask.** `supervision_export` multiplies `presence_mask` by the
   broadcast `validity_mask` (and masks chain-gap *edges* for graph features).
   Carried unchanged through `supervision_crop` (a crop is just an index slice).

4. **Gate — coverage instead of drop.** `prepare_for_supervision` (a new mode, or
   a `RepairPolicy` knob) stops dropping on localizable hazards and reports a
   **coverage fraction** over the right denominator — *valid protein residues /
   exportable protein residues* (NOT total, which includes ligands/waters/excluded
   chains). Report several (backbone-valid, all-heavy-valid, per-profile), and tie
   the default to crop behaviour ("expected unmasked residues per crop"). A policy
   keeps a structure iff coverage ≥ threshold; structure-wide hazards still drop.

## Coverage calibration (measured)

Per-residue *completeness* on the 1k sample (`validation/eval_residue_coverage.py`;
a residue is complete iff every atom37 slot its type expects is present):

- **87%** of all residues (798,360 / 918,091) are complete coordinate labels —
  versus the structure-level gate's ~15% *structure* yield.
- Per-structure coverage: **median 0.95**, mean 0.91, p10 0.79, p25 0.88.
- Structures at coverage ≥ t:

  | coverage ≥ | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 1.0 |
  |---|---|---|---|---|---|---|
  | structures kept (of 974) | 959 | 916 | **866** | 692 | 486 | 142 |
  | frac | 99% | 94% | **89%** | 71% | 50% | **15%** |

The `= 1.0` column (142, 15%) is exactly today's whole-complete yield — masking
turns that into **89% at coverage ≥ 0.8**, recovering the 87% residue-level pool.

**The threshold's job changes.** With masking, a partial structure is not
*corrupted* — it contributes its complete residues and masks the rest. So the
coverage floor is a **quality / crop-efficiency knob** (skip the sparse,
mostly-modelled tail so a fixed-size crop isn't wasted on masked positions), NOT
a corruption guard like the clashscore gate. Recommended default **≥ 0.8**: keeps
89% of structures, cuts the worst ~10% (p10 = 0.79). A higher floor (0.9) drops
structures over a single missing loop that masking would handle fine.

## Open decisions (for claudex)
- **Clash pairwise attribution.** Mask both residues in every severe-clash pair
  (conservative) vs only the worse-resolved one. Start conservative (both).
- **Edge masking for graph labels.** Chain gaps and clashes are pairwise — decide
  the edge-mask semantics for neighbor-graph / contact labels (mask any edge
  incident to a masked residue? mask only the gap edge?).
- **Reconstructed residues.** Presence-mask says present, validity-mask says
  "not observed". Confirm this is just the existing observed/reconstructed mask
  generalized — don't build a second mechanism.
- **Interaction with `reconstruct`.** With per-residue masking, `reconstruct` is
  largely unnecessary for coordinate labels (mask the missing residue instead of
  fabricating it). reconstruct stays useful only when a *downstream* step needs a
  geometrically complete structure (e.g. energy). Document the divide.

## Scope

- **v1**: localize missing-atoms + severe-clash + altloc + reconstructed +
  chirality into per-residue `node_validity(profile)`; combine it into each
  EXISTING label mask with that label's dependency structure (atom / pseudo-beta /
  torsion-with-neighbours / frame / pair / edge) — NOT one broadcast mask; emit
  the combined loss masks in the export; coverage-based gate with the *measured*
  ≥ 0.8 default; boundary-invalidate neighbour-dependent masks after crop.
- **Explicitly deferred** (name them so silence ≠ "handled"): occupancy-below-
  threshold, high-B-factor / low-confidence regions, SEQRES-vs-ATOM unresolved
  residues, duplicate residue/atom records, zero/malformed coordinates,
  distance-based chain breaks (not just declared gaps), cis/trans peptide outliers
  if torsions are supervised, NMR model selection, resolution/method filters,
  nucleic-acid / ligand / PTM handling.
- **Out of scope for v1**: per-edge graph masking semantics for contact labels
  (decide the rule, build later); resolution-aware coverage threshold; metal
  first-shell energy/interface attribution.

## Validation plan

Re-run `validation/eval_prepare_diverse.py` with a `--mask` mode reporting the
**residue-level** yield (valid residues / total residues across the corpus) — the
honest number that replaces the structure-level 10–15%. Expectation: a large
jump, because most dropped structures are mostly-complete.
