# proteon API stability

proteon's top-level Python surface is split into two tiers. The split is the
public contract, machine-checked by `tests/test_public_api_surface.py` (the
**stable** set is frozen against a checked-in snapshot, so growing it is a
deliberate, reviewed act). Introspect at runtime:

```python
import proteon
proteon.__stable__         # frozenset of stable top-level names
proteon.__experimental__   # frozenset of experimental top-level names
proteon.experimental       # canonical namespace for experimental APIs
```

## What "stable" means — the 5 gates

A symbol is **stable** only if *all five* hold. This is deliberately strict;
when in doubt, a symbol is experimental.

1. **Documented I/O** — inputs, outputs, units, and coordinate conventions are
   specified.
2. **Defined error semantics** — what it raises / returns on bad input is
   specified and tested.
3. **Back-compat promise** — within a major version, existing callers keep
   working, or get an explicit deprecation cycle + migration.
4. **Validation evidence** — correctness is gated against a named oracle, with
   the sample size and tolerance stated (not just "doesn't crash").
5. **Schema/versioning policy** — any serialized output carries a version and a
   documented evolution policy.

"Robust" (survives 47k structures without crashing) is **not** the same as
"correct" and does **not**, by itself, earn the stable tier.

## Stable tier (the strict pure-compute core)

One oracle-validated quantity per call, fixed signature. proteon promises to
keep these working.

| Area | Symbols | Validation |
|---|---|---|
| Structure I/O | `load`, `load_pdb`, `load_mmcif`, `save`, `save_pdb`, `save_mmcif`, `batch_load`, tolerant/rescue loaders | 96.4% load on 47k real PDB |
| Data model | `Atom`, `Chain`, `Model`, `Residue`, `Structure` | core types |
| Alignment | `tm_align`, `mm_align`, `soi_align`, `flex_align` (+ one/many variants, result types) | 0.003 median TM drift vs USAlign, 4,656 pairs |
| SASA | `total_sasa`, `residue_sasa`, `atom_sasa`, `relative_sasa` (+ batch) | 0.17% median vs Biopython, 1,000 PDBs |
| DSSP | `dssp`, `dssp_array`, `batch_dssp`, `load_and_dssp` | oracle-gated |
| H-bonds | `backbone_hbonds`, `geometric_hbonds`, `hbond_count` (+ batch) | geometric oracle |
| Geometry | `kabsch_superpose`, `rmsd`, `rmsd_no_super`, `tm_score`, `apply_transform`, `assign_secondary_structure` | pure math |
| Analysis | `backbone_dihedrals`, `contact_map`, `distance_matrix`, `extract_ca_coords`, `radius_of_gyration`, `centroid`, `dihedral_angle`, `to_dataframe` (+ batch / `load_and_*`) | pure geometry |
| Forcefield | `compute_energy`, `minimize_hydrogens`, `minimize_structure`, `load_and_minimize_hydrogens`, `gpu_available`, `gpu_info` (+ batch) | AMBER96 ≤0.5%, OBC GB ≤1% vs OpenMM |

## Experimental tier

Validated but **not** contract-frozen — APIs and schemas may change without a
deprecation cycle. Canonical access is `proteon.experimental.*`; flat top-level
access is kept for back-compat and will warn in a future minor.

- **`prepare` + structure-supervision corpus pipeline** — `prepare`,
  `batch_prepare`, coverage/masking, assembly builder, supervision build/export,
  corpus/sequence/training/cluster release builders. Robust at 47k (zero
  crashes) but the prep heuristics and on-disk schemas still move; output is
  schema-versioned, not behaviour-frozen.
- **Hydrogen placement** — `place_*_hydrogens`, `reconstruct_fragments`
  (chemically loaded; affects downstream energies).
- **Structural search / MSA / templates** — `search`, `build_search_db`,
  `MsaSearch`, template retrieval + features.
- **Arrow/Parquet export** — `to_arrow`, `to_parquet`, … (columnar contract not
  yet frozen).
- **Electrostatics (BEM)** — `born_energy`, `surface_potential`, surface-format
  parsers; built on a DRAFT formulation.
- **Vina docking** — `dock`, `score_only`, `local_only`; port roadmap open.
- **MD** — `run_md` (far less validated than `minimize`).
- **Selection** — `select` (query grammar not frozen).
- **Failure taxonomy** — `classify_exception`, loader-failure analysis (classes
  still being discovered).
- **`RustWrapperObject`** — implementation base class; candidate for de-export.

## Adding to the stable tier

1. Make the symbol pass all 5 gates above.
2. Move it from an experimental group to a stable group in
   `packages/proteon/src/proteon/__init__.py`.
3. Update the frozen snapshot in `tests/test_public_api_surface.py` in the same
   change, and note it in the changelog. The test will otherwise fail — that is
   the point.
