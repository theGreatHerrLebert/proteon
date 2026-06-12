# Vina Roadmap

**Last updated: 2026-04-24**

This file tracks what's not done yet for `proteon-vina` — the
AutoDock-Vina scorer + local-optimiser port. For what's already
covered, read the crate source (`proteon-vina/src/`), the parity
test suite (`proteon-vina/tests/`), and the Python bindings at
`packages/proteon/src/proteon/vina.py`.

## Shipped (rebased onto current main as `proteon-vina`)

All five Vina scoring terms (two Gaussians, repulsion, hydrophobic,
non-directional H-bond) with published weights; 32 XS atom types +
AD→element + covalent-radius tables; PDBQT parser with torsion-tree
fragment IDs and rotatable-bond capture; bond inference filtered by
fragment-mask mobility; macrocycle closure support
(`is_closure_clash` + `ad_is_heteroatom`); precalculated pairwise
tables (factor 32, linear-interpolated `eval_deriv`); `score_only`
returning the upstream 8-component vector to ≤ 1 mkcal/mol across
four fixture families (1iep kinase, 1fpu ring-rooted imatinib, 1s63
zinc metalloprotein, BACE_1 macrocycle) and 22 multi-pose cases;
conformation + torsion-tree forward pass with FD-validated
analytical gradient; Armijo-backtracking BFGS; `local_only`
matching `vina --local_only` to ≤ 80 mkcal/mol on drug-like and
≤ 1.5 kcal/mol on high-DoF macrocycles; PyO3 bindings with
`VinaScoreComponents` / `BfgsOutcome` / `VinaLocalOnlyOutcome`
classes + single-call + batch APIs that parse receptor and the
~2 MB precalc table once and iterate ligands on a rayon pool;
`proteon.score_only` / `proteon.local_only` / `proteon.batch_*`
promoted to the top-level `proteon.*` namespace alongside
`proteon.vina.*` submodule access.

168 Rust tests + 19 Python tests; clippy clean.

## In flight / not done yet

Ordered by value / effort trade-off — "do first" is at the top.

---

## Pre-merge checklist

- [x] Re-verified on a clean checkout of current `main`: the crate builds,
      `cargo test -p proteon-vina` is 168/168 green, `cargo clippy -p
      proteon-connector` is clean, and `pytest tests/test_vina.py` is 19/19.
- [x] `THIRD_PARTY_NOTICES.md` §6 added for the Apache-2.0 AutoDock Vina
      lineage — proteon-vina is the only non-MIT crate in the workspace.
- [ ] After merge: tag a pre-release so users can pin against a known-good
      revision while the next minor stabilises.

> Rebase note: this landed by re-applying the self-contained `proteon-vina/`
> crate onto current `main` (verified portable — its only dep is `thiserror`)
> plus thin re-wired bindings, rather than merging the stale `docking-vina`
> branch, which conflicted with the connector's compute → `proteon-core`
> refactor.

**Not a blocker** — the upstream AutoDock-Vina fork at
`/scratch/TMAlign/AutoDock-Vina` currently has our debug-print patch
reverted (see commit history of that repo). If that upstream disk got
re-cloned between sessions, regenerating the parity references would
take ~5 min per fixture — no behavioural change.

---

## Phase E — Monte Carlo outer search (full docking)

The scorer + local optimiser is now the reusable kernel a full docking
run sits on top of. Upstream's `Vina::global_search` is:

1. Initial random placement in the box.
2. Monte-Carlo step on the Conf: perturb translation / orientation /
   one torsion, run BFGS (already have it), accept/reject by
   Metropolis at T = 1.2 (upstream default).
3. Repeat `global_steps` times (default 2500 per replicate).
4. Multiple replicates in parallel; collect top-N poses, cluster by
   RMSD, report.

Estimated effort: **4–6 focused hours**. Most of the interesting parts
are already done — random Conf generation, MC mutation rules, and the
parallel replicate loop. The parity conversation shifts from
"bit-for-bit energies" to "find a pose within X Å RMSD of upstream's
best pose on Y% of runs" — statistical rather than deterministic.

Suggested sub-phases:

- **E.1** — `mutate.rs`: random Conf perturbation (translation,
  quaternion, torsion). Upstream's step sizes are in `monte_carlo.h`.
- **E.2** — `mc.rs`: the MC loop. Needs an RNG seed arg for
  reproducibility; use `rand_chacha` (proteon already pins this
  elsewhere).
- **E.3** — `global_search.rs`: parallel replicates over rayon, top-N
  collection, RMSD clustering.
- **E.4** — `vina --docking` equivalent Python binding + fixture test
  against upstream's final pose RMSD.

---

## Scaling / performance

We currently do O(N_ligand × N_receptor) pair evaluation for every
score. For large receptors (e.g. full proteins rather than the trimmed
pocket PDBQTs in our fixtures) this is the bottleneck.

- **Grid rasterisation** (Phase C.5 we deferred). Upstream pre-computes
  a 3D grid of per-XS-type affinities; ligand evaluation becomes
  `N_ligand_atoms × O(1)` trilerp lookups instead of O(N_rec) pair
  evals. Changes nothing for correctness; ~10–50× speedup for large
  receptors. Sits behind a `GridCache` struct with an optional
  `use_grid: bool` switch in the `LocalOnlyOptions`.
- **SIMD** in the pair loop. `precalc.eval_deriv` is currently scalar;
  a Vec4 version over 4 pairs at once is ~2× on x86. Low priority
  until a real profile points at it.
- **GPU port.** Proteon's CUDA infrastructure (see `proteon-connector/src/forcefield/`)
  is a reasonable template. Only worth it for VS libraries > 1M
  ligands — below that, 6× CPU parallel scaling is already enough.

---

## Coverage extensions

### Flex side chains
Upstream's `--flex` argument lets specific receptor side chains move
during docking. Plumbing:
1. Extend PDBQT parser to accept a separate flex PDBQT with `BEGIN_RES`
   / `END_RES` markup.
2. Treat flex residue atoms as a second movable "ligand" whose tree
   roots at the backbone N–Cα bond.
3. Update `score_only` so `flex_grids`, `inter_pairs`, `intra_pairs`
   become non-zero.

Estimated effort: **1–2 days**. Mostly plumbing; the scorer already
structurally separates ligand / flex / receptor in the component names.

### AutoDock4 and Vinardo scoring functions
We implemented Vina SF only. Upstream also supports:
- **AutoDock4** (`SF_AD4`): 12-6 vdW, directional H-bond, electrostatic,
  solvation. Different atom types, different precalc tables. ~800 LOC
  to port.
- **Vinardo** (`SF_VINARDO`): same five terms as Vina but different
  widths/cutoffs + weights. Trivial extension — could share the
  `precalculate` machinery with a different term parameter set.

Vinardo is the lower-hanging fruit and worth having for benchmarking.

### More fixtures
Current corpus covers kinase + amide + macrocycle + zinc. Missing
meaningfully:
- Nucleotide-binding (sulphur, phosphate) — any G-protein binder.
- Heme / iron-sulphur clusters — already typed correctly via Met_D
  but untested on a real receptor.
- Protein-protein interface inhibitors — larger ligands with more
  rotatable bonds than typical drug-likes.

Add by: running `vina --score_only` on a PDBbind core-set complex,
vendoring the PDBQTs, updating `tests/fixtures/pairs/<name>/` plus
`upstream.ref`.

---

## Ecosystem / ergonomics

- **File-path convenience wrapper.** Right now callers do
  `score_only(open(rec).read(), open(lig).read())`. A thin
  `score_only_from_paths(receptor: Path, ligand: Path)` would save a
  line and is ~5 LOC.
- **CLI binary.** `proteon-bin` could grow a `vina_score` / `vina_dock`
  subcommand that mirrors the Python API. Useful for shell-based
  workflows.
- **Jupyter example notebook.** A `notebooks/vina_virtual_screening.ipynb`
  walking through: load receptor PDB → prepare PDBQT via Meeko → score
  a small library → rank → visualise top 3 with nglview. Drives user
  adoption once v0.1.3 ships.

---

## Known rough edges

- **`proteon.vina.local_only` vs upstream drift** is up to ~1 kcal/mol
  on high-DoF ligands (macrocycle, ring-rooted imatinib). Cause: BFGS
  line-search tie-breaks differ between implementations on non-convex
  surfaces. Not a bug; documented in
  `src/local_only.rs::check_local_only_parity`. Likely tightens if we
  track upstream's exact Armijo break condition or tighten the
  gradient-norm convergence from `1e-5` to `1e-7`.
- **FD force parity on real fixtures** is deliberately skipped: the
  Vina pair potentials have piecewise-linear kinks (slope_step in
  hydrophobic + H-bond + the outer cutoff) that corrupt any
  central-difference estimator. Synthetic-system FD gates at 5e-3
  relative cover the math; real-fixture integration is covered by
  the local_only parity gate.
- **Licensing mixing.** The workspace is MIT except for `proteon-vina`,
  which is Apache-2.0 to match upstream AutoDock-Vina. When shipping
  a wheel, ensure both NOTICE files are included — maturin currently
  does this automatically because both `LICENSE` files sit in their
  respective crate roots.
