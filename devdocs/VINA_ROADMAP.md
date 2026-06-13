# Vina Roadmap

**Last updated: 2026-06-12**

This file tracks what's not done yet for `proteon-vina` — the
AutoDock-Vina scorer + local-optimiser + Monte-Carlo docking port. For
what's already covered, read the crate source (`proteon-vina/src/`), the
parity test suite (`proteon-vina/tests/`), and the Python bindings at
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

**Phase E — full docking (Monte-Carlo global search).** Random
placement + `mutate → BFGS → Metropolis` per replicate
(`mutate.rs`, `mc.rs`), parallel replicates with cross-replicate
RMSD clustering and box / autobox handling (`global_search.rs`), a
linear out-of-box confinement penalty (`BoxPenalty`) that replaces
upstream's grid out-of-bounds slope so the unconstrained translation
DoF can't drift the ligand into empty space, and a `proteon.dock` /
`proteon.vina.dock` binding returning ranked `VinaDockPose` modes.
Seeded (`ChaCha8Rng`) so a run is reproducible. Docking parity is
statistical: on 1iep the top modes redock the crystal pose to
**< 1 Å** (deterministic Rust test `tests/redock.rs`).

183 Rust tests + 22 Python tests; clippy clean.

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

## Phase E — Monte Carlo outer search (full docking) — DONE

Shipped (`mutate.rs`, `mc.rs`, `global_search.rs`, `py_vina::dock`,
`proteon.dock`). Upstream's `Vina::global_search`, reproduced:

1. Random placement in the box (`randomize_conf`).
2. Per step: mutate one DoF (translation / orientation / one torsion,
   `mutate_conf`), BFGS-minimise, Metropolis accept at T = 1.2.
3. `global_steps` per replicate (default 2500).
4. `exhaustiveness` replicates in parallel (rayon), pools merged and
   RMSD-clustered into ranked `VinaDockPose` modes.

Notes for whoever extends this:

- **The roadmap's earlier "most parts already done" was wrong** — none
  of `mutate` / `mc` / the replicate loop existed, and no `rand` /
  `rand_chacha` was pinned anywhere. All of it is new here.
- **Box confinement was the non-obvious correctness fix.** Without a
  receptor grid, BFGS's free translation DoF drifts the ligand out of
  the box into empty space (no contacts → energy ≈ 0, a spurious
  "minimum"). `BoxPenalty` (in `local_only.rs`) adds a linear per-atom
  out-of-box penalty + restoring force, mirroring upstream's grid
  out-of-bounds slope. Without it, docking silently fails.
- Search and final scoring both use the authentic `v = 1000` cap (not
  upstream's tighter search-time `hunt_cap`) so the BFGS gradient stays
  exactly consistent with the parity-validated `local_only` energy.
- Parity is statistical: 1iep redocks to < 1 Å (`tests/redock.rs`).

Still open under Phase E for a follow-up:

- **Grid rasterisation** (see Scaling below) — would replace `BoxPenalty`
  with real out-of-bounds affinity grids and give the 10–50× speedup;
  the per-pair `O(N_rec)` cost currently makes large-receptor docking
  slow (the trimmed-pocket fixtures dock in seconds; a full protein
  would not).
- **`hunt_cap` during search** — upstream tightens the cap while
  searching to escape clashes faster; we left it authentic. Worth
  revisiting if random placements are seen to waste BFGS budget.
- **Multi-seed redock benchmark** — current validation is one fixture,
  one seed. A success-rate-over-N-seeds sweep on a PDBbind subset would
  quantify docking quality properly.

---

## Scaling / performance

- **Grid rasterisation — DONE** (`grid.rs`, `cache.rs`). Per-XS-type 3-D
  affinity grids precompute the whole-receptor field once; ligand scoring
  is then `N_ligand × O(1)` trilinear lookups instead of `O(N_receptor)`
  pair evals, and the grid's out-of-bounds slope subsumes `BoxPenalty`.
  Exposed as `use_grid` on `McParams` / `DockParams` and the
  `proteon.dock` binding (off by default — the exact pair path stays the
  parity-validated one). Measured **~3.3× end-to-end** docking speedup on
  the 2229-atom 1iep receptor (5.1 s vs 17.0 s, same seed/budget),
  including the one-time grid build; the speedup grows with receptor size
  since per-pose cost no longer scales with `N_receptor`. The grid
  trilinearly smooths the steep potential, so it's a slight approximation
  (redock 1.1 Å vs 0.4 Å exact on 1iep; energy a few % shallower) — it
  converges to the pair energy as granularity → 0 (cache parity test).
  Remaining: `szv_grid`-style receptor bucketing would speed the *build*
  for very large receptors (build is currently O(voxels × N_receptor),
  rayon-parallel over slabs).
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
