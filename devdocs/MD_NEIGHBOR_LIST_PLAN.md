# MD_NEIGHBOR_LIST_PLAN — make MD nonbonded O(N) via the existing NbCache

Status: DRAFT (pre-implementation). Tier-2 scaling fix — the **MD half** of the
"OBC GB + MD neighbor lists" audit item. The OBC GB half (a GB cutoff changes the
physics and needs OpenMM-cutoff oracle re-validation) is **out of scope** — separate.

## 1. Problem

`md::velocity_verlet_constrained` (and `velocity_verlet`, which delegates to it) calls
the **all-pair** `compute_energy_and_forces` every step (`md.rs:408,475,519`), so MD is
O(N²)/step — impractical past a few thousand atoms. Meanwhile a cutoff-aware
neighbor-list energy path (`compute_energy_and_forces_nbl`) already exists and is
cross-path parity-tested vs all-pair, and `minimize.rs` already wraps it in **`NbCache`**
(NBL build + drift-refresh + silent GPU dispatch, gated at `MIN_NBL_THRESHOLD=2000`).

## 2. Fix — reuse `NbCache` in the MD loop (revised per claudex)

- **[CRITICAL bug fix] GPU NBL re-upload failure must drop GPU state.** Today
  `NbCache::refresh` (`minimize.rs:82-95`) installs the rebuilt CPU list but, if
  `gpu.refresh_nbl()` fails, only `eprintln!`s and **leaves GPU dispatch enabled** → the
  next eval uses STALE GPU pairs. In minimize a bad step is rejected; in MD one stale
  force eval permanently corrupts the trajectory. Fix: on re-upload failure, **set
  `self.gpu = None`** (drop to the CPU NBL path) — a silent, safe degrade. Affects both
  callers; do it as part of the relocation.
- **Share `NbCache` + an explicit execution policy.** Move it to a new
  `forcefield/nb_cache.rs` (`pub(crate)`); `minimize.rs` switches to
  `use super::nb_cache::NbCache`. Add an explicit `NbExec { AllPair, CpuNbl, Auto }`
  (default `Auto` = the current `>= MIN_NBL_THRESHOLD` gate) instead of a special
  forced-NBL constructor — `CpuNbl` lets tests force the CPU-NBL path on a small fixture
  and run clean A/B trajectory comparisons with **no GPU involvement**. Preserve exact
  threshold semantics (`>= 2000`; note `energy.rs` uses `> threshold` elsewhere — do not
  change either).
- **MD uses it.** In `velocity_verlet_constrained`: build `NbCache::new(&pos, …)` BEFORE
  the initial force eval; replace the three `compute_energy_and_forces` calls with
  `nbc.energy_and_forces(...)`; **`nbc.refresh(&pos, topo)` after the position update AND
  after SHAKE, immediately before the force eval**. The final-frame energy should **reuse
  the last loop iteration's `new_energy`** (no extra eval; covers `n_steps==0` via the
  initial energy) rather than recomputing at `md.rs:519`.

## 3. Result-parity claims (qualified per claudex)

- The NBL path computes the **same cutoff'd nonbonded energy/forces** as all-pair (it
  only skips pairs beyond `cutoff`, which contribute 0 in both). NBL-vs-all-pair
  **force** parity (1e-6) is already tested (`energy.rs:~2001`) — but only on static
  coordinates; this PR adds the *dynamic* cases (post-drift, post-rebuild, pair-crossing).
- **Below `MIN_NBL_THRESHOLD` (2000)**: `Auto` stays all-pair on CPU → existing MD tests
  (1crn, 660 atoms) are **bit-for-bit identical** (no-regression guard).
- **Above threshold (CPU-NBL)**: do NOT claim bit-for-bit — NBL pair ordering changes
  float accumulation order. Guarantee is per-eval force parity (tight tol) + bounded NVE
  energy drift; trajectories may diverge by chaotic amplification over long runs.
- **GPU-NBL**: weaker still (atomic force accumulation is non-deterministic); validate
  separately (bounded drift), not vs CPU bit-for-bit.

## 4. Tests (expanded per claudex)
- All existing `md.rs` tests pass **unchanged** (sub-threshold → all-pair). No-regression.
- **Force parity, dynamic** (via `NbExec::CpuNbl` on 1crn): (a) after sub-threshold
  cumulative drift with **no** rebuild; (b) immediately **after** a rebuild; (c) a
  constructed pair moved from outside `cutoff+skin` to inside `cutoff` — forces match
  all-pair to 1e-6 in every case.
- **Forced-NBL NVE conservation** spanning several rebuilds (CpuNbl) — bounded total-energy
  drift, finite throughout.
- **Short trajectory parity** (CpuNbl vs AllPair) on 1crn over a few NVE **and**
  SHAKE-constrained steps — final coords/velocities within a short-horizon tolerance.
- **GPU (cfg cuda, skips without device):** the refresh-failure path drops GPU→CPU
  (inject a failure or assert the degrade); a multi-step NVE that triggers ≥1 rebuild
  stays finite + bounded-drift.

## 5. Non-goals
- **OBC GB stays O(N²)** even after this (its `_nbl` is a dead stub delegating to all-pair;
  a GB cutoff changes the physics and needs OpenMM-cutoff oracle re-validation) — so the
  win is "MD nonbonded O(N) **for the non-GB nonbonded**", a separate item for OBC.
- Changing `MIN_NBL_THRESHOLD` or the cutoff/switching policy.
- MD performance beyond the nonbonded (bonded terms are already O(N)).

## 6. Files
- `proteon-core/src/forcefield/nb_cache.rs` (new — relocated `NbCache` + `NbExec` +
  GPU-failure degrade; mechanically-reviewable cut/paste from `minimize.rs`)
- `proteon-core/src/forcefield/minimize.rs` (import the moved `NbCache`; unchanged behavior)
- `proteon-core/src/forcefield/md.rs` (use `NbCache` in the loop)
- `proteon-core/src/forcefield/mod.rs` (declare `nb_cache`)

## 7. Review log (claudex)
**Critical bug found:** GPU NBL re-upload failure leaves stale GPU pairs enabled →
trajectory corruption in MD; fix = drop GPU state on failure (§2). Adopted: explicit
`NbExec` policy over a forced-NBL constructor (#6); refresh after SHAKE + reuse final
energy + handle `n_steps==0` (#2); qualify parity claims — bit-identical only below
threshold, tolerances above, CPU-NBL vs GPU-NBL separately (#4); expanded dynamic test
matrix — post-drift / post-rebuild / pair-crossing force parity, forced-NBL NVE over
several rebuilds, SHAKE trajectory parity, GPU refresh-failure + multi-step (#3,#5,#6);
OBC stays O(N²) so the scope claim is "non-GB nonbonded" (#5); relocation as a clean
mechanical diff since `minimize.rs` just merged (#7). No findings rejected.
