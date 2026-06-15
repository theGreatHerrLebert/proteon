# GB_CUTOFF_PLAN — neighbor-list / cutoff path for OBC GB (kill the last O(N²))

Status: DRAFT (pre-implementation). Tier-2 weak-spot: every nonbonded term got NBL
acceleration via the shared `NbCache` (#152) **except OBC GB**, which stays all-pairs
O(N²) on both passes — so large-system minimize/MD is GB-bound. Goal: a cutoff /
neighbor-list path for OBC GB that (a) matches **OpenMM `CutoffNonPeriodic` GBSAOBCForce**
to the same tolerance the NoCutoff path already meets, and (b) leaves the exact all-pairs
path as the default for small systems and the oracle.

## 0. Scope decision — REVISED after claudex (the size gate must NOT change physics)
**The deliverable is an explicit, opt-in `CutoffNonPeriodic` GB *method*, not a transparent
speedup of the default.** claudex #7: a size gate that flips GB from NoCutoff (exact) below
2000 atoms to CutoffNonPeriodic (approx) above it silently changes the Hamiltonian with system
size — wrong. The cutoff-vs-NoCutoff choice is a **force-field / nonbonded-method setting**; the
size gate may only pick the *implementation* (all-pairs loop vs cell list) of the selected method.

Precedent (verified): LJ/Coulomb apply the **same** `nonbonded_cutoff` (15 Å) + cubic switch in
**both** the all-pairs path (`energy.rs:527`) and the NBL path (`energy.rs:728`) — the 2000 gate
is implementation-only and results are consistent across it. GB must mirror this *within a method*:

- **Default unchanged:** GB stays **NoCutoff, all-pairs, O(N²)**, byte-for-byte as today — the
  validated oracle (≤5% GB / ≤1% total vs OpenMM NoCutoff) is preserved. NoCutoff GB is
  inherently O(N²) (OpenMM's is too); we do NOT approximate the default.
- **New opt-in method `CutoffNonPeriodic` GB** (cutoff = `nonbonded_cutoff`, matching the LJ
  cutoff and OpenMM): applies the cutoff + RF shift in **both** GB paths (all-pairs and cell-list),
  so the 2000 gate is implementation-only here too. This is where the O(N) win lives — it gives
  large-system GB MD an O(N) path it currently lacks, at OpenMM-CutoffNonPeriodic accuracy.
- **CPU first.** GPU cutoff-GB kernels are a separate follow-up. Until they exist, selecting
  CutoffNonPeriodic GB must **force the CPU path for GB** — the GPU OBC kernels are all-pairs, so
  GPU dispatch would silently give NoCutoff GB and make the Hamiltonian depend on CUDA
  availability (claudex #6). Gate GPU off for GB when the cutoff method is active.

## 1. The two passes and why a cutoff is non-trivial
OBC GB is two all-pairs passes (per the code map):
1. **Born-radius descreening integral** (`compute_born_radii*`, `gb_obc.rs:206–280`): each
   atom's `R_eff` is an HCT integral over all other atoms. The integrand decays fast
   (~1/r⁴-ish), so truncation error is small — but truncating changes **every** Born radius
   (less descreening ⇒ larger radii). Must match OpenMM's truncation exactly.
2. **GB polarization energy/forces** (`gb_obc_energy_and_forces`, `gb_obc.rs:476–606`): the
   `q_i q_j / f_GB(r)` term. f_GB ≈ r at long range, so this term is **1/r long-range** — the
   same truncation problem as bare Coulomb. Whether OpenMM applies a reaction-field-style
   shift here or hard-truncates is the **#1 correctness question** (see §3).

Plus the force path's **third loop** — the chain-rule HCT spread (`gb_obc.rs:547–606`) — which
propagates `born_forces[i]` back through the pass-1 integrand. **Invariant:** the spread loop
must iterate **exactly the same pair set** as the pass-1 integral, or forces stop being the
gradient of the energy. This is the easiest correctness trap to trip when introducing a cutoff.

## 2. The neighbor list for GB is NOT the LJ neighbor list
`NeighborList::build(coords, cutoff, excluded_pairs, pairs_14)` **skips** `excluded_pairs`
(1-2/1-3) — `neighbor_list.rs:137,156`. **GB has no exclusions**: both passes sum over every
pair including 1-2/1-3/1-4. So GB cannot reuse the LJ pair list.
- **Build a GB-specific list:** `NeighborList::build(coords, gb_cutoff, &empty, &empty)` — all
  spatial pairs within cutoff, no exclusions, `is_14` irrelevant. The cell-list builder already
  supports this; no new data structure.
- **Cutoff choice:** `gb_cutoff = nonbonded_cutoff` (what OpenMM `CutoffNonPeriodic` uses for
  GB). Reuses the existing 2 Å Verlet buffer + drift refresh. (If parity needs a larger GB
  cutoff than the LJ cutoff, the GB list is independent so it can carry its own — decided by
  the oracle in §5, not assumed.)
- **Where it lives:** add a GB pair list alongside the LJ one. Option A — a second
  `Option<NeighborList>` field in `NbCache` (built/refreshed together, same drift criterion).
  Option B — GB owns its own `NeighborList` threaded through the GB calls. Lean A (one drift
  check, one refresh site, GPU upload hook later). Decide in review.

## 3. OpenMM CutoffNonPeriodic semantics — RESOLVED from reference source
Read from OpenMM `platforms/reference/src/SimTKReference/ReferenceObc.cpp` (master):
- **Pass 1 `computeBornRadii`:** hard truncation —
  `if (getUseCutoff() && r > getCutoffDistance()) continue;`. No correction; truncating simply
  drops far descreeners (Born radii grow). Replicate: pass-1 integral runs over the GB pair
  list (which is exactly the in-cutoff set).
- **Pass 2 `computeBornEnergyForces`:** for **distinct** in-cutoff pairs, a reaction-field-style
  shift is subtracted: `energy -= preFactor · q_i · q_j / cutoffDistance` (same `preFactor`/
  scaling as the Gpol term). **The shift is distance-independent ⇒ contributes ZERO force** —
  it changes the energy only. So: forces come purely from the truncated Gpol+chain terms; the
  cutoff adds one constant-per-pair energy correction. (Confirm the exact `preFactor` placement
  against the surrounding lines when coding — it wraps the whole pair energy.)
- **Self term (i==j):** `energy *= 0.5`, **no** RF shift — unchanged from the all-pairs path.
- **Acceptance:** reproduce OpenMM `CutoffNonPeriodic` to ≤ the NoCutoff tolerances (≤5% GB
  component / ≤1% total on crambin) **at the same cutoff**, AND analytic forces match finite
  differences on the cutoff path (not just the all-pairs path). Because the RF shift is
  forceless, the FD test also independently checks we did NOT accidentally give it a gradient.

## 4. Implementation outline (with claudex #2/#5 correctness essentials)
1. `compute_born_radii_with_chain_nbl(coords, gb_pairs, cutoff, …)` — pass-1 integral over the
   GB pair list. **Each unordered pair expands to BOTH directed contributions** (j→R_i and i→R_j
   differ: radii/scales are per-atom), since the cell list stores each pair once (claudex #2).
   **Apply an explicit `r² ≤ cutoff²` predicate** — the list carries a 2 Å Verlet buffer, so
   iterating it raw uses the wrong physical domain (claudex #2). Same near-overlap guard
   (`r² < 0.01`) as all-pairs.
2. `gb_obc_energy_and_forces_nbl` (replaces the `gb_obc.rs:619` stub) — pass-2 energy/direct
   forces + pass-3 HCT spread over the **same** `gb_pairs` with the **identical** `r² ≤ cutoff²`
   predicate and overlap guard as pass 1 (domain equality is necessary for gradient consistency
   but **not sufficient** — the predicate and guard must be byte-identical across passes, and
   pass-2 must accumulate ∂/∂R_i and ∂/∂R_j into `born_forces` for **both** atoms before the
   chain transform, exactly as `gb_obc.rs:529`). Pass-2 cutoff treatment = §3 (RF shift on
   distinct pairs, forceless; self term halved).
3. Method plumbing: add the nonbonded-method/GB-cutoff setting (§0). When GB is NoCutoff →
   all-pairs path unchanged (`energy.rs:643`). When GB is CutoffNonPeriodic → apply cutoff+shift
   in BOTH the all-pairs GB path AND the `_nbl` path; the 2000 gate selects only which.
4. NbCache: hold the LJ list and the GB (exclusion-free) list as one bundle, **rebuilt
   atomically from the same reference coords** under one drift check (claudex #4).
5. **Thread the GB list through `compute_energy_auto` and `compute_energy_and_forces_auto`**
   (`energy.rs:111/370`) — they currently build/pass only the one exclusion-filtered LJ list;
   adding an NbCache field does not reach those direct paths. Build the GB list there too; do
   NOT rebuild it inside each GB call (claudex #5).

## 5. Tests / oracle (revised per claudex #1/#3 + test additions)
- **Large-cutoff reduction (CORRECTED — claudex #1):** at GB cutoff ≥ box diameter, the cutoff
  path must match the all-pairs CutoffNonPeriodic path's **forces and Born radii to ~1e-9**, and
  its **energy to all-pairs-cutoff energy** (NOT NoCutoff — the RF shift `−Σ preFactor q_i q_j/
  cutoff` is present even when all pairs are in range). Equivalent check: cutoff-energy =
  NoCutoff-energy + analytic shift sum. This catches double-count / missing-pair / asymmetric-
  spread bugs without the shift confusing the assertion.
- **Buffer correctness:** pairs with `cutoff < r ≤ cutoff+buffer` (in the list but outside the
  physical cutoff) must contribute **zero** to energy and forces — proves the explicit `r²≤cutoff²`
  predicate is applied, not raw list iteration (claudex #2).
- **FD forces on the cutoff path (claudex #3):** extend the FD tests (`gb_obc.rs:882,919`) with a
  cutoff variant whose pair distances are all **comfortably away** from the cutoff (the potential
  is discontinuous at the boundary — FD is invalid for near-cutoff pairs). Separately, a
  near-boundary test that asserts the **energy jump** across the cutoff matches the shifted
  truncation (NOT differentiability). Newton-3 (Σforce=0) on the cutoff path. The forceless RF
  shift is independently checked here (FD must see no force from it).
- **energy-only ≡ energy-from-force-kernel** on the cutoff path (claudex test add).
- **Cross-path parity (tolerance, not bit-exact — claudex):** AllPair-impl vs CpuNbl-impl of the
  **same** CutoffNonPeriodic method agree to a tight numerical tolerance (iteration/accumulation
  order differs, so not bit-identical).
- **Stale-list across cutoff:** move an atom across the physical cutoff *without* tripping the
  Verlet rebuild; document/verify the (small, OpenMM-absent) inconsistency is bounded by the
  buffer — same semantics the LJ NBL already accepts.
- **OpenMM CutoffNonPeriodic oracle** — new `validation/` script mirroring `amber96_obc_oracle.py`
  with `nonbondedMethod=CutoffNonPeriodic` + matching cutoff both sides; crambin; ≤5% GB / ≤1%
  total. PLUS **small-system exact** OpenMM comparisons (energy + per-atom forces) on neutral and
  charged systems with heterogeneous radii/scales at several cutoffs — the 5% crambin tolerance
  alone can hide a sign/prefactor/missing-pair error (claudex). Wire into `tests/oracle/` (skips
  without OpenMM).
- **NoCutoff regression:** the default path's existing oracle + Rust tests stay green unchanged.

## 6. Non-goals
- **NbCache GB-list caching (DEFERRED to a follow-up PR).** This PR builds the
  exclusion-free GB list inside `compute_energy_and_forces_nbl` — O(N) per eval (vs the
  prior O(N²) all-pairs), correct and validated. Caching it in `NbCache` alongside the LJ
  list (one drift refresh) removes the per-eval cell-grid rebuild for iterative callers
  (minimizer line searches, MD steps). It ripples the energy-fn signatures (a
  `gb_nbl: Option<&NeighborList>` param threaded through `compute_energy{,_and_forces}_nbl`
  + the `*_auto` builders + `NbCache`) and needs its own AllPair-vs-cached parity test, so
  it lands as a focused follow-up rather than bloating this correctness PR. (codex code-review
  P2; plan §4.4.)
- GPU cutoff kernels (separate PR).
- Changing the default solvent path (CHARMM19+EEF1 stays the validated production path; this is
  the AMBER96+OBC secondary workflow).
- Periodic GB / salt screening / variable dielectric.

## 7. Risks
- **Pass-2 long-range truncation** is the real risk — if OpenMM hard-truncates without a shift,
  cutoff GB carries an intrinsic error vs NoCutoff; the oracle target is OpenMM-at-cutoff (not
  NoCutoff), so we match OpenMM's approximation, not ground truth. State this explicitly so the
  cutoff path isn't mistaken for the exact path.
- **Energy/force consistency under truncation** (pass-1 domain ≡ pass-3 spread domain).
- **Born-radius drift**: the GB list must refresh on the same Verlet criterion as the LJ list,
  or stale pairs bias Born radii during MD.

## 8. Open questions for review
1. ~~Pass-2 RF shift vs hard-truncate?~~ **RESOLVED (§3):** RF shift `-preFactor·q_i q_j/cutoff`
   on distinct in-cutoff pairs, forceless; Born radii hard-truncated; self term halved.
2. ~~Same cutoff as LJ, or GB-specific?~~ **RESOLVED:** same `nonbonded_cutoff` as LJ (matches
   OpenMM CutoffNonPeriodic; the GB list is independent so a larger GB cutoff stays possible if a
   future oracle shows it's needed).
3. ~~NbCache Option A vs B?~~ **RESOLVED (claudex #4):** A — LJ+GB lists as one bundle in NbCache,
   atomic rebuild, one drift check.
4. ~~>2000 size gate for GB?~~ **RESOLVED (claudex #7):** the gate is implementation-only within a
   method; cutoff-vs-NoCutoff is an explicit setting (§0). Default = NoCutoff O(N²) unchanged.

## 9. Review log (claudex)
All 7 findings adopted (none rejected): #1 large-cutoff test compares forces/radii to all-pairs +
energy to all-pairs-*cutoff* (the RF shift means it ≠ NoCutoff); #2 explicit `r²≤cutoff²` past the
buffer + both directed HCT contributions + dual Born-radius derivative accumulation; #3 RF shift is
force-free but the potential is discontinuous at the cutoff → FD only away from boundary, separate
boundary-jump test; #4 LJ+GB list bundle, atomic rebuild; #5 thread the GB list through the
`*_auto` direct paths, not just NbCache; #6 force CPU for GB under the cutoff method until GPU
kernels exist; #7 **the big one** — opt-in CutoffNonPeriodic *method*, not a size-gated physics
switch; default NoCutoff GB stays exact O(N²). Q1 (RF shift) pre-resolved from OpenMM ReferenceObc.cpp.
