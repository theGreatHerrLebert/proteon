# GPU_CUTOFF_GB_PLAN — CutoffNonPeriodic OBC GB on the GPU

Status: DRAFT (pre-implementation). Closes the last GB-cutoff gap: #154 added the
CutoffNonPeriodic GB *method* on CPU but the GPU OBC kernels are NoCutoff-only, so
`GpuStructState::new` refuses the method and cutoff GB falls back to CPU. This adds the cutoff
+ reaction-field shift to the GPU kernels so cutoff GB runs on-device, CPU-parity-gated.

## 1. Scope
- The GPU OBC kernels are **all-pairs** (no neighbor list) — for NoCutoff AND for this cutoff
  work. The cutoff is applied as a per-pair predicate in the existing all-pairs kernels (each
  thread already loops all j); this gives OpenMM-correct cutoff GB on GPU. A GPU GB *neighbor
  list* (true O(N) work) is a separate, larger optimization and is OUT of scope.
- NoCutoff must stay **byte-identical** (the existing `gpu_obc_matches_cpu_on_crambin` parity
  test is the regression guard).

## 2. The INFINITY-sentinel trick (no NoCutoff branch)
Thread two scalars into the pair kernels: `cutoff_sq` and `rc`. For the cutoff method
`(rc², rc)`; for NoCutoff `(f64::INFINITY, f64::INFINITY)`. Then the kernels ALWAYS run:
- `if (r2 > cutoff_sq) continue/return;` — never true for `INFINITY`, so NoCutoff skips nothing.
- energy shift `energy_i -= 0.5 * pqi * charges[j] / rc;` — `x/INFINITY == 0.0`, and
  `energy_i -= 0.0 == energy_i` exactly in IEEE, so NoCutoff is unchanged bit-for-bit.
No `if (cutoff)` branching; NoCutoff numerics are provably identical.

## 3. Kernel changes (`obc_kernel.cu`)
1. **`obc_born_radii`** — add `double cutoff_sq`. After the `if (r2 < 0.01) continue;` guard
   (line 58), add `if (r2 > cutoff_sq) continue;`. (Truncates the descreening integral, matching
   CPU `compute_born_radii_with_chain` cutoff branch.)
2. **`obc_energy_forces_direct`** — add `double cutoff_sq, double rc`. In the `j != i` branch,
   after computing `r2` (line 153): `if (r2 > cutoff_sq) continue;`. After
   `energy_i += 0.5 * gpol;` (line 164): `energy_i -= 0.5 * pqi * charges[j] / rc;`.
   **Why 0.5·pqi·q_j/rc:** the kernel is rowwise — thread i adds `0.5·gpol` per partner j and
   thread j adds the other half, so the full pair energy is `gpol` per unordered pair. The CPU
   subtracts `pqi·q_j/rc` ONCE per unordered pair; on GPU each of the two threads subtracts half
   ⇒ `0.5·pqi·q_j/rc` per ordered pair, summing to the same total. (`pqi=pre_factor·q_i`, so the
   term is symmetric `pre_factor·q_i·q_j/rc`.) The shift is **distance-independent ⇒ forceless**
   (no change to `fxi/fyi/fzi` or `bf_i`), matching OpenMM/CPU.
   Self term (`j==i`) unchanged: halved, no shift.
3. **`obc_force_spread`** — add `double cutoff_sq`. After `if (r2 < 0.01) return;` (line 226):
   `if (r2 > cutoff_sq) return;`. The spread MUST truncate on the SAME predicate as pass 1 (Born
   integral) or forces stop being the gradient of the truncated energy — `cutoff_sq` is the same
   value passed to both kernels.
4. `obc_chain_transform` — no change (per-atom, no pairs).

## 4. gpu.rs changes
- **Remove the #154 guard** (`if ff.gb_cutoff().is_some() { return Err(...) }` in
  `GpuStructState::new`) — the kernels now support the method.
- Add fields `obc_cutoff_sq: f64`, `obc_rc: f64`, set:
  `match ff.gb_cutoff() { Some(rc) => (rc*rc, rc), None => (f64::INFINITY, f64::INFINITY) }`.
  (Only meaningful when `obc_enabled`; harmless otherwise.)
- Thread the args into the 3 launches at the SAME positions as the kernel signatures:
  `obc_born_radii` gets `cutoff_sq` after `n_atoms`; `obc_energy_forces_direct` gets
  `cutoff_sq, rc` after `include_self_term`; `obc_force_spread` gets `cutoff_sq` after `n_atoms`.
  (cudarc arg order must match exactly.)

## 5. NbCache interaction
With the guard gone, `GpuStructState::new` succeeds under cutoff GB, so NbCache uses the GPU
path for cutoff GB on large systems (Auto + device present). The GPU path is all-pairs-with-
predicate (correct, O(N²) parallel); the CPU fallback still uses the cached `gb_nbl` (#155,
O(N)). Both are parity-tested ⇒ correctness is path-independent. The `gb_nbl` is simply unused
when the GPU path is taken — no NbCache change needed. (The earlier "Hamiltonian depends on CUDA
availability" concern is resolved: GPU and CPU now compute the SAME cutoff method.)

## 6. Tests
- **`gpu_obc_cutoff_matches_cpu_on_crambin`** (new, `minimize.rs` gpu_parity_tests): crambin +
  `amber96_obc_cutoff()` with a cutoff that actually truncates (e.g. `cutoff_override = 8 Å`).
  GPU energy+forces vs the CPU cutoff path (`compute_energy_and_forces` all-pairs cutoff). Same
  tolerances as the existing OBC test: energy 1e-3, solvation 1e-3, max force 1e-3 (atomicAdd
  non-determinism). Skips when no GPU.
- **NoCutoff regression:** the existing `gpu_obc_matches_cpu_on_crambin` must stay green
  unchanged (proves the INFINITY sentinel didn't perturb NoCutoff).
- CPU-only build (`cargo test -p proteon-core`) unaffected (kernels are `cuda`-gated).

## 7. Validation
Build + run on the local RTX 2070 (CUDA 12.4, NVRTC runtime compile): `cargo test -p proteon-core
--features cuda forcefield::minimize::gpu_parity_tests`. Confirm both OBC parity tests pass.

## 8. Non-goals / risks
- GPU GB neighbor list (true O(N) GPU work) — separate.
- Risk: arg-order mismatch in cudarc launches (compile-checked types, but order is positional —
  double-check against the kernel signatures). Risk: the spread predicate must use the identical
  `cutoff_sq` as the Born kernel (energy/force consistency) — it does (one field, passed to both).

## 9. Review log (claudex) — adopted
Kernel math CONFIRMED correct (0.5 RF factor sums to `pre·q_i·q_j/rc`; shift forceless; the three
truncation sites — Born integral, direct energy/force/born_forces, HCT spread — are complete; r==rc
included, matching CPU). Revisions:
1. **Soften §2:** the INFINITY sentinel is bit-preserving for *valid finite* inputs only
   (`x − 0.0 == x`); a NaN/overflowed `r2` is processed identically to today (`NaN > ∞`, `∞ > ∞`,
   `finite > ∞` all false) so there's no NoCutoff regression, but it's not a universal byte-identity
   claim. Guard the cutoff case `rc.is_finite() && rc > 0.0` (it is, via `nonbonded_cutoff`).
2. **Move the NbCache method-consistency check BEFORE GPU dispatch** (it currently sits in the CPU
   fallback at `nb_cache.rs`): GPU constants are captured at `GpuStructState::new`, so a caller
   reusing an `NbCache` with a changed GB method would silently keep the GPU's old cutoff. Assert
   `params.gb_cutoff() == self.gb_cutoff` at the top of `energy`/`energy_and_forces`, before the
   GPU branch.
3. **Tests (§6) expanded:** (a) a charged 2-atom GPU test — large finite cutoff vs NoCutoff: verify
   the analytic energy shift AND identical GPU forces (isolates the forceless shift); (b) pairs just
   inside / exactly at / just outside the cutoff; (c) GPU finite-difference forces away from the
   boundary (CPU FD alone can't catch a missing GPU *spread* predicate); (d) exercise real
   `NbCache + Auto` dispatch above the GPU threshold, not only direct `GpuStructState`; (e) compare
   the solvation component separately (total-energy parity can mask compensating errors).
