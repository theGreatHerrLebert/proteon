# Codex code review — electrostatics scaffold + P0 oracle (2026-06-10)

`codex exec` (gpt-5.5) over the `feat/electrostatics` diff vs `origin/main`
(scaffold crate + NESSie harness). Confirmations and dispositions below.

## Confirmed sound (the questions that mattered)

- **Kernel subset is bit-exact.** `KERNEL_SUBSET=32` (`model.elements[1:32]` +
  centroids) is the *bit-exact top-left block* of NESSie's full collocation
  operator: both `laplacecoll!`/`regularyukawacoll!` matrix overloads assign
  `dest[o,e]` from a single per-pair evaluation — no global normalization, ordering
  dependence, or accumulation (Yukawa's only preprocessing is an independent
  per-element `TriangleQuad`). The subset is a valid kernel oracle.
- **`analytic_dump` fix correct.** `bornion` ∈ `NESSie.TestModel`; charge is
  `ion.charge.val`; radius in Å; `rfenergy(LocalES|NonlocalES, ion)` returns kJ/mol.
- **Core model matches NESSie.** Field meanings, charge units, and
  `yukawa()=√(εΣ/ε∞)/λ` all faithful.

## Fixed in this pass

| # | Sev | Finding | Fix |
|---|-----|---------|-----|
| 2 | High | `post_dump` not self-contained/reproducible (default GMRES; omitted model/method/params) | Switched to deterministic `:blas`; now emits `method`, `elements`, `charges`, `params`, `num_elements`. Fixtures regenerated. |
| 5 | Med | Silent truncation `min(n,len)`; no subset provenance; solve filenames `NESSie.LocalES` | `kernel_subset` now `error`s if mesh < n; dumps carry `subset{full_num_elements,size,indices}`; `LOC_NAME` map → clean `LocalES`/`NonlocalES` in names + fields (matches README). |
| 4 | Med | Born API overstates generality (NESSie assumes εΩ=1; `(1/εΣ−1)`) | Documented the εΩ=1 invariant; `born_rfenergy` now takes `Locality`, not `nonlocal: bool`. |
| 6 | Low | `PotentialKind::Double` doc baked in the ½-jump | Corrected: collocation is principal-value only; the `2π` jump is added at assembly. |
| — | — | `model.rs` doc claimed `εΣ=ε∞`+`λ→0` ⇒ local is a clean operator identity | Corrected to match formulation §10 (not an operator collapse; only the limiting potential, after eliminating `w`). |

## Deferred to the implementing phase (TODOs encoded in-code)

- **#1 (High) — solve/post API binding + `Result`.** `rfenergy(model, cauchy)` lets a
  caller pair Cauchy data with the wrong model; `solve_*` returns
  `(result, stats{converged})` a caller can ignore. These are `unimplemented!()`
  stubs — reshaping now is churn. Encoded as TODOs: P4 `solve_*` must return `Result`
  (non-convergence/non-finite = hard error) and bind the model it solved on (a
  `SolvedBem`), which `post` then consumes. Codex itself framed this as "change the
  signatures before implementing."
- **#3 (Med) — `Params` validation / private fields.** Already acknowledged in-code
  as the P4 hardening (`Params::new -> Result`); kept the `debug_assert` guard for
  now since `Params` is not yet executed.

All Rust checks green (`cargo fmt`/`clippy -p proteon-electrostatics`); fixtures
regenerated from NESSie v1.5.1.
