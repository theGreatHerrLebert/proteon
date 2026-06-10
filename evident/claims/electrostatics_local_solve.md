# Local BEM solve: proteon vs NESSie (P4)

Operational case writeup for the local-Cauchy-data parity claim in
`claims/electrostatics_local_solve.yaml`.

## Problem

P4 is where the collocation kernels become a *solvable system*. The local
(Poisson) BEM solve recovers the surface Cauchy data — `u = γ₀int(φ*)` and
`q = γ₁int(φ*)` — from the molecular potential of the point charges, by solving two
sequential `numelem × numelem` systems (`proteon-electrostatics/src/{system,solve}.rs`,
ported from NESSie `src/bem/{local,implicit}.jl`):

```
Stage 1:  M·u = b₁,   M = 2π(1 + εΩ/εΣ)·I + (εΩ/εΣ − 1)·K,   b₁ = (K − 2π)·umol − (εΩ/εΣ)·V·qmol
Stage 2:  V·q = b₂,   b₂ = (2π·I + K)·u
```

`V`/`K` are the single/double-layer Laplace collocation matrices (P2); `umol`/`qmol`
are the molecular potential and its normal derivative; `2π = 4π·σ` is the ½-jump.
This is the largest single step of the port: it adds assembly, a matrix-free
operator, a rolled GMRES, and the first values that depend on *all* the prior layers
at once.

## Trust Strategy

Validation, three independent angles:

1. **NESSie Cauchy-data parity.** NESSie's `:blas` path is the *exact* dense-LU
   solution of the same system, so it is the ground-truth target. proteon assembles
   the system on byte-identical geometry and solves with its own GMRES; `u, q, umol,
   qmol` must match.

2. **LU vs GMRES (and LU vs fixture).** A dense LU (nalgebra, test-only) of
   proteon's *own* explicit `M`/`V`:
   - vs the **fixture** → entrywise assembly check (the fixture is NESSie's LU
     solution, so a wrong `M`/`V`/`b` would show up here);
   - vs **GMRES** → solver-parity (the iterative solver reached the true solution,
     not a self-consistent wrong one).

3. **True residual.** `‖A·x − b‖/‖b‖` for both stages, gated directly — the
   load-bearing "the solution actually solves the system" check, independent of both
   the oracle and the LU. Iteration count is a *measured* stat, not a gate.

## Inputs

- **Fixture:** `solve_LocalES_blas_na.json` — NESSie's local Cauchy data on the
  512-element Born sodium mesh (`εΩ=1, εΣ=78`, one unit charge at the origin), pinned
  to NESSie.jl 1.5.1. The test consumes its stored normals + charges + params
  verbatim.

A note on units: NESSie labels `u`/`q` as "premultiplied by 4π·ε0", but the code
never multiplies by `ε0` (it divides the raw charge sum by `εΩ` only) — so neither
the fixture nor proteon carries an `ε0` factor.

## Tolerances

- Cauchy data vs NESSie: `u`/`q` **< 1e-7** (GMRES-converged vs exact LU);
  `umol`/`qmol` **< 1e-10** (direct evaluations, no solve).
- LU vs fixture **< 1e-9**; LU vs GMRES **< 1e-7**.
- True residual **< 1e-8** (both stages).

## What this does NOT cover

- **One structure, one parameter set.** Multi-charge systems, extreme dielectric
  ratios, and ill-conditioned meshes are unexercised — a protein-scale solve is a
  follow-up (after the P6.5 mesh-acceptance + scaling work).
- **Scaling.** The dense O(N²) assembly + matvec are used here; the O(N) matrix-free
  `K·x` and a fast-summation backend (plan §6) are not yet implemented, so this says
  nothing about large-N memory/time.
- **Hard systems.** Jacobi-preconditioned GMRES converges well here; it may stagnate
  on refined/nonlocal systems. The true-residual gate would catch that (GMRES errors
  via `SolveError::NotConverged`), but the preconditioner may need revisiting.

## How to run

```bash
cargo test -p proteon-electrostatics      # lib + all parity test files
```

No Julia at gate time: the NESSie values are the checked-in fixture, reproduced in
the standard Rust CI job. `nalgebra` is a test-only dev-dependency (the LU oracle).
