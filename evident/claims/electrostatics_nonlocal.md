# Nonlocal 3-block solve + nonlocal energy (P6)

Operational case writeup for the nonlocal claim in
`claims/electrostatics_nonlocal.yaml`.

## Problem

P6 is the differentiator. Nonlocal (Lorentz cavity / Yukawa) continuum
electrostatics models the *structured* solvent — the finite correlation length of
water's response — which the standard local Poisson-Boltzmann tools (APBS, DelPhi,
…) do not. It replaces the local 2-stage solve with a single **coupled 3-block
system** for `(u, q, w)`, where `w = γ₀ext(Ψ)` is a genuinely new unknown
(`proteon-electrostatics/src/{system,solve,post}.rs`, ported from
NESSie `src/bem/nonlocal.jl`).

It is also the most intricate assembly in the port: a `3n × 3n` operator over four
collocation matrices (Laplace `V`/`K`, regular-Yukawa `Vy`/`Ky`), and two subtleties
the formulation spec §6 flags as critical:

1. **Two distinct "diagonals."** The algebraic matrix diagonal has `−diag(V)` in the
   middle block, but NESSie's `diag(A)` method — the vector fed to the Jacobi
   preconditioner — uses `+diag(V)`. proteon keeps them separate: the preconditioner
   uses NESSie's `+diag(V)` form.
2. **The RHS** is `[b1; 0; 0]` (the explicit assembly's dimensionally-sound form);
   NESSie's implicit path builds only the length-`n` `b1`, an apparent source defect
   resolved against the running oracle.

## Trust Strategy

Validation, four angles:

1. **Nonlocal Cauchy-data parity.** `u, q, w, umol, qmol` vs NESSie's `:blas` (exact
   LU) `solve_NonlocalES` fixture. Getting **`w`** right is the load-bearing check —
   it is the unknown that does not exist in the local problem.
2. **Nonlocal post parity.** `rfenergy` + `espotential` over Ω/Σ/Γ vs NESSie's
   `post_nonlocal`. The **Σ branch** is the nonlocal one (it pulls in `w` and the
   regular-Yukawa kernels).
3. **Nonlocal → local limit.** As the correlation length `λ` shrinks, the nonlocal
   Born energy approaches the local one — asserted at the *analytic* level, where it
   is clean. (The operator-level collapse is **not** asserted; spec §10 shows it is
   not a clean identity.)
4. **BEM vs analytic Born.** The nonlocal BEM energy of a central charge in a
   triangulated sphere vs the closed-form nonlocal Born energy — the whole nonlocal
   stack against analytic physics.

## Tolerances

- Cauchy data (`u`,`q`,`w`) vs NESSie: **< 1e-6** (umol/qmol < 1e-10).
- Nonlocal post (rfenergy + Ω/Σ/Γ): **< 1e-6**.
- Nonlocal → local Born limit: **< 1%** at `λ = 0.1`, gap shrinking monotonically.
- BEM vs nonlocal Born: **< 5%** at subdivisions 1/2/3.

## What this does NOT cover — the honest caveat

The BEM-vs-Born nonlocal gate is **Radon-cubature-floor-limited**. Unlike the
analytic-Laplace local case (which converges *monotonically* to Born), the nonlocal
energy **plateaus** at a few percent: the fixed 7-point Radon cubature for the
regular-Yukawa kernel loses accuracy for near-neighbour elements as the mesh refines.
So this gate pins the *physics* (a sign or coefficient error in the 3-block assembly
would be wildly off, not a few percent), not the convergence rate. **Tight nonlocal
convergence — and protein-scale nonlocal solves — need the P6.5 near-singular
remediation** (adaptive subdivision / corrected quadrature).

Also: one structure, one parameter set; and the plan's headline nonlocal analytic
model (Xie's multi-charge sphere, Bessel functions) is not yet ported — only the
single-charge Born nonlocal model is gated.

## How to run

```bash
cargo test -p proteon-electrostatics      # lib + all parity/limit/convergence tests
```

No Julia at gate time; the NESSie values are checked-in fixtures.
