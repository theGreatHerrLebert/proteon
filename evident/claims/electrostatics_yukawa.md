# Regular-Yukawa collocation: proteon vs NESSie (P3)

Operational case writeup for the regular-Yukawa-collocation parity claim in
`claims/electrostatics_yukawa.yaml`.

## Problem

The L2 layer of the BEM electrostatics port is the **regular part of the Yukawa
potential** — Yukawa minus Laplace — integrated over each triangle with a 7-point
Radon cubature (`proteon-electrostatics/src/{quadrature,yukawa}.rs`, ported from
NESSie `src/Radon.jl` + `src/base/quadrature.jl`).

Splitting off the singular Laplace part (handled analytically in `laplace.rs`)
leaves a smooth integrand the cubature can resolve:

- single (×4π): `(e^(−yukawa·r) − 1) / r`
- double (×4π): `(1 − (1 + yukawa·r)·e^(−yukawa·r)) · (x−r')·n / r³`

The fragile bit is **catastrophic cancellation**: for small `yukawa·r` the closed
forms subtract two nearly-equal quantities, so NESSie switches to an alternating-
series expansion below `scalednorm = 0.1`, and uses explicit `r → 0` limits
(`−yukawa` for single, `yukawa²/(2√3)` for double). A wrong series coefficient, a
wrong branch threshold, or a wrong limit would be invisible on most entries but
corrupt the near-field — exactly where BEM assembly is most sensitive.

## Trust Strategy

Validation, layered:

1. **NESSie cross-implementation parity.** NESSie's `regularyukawacoll!` is the
   reference. We rebuild its `yukawa_dump` single + double matrices entry-by-entry
   from proteon's collocation on byte-identical geometry, at the fixture's `yukawa`
   exponent. The port mirrors NESSie operation-for-operation (same Radon
   points/weights, same series guard, same `×2·area`), so agreement is to libm
   precision.

2. **NESSie-independent numerical quadrature.** The single-layer collocation is
   compared to a fine centroid-subdivision integral of the *physical* regular
   kernel `e^(−yr)/r − 1/r` at non-singular points — independent of NESSie and of
   the Radon algebra. It catches a wrong integrand, a wrong `2·area` factor, or a
   bad barycentric→world mapping.

3. **Limits, continuity, and the L0 cubature.** The `r → 0` limits, series/closed-
   form continuity across the `0.1` boundary, and the cubature's own invariants
   (weights sum to ½, points inside the reference triangle, monomials exact to
   total degree 5) — none of which need an oracle.

## Inputs

- **Fixture:** `proteon-electrostatics/tests/fixtures/nessie/yukawa_na.json` —
  NESSie's `regularyukawacoll!` single + double matrices on a 32-element subset of
  the bundled Born sodium mesh, at the model's yukawa exponent, pinned to NESSie.jl
  1.5.1 (tree-hash `c6c0c478`). 2 × 32 × 32 = 2048 gated entries. The subset is a
  bit-exact block of the full operator (the cubature has no cross-element coupling).
- **Quadrature / limit / cubature:** a synthetic equilateral triangle, far
  observation points, and direct evaluations of the kernel at / near `r = 0` and
  across the `0.1` branch.

The test consumes the fixture's *stored* per-element normal verbatim
(`Tri::with_normal`) and the fixture's `yukawa`, so the integrand and the
double-layer `(x−r')·n` term run on geometry byte-identical to NESSie.

## Tolerances

- Collocation vs NESSie: **max abs < 1e-10**, **max rel < 1e-10** (rel only where
  |reference| > 1e-9) — orders of magnitude looser than observed libm divergence.
- Quadrature residual (single layer): **< 1e-4 relative** at non-singular points.
- Limits / continuity / cubature: **pass_rate == 1.0**.

## What this does NOT cover

- The **near-singular regime** (nearly-touching non-self elements), where the
  fixed 7-point Radon rule is itself inaccurate — adaptive subdivision is the
  mandatory P6.5 remediation, gated against high-precision quadrature there.
- An **independent double-layer quadrature** (the from-physics single-layer one is
  here; the double layer is gated against NESSie + the limit/continuity checks).
- The **deep-cancellation tail** (`scalednorm` far below 0.1) is only covered
  indirectly via the fixture; a targeted high-precision series check would
  strengthen it.

## How to run

```bash
cargo test -p proteon-electrostatics      # all gates (lib L0/L2 + parity test files)
```

No Julia is needed at gate time: the NESSie values are the checked-in fixture,
reproduced in the standard Rust CI job. Regenerate the fixture (only when
intentionally re-pinning NESSie) via the harness in `tools/oracle/nessie/`.
