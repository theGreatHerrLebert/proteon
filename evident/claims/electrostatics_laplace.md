# Laplace collocation: proteon vs NESSie (P2)

Operational case writeup for the Laplace-collocation parity claim in
`claims/electrostatics_laplace.yaml`.

## Problem

`proteon-electrostatics` ports NESSie.jl's boundary-element continuum
electrostatics into pure Rust. The first numerical layer (L1) is the
**Rjasanow analytic Laplace collocation**: the single- and double-layer
Laplace potential of a triangle at an observation point ξ, computed in
closed form via projection onto the element plane and the InPlane /
InSpace branches (`proteon-electrostatics/src/laplace.rs`, ported from
NESSie `src/Rjasanow.jl`).

This kernel is the building block the BEM system matrices assemble from;
every downstream Cauchy-data solve and reaction-field energy rests on it.
A transcription error here — a sign, a missing ½, a wrong InPlane↔InSpace
branch, the Julia-`sign` vs Rust-`signum` `0` discrepancy — would
propagate silently into every energy. The closed form is intricate, so an
entry-level oracle is the cheapest insurance.

## Trust Strategy

Validation, two independent oracles:

1. **NESSie cross-implementation parity.** NESSie's `laplacecoll!` is the
   reference Julia implementation of the same Rjasanow algebra. We rebuild
   its `collocation_dump` matrices entry-by-entry from proteon's
   kernel on byte-identical geometry. Because the port mirrors NESSie
   operation-for-operation, agreement is to libm precision; any structural
   error shows up far above the noise floor.

2. **NESSie-independent numerical quadrature.** Cross-implementation parity
   cannot catch a transcription error *shared* by both renderings of the
   same closed form. So we also integrate the defining surface integrals
   directly — `∫_T 1/|ξ−r'| dA` (single), `∫_T (ξ−r')·n/|ξ−r'|³ dA`
   (double) — by centroid subdivision at non-singular points, and require
   the analytic value to match. This gate knows nothing about Rjasanow or
   NESSie; it knows only the physics the closed form is supposed to encode.

Five metamorphic invariants (orientation, vanishing in-plane double layer,
rigid-motion invariance) pin the conventions both oracles assume.

## Inputs

- **Fixture:** `proteon-electrostatics/tests/fixtures/nessie/collocation_na.json`
  — NESSie's own `laplacecoll!` single + double matrices on a 32-element
  subset of its bundled Born sodium mesh (`data/born/na.off`), emitted by
  `tools/oracle/nessie/harness.jl`, pinned to NESSie.jl 1.5.1 (tree-hash
  `c6c0c478`) via the committed `Manifest.toml`. That is 2 × 32 × 32 = 2048
  gated entries. The subset is a bit-exact top-left block of the full
  operator (the kernel has no cross-element coupling).
- **Quadrature / metamorphic:** a synthetic equilateral triangle and a set
  of well-separated observation points (some on each side of the element
  plane, so the double layer exercises its sign flip).

The test consumes the fixture's *stored* per-element normal verbatim
(`Tri::with_normal`), so projection and the double-layer sign run on
geometry byte-identical to NESSie; `distorig = normal·v1` reproduces
NESSie's `props`.

## Tolerances

- Collocation vs NESSie: **max abs < 1e-10**, **max rel < 1e-11** (rel only
  where |reference| > 1e-9). Both are orders of magnitude looser than the
  observed asin/log/sqrt libm divergence, so they trip on structure, not
  noise.
- Quadrature residual: **< 1e-4 relative** at non-singular points
  (256×256 centroid subdivision).
- Metamorphic: **pass_rate == 1.0** over the five invariants.

## What this does NOT cover

- The **singular self / diagonal (InPlane) term** is gated against NESSie
  only — the numerical-quadrature oracle runs at non-singular points where
  centroid subdivision converges. An independent singular oracle (Duffy /
  corrected quadrature, or a CAS closed form for the equilateral self entry)
  is owed before assembled energies are trusted (P4/P5).
- One mesh at one parameter set. The **near-singular corpus** that actually
  stresses the InSpace forms near the surface — the regime BEM assembly
  cares about most — is the natural P2 follow-up.

## How to run

```bash
cargo test -p proteon-electrostatics      # all three gates (lib + 2 test files)
```

No Julia is needed at gate time: the NESSie values are the checked-in
fixture, reproduced in the standard Rust CI job. Regenerate the fixture
(only when intentionally re-pinning NESSie) via the harness in
`tools/oracle/nessie/`.
