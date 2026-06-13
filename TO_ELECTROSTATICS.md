# Plan: port NESSie.jl's (non)local protein electrostatics to proteon

## 0. Goal & verdict

Port NESSie.jl's **boundary-element (BEM) continuum-electrostatics** solver into
pure Rust in proteon, validated the proteon way: **port, then gate every layer
against NESSie.jl as an oracle** via EVIDENT claims — with the bonus that NESSie
ships **closed-form analytical models** (Born ion, Xie multi-charge spheres) that
serve as *independent* ground truth above the cross-implementation parity.

This adds a **high-accuracy reference / research tier** of continuum
electrostatics alongside proteon's existing *approximate, fast* solvation tier
(OBC Generalized Born in `proteon-core/src/forcefield/gb_obc.rs`, EEF1): solve the
Poisson (local) or nonlocal-Poisson (Lorentz cavity / Yukawa) problem on the
molecular surface and read off reaction-field energies and potentials. The
**nonlocal** capability is a genuine differentiator — APBS/DelPhi/etc. do local PB
only.

**Positioning (decided, see §6 O(N²) ceiling):** this is **not** a drop-in GB
replacement for routine protein workloads. Ship it as a high-accuracy
reference/validation tier with explicit triangle-count and runtime limits.
Nonlocal protein-scale BEM and any force-field-energy-component wiring proceed
**only** if benchmarks establish a useful bounded regime or fast summation gets a
funded phase — not by default.

**Portability verdict (from reading the source): the kernel port is a medium lift;
a fully validated protein-scale solver is a large one.** NESSie is MIT, ~pure
Julia, and the numerical core has no exotic dependencies — the BEM assembly +
collocation kernels are self-contained (`LinearAlgebra`, `Distances`,
`SpecialFunctions` for the analytical models only). The two things proteon must
*acquire* rather than port are (a) a preconditioned **GMRES** (NESSie uses
`IterativeSolvers.gmres` + `Preconditioners.DiagonalPreconditioner`) and (b) a
**matrix-free / implicit system-matrix** representation (NESSie uses
`ImplicitArrays`) — both map onto patterns proteon already has from the
NBL/implicit energy paths. Author is Thomas Kemmer (third-party, like BALL /
MMseqs2), so this is an **oracle port**, not a merge.

**Effort split (be honest — three tiers, not two):**
- **Medium** — formulation/convention spec (§1b) + L0–L2 **kernel parity**
  (Laplace + regular-Yukawa collocation gated vs NESSie *and* numerical quadrature).
- **Large** — a *reliable local solve* + the scientific validation around it
  (assembly correctness, GMRES robustness, potential-level convergence on analytic
  spheres, physical invariants). This is itself a substantial validated-numerics
  project; P0 and P5 are **not** "medium" (P0 alone carries six Julia exporters,
  singular high-precision quadrature, external CAS fixtures, and a sphere mesher).
- **Larger / experimental** — the SES-fed **nonlocal protein** solver: gated on
  near-singular remediation (§6) and O(N²) scaling evidence, not assumed.

The matvec is O(N²) without a fast summation method (§6); do not scope it away.

**The hard part is not dependencies — it is (i) pinning the exact boundary-integral
formulation and unit conventions** (the first deliverable, §1b — a single
convention bug can hide behind close NESSie parity and a compensating Born energy),
**(ii) numerical fidelity of the singular *and near-singular* collocation
integrals** (Rjasanow's analytic Laplace potential — whose InPlane closed form *is*
the self/diagonal term — and Radon's 7-point Yukawa cubature, which is **not**
accurate for nearly-touching non-self elements), **and (iii) the dependency on a
correct molecular surface mesh.** This plan sits **downstream of
`TO_SES_TRIANGULATION.md`** — BEM operates on exactly the SES triangles that port
is about to produce. Local Poisson lands first (broad appeal, 2-block system);
nonlocal follows (3-block system, the differentiator).

## 1. The pipeline (as it exists in NESSie)

```
surface mesh (Triangle{T}) + point charges (Charge{T}) + Option{εΩ,εΣ,ε∞,λ}
  │
  │   observation points Ξ = triangle centroids; yukawa exponent = √(εΣ/ε∞)/λ
  │
  ├─ Rjasanow.laplacecoll        src/Rjasanow.jl
  │     ANALYTIC single/double-layer Laplace potential per triangle, with the
  │     observation point projected onto the element plane; InPlane vs InSpace
  │     closed forms; degenerate-triangle guards at _etol(T). (×4π premultiplied.)
  │
  ├─ Radon.regularyukawacoll     src/Radon.jl
  │     REGULAR part of the Yukawa potential (Yukawa − Laplace) via a 7-point
  │     Radon cubature over each triangle; alternating-series expansion guards
  │     cancellation for small yukawa·|x−ξ|. (×4π premultiplied.)
  │
  ├─ BEM.solve(LocalES|NonlocalES, model; method=:gmres|:blas)   src/bem/{local,nonlocal,implicit}.jl
  │     LOCAL  : 2-block system → Cauchy data (u = γ₀φ, q = γ₁φ).
  │     NONLOCAL: 3-block system (3·numelem) → (u, q, w = γ₀ext Ψ).
  │     :blas  = explicit dense matrices (O(N²) memory);
  │     :gmres = implicit matrices + DiagonalPreconditioner (O(N) memory).
  │     Returns LocalBEMResult / NonlocalBEMResult {model, u, q, [w], umol, qmol}.
  │
  └─ BEM.post                    src/bem/post.jl
        rfenergy(bem)                      → reaction-field energy (kJ/mol)  ← headline
        espotential(:Ω|:Σ|:Γ, ξ, bem)      → electrostatic potential by domain
        molpotential / rfpotential
```

Independent analytical oracles shipped *inside* NESSie (`src/testmodel/`):

```
TestModel.BornIon / bornion(name)          closed-form Born solvation energy of a single ion
TestModel.LocalXieModel                    multi-charge dielectric sphere, LOCAL closed form
TestModel.NonlocalXieModel1 / …Model2      multi-charge sphere, NONLOCAL closed form (Bessel I/K)
```

These give exact reaction-field energies/potentials for spherically-symmetric
systems — the strongest possible gate, independent of NESSie's own BEM path.

## 1b. Formulation & convention spec — the FIRST deliverable (P0.5)

Before any kernel is ported, transcribe NESSie's actual boundary-integral system
into the plan **algebraically**, from `src/bem/{local,nonlocal,implicit}.jl` and
`src/base/{potentials,constants}.jl`. "2-block / 3-block" is not a specification.
Close NESSie parity can faithfully reproduce a convention mistake while the Born
energy compensates in post-processing — so the conventions must be pinned, written
down, and unit-tested *as a spec*, not inferred during the port. The spec must
state, for **both** the local and nonlocal systems:

- **Normal orientation** (outward from solute Ω) and the sign convention it implies
  for the double-layer operator.
- **Trace-jump / solid-angle terms** — the `±½I` on the diagonal. NESSie encodes
  the ½ as the `σ = 0.5` constant (`base/constants.jl`); confirm where it enters
  each block and that the piecewise-flat collocation reproduces the flat-surface
  solid angle exactly.
- **Single- and double-layer operator definitions** (`V`, `K`, `K'`) and which
  kernel (Laplace vs regular-Yukawa) populates each block.
- **Definition of each unknown** — is `q` the normal derivative `∂φ/∂n`, a
  dielectric flux `ε ∂φ/∂n`, or a scaled quantity? What is `w = γ₀ext Ψ`?
- **Dielectric factors** multiplying each block (`εΩ`, `εΣ`, `ε∞`) and the Yukawa
  exponent `√(εΣ/ε∞)/λ`.
- **Discretization, not just symbols** (two impls can share symbols and still
  differ): the **trial basis** (piecewise-constant per element) and the
  **collocation scheme** — this is **point collocation** (Dirac test functionals at
  triangle centroids), *not* a Galerkin method with piecewise-constant test
  *spaces*; state it explicitly so no one reads "piecewise-constant test space" as
  Galerkin. Also **area weighting**, **unknown ordering**
  within and across blocks, and the **RHS source vectors** for each block.
- **Exterior condition & gauge**: the decay/radiation condition at infinity and any
  potential-gauge choice; **net-charge** behavior; **topology-aware assignment** of
  each charge to its dielectric region (solute component vs solvent vs disconnected
  cavity).
- **The full unit chain**: Coulomb-source normalization, the `4π` premultiplier on
  collocation results, `ε0`, Å→m, the `10¹⁰·ec` elementary-charge factor, and the
  final kJ/mol of `rfenergy`. Resolve where each `4π` and `ε0` enters.

**Acceptance for P0.5** (the spec is "done" when):
1. Every block matrix + RHS is written out algebraically with operator arguments,
   normal-derivative directions, and dielectric factors explicit.
2. A **hand-computable single fixture** — one operator-matrix *entry* (one
   source-element / one observation-element pair) and one RHS entry — is derived by
   hand and checked in, so the first ported kernel value has a non-NESSie reference.
3. The **unit/dimensional chain** is tested in isolation by feeding an *injected
   analytic reaction potential* through the `rfenergy` post-processing and
   recovering the closed-form energy — i.e. validate the conversion arithmetic
   **without** any kernel/assembly/solve. (A real Born-ion *BEM* end-to-end test
   needs kernels + assembly and therefore belongs at **P5**, not here.)

Deliverable: `devdocs/ELECTROSTATICS_FORMULATION.md` with the above, wired as the
first EVIDENT artifact. This section gates P2 onward.

## 2. Where it goes in proteon

New workspace crate **`proteon-electrostatics/`** (pyo3-free, sibling to
`proteon-core`), consuming proteon's own SES mesh + charges:

```
proteon-electrostatics/src/
  model.rs        Charge{pos,val}, BemModel{ tris: &[Triangle], charges, params:Option }
                  Option{ eps_omega, eps_sigma, eps_inf, lambda }; yukawa() = √(εΣ/ε∞)/λ
                  (Triangle props — centroid/normal/area — reused from proteon-core::surface)
  quadrature.rs   Radon 7-point triangle cubature points/weights (TriangleQuad)
  laplace.rs      Rjasanow analytic single/double-layer Laplace collocation
                  (InPlane/InSpace closed forms + projection + _etol guards)
  yukawa.rs       Radon regular-Yukawa collocation (Yukawa − Laplace) + series guard
  system.rs       implicit (matrix-free) 2-block (local) / 3-block (nonlocal) operators
  solve.rs        GMRES + diagonal preconditioner; LocalResult / NonlocalResult
  post.rs         rfenergy, espotential(domain), molpotential, rfpotential
  analytic.rs     Born / Xie closed-form models  ← in-tree, for self-validating tests
  lib.rs          public API: solve_local / solve_nonlocal / rfenergy / espotential
```

- **Geometry kernel reuse:** triangle centroid/normal/area, point-to-plane
  distance/projection, and dot products live in `proteon-core::surface::geom`
  (built by the SES port). Do **not** re-derive — share, same epsilons.
- **GMRES + preconditioner:** roll our own matrix-free GMRES in `solve.rs` (decided
  — op-order control for parity + zero dep). **Budget it honestly: this is not 150
  lines.** A trustworthy restarted GMRES needs modified Gram-Schmidt, restart,
  happy-breakdown detection, residual replacement, stagnation handling, and
  diagnostics. Start with a scalar **Jacobi** preconditioner to mirror NESSie, but
  expect it to be weak on refined or poor-quality meshes and on the nonlocal block
  system — design the operator interface so a **block-diagonal** preconditioner can
  drop in. Gate the **true (unpreconditioned) residual** and per-block residuals,
  not just Cauchy-data parity.
- **Dense direct solve for tests:** a vetted **LU/QR** (e.g. `nalgebra`/`faer` on
  small fixtures) is required as the **solver** oracle (LU-vs-GMRES) — see L3.
  Assembly correctness is gated separately (entrywise vs NESSie `system_dump`); LU of
  proteon's *own* matrix would inherit any assembly bug. Test-only, not on the
  production path.
- **Bessel I/K** for the Xie nonlocal analytical model: a `special-functions`/
  `puruspe`-style crate, or port the few needed `besseli`/`besselk` evaluations.
- **Mesh hand-off:** proteon's SES mesher emits `Mesh{verts,normals,tris}`; the BEM
  observation set is the triangle centroids. A thin adapter replaces NESSie's
  `Format` readers (`.off/.hmo/.msms/.pqr`) — proteon already holds charges from
  force-field params, so PQR ingestion is optional, not on the critical path.
- Python exposure later via `proteon-connector/src/py_electrostatics.rs`; CLI via
  a `proteon rfenergy` / `proteon electrostatics` subcommand (same shape as the
  analysis CLI).

## 3. The oracle: a NESSie.jl JSON-emitting harness

NESSie is Julia, so the oracle is a **thin Julia harness** (mirrors how `ball-py`
is the BALL oracle and how `BALLJL` already appears in `devdocs/ORACLE.md`). It
dumps deterministic JSON fixtures that the Rust tests load and gate against. No
new runtime dep for proteon — the harness runs offline to *generate* fixtures,
which are checked in (small) or regenerated by a documented script.

`tools/oracle/nessie/` (Julia project pinning `NESSie@1.5`):

- `collocation_dump(mesh, ptype) -> json` — per-element single/double-layer
  **Laplace** collocation values (`Rjasanow.laplacecoll`) for a fixed observation
  set. L1 gate. Include element geometry + observation points so the Rust side is
  fully determined.
- `yukawa_dump(mesh, ptype, yukawa) -> json` — per-element **regular Yukawa**
  collocation values (`Radon.regularyukawacoll`). L2 gate. Sweep `yukawa` across
  the small/large-argument branch boundary (the series-guard threshold `0.1`).
- `solve_dump(model, locality, method) -> json` — full Cauchy data `u,q,[w]`,
  plus `umol,qmol`, for both `:gmres` and `:blas`. L3 gate. Pin `εΩ,εΣ,ε∞,λ`.
- `system_dump(model, locality) -> json` — NESSie's **explicit dense system matrix
  blocks + RHS vectors** (the `:blas` assembly) on a small mesh. This is the real
  **assembly oracle**: gate proteon's assembled blocks/RHS **entrywise** against
  NESSie (see L3 — an LU solve of proteon's *own* matrix validates the solver, not
  the assembly).
- `post_dump(model, locality) -> json` — `rfenergy`, and `espotential` sampled at
  a fixed point set across domains `:Ω/:Σ/:Γ`. L4 gate (NESSie parity).
- `analytic_dump() -> json` — closed-form `BornIon` energies and `Local/Nonlocal
  XieModel` energies/potentials. **L4 independent ground truth** (NESSie's own
  analytical lineage, separate from the BEM path).

Pin NESSie version + git sha + Julia version into every dump. Reuse NESSie's own
fixtures: `NESSie.jl/data/born/{na,k,…}.{off,pqr}` and `data/xie/{2LZX.pqr,
unitsphere.off}` are ready-made, physically meaningful inputs.

**NESSie parity is necessary but not independent.** NESSie's analytical models are
mathematically independent of its BEM path, but *porting those formulas into Rust*
re-introduces shared transcription + special-function risk. Add genuinely
independent checks so the gate cannot be passed by a correlated error:

- **A third kernel oracle: high-precision numerical quadrature** of the Laplace and
  Yukawa kernels, generated offline and checked in. **Needs a concrete
  singular-integration design, not "adaptive quadrature" hand-waving** (otherwise
  this is an open-ended numerical-analysis project): use **Duffy transformations**
  for the on-panel single-layer singularity, **panel subdivision** for near-singular
  pairs, an explicit **separation of the double-layer principal value from its jump
  term**, a stated working precision (f128 / mpmath) and **error estimator**, and a
  documented target tolerance. Gate the analytic collocation kernels (L1/L2) against
  it — *especially* near-singular, where "regular Yukawa + Laplace ≈ full Yukawa" is
  definitional and shares errors. Scope this design as part of P0; if it balloons,
  it is a signal the kernel accuracy story itself is the hard part.
- **Externally-generated closed-form fixtures** for Born/Xie, computed in an
  independent CAS/mpmath at high precision and checked in — so the Rust `analytic.rs`
  is gated against a non-NESSie evaluation of the same formula (catches Bessel /
  transcription bugs the "oracle for the oracle" note already flags).
- **A mature local-PB cross-check (APBS or similar)** on matched spherical geometry
  for the local Poisson energy — a fully independent solver lineage.
- **An `analytic_sphere_mesh(radius, density)` generator** (vertices projected onto
  the exact sphere) shipped in-tree. The Born/Xie convergence claim runs on **these
  exact meshes, not the SES mesher** — decoupling BEM convergence from SES geometry
  convergence (see L4 / Q1).

## 4. Decomposition into independently-testable layers (the core ask)

Each layer is callable and gated on its own. Collocation/Cauchy-data values are
**numerically reproducible** (unlike a non-unique mesh), so gate on **tolerance**,
not exact equality — NESSie targets ~4–5 dp vs C++ under `-ffast-math`; proteon-vs-
Julia should agree tighter (both IEEE, no fast-math). Use type-aware tolerances
(`_etol`: 1.45e-8 f64, 3.45e-4 f32).

**L0 — Geometry + quadrature (`quadrature.rs`, geom reuse).** No NESSie needed.
- Unit/property tests: triangle centroid/normal/area; point→plane projection;
  Radon 7-point weights sum to 1 and integrate low-order polynomials exactly over
  a reference triangle; cubature points lie in-triangle (barycentric ≥ 0).
- *Independently testable: fully.*

**L1 — Laplace collocation (`laplace.rs`).** Oracles: `collocation_dump` **and**
high-precision numerical quadrature (§3).
- Gate per-element single- **and** double-layer values vs NESSie within tol, *and*
  vs the independent numerical-quadrature oracle (catches a correlated NESSie/Rust
  transcription error).
- **Self/diagonal term is explicit.** The InPlane closed form *is* the self term
  (ξ on its own element); the ½ solid-angle jump is the `σ` constant. Test the
  self-element value directly against numerical quadrature of the principal-value
  integral, not only against NESSie.
- **Near-singular non-self interactions.** Two close-but-not-coincident triangles
  (SES re-entrant clefts) are the accuracy risk. Add a near-singular micro-corpus
  (decreasing gap) gated against high-precision quadrature; document the accuracy
  floor where the analytic-projection form degrades.
- **Metamorphic** (catches frame bugs the analytic form can hide): invariance
  under rigid motion + **cyclic** vertex permutation (invariant), while an **odd**
  permutation reverses orientation and must flip the double-layer sign; sign flip
  under normal flip;
  InPlane↔InSpace continuity as the observation point crosses the element plane;
  exercise every degenerate guard (ξ on an edge-line, φ=±π/2, φ₁=φ₂).
- *Independently testable: yes, element-by-element, no solve.*

**L2 — Regular Yukawa collocation (`yukawa.rs`).** Oracles: `yukawa_dump` **and**
numerical quadrature (§3).
- Gate per-element values vs NESSie across the **series/closed-form branch
  boundary** (yukawa·|x−ξ| around 0.1) — the cancellation guard is the fragile
  part; test on both sides and at the limit `|x−ξ|→0` (returns `−yukawa` SL,
  `yukawa²/(2√3)` DL).
- **7-point cubature is not enough for near-singular elements.** Gate the
  near-singular micro-corpus against high-precision adaptive quadrature; where the
  fixed 7-point rule misses tolerance, document it as a known limitation (and a
  later adaptive/subdivision path) rather than letting "regular Yukawa + Laplace ≈
  full Yukawa" mask it — that identity is definitional and shares the error.
- *Independently testable: yes, on collocation values.*

**L3 — System assembly + solve (`system.rs`, `solve.rs`).** Oracle: `solve_dump`.
- **L3a local** (2-block) first, then **L3b nonlocal** (3-block). Gate the Cauchy
  data `u,q,[w]` vs NESSie within tol.
- **Three distinct checks — don't conflate them:**
  1. *Operator parity* — assemble an explicit dense operator and gate
     `implicit·x == explicit·x`. Tests **indexing/matvec only**, not correctness of
     the assembled entries.
  2. *Assembly correctness* — gate proteon's assembled **matrix blocks + RHS
     entrywise** against NESSie's `system_dump`. **A dense LU/QR of proteon's *own*
     matrix does NOT validate assembly** — it inherits any assembly bug; it only
     confirms the iterative solver reaches the direct solution. The entrywise
     matrix comparison (or, stronger, reconstructing entries from the
     numerical-quadrature oracle) is the actual assembly gate.
  3. *Solver correctness* — LU/QR of the explicit matrix vs GMRES on the same
     operator, on small fixtures: confirms GMRES converges to the direct solution
     (this is what the dense factorization is *for*).
- **Gate the true residual.** GMRES `restart`/preconditioner choices change
  *iterations*, not the converged solution — gate the converged Cauchy data and the
  **true unpreconditioned per-block residual** (assert it falls below tol);
  iteration count and condition estimates are **measured**. Add an
  iteration/conditioning sweep over mesh refinement and dielectric contrast `εΣ/εΩ`.
- *Independently testable: yes — dumped Cauchy data + operator parity + independent
  LU.*

**L4 — Post-processing & the scientific claim (`post.rs`, `analytic.rs`).**
Oracle: `post_dump` (NESSie parity) **and** `analytic_dump` (closed form).
- **Headline gated claim:** the *reaction-field potential and energy* of a Born ion
  and of Xie spheres converge to the **closed-form** values as mesh density rises
  (local *and* nonlocal), on **exact analytic sphere meshes** (`analytic_sphere_mesh`,
  not the SES mesher — decouples BEM convergence from SES geometry).
- **Energy alone is not enough.** A single scalar `rfenergy` hides spatially
  compensating potential errors, sign flips, and charge-ordering mistakes. Gate at
  the **potential level** too:
  - reaction potential `φ_rf` at each charge site (interior, away from Γ — a clean
    evaluation, no singular kernel),
  - `φ` at interior/exterior radial sample points,
  - energy assembled two independent ways — NESSie's `rfenergy` path **and**
    `½ Σ qᵢ φ_rf(rᵢ)` from the potentials — gated to agree.
  - Evaluating `φ` *on* Γ must distinguish the **limiting trace** (jump-corrected)
    from a direct on-surface kernel evaluation — pick one and state it; they differ
    by the jump term.
- **Convergence — specify the gate precisely.** Collocation BEM on independently
  generated meshes converges **non-monotonically**, so gate an **observed
  convergence order** with a *stated norm* (e.g. relative L²(Γ) on the Cauchy data,
  abs error on energy), an *expected minimum rate*, and a *defined refinement
  family*. Note: centroid-projected sphere refinements are **connectivity-nested but
  not geometrically nested** — so also validate **geometric refinement** (the
  triangulation approaching the true sphere), not density alone. Richardson
  extrapolation is a **measured** diagnostic, not a gate. Do not require monotone
  descent.
- **Invariants — and what kind each one is (don't over-claim):**
  - zero charge ⟹ zero reaction energy — *independent physical gate*.
  - `εΣ = ε∞` **and** `κ→0` ⟹ the nonlocal model collapses to the local one —
    gate **only after confirming it algebraically from the formulation** (§1b); it
    is model-specific, not self-evident.
  - **dielectric-flux continuity across Γ** — likely *imposed algebraically* by the
    formulation, so it is a **consistency/sanity check**, not independent
    validation. Label it as such.
  - **reciprocity** — needs the *precise discrete weighted inner product* of the
    formulation; a collocation matrix is generally **not** naively symmetric. Gate
    the correct discrete pairing or drop it.
  - **orientation reversal is NOT a physical invariant** — it is an **operator-sign
    unit test**. Production should *normalize or reject* inward-oriented meshes;
    deliberately flipping normals only tests that the sign convention is wired right.
- Parity gate: `rfenergy` / `espotential(:Ω/:Σ/:Γ)` vs NESSie on the `data/born`
  and `data/xie` fixtures **at a fixed identical mesh** within tol (tight parity);
  vs analytic on the sphere ladder (science gate). **APBS** on matched spherical
  geometry is a **loose cross-method benchmark** (it carries its own grid /
  boundary-discretization error and cannot match a triangulated boundary exactly) —
  measured, not a tight gate.
- *Independently testable: yes — closed form + the zero-charge/algebraically-confirmed
  invariants need no oracle.*

This mirrors the EVIDENT "gated claim vs measured stat" split: kernel parity (vs
NESSie *and* numerical quadrature), assembly correctness (entrywise block/RHS vs
NESSie `system_dump`), solver correctness (LU vs GMRES), true residual,
potential-and-energy-converge-to-analytic, and the algebraically-confirmed physical
invariants are **gated**; GMRES iteration/condition sweeps, extrapolation,
wall-clock, vertex/triangle counts, and the APBS cross-method benchmark are
**measured**.

## 5. Phasing (each phase = mergeable PR + its EVIDENT claim)

- **P0** NESSie Julia oracle harness (`tools/oracle/nessie/`) emitting the six
  dumps + pinned versions, **plus** the independent numerical-quadrature kernel
  oracle, external closed-form fixtures, and `analytic_sphere_mesh`. Nothing ported
  yet — stand up the oracles. *(unblocks all)*
- **P0.5** **Formulation & convention spec (§1b)** — block matrices written out,
  unit chain resolved, dimensional-analysis + injected-analytic-potential unit test
  green (no kernel/assembly/solve — see §1b).
  No kernel yet. EVIDENT artifact: electrostatics-convention-spec. *(gates P2+)*
- **P1** `proteon-electrostatics` crate skeleton + `model.rs` + `quadrature.rs`.
  L0 tests. No oracle dep.
- **P2** `laplace.rs` → L1 gated on `collocation_dump` **+ numerical quadrature**
  (incl. self + near-singular corpus). EVIDENT claim: laplace-collocation-parity.
- **P3** `yukawa.rs` → L2 gated on `yukawa_dump` **+ numerical quadrature**.
  EVIDENT claim: regular-yukawa-collocation-parity.
- **P4** `system.rs` + `solve.rs` **local** → L3a. EVIDENT claims:
  local-cauchy-data-parity-vs-nessie + operator-parity + **entrywise-assembly-parity**
  (blocks/RHS vs `system_dump`) + LU-vs-GMRES-solver-parity
  + true-residual.
- **P5** `post.rs` + `analytic.rs` **local** → L4 local on **analytic sphere
  meshes**. EVIDENT headline: **born-rf-potential-and-energy-converge-to-analytic**
  (+ Xie local + invariants + APBS local cross-check). First publishable result.
- **P6** **nonlocal** solve (3-block) → L3b + L4 nonlocal. EVIDENT headline:
  **nonlocal-xie-rf-converges-to-analytic** + nonlocal→local limit invariant — the
  differentiator.
- **P6.5** **Mesh-acceptance + near-singular remediation + scaling gate** (before
  any protein run): mesh-quality preconditions (§6), charge-placement policy,
  true-residual / conditioning behavior on SES-fed meshes, the **mandatory
  near-singular remediation** (adaptive subdivision / corrected quadrature) passing
  a measured-error gate on realistic SES meshes, multiple-component & cavity
  handling, and a documented O(N²) scaling ceiling with enforced triangle/runtime
  limits. **This phase can block indefinitely** if near-singular remediation proves
  hard — that is the gate working, not a bug. *(hard-gates P7)*
- **P7** Python exposure (`py_electrostatics.rs`) + `proteon rfenergy` CLI
  (reads structure → SES mesh → energy), same pattern as the analysis CLI.
  **The API must surface convergence/residual/mesh-quality diagnostics and refuse
  or warn on inputs that fail the P6.5 preconditions** — no silent arbitrary-protein
  runs.
- **P8** (**conditional, not default**) fast-summation (FMM/treecode/H-matrix) +/or
  GPU to break the O(N²) ceiling, *then* — only if a useful bounded regime is
  demonstrated — wire as a solvation **energy component** alongside GB/EEF1. Without
  that evidence the tier stays reference/research-only (per §0 positioning).
  NESSie's `CuNESSie.jl` pre-maps the GPU kernels but lowers the constant, not the
  O(N²).

## 6. Risks / things to watch

- **Singular collocation epsilons** — the correctness story for L1/L2. Port the
  Rjasanow InPlane/InSpace closed forms and the Radon series guard with **identical
  type-specific tolerances** (`_etol`: 1.45e-8 f64, 3.45e-4 f32; `0.1` series
  threshold). Gate both sides of every branch with a dedicated micro-corpus.
- **The ×4π convention** — NESSie's collocation results are premultiplied by 4π.
  Carry the factor identically through assembly or the energies are off by 4π;
  make it explicit in one place and test a scalar against analytic.
- **O(N) memory, O(N²) matvec — the scaling ceiling.** Implicit assembly is O(N)
  *memory*, but each GMRES matvec is O(N²) without a fast-summation method
  (FMM / treecode / hierarchical matrices). State this prominently: matrix-free
  GMRES alone becomes impractical on large proteins. NESSie has the same ceiling and
  punts to CuNESSie for GPU — but GPU lowers the constant, not the O(N²). Real
  protein-scale needs fast summation (P8), and the API must not pretend otherwise.
- **Mesh-quality gates (BEM is notoriously mesh-sensitive).** A density ladder is
  meaningless without quality controls. Gate (or measure) before trusting a result:
  watertightness, manifold edges, consistent outward orientation, signed volume,
  connected components; min angle, aspect ratio, area distribution, zero-area faces;
  self-intersections and near-contact separation; and **charge-to-surface distance
  relative to local element size**. Refining a geometrically biased SES converges to
  the *wrong domain* — the BEM analogue of the SES plan's "closed but geometrically
  wrong". Reuse proteon-core's SES mesh invariants here.
- **Charge-placement policy.** Charges on or very near Γ need a **scale-aware**
  policy, not a fixed absolute epsilon: reject with a tolerance proportional to
  local element size, or implement specialized near-singular evaluation. Decide and
  test at the boundary.
- **Multiple components, cavities, and charge→region assignment.** Proteins give
  disconnected solute components and interior solvent cavities. The formulation must
  define per-component systems and a **topology-aware assignment** of each charge to
  its dielectric region (which solute component, or a buried cavity) — a point-in-
  component test, not just a distance check. A charge mis-assigned to the wrong
  region is a silent correctness bug no parity test on a single sphere will catch.
  Specify in §1b; gate on a two-component / one-cavity fixture.
- **Near-singular integration in re-entrant clefts — the single highest failure
  risk.** SES toric/spheric reentrant regions create close-but-not-coincident
  element pairs; the fixed 7-point Radon rule and the analytic-projection Laplace
  form lose accuracy there, and it couples with system conditioning. **"Document the
  accuracy floor" is not enough**: if realistic SES meshes exceed that floor, the
  solver has *no valid production path*. Therefore a **mandatory remediation** —
  adaptive panel subdivision or a corrected near-singular quadrature rule, with a
  measured-error gate showing realistic meshes stay within tolerance — is a
  **blocking precondition of P6.5/P7**, not a deferred "eventual fix". Until it
  exists and passes, SES-fed protein results do not ship.
- **Mesh dependency / sequencing** — useless until proteon's SES mesher is solid;
  `SES_ROBUSTNESS_CODEX.md` shows that is still stabilizing. Until then, gate kernel
  + solve layers on **analytic sphere meshes** (exact geometry) and NESSie's
  **bundled** `.off` meshes (identical input → tight parity); defer proteon-SES-fed
  runs to after SES + the P6.5 acceptance gate. Don't conflate "BEM bug" with
  "mesh bug" — the analytic-sphere path exists precisely to separate them.
- **GMRES robustness, not just convergence** — gate the **true unpreconditioned
  per-block residual** below tol, plus condition/iteration sweeps over refinement and
  dielectric contrast. Scalar Jacobi may be too weak for refined/nonlocal systems;
  keep the operator interface open to a block-diagonal preconditioner. A
  non-converging system is the real failure (cap iterations, assert residual,
  surface it — never return an unconverged solution silently).
- **Float reproducibility** — NESSie is generic over `Float64`/`Float32`. proteon
  should be `f64` for the solver; preserve op order in the collocation predicates
  (same discipline as the TM-align port). No `-ffast-math` either side → expect
  *tighter* than the 4–5 dp NESSie quotes vs C++.
- **`:blas` memory** — the explicit path is O(N²) memory; keep it for small-system
  parity tests only, exactly as NESSie warns. Production path is implicit + GMRES.
- **Bessel functions** — the Xie nonlocal analytic model needs `besseli`/`besselk`.
  Validate the chosen Rust impl against NESSie's `SpecialFunctions` outputs before
  trusting it as "ground truth" (an oracle for the oracle).
- **Scope creep** — do **not** port NESSie's FEM/volume-element side, the Gmsh
  meshing, or the `hmo/msms/stl/vtk/obj/xml3d` format zoo. Surface BEM core only,
  fed by proteon's mesh. Local first; nonlocal is the only must-have extension.
- **Physical-input domain** — define accepted inputs (probe radius for the SES,
  dielectric constants, λ units = Å) and behavior on degeneracies (zero-area
  triangles, charges on the surface, non-watertight mesh). Decide explicitly
  whether parity-with-NESSie includes its behavior on pathological input.

## 7. Open questions

**Resolved (claudex round, 2026-06-09):**

- **Q1 — Mesh provenance for the science gate → RESOLVED.** Use the in-tree
  `analytic_sphere_mesh` (vertices on the exact sphere) for the Born/Xie convergence
  claim, decoupling BEM convergence from SES geometry. A *second*, explicitly
  coupled SES+BEM claim runs later, after SES + the P6.5 acceptance gate.
- **Q2 — GMRES → DECIDED: roll our own** (op-order control + zero dep), but budgeted
  as non-trivial (MGS, restart, breakdown, residual replacement; §2). Vetted dense
  LU/QR is pulled in **test-only** as the **solver** oracle (LU-vs-GMRES); assembly
  is gated entrywise vs NESSie `system_dump`.
- **Q3 — Charge source → SETTLED:** PQR for NESSie-parity fixtures, force-field
  partial charges for production. Both paths supported; PQR reader is test-scoped.
- **Q4 — Energy-model placement → PARTLY SETTLED:** ship standalone in
  `proteon-electrostatics` first (P5/P6); force-field integration as a solvation
  component (P8) is the only piece left open.
- **Q5 — Oracle fixture size → SETTLED:** small checked-in corpus (Born ions, Xie
  spheres, one small protein); dense convergence runs regenerate-on-demand, not
  checked in.

**Still genuinely open:**

1. **Nonlocal parameter defaults.** Pin `ε∞` and `λ` (NESSie default λ = 20 Å) so
   the nonlocal claim is reproducible and comparable to NESSie's defaults.
2. **Fast-summation path.** Which method (FMM / treecode / H-matrix) and when —
   needed for protein-scale (§6 ceiling). Defer the choice to P8, but it determines
   whether protein-scale BEM is ever competitive with the existing GB tier.
3. **APBS cross-check feasibility.** Is a matched-geometry APBS run cheap to wire as
   an offline fixture generator, or does the local-Poisson invariant set (Gauss /
   reciprocity / dielectric-equality) suffice as the independent local gate?

## MSRV / deps

Rust workspace MSRV (1.75+). Production deps kept minimal: a Bessel/special-functions
crate (Xie analytic only); the matrix-free GMRES is in-crate (rolled, not a dep —
but budgeted as real solver code, not 150 lines). A dense LU/QR crate
(`nalgebra`/`faer`) is a **dev-dependency** for the solver oracle (LU-vs-GMRES), not
on the production path; assembly correctness is gated entrywise vs NESSie
`system_dump`. `proteon-electrostatics` is pyo3-free; Python/GPU exposure
follows the established connector pattern. Oracle harness is an offline Julia project
(`NESSie@1.5`), not a proteon build/runtime dependency; the numerical-quadrature and
external closed-form fixtures it generates are checked in.

---

*Three independent Codex (gpt-5.5) reviews, 2026-06-09 —
`TO_ELECTROSTATICS.codex-review.md` (v1), `…-v2.md` (v2), `…-v3.md` (v3 final pass).*
**v3 verdict: GO for P0–P3** — kernel-parity gates technically sound, no
architectural replanning needed; the wording/spec defects it flagged are fixed
(point-collocation vs Galerkin in §1b; cyclic-vs-odd vertex permutation; LU =
solver oracle not assembly; six dumps; injected-potential P0.5 wording).*

*v1 → v2: §1b convention spec, potential-level + invariant gates, analytic-sphere
meshes, numerical-quadrature third kernel oracle, mesh-quality + charge-placement
gates, O(N²) scaling ceiling, GMRES robustness, P0.5/P6.5 phases, effort split.*

*v2 → v3: fixed the dense-LU "assembly oracle" error (now entrywise block/RHS parity
vs NESSie `system_dump`; LU validates only the solver); resolved the P0.5
self-contradiction (unit-chain test via injected potential at P0.5, Born-ion BEM
test moved to P5); added the §1b acceptance checklist + hand-computable single-entry
fixture; corrected ungateable L4 gates (orientation-reversal = sign test not
invariant; flux-continuity = consistency check; convergence needs norm+rate+family +
geometric refinement; reciprocity needs the discrete inner product; APBS = loose
benchmark); concrete singular-integration design (Duffy/subdivision/jump-separation);
near-singular remediation made a **mandatory** P6.5 gate; multiple-component/cavity +
charge→region assignment; three-tier effort split; and repositioned the whole tier as
research/reference (not a routine GB replacement).*
