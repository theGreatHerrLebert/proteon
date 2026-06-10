//! proteon-electrostatics — boundary-element (BEM) continuum protein electrostatics.
//!
//! A **high-accuracy reference / research tier** of solvation electrostatics,
//! sibling to proteon's fast approximate tier (OBC GB / EEF1 in `proteon-core`).
//! It solves the Poisson (local) or nonlocal-Poisson (Lorentz cavity / Yukawa)
//! problem on a triangulated molecular surface and reads off reaction-field
//! energies and potentials. **No Python / PyO3 dependency** (exposure is added
//! later via `proteon-connector`).
//!
//! **Under construction.** The lower layers are landing phase by phase; the plan,
//! phasing, and per-layer EVIDENT gates live in `TO_ELECTROSTATICS.md` at the repo
//! root. Ported from [NESSie.jl](https://github.com/tkemmer/NESSie.jl) (MIT,
//! Thomas Kemmer) and gated against it — and against NESSie's own closed-form
//! Born/Xie models — exactly as proteon ports BALL/MMseqs2.
//!
//! Positioning (decided): **not** a drop-in GB replacement for routine workloads.
//! The matvec is O(N²) without fast summation; ship with explicit size/runtime
//! limits. See `TO_ELECTROSTATICS.md` §0 / §6.
//!
//! # Pipeline (the layers, bottom-up)
//! - [`model`]      — `BemModel`, `Charge`, `Params`, `Domain`, Yukawa exponent.
//! - [`quadrature`] — Radon 7-point triangle cubature (L0).
//! - [`laplace`]    — Rjasanow analytic single/double-layer Laplace collocation (L1) ✅ P2.
//! - [`yukawa`]     — Radon regular-Yukawa (Yukawa − Laplace) collocation (L2).
//! - [`system`]     — implicit (matrix-free) 2-block / 3-block operators (L3).
//! - [`solve`]      — rolled matrix-free GMRES + preconditioner; result types (L3).
//! - [`post`]       — reaction-field energy and potentials (L4).
//! - [`analytic`]   — Born / Xie closed forms + analytic sphere mesh (L4 ground truth).
//!
//! # Status
//! - **P0 / P0.5** ✅ — NESSie oracle fixtures (`tests/fixtures/nessie/`) + the
//!   formulation/convention spec (`devdocs/ELECTROSTATICS_FORMULATION.md`, plan §1b).
//! - **P2** ✅ — [`laplace`]: gated vs NESSie `collocation_dump`, an independent
//!   numerical-quadrature oracle, and metamorphic invariants.
//! - **P3+** — [`yukawa`], [`system`], [`solve`], [`post`], [`analytic`] remain
//!   `unimplemented!()` stubs keyed to their gating phase. The convention spec still
//!   gates anything energy-level: a single convention bug hides behind close parity.

// Scaffolding: the stubs below intentionally take parameters they do not yet use.
// Remove this allow as each phase (P2 = laplace, P3 = yukawa, P4 = system/solve,
// P5 = post/analytic) lands and the bodies are implemented.
#![allow(unused_variables, dead_code, clippy::needless_pass_by_value)]

pub mod analytic;
pub mod laplace;
pub mod model;
pub mod post;
pub mod quadrature;
pub mod solve;
pub mod system;
pub mod yukawa;

pub use laplace::{laplace_collocation, Tri};
pub use model::{BemModel, Charge, Domain, Locality, Params, PotentialKind};
pub use post::{espotential, rfenergy};
pub use solve::{
    solve_local, solve_nonlocal, CauchyData, LocalResult, NonlocalResult, SolveConfig, SolveStats,
};
pub use system::{BlockLayout, LinearOperator, Preconditioner};
