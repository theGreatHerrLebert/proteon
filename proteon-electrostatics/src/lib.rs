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
//! phasing, and per-layer EVIDENT gates live in `devdocs/TO_ELECTROSTATICS.md` at the repo
//! root. Ported from [NESSie.jl](https://github.com/tkemmer/NESSie.jl) (MIT,
//! Thomas Kemmer) and gated against it — and against NESSie's own closed-form
//! Born/Xie models — exactly as proteon ports BALL/MMseqs2.
//!
//! Positioning (decided): **not** a drop-in GB replacement for routine workloads.
//! The matvec is O(N²) without fast summation; ship with explicit size/runtime
//! limits. See `devdocs/TO_ELECTROSTATICS.md` §0 / §6.
//!
//! # Pipeline (the layers, bottom-up)
//! - [`model`]      — `BemModel`, `Charge`, `Params`, `Domain`, Yukawa exponent.
//! - [`quadrature`] — Radon 7-point triangle cubature (L0) ✅ P3.
//! - [`laplace`]    — Rjasanow analytic single/double-layer Laplace collocation (L1) ✅ P2.
//! - [`yukawa`]     — Radon regular-Yukawa (Yukawa − Laplace) collocation (L2) ✅ P3.
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
//! - **P3** ✅ — [`quadrature`] (Radon 7-point, L0) + [`yukawa`]: gated vs NESSie
//!   `yukawa_dump`, an independent single-layer quadrature oracle, the `r → 0`
//!   limits, and series/closed-form continuity.
//! - **P4 (local)** ✅ — [`system`] (collocation matrices + the implicit local
//!   operator) + [`solve`] (matrix-free GMRES, two-stage local solve): Cauchy data
//!   gated vs NESSie `solve_dump`, LU-vs-GMRES, entrywise assembly, true residual.
//! - **P5 (local)** ✅ — [`post`] (`rfenergy` + `espotential`) + [`analytic`] (Born
//!   closed form): energy + potentials gated vs NESSie `post_dump`, and the BEM
//!   energy converges to the analytic Born energy on refined sphere meshes — the
//!   science gate, independent of any BEM path.
//! - **P6 (nonlocal)** ✅ — the coupled 3-block `(u,q,w)` solve
//!   ([`system::NonlocalOperator`] and [`solve::solve_nonlocal`]) plus nonlocal
//!   post-processing: Cauchy data, energy, and potentials gated vs NESSie, the
//!   nonlocal→local Born limit, and BEM-vs-Born to a few % (Radon-floor-limited;
//!   tight nonlocal convergence is the P6.5 follow-up). The genuine differentiator —
//!   APBS/DelPhi do local PB only.
//! - **Follow-ups** — P6.5 near-singular remediation (adaptive quadrature) for tight
//!   nonlocal convergence + protein-scale meshes; P7 Python/CLI exposure; the Xie
//!   multi-charge analytic model. The convention spec still gates anything energy-level.

// Scaffolding: the stubs below intentionally take parameters they do not yet use.
// Remove this allow as each phase (P2 = laplace, P3 = yukawa, P4 = system/solve,
// P5 = post/analytic) lands and the bodies are implemented.
#![allow(unused_variables, dead_code, clippy::needless_pass_by_value)]

pub mod adaptive;
pub mod analytic;
pub mod fastsum;
pub mod format;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod laplace;
pub mod model;
pub mod post;
pub mod quadrature;
pub mod quality;
pub mod solve;
pub mod surface;
pub mod system;
pub mod xie;
pub mod yukawa;

pub use adaptive::{adaptive_regular_yukawa_collocation, duffy_self_single, AdaptiveConfig};
pub use analytic::{
    analytic_sphere_mesh, born_rfenergy, concentric_kirkwood_rfenergy, concentric_shell_rfenergy,
    kirkwood_rfenergy,
};
pub use format::{read_hmo, read_msms, read_off, read_pqr, write_off};
#[cfg(feature = "cuda")]
pub use gpu::{solve_local_gpu, solve_nonlocal_gpu};
pub use laplace::laplace_collocation;
pub use model::{BemModel, Charge, Domain, Locality, Params, PotentialKind, Tri};
pub use post::{espotential, rfenergy, ENERGY_FACTOR, POTPREFACTOR};
pub use quadrature::{radon7, TriangleQuad};
pub use quality::{QualityIssue, QualityReport, Severity, TopologyReport};
pub use solve::{
    dense_matrix_bytes, solve_local, solve_local_elements, solve_local_elements_auto,
    solve_local_elements_treecode, solve_nonlocal, solve_nonlocal_elements,
    solve_nonlocal_elements_auto, solve_nonlocal_elements_q, solve_nonlocal_elements_treecode,
    CauchyData, LocalResult, NonlocalResult, SolveConfig, SolveError, SolveStats,
    DENSE_MATRIX_BUDGET,
};
pub use surface::{
    solve_surface, FastSummation, SurfaceSolution, SurfaceSolveError, SurfaceSolveOptions,
    MEM_BUDGET, N_WARN,
};
pub use system::{
    laplace_matrices, mol_potentials, yukawa_matrices, yukawa_matrices_q, BlockLayout,
    DenseOperator, JacobiPreconditioner, LinearOperator, LocalOperator, NonlocalOperator,
    Preconditioner, Quadrature, TWO_PI,
};
pub use xie::{scalemodel, XieKind, XieModel};
pub use yukawa::regular_yukawa_collocation;
