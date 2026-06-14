//! P8 fast summation — breaking the O(N²) BEM matvec ceiling (plan
//! `devdocs/TO_ELECTROSTATICS_P8.md`).
//!
//! **Status: P8.1–P8.4 complete and wired.** The treecode operators
//! ([`operator::CollocationTreecode`] for Laplace `V`/`K`, [`operator::YukawaTreecode`]
//! for regular-Yukawa `Vy`/`Ky`) drive both the local and nonlocal surface solve
//! (`surface::solve_surface`), opt-in via `SurfaceSolveOptions::fast_summation` and
//! auto-enabled as the over-budget fallback. The M2M upward pass is in use; the M2L/L2P
//! FMM building blocks are gated bit-true but the full downward pass (L2L + interaction
//! lists + accelerated M2L) is deferred — the v1 treecode is an **O(N)-memory** unlock,
//! not a speed win (plan §5).
//!
//! The dense `K`/`V` matvecs are O(N²); the treecode approximates the *far field* in
//! O(N log N) while the near field keeps the exact Rjasanow analytic collocation
//! (`laplace::laplace_collocation`), so the reference-tier accuracy is preserved.
//!
//! The key correctness point (codex review): the far field is **not** a centroid
//! collapse (`∫_T G dS ≈ A·G(ξ,c)`, whose `O(h²/r³)` error a higher expansion order
//! cannot reduce). It uses **panel-aware moments** — the cluster expansion basis is
//! integrated over each panel exactly, so accuracy is genuinely controlled by the
//! expansion order `p` and the admissibility ratio `θ`. See [`cheb`] (Chebyshev
//! barycentric interpolation) and [`cubature`] (panel integrals).
//!
//! This module is built bottom-up and gated at each layer:
//! - [`cheb`]      — 1-D Chebyshev (2nd-kind) nodes, barycentric Lagrange basis.
//! - [`cubature`]  — runtime Gauss–Legendre + triangle cubature for panel integrals.
//! - [`expansion`] — panel-aware BLTC cluster expansions (scalar + dipole moments).
//! - [`cartesian`] — panel-aware Cartesian multipole + Coulomb Taylor recurrence;
//!   M2M upward pass; M2L/L2P FMM building blocks (gated, downward pass deferred).
//! - [`octree`]    — vertex-enclosing octree (admissibility sees true panel extent).
//! - [`operator`]  — `CollocationTreecode` / `YukawaTreecode: LinearOperator`.

pub mod cartesian;
pub mod cheb;
pub mod cubature;
pub mod expansion;
pub mod octree;
pub mod operator;
