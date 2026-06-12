//! P8 fast summation — breaking the O(N²) BEM matvec ceiling (plan
//! `devdocs/TO_ELECTROSTATICS_P8.md`).
//!
//! **Status: P8.1 — isolated summation harness, not yet wired into any operator.**
//! The dense `K`/`V` matvecs are O(N²); a treecode approximates the *far field* in
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
//! - (next) panel-aware single/double-layer cluster expansions (BLTC + Cartesian),
//!   then the octree, then a `TreecodeOperator: LinearOperator`.

pub mod cheb;
pub mod cubature;
pub mod expansion;
