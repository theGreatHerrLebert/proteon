//! Radon 7-point triangle cubature (L0).
//!
//! Mirrors NESSie's `TriangleQuad` / `quadraturepoints` (`src/base/quadrature.jl`).
//! The 7-point Radon rule integrates the regular Yukawa kernel over each triangle
//! (`yukawa.rs`). **Port the points/weights from NESSie's source — do not invent
//! them** (faithful-port discipline).
//!
//! L0 gate (no oracle needed): weights sum to 1, low-order polynomials integrate
//! exactly over a reference triangle, points lie inside (barycentric ≥ 0).

use proteon_core::surface::geom::Vec3;

/// Precomputed cubature for one triangle: world-space points + weights, plus the
/// element's normal and area (NESSie caches these to avoid recomputation).
#[derive(Debug, Clone)]
pub struct TriangleQuad {
    /// Cubature points in world space (7 for the Radon rule).
    pub points: Vec<Vec3>,
    /// Cubature weights (sum to 1).
    pub weights: Vec<f64>,
    /// Unit outward normal of the triangle.
    pub normal: Vec3,
    /// Triangle area.
    pub area: f64,
}

/// Build the 7-point Radon cubature for a triangle `(v1, v2, v3)`.
///
/// TODO(P1/P3): port the barycentric points + weights from NESSie
/// `base/quadrature.jl` (`QuadPts2D` / the Radon 7-point set) and map to world
/// space. Cache the normal + area.
#[must_use]
pub fn radon7(v1: Vec3, v2: Vec3, v3: Vec3) -> TriangleQuad {
    unimplemented!("P1/P3: port Radon 7-point rule from NESSie base/quadrature.jl")
}
