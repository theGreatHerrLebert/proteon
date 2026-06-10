//! Rjasanow analytic single/double-layer Laplace collocation (L1).
//!
//! Port of NESSie's `Rjasanow` module (`src/Rjasanow.jl`): the **analytic** Laplace
//! potential of a triangle at an observation point ξ, via projection onto the
//! element plane and the InPlane/InSpace closed forms. The InPlane form **is the
//! self/diagonal term** (ξ on its own element); the ½ solid-angle jump is the `σ`
//! constant.
//!
//! Conventions (sign, ×4π premultiplier, normal orientation) come from the §1b
//! formulation spec — pin them there before trusting these values.
//!
//! # Gates (P2)
//! - per-element value vs NESSie `collocation_dump` **and** the independent
//!   high-precision numerical-quadrature oracle (catches correlated transcription
//!   errors), incl. a self + near-singular micro-corpus.
//! - metamorphic: rigid-motion invariance; cyclic vertex permutation invariant,
//!   **odd** permutation flips the double-layer sign; InPlane↔InSpace continuity;
//!   every degenerate guard (ξ on an edge-line, φ=±π/2, φ₁=φ₂).
//!
//! Tolerances are type-specific in NESSie (`_etol`: 1.45e-8 f64). proteon is f64.

use crate::model::PotentialKind;
use proteon_core::surface::geom::Vec3;

/// Common Laplace tolerance (NESSie `_etol` for f64).
pub const ETOL_F64: f64 = 1.45e-8;

/// Analytic single/double-layer Laplace collocation of a triangle at ξ.
///
/// Result is **premultiplied by 4π** (NESSie convention; see §1b unit chain).
///
/// TODO(P2): port `Rjasanow._laplacepot` (InPlane/InSpace closed forms),
/// `_projectξ!`, and the degenerate-triangle guards.
#[must_use]
pub fn laplace_collocation(kind: PotentialKind, xi: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> f64 {
    unimplemented!("P2: port Rjasanow analytic Laplace collocation (src/Rjasanow.jl)")
}
