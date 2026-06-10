//! Post-processing: reaction-field energy and potentials (L4).
//!
//! Port of NESSie's `bem/post.jl`. Consumes solved Cauchy data and the model to
//! produce the scientific outputs. Generic over [`CauchyData`] so the same API
//! serves local and nonlocal results (no P6 breaking change).
//!
//! **Energy alone is not enough** (plan §4 L4): a scalar `rfenergy` hides
//! compensating potential errors. Gate the **potential level** too — `φ_rf` at each
//! charge, radial samples, and energy assembled two independent ways (NESSie's
//! `rfenergy` path *and* `½ Σ qᵢ φ_rf(rᵢ)`). On-Γ evaluation must use the
//! jump-corrected **limiting trace**, not a raw on-surface kernel value.
//!
//! # Gates (P5/P6)
//! - vs NESSie `post_dump` at a **fixed identical mesh** (tight parity).
//! - vs `analytic` closed forms on the analytic-sphere ladder (science gate);
//!   convergence by **rate in a stated norm**, not monotonicity; validate
//!   geometric refinement, not density alone.
//!
//! # API note (P5, codex review)
//! These free functions take `model` and `cauchy` **independently**, so a caller can
//! pair Cauchy data with the wrong model (mismatched mesh size/order, charges, or
//! params) and get a silently-wrong energy. When implementing, bind them: have
//! [`crate::solve`] return a value that **owns or borrows the model it was solved
//! on** (e.g. `SolvedBem<'a>`), and take that here — don't keep the two-arg form.

use crate::model::{BemModel, Domain};
use crate::solve::CauchyData;
use proteon_core::surface::geom::Vec3;

/// Reaction-field energy (kJ/mol). NESSie: `rfenergy(bem)`.
///
/// TODO(P5): port `NESSie.rfenergy`; cross-check against `½ Σ qᵢ φ_rf(rᵢ)`.
#[must_use]
pub fn rfenergy(model: &BemModel, cauchy: &dyn CauchyData) -> f64 {
    unimplemented!("P5: port rfenergy (src/bem/post.jl)")
}

/// Electrostatic potential at ξ in the given domain. NESSie: `espotential(:Ω/:Σ/:Γ, …)`.
///
/// TODO(P5): port `NESSie.espotential` per-domain (`:Ω`/`:Σ`/`:Γ`); for `Gamma`
/// return the limiting (jump-corrected) trace.
#[must_use]
pub fn espotential(domain: Domain, xi: Vec3, model: &BemModel, cauchy: &dyn CauchyData) -> f64 {
    unimplemented!("P5: port espotential (src/bem/post.jl)")
}
