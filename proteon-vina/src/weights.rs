// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Constants from AutoDock-Vina src/lib/vina.h / vina.cpp (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! Published Vina scoring weights (2021 refit; identical to the
//! defaults in upstream `Vina::set_vina_weights` in `vina.h:123`).
//!
//! The five pair-potential weights multiply the corresponding terms in
//! `potentials.rs`. `W_ROT` is the published per-rotatable-bond
//! penalty; it is applied via the `num_tors_div` conf-independent
//! transform (`Ninter / (1 + W_ROT_INTERNAL * N_rot)`) during total-energy
//! assembly, not directly in the pair-potential sum.
//!
//! `W_GLUE` is the weight on the macrocycle linear-attraction glue term
//! and is not exercised by v0 (rigid, non-macrocyclic scoring).

/// Weight on `gaussian(offset=0, width=0.5)`. Attractive (negative).
pub const W_GAUSS1: f64 = -0.035579;

/// Weight on `gaussian(offset=3.0, width=2.0)`. Attractive (negative).
pub const W_GAUSS2: f64 = -0.005156;

/// Weight on the quadratic repulsion term (positive; penalizes clash).
pub const W_REPULSION: f64 = 0.840245;

/// Weight on the hydrophobic contact term. Attractive (negative).
pub const W_HYDROPHOBIC: f64 = -0.035069;

/// Weight on the non-directional H-bond term. Attractive (negative).
pub const W_HYDROGEN: f64 = -0.587439;

/// Weight on the macrocycle-closure linear-attraction glue term.
/// Out of scope for v0; kept here for completeness.
pub const W_GLUE: f64 = 50.0;

/// Published per-rotatable-bond penalty (the value the user-facing
/// `weight_rot` takes). Applied via `num_tors_div`; see
/// `w_rot_internal` below.
pub const W_ROT: f64 = 0.05846;

/// Internal conf-independent weight derived from `W_ROT`.
/// Upstream (`vina.cpp:230`) stores `5 * weight_rot / 0.1 - 1`.
#[must_use]
pub const fn w_rot_internal() -> f64 {
    5.0 * W_ROT / 0.1 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_match_published_defaults() {
        // Exact-equality checks — these are constants reproduced from
        // the upstream header (`vina.h:123`), not measurements.
        assert_eq!(W_GAUSS1, -0.035579);
        assert_eq!(W_GAUSS2, -0.005156);
        assert_eq!(W_REPULSION, 0.840245);
        assert_eq!(W_HYDROPHOBIC, -0.035069);
        assert_eq!(W_HYDROGEN, -0.587439);
        assert_eq!(W_GLUE, 50.0);
        assert_eq!(W_ROT, 0.05846);
    }

    #[test]
    fn w_rot_internal_matches_cpp_transform() {
        // 5 * 0.05846 / 0.1 - 1 = 2.923 - 1 = 1.923.
        assert!((w_rot_internal() - 1.923).abs() < 1e-12);
    }
}
