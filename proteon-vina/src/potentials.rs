// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/potentials.h (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! Vina scoring-function terms.
//!
//! Each term is a pure function of the pair of XS atom types and the
//! inter-atomic distance `r` (Å). Below the per-term cutoff the term
//! is zero. The published Vina scoring function is a weighted sum of
//! `gaussian1`, `gaussian2`, `repulsion`, `hydrophobic`, and
//! `non_dir_h_bond` (see `weights.rs`).
//!
//! Vinardo and AD4.2 variants are intentionally out of scope for v0.

use crate::atom_types::{self, XsType};

/// Cutoff distance shared by all v0 Vina terms (Å). Matches the upstream
/// `ScoringFunction::SF_VINA` branch in `scoring_function.h`.
pub const VINA_CUTOFF: f64 = 8.0;

/// Sum of Vina vdW radii for a pair of XS types. Glue atoms are treated
/// as having optimal distance 0 (matches `optimal_distance` in upstream
/// `potentials.h`).
#[inline]
#[must_use]
pub fn optimal_distance(t1: XsType, t2: XsType) -> f64 {
    if atom_types::is_glue(t1) || atom_types::is_glue(t2) {
        0.0
    } else {
        atom_types::radius(t1) + atom_types::radius(t2)
    }
}

/// Piecewise-linear step from 0 at `x_bad` to 1 at `x_good`.
/// Mirrors `slope_step` in upstream `potentials.h`. Works for both
/// `x_bad < x_good` and `x_bad > x_good` orderings.
#[inline]
#[must_use]
pub fn slope_step(x_bad: f64, x_good: f64, x: f64) -> f64 {
    if x_bad < x_good {
        if x <= x_bad {
            return 0.0;
        }
        if x >= x_good {
            return 1.0;
        }
    } else {
        if x >= x_bad {
            return 0.0;
        }
        if x <= x_good {
            return 1.0;
        }
    }
    (x - x_bad) / (x_good - x_bad)
}

/// Vina gaussian term: `exp(-((r - (r_opt + offset)) / width)^2)`, zero
/// beyond `cutoff`. Used twice with different `(offset, width)` pairs
/// to form the two Vina attractive gaussians.
#[inline]
#[must_use]
pub fn gaussian(t1: XsType, t2: XsType, r: f64, offset: f64, width: f64, cutoff: f64) -> f64 {
    if r >= cutoff {
        return 0.0;
    }
    let d = r - (optimal_distance(t1, t2) + offset);
    let z = d / width;
    (-z * z).exp()
}

/// Vina repulsion term: `(r - r_opt - offset)^2` while below the
/// optimum (i.e. the atoms are too close), else 0. Zero beyond
/// `cutoff`.
#[inline]
#[must_use]
pub fn repulsion(t1: XsType, t2: XsType, r: f64, offset: f64, cutoff: f64) -> f64 {
    if r >= cutoff {
        return 0.0;
    }
    let d = r - (optimal_distance(t1, t2) + offset);
    if d > 0.0 {
        0.0
    } else {
        d * d
    }
}

/// Vina hydrophobic term. For pairs that are both hydrophobic,
/// piecewise-linear 1 → 0 as `r - r_opt` rises from `good` to `bad`.
/// Zero for any non-hydrophobic pair and beyond `cutoff`.
#[inline]
#[must_use]
pub fn hydrophobic(t1: XsType, t2: XsType, r: f64, good: f64, bad: f64, cutoff: f64) -> f64 {
    if r >= cutoff {
        return 0.0;
    }
    if atom_types::is_hydrophobic(t1) && atom_types::is_hydrophobic(t2) {
        slope_step(bad, good, r - optimal_distance(t1, t2))
    } else {
        0.0
    }
}

/// Vina non-directional H-bond term. For donor/acceptor pairs,
/// piecewise-linear 1 → 0 as `r - r_opt` rises from `good` (typically
/// negative, i.e. slightly compressed) to `bad`. Zero for any
/// non-H-bonding pair and beyond `cutoff`.
#[inline]
#[must_use]
pub fn non_dir_h_bond(
    t1: XsType,
    t2: XsType,
    r: f64,
    good: f64,
    bad: f64,
    cutoff: f64,
) -> f64 {
    if r >= cutoff {
        return 0.0;
    }
    if atom_types::h_bond_possible(t1, t2) {
        slope_step(bad, good, r - optimal_distance(t1, t2))
    } else {
        0.0
    }
}

/// Weighted sum of the five Vina pair terms with published parameters.
/// This is the pair contribution `ScoringFunction::SF_VINA::eval(t1, t2, r)`
/// from upstream — i.e. the kernel that `precalculate` samples to build
/// its tables. The macrocycle glue `linearattraction` term is out of
/// scope for v0 and not included.
#[inline]
#[must_use]
pub fn vina_pair_energy(t1: XsType, t2: XsType, r: f64) -> f64 {
    use crate::weights::{W_GAUSS1, W_GAUSS2, W_HYDROGEN, W_HYDROPHOBIC, W_REPULSION};
    W_GAUSS1 * gaussian(t1, t2, r, 0.0, 0.5, VINA_CUTOFF)
        + W_GAUSS2 * gaussian(t1, t2, r, 3.0, 2.0, VINA_CUTOFF)
        + W_REPULSION * repulsion(t1, t2, r, 0.0, VINA_CUTOFF)
        + W_HYDROPHOBIC * hydrophobic(t1, t2, r, 0.5, 1.5, VINA_CUTOFF)
        + W_HYDROGEN * non_dir_h_bond(t1, t2, r, -0.7, 0.0, VINA_CUTOFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPS: f64 = 1e-12;

    // --- optimal_distance --------------------------------------------------

    #[test]
    fn optimal_distance_sums_radii() {
        // C_H (1.9) + O_A (1.7) = 3.6
        assert_relative_eq!(
            optimal_distance(XsType::CH, XsType::OA),
            3.6,
            epsilon = EPS
        );
    }

    #[test]
    fn optimal_distance_glue_collapses_to_zero() {
        assert_eq!(optimal_distance(XsType::G0, XsType::CH), 0.0);
        assert_eq!(optimal_distance(XsType::CH, XsType::G0), 0.0);
        assert_eq!(optimal_distance(XsType::G0, XsType::G0), 0.0);
    }

    // --- slope_step --------------------------------------------------------

    #[test]
    fn slope_step_ascending() {
        // x_bad=1.5, x_good=0.5: as x drops from 1.5 to 0.5, value rises 0→1.
        assert_eq!(slope_step(1.5, 0.5, 2.0), 0.0);
        assert_eq!(slope_step(1.5, 0.5, 1.5), 0.0);
        assert_eq!(slope_step(1.5, 0.5, 0.5), 1.0);
        assert_eq!(slope_step(1.5, 0.5, -0.3), 1.0);
        assert_relative_eq!(slope_step(1.5, 0.5, 1.0), 0.5, epsilon = EPS);
    }

    #[test]
    fn slope_step_descending() {
        // x_bad=-0.7, x_good=0.0: x rising from -0.7 to 0.0 moves 0→1.
        assert_eq!(slope_step(-0.7, 0.0, -1.0), 0.0);
        assert_eq!(slope_step(-0.7, 0.0, -0.7), 0.0);
        assert_eq!(slope_step(-0.7, 0.0, 0.0), 1.0);
        assert_eq!(slope_step(-0.7, 0.0, 0.5), 1.0);
        assert_relative_eq!(slope_step(-0.7, 0.0, -0.35), 0.5, epsilon = EPS);
    }

    // --- gaussian ----------------------------------------------------------

    #[test]
    fn gaussian_is_one_at_optimum() {
        // gauss1: offset=0, width=0.5. r = r_opt → d=0 → exp(0)=1.
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        assert_relative_eq!(
            gaussian(XsType::CH, XsType::CH, r_opt, 0.0, 0.5, VINA_CUTOFF),
            1.0,
            epsilon = EPS
        );
    }

    #[test]
    fn gaussian_offset_shifts_peak() {
        // gauss2: offset=3, width=2. Peak at r = r_opt + 3.
        let r_opt = optimal_distance(XsType::CH, XsType::OA);
        assert_relative_eq!(
            gaussian(XsType::CH, XsType::OA, r_opt + 3.0, 3.0, 2.0, VINA_CUTOFF),
            1.0,
            epsilon = EPS
        );
        // At r = r_opt, d = -3, z = -1.5, exp(-2.25)
        let expected = (-2.25_f64).exp();
        assert_relative_eq!(
            gaussian(XsType::CH, XsType::OA, r_opt, 3.0, 2.0, VINA_CUTOFF),
            expected,
            epsilon = EPS
        );
    }

    #[test]
    fn gaussian_zero_beyond_cutoff() {
        assert_eq!(
            gaussian(XsType::CH, XsType::CH, VINA_CUTOFF, 0.0, 0.5, VINA_CUTOFF),
            0.0
        );
        assert_eq!(
            gaussian(XsType::CH, XsType::CH, 9.0, 0.0, 0.5, VINA_CUTOFF),
            0.0
        );
    }

    // --- repulsion ---------------------------------------------------------

    #[test]
    fn repulsion_is_zero_at_and_beyond_optimum() {
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        assert_eq!(repulsion(XsType::CH, XsType::CH, r_opt, 0.0, VINA_CUTOFF), 0.0);
        assert_eq!(
            repulsion(XsType::CH, XsType::CH, r_opt + 1.0, 0.0, VINA_CUTOFF),
            0.0
        );
    }

    #[test]
    fn repulsion_quadratic_below_optimum() {
        // r_opt = 3.8 for CH-CH. At r=3.3, d=-0.5, expected 0.25.
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        assert_relative_eq!(
            repulsion(XsType::CH, XsType::CH, r_opt - 0.5, 0.0, VINA_CUTOFF),
            0.25,
            epsilon = EPS
        );
    }

    // --- hydrophobic -------------------------------------------------------

    #[test]
    fn hydrophobic_only_hydrophobic_pairs() {
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        // Both hydrophobic: non-zero in the ramp zone.
        assert!(hydrophobic(XsType::CH, XsType::CH, r_opt, 0.5, 1.5, VINA_CUTOFF) > 0.0);
        // Mixed: zero.
        assert_eq!(
            hydrophobic(XsType::CH, XsType::OA, r_opt, 0.5, 1.5, VINA_CUTOFF),
            0.0
        );
    }

    #[test]
    fn hydrophobic_ramp_values() {
        // At r = r_opt: r - r_opt = 0, which sits below good=0.5 in an
        // "x_bad=1.5, x_good=0.5" slope. Step returns 1.0 (ideal contact).
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        assert_eq!(
            hydrophobic(XsType::CH, XsType::CH, r_opt, 0.5, 1.5, VINA_CUTOFF),
            1.0
        );
        // At r - r_opt = 1.0 (midway through the ramp): 0.5.
        assert_relative_eq!(
            hydrophobic(XsType::CH, XsType::CH, r_opt + 1.0, 0.5, 1.5, VINA_CUTOFF),
            0.5,
            epsilon = EPS
        );
        // At r - r_opt = 1.5 (past "bad"): 0.
        assert_eq!(
            hydrophobic(XsType::CH, XsType::CH, r_opt + 1.5, 0.5, 1.5, VINA_CUTOFF),
            0.0
        );
    }

    // --- non_dir_h_bond ----------------------------------------------------

    #[test]
    fn h_bond_only_for_donor_acceptor() {
        let r_opt = optimal_distance(XsType::ND, XsType::OA);
        assert!(non_dir_h_bond(XsType::ND, XsType::OA, r_opt - 0.5, -0.7, 0.0, VINA_CUTOFF) > 0.0);
        // Both acceptors: zero.
        assert_eq!(
            non_dir_h_bond(XsType::OA, XsType::OA, r_opt - 0.5, -0.7, 0.0, VINA_CUTOFF),
            0.0
        );
    }

    #[test]
    fn h_bond_ramp_values() {
        // slope_step(bad=0, good=-0.7, x = r - r_opt)
        // At r - r_opt = -0.7 (or less): 1 (tight H-bond).
        let r_opt = optimal_distance(XsType::ND, XsType::OA);
        assert_eq!(
            non_dir_h_bond(XsType::ND, XsType::OA, r_opt - 0.7, -0.7, 0.0, VINA_CUTOFF),
            1.0
        );
        // At r - r_opt = -0.35 (midway): 0.5.
        assert_relative_eq!(
            non_dir_h_bond(XsType::ND, XsType::OA, r_opt - 0.35, -0.7, 0.0, VINA_CUTOFF),
            0.5,
            epsilon = EPS
        );
        // At r - r_opt = 0 (or more): 0 (broken).
        assert_eq!(
            non_dir_h_bond(XsType::ND, XsType::OA, r_opt, -0.7, 0.0, VINA_CUTOFF),
            0.0
        );
    }

    // --- vina_pair_energy --------------------------------------------------

    #[test]
    fn vina_pair_energy_symmetric() {
        let r = 3.5;
        assert_eq!(
            vina_pair_energy(XsType::CH, XsType::OA, r),
            vina_pair_energy(XsType::OA, XsType::CH, r),
        );
    }

    #[test]
    fn vina_pair_energy_zero_beyond_cutoff() {
        assert_eq!(vina_pair_energy(XsType::CH, XsType::CH, VINA_CUTOFF), 0.0);
        assert_eq!(vina_pair_energy(XsType::CH, XsType::CH, 100.0), 0.0);
    }

    #[test]
    fn vina_pair_energy_hydrophobic_contact_is_negative() {
        // CH-CH at optimum: gauss1 ≈ 1, gauss2 small, repulsion 0,
        // hydrophobic = 1, hbond = 0. Sum is dominated by the two
        // attractive gaussians + hydrophobic term.
        let r = optimal_distance(XsType::CH, XsType::CH);
        let e = vina_pair_energy(XsType::CH, XsType::CH, r);
        assert!(e < 0.0, "expected attractive pair energy, got {e}");
    }

    #[test]
    fn vina_pair_energy_clash_is_repulsive() {
        // Well below optimum: repulsion (positive weight × positive d²)
        // dominates. Note gauss1 is also positive here because d is in
        // the attractive gaussian's tail.
        let r_opt = optimal_distance(XsType::CH, XsType::CH);
        let e_clash = vina_pair_energy(XsType::CH, XsType::CH, r_opt - 1.0);
        let e_opt = vina_pair_energy(XsType::CH, XsType::CH, r_opt);
        assert!(
            e_clash > e_opt,
            "clash ({e_clash}) should be less favourable than optimum ({e_opt})"
        );
    }

    #[test]
    fn all_terms_zero_at_cutoff() {
        let cutoff = VINA_CUTOFF;
        assert_eq!(gaussian(XsType::CH, XsType::CH, cutoff, 0.0, 0.5, cutoff), 0.0);
        assert_eq!(gaussian(XsType::CH, XsType::CH, cutoff, 3.0, 2.0, cutoff), 0.0);
        assert_eq!(repulsion(XsType::CH, XsType::CH, cutoff, 0.0, cutoff), 0.0);
        assert_eq!(
            hydrophobic(XsType::CH, XsType::CH, cutoff, 0.5, 1.5, cutoff),
            0.0
        );
        assert_eq!(
            non_dir_h_bond(XsType::ND, XsType::OA, cutoff, -0.7, 0.0, cutoff),
            0.0
        );
    }
}
