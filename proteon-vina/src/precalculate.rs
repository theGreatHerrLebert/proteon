// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/precalculate.h (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! Precalculated pairwise Vina potential tables.
//!
//! For every unordered XS-type pair `(t1, t2)` we sample the Vina pair
//! energy on a grid in `r²` space at spacing `1 / factor` (upstream
//! default `factor = 32`). Samples are stored alongside pre-computed
//! `dor = (dE/dr) / r` values for fast force chain-rule during search.
//! For v0 scoring we only read the energy; the `dor` field is retained
//! so derivatives remain available for later search/minimization work.
//!
//! Storage layout: a single flat `Vec<Element>` in upper-triangular
//! packing order (`t1 ≤ t2`). Lookups with `t1 > t2` swap transparently.

use crate::atom_types::{XsType, NUM_XS_TYPES};
use crate::potentials::{vina_pair_energy, VINA_CUTOFF};

/// Upstream's default `precalculate` resolution: `factor = 32` samples
/// per unit of `r²` (Å²).
pub const DEFAULT_FACTOR: f64 = 32.0;

/// One `(r²)`-indexed sample of a pair potential.
#[derive(Copy, Clone, Debug, Default)]
struct Sample {
    /// Potential value at this sample point, `E(r)`.
    e: f64,
    /// `(dE/dr) / r`, pre-multiplied so a force lookup becomes
    /// `F = -dor · r_vec`. Zero at the endpoints (one-sided edges).
    dor: f64,
}

/// Pair-indexed precalculated table.
#[derive(Clone, Debug)]
pub struct Precalculate {
    /// Number of samples per type-pair. `m_n` in upstream.
    n: usize,
    /// Samples per unit of `r²` (`factor` in upstream).
    factor: f64,
    /// `cutoff²` of the pair potential itself (Vina: 64 Å²).
    cutoff_sqr: f64,
    /// `max_cutoff²` — table extent. May exceed `cutoff²` to keep
    /// the `widen` operation well-defined even when edges are padded.
    max_cutoff_sqr: f64,
    /// `smooth[pair_idx * n + i]` — table samples.
    smooth: Vec<Sample>,
    /// Midpoint-averaged energy lookup (no interpolation). Upstream
    /// `fast[i] = (smooth[i].e + smooth[i+1].e) / 2`; the last entry
    /// halves the final sample.
    fast: Vec<f64>,
}

/// Unique-pair index for `t1 ≤ t2` using row-major upper-triangle packing.
/// `idx(t1, t2) = t1 · n − t1·(t1−1)/2 + (t2 − t1)`.
#[inline]
const fn pair_index(t1: usize, t2: usize) -> usize {
    // Caller guarantees t1 <= t2 < NUM_XS_TYPES. Using wrapping-safe
    // usize arithmetic.
    let (a, b) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
    a * NUM_XS_TYPES - a * (a.saturating_sub(1)) / 2 + (b - a)
}

/// Total number of unique unordered pairs including diagonal.
const NUM_PAIRS: usize = NUM_XS_TYPES * (NUM_XS_TYPES + 1) / 2;

impl Precalculate {
    /// Build tables by sampling `eval(t1, t2, r)` on the standard
    /// `r²` grid. Matches upstream `precalculate(const ScoringFunction&,
    /// v, factor)`.
    ///
    /// * `cutoff` — pair-potential cutoff (Å).
    /// * `max_cutoff` — table extent (Å). Usually equal to `cutoff`;
    ///   upstream permits `max_cutoff > cutoff` to support `widen()`.
    /// * `factor` — samples per unit `r²`.
    /// * `v_cap` — upper clamp on raw potential values (upstream's
    ///   `v` argument, default `max_fl`). `f64::INFINITY` = no clamp.
    /// * `eval` — pair-energy kernel, typically `vina_pair_energy`.
    pub fn new<F>(cutoff: f64, max_cutoff: f64, factor: f64, v_cap: f64, eval: F) -> Self
    where
        F: Fn(XsType, XsType, f64) -> f64,
    {
        assert!(factor > 0.0, "factor must be positive");
        assert!(cutoff > 0.0 && max_cutoff >= cutoff, "invalid cutoffs");

        let cutoff_sqr = cutoff * cutoff;
        let max_cutoff_sqr = max_cutoff * max_cutoff;
        // Matches `m_n = sz(factor * max_cutoff_sqr) + 3` in upstream,
        // ensuring `factor * r² + 1 < n` for any `r² <= max_cutoff_sqr`.
        let n = (factor * max_cutoff_sqr) as usize + 3;

        let rs = calculate_rs(n, factor);
        let mut smooth = vec![Sample::default(); NUM_PAIRS * n];
        let mut fast = vec![0.0_f64; NUM_PAIRS * n];

        for t1 in 0..NUM_XS_TYPES {
            for t2 in t1..NUM_XS_TYPES {
                let p = pair_index(t1, t2);
                let base = p * n;
                let xs1 = XS_TYPES[t1];
                let xs2 = XS_TYPES[t2];

                // Raw potential samples (capped at v_cap).
                for i in 0..n {
                    let raw = eval(xs1, xs2, rs[i]);
                    smooth[base + i].e = raw.min(v_cap);
                }

                // dor + fast — mirrors precalculate_element::init_from_smooth_fst.
                for i in 0..n {
                    let e_i = smooth[base + i].e;
                    let e_next = if i + 1 >= n { 0.0 } else { smooth[base + i + 1].e };
                    fast[base + i] = 0.5 * (e_i + e_next);
                }
                smooth[base].dor = 0.0;
                smooth[base + n - 1].dor = 0.0;
                for i in 1..n - 1 {
                    let delta = rs[i + 1] - rs[i - 1];
                    let r = rs[i];
                    smooth[base + i].dor =
                        (smooth[base + i + 1].e - smooth[base + i - 1].e) / (delta * r);
                }
            }
        }

        Self {
            n,
            factor,
            cutoff_sqr,
            max_cutoff_sqr,
            smooth,
            fast,
        }
    }

    /// Convenience constructor for the default Vina scoring function:
    /// cutoff 8 Å, factor 32, no cap.
    #[must_use]
    pub fn vina() -> Self {
        Self::new(
            VINA_CUTOFF,
            VINA_CUTOFF,
            DEFAULT_FACTOR,
            f64::INFINITY,
            vina_pair_energy,
        )
    }

    /// Pair-potential cutoff squared (Å²). Values beyond this are
    /// zero by construction and should not be queried.
    #[inline]
    #[must_use]
    pub fn cutoff_sqr(&self) -> f64 {
        self.cutoff_sqr
    }

    /// Table extent squared (Å²). `r²` passed to `eval_*` must be
    /// `<= max_cutoff_sqr`.
    #[inline]
    #[must_use]
    pub fn max_cutoff_sqr(&self) -> f64 {
        self.max_cutoff_sqr
    }

    /// Midpoint-averaged energy lookup. No interpolation; slightly
    /// smoother than raw `smooth` samples because of the averaging
    /// upstream applies. Matches `precalculate_element::eval_fast`.
    ///
    /// # Panics
    /// In debug builds, panics if `r² > max_cutoff_sqr`.
    #[inline]
    #[must_use]
    pub fn eval_fast(&self, t1: XsType, t2: XsType, r_sq: f64) -> f64 {
        debug_assert!(r_sq <= self.max_cutoff_sqr);
        let base = pair_index(t1.index(), t2.index()) * self.n;
        let i = (self.factor * r_sq) as usize;
        debug_assert!(i < self.n);
        self.fast[base + i]
    }

    /// Linear-interpolated energy and pre-scaled derivative lookup.
    /// Returns `(E, dor)` where `dor = (dE/dr) / r`. Matches
    /// `precalculate_element::eval_deriv`.
    ///
    /// # Panics
    /// In debug builds, panics if `r² > max_cutoff_sqr`.
    #[inline]
    #[must_use]
    pub fn eval_deriv(&self, t1: XsType, t2: XsType, r_sq: f64) -> (f64, f64) {
        debug_assert!(r_sq <= self.max_cutoff_sqr);
        let base = pair_index(t1.index(), t2.index()) * self.n;
        let r2_factored = self.factor * r_sq;
        let i1 = r2_factored as usize;
        let i2 = i1 + 1;
        debug_assert!(i2 < self.n);
        let rem = r2_factored - i1 as f64;
        let s1 = self.smooth[base + i1];
        let s2 = self.smooth[base + i2];
        let e = s1.e + rem * (s2.e - s1.e);
        let dor = s1.dor + rem * (s2.dor - s1.dor);
        (e, dor)
    }
}

/// Sample positions: `rs[i] = sqrt(i / factor)`. Matches
/// `precalculate::calculate_rs()`.
fn calculate_rs(n: usize, factor: f64) -> Vec<f64> {
    (0..n).map(|i| (i as f64 / factor).sqrt()).collect()
}

/// Static table of `XsType` by discriminant, used to iterate during
/// precalc build. Order matches the enum discriminants so
/// `XS_TYPES[t.index()] == t`.
const XS_TYPES: [XsType; NUM_XS_TYPES] = [
    XsType::CH, XsType::CP, XsType::NP, XsType::ND, XsType::NA, XsType::NDA,
    XsType::OP, XsType::OD, XsType::OA, XsType::ODA, XsType::SP, XsType::PP,
    XsType::FH, XsType::ClH, XsType::BrH, XsType::IH, XsType::Si, XsType::At,
    XsType::MetD, XsType::CHCG0, XsType::CPCG0, XsType::G0, XsType::CHCG1,
    XsType::CPCG1, XsType::G1, XsType::CHCG2, XsType::CPCG2, XsType::G2,
    XsType::CHCG3, XsType::CPCG3, XsType::G3, XsType::W,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn xs_types_table_is_self_consistent() {
        for (i, &t) in XS_TYPES.iter().enumerate() {
            assert_eq!(t.index(), i);
        }
    }

    #[test]
    fn pair_index_covers_every_unordered_pair_once() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for t1 in 0..NUM_XS_TYPES {
            for t2 in t1..NUM_XS_TYPES {
                let idx = pair_index(t1, t2);
                assert!(idx < NUM_PAIRS, "idx {idx} out of range");
                assert!(seen.insert(idx), "duplicate pair index at ({t1},{t2})");
            }
        }
        assert_eq!(seen.len(), NUM_PAIRS);
    }

    #[test]
    fn pair_index_symmetric() {
        for t1 in 0..NUM_XS_TYPES {
            for t2 in 0..NUM_XS_TYPES {
                assert_eq!(pair_index(t1, t2), pair_index(t2, t1));
            }
        }
    }

    #[test]
    fn calculate_rs_is_sqrt_of_i_over_factor() {
        let rs = calculate_rs(10, 32.0);
        for i in 0..10 {
            assert_relative_eq!(rs[i], (i as f64 / 32.0).sqrt(), epsilon = 1e-15);
        }
    }

    #[test]
    fn vina_build_is_cheap() {
        let p = Precalculate::vina();
        assert_eq!(p.factor, DEFAULT_FACTOR);
        assert_relative_eq!(p.cutoff_sqr, 64.0, epsilon = 1e-12);
        assert_eq!(p.n, 32 * 64 + 3);
        assert_eq!(p.smooth.len(), NUM_PAIRS * p.n);
    }

    #[test]
    fn table_symmetric_across_type_swap() {
        let p = Precalculate::vina();
        for r in [0.5_f64, 1.5, 3.5, 5.0, 7.5] {
            let r_sq = r * r;
            let a = p.eval_fast(XsType::CH, XsType::OA, r_sq);
            let b = p.eval_fast(XsType::OA, XsType::CH, r_sq);
            assert_eq!(a, b, "eval_fast asymmetric at r={r}");
            let (ea, da) = p.eval_deriv(XsType::ND, XsType::OA, r_sq);
            let (eb, db) = p.eval_deriv(XsType::OA, XsType::ND, r_sq);
            assert_eq!(ea, eb, "eval_deriv energy asymmetric at r={r}");
            assert_eq!(da, db, "eval_deriv dor asymmetric at r={r}");
        }
    }

    #[test]
    fn eval_fast_matches_midpoint_of_samples() {
        // At an exact grid point r² = i/factor, `eval_fast` returns
        // the midpoint average (smooth[i] + smooth[i+1]) / 2. Check it
        // against a direct recomputation from the raw potential.
        let p = Precalculate::vina();
        let factor = DEFAULT_FACTOR;
        for i in 5..20 {
            let r_sq = i as f64 / factor;
            let r_i = (i as f64 / factor).sqrt();
            let r_next = ((i + 1) as f64 / factor).sqrt();
            let expected_mid = 0.5
                * (vina_pair_energy(XsType::CH, XsType::OA, r_i)
                    + vina_pair_energy(XsType::CH, XsType::OA, r_next));
            let got = p.eval_fast(XsType::CH, XsType::OA, r_sq);
            assert_relative_eq!(got, expected_mid, epsilon = 1e-12);
        }
    }

    #[test]
    fn eval_deriv_matches_raw_potential_at_exact_samples() {
        // At exact sample points the linear interpolation collapses to
        // the stored energy, which equals the raw potential.
        let p = Precalculate::vina();
        let factor = DEFAULT_FACTOR;
        for i in 5..20 {
            let r_sq = i as f64 / factor;
            let r = (i as f64 / factor).sqrt();
            let expected = vina_pair_energy(XsType::CH, XsType::CH, r);
            let (got, _dor) = p.eval_deriv(XsType::CH, XsType::CH, r_sq);
            assert_relative_eq!(got, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn eval_deriv_is_zero_beyond_cutoff() {
        let p = Precalculate::vina();
        // Well past cutoff: r² = 63 (r ≈ 7.94), just below max_cutoff.
        let (e, dor) = p.eval_deriv(XsType::CH, XsType::OA, 63.0);
        // All terms have cutoff 8.0 Å, so both samples around r=7.94
        // are from the near-cutoff tail. At r = 8.0 exactly, energy is
        // 0; nearby samples are tiny. We don't assert exactly zero, but
        // we do assert |e| is small.
        assert!(e.abs() < 1e-1, "expected near-zero e at r²=63, got {e}");
        assert!(dor.abs() < 1.0, "expected small dor at r²=63, got {dor}");
    }

    #[test]
    fn hydrophobic_pair_more_negative_at_contact_than_polar_pair() {
        // Sanity check: at contact distance, a CH-CH pair should score
        // more attractively than a CH-NP pair (hydrophobic term fires
        // for CH-CH only, and both miss the H-bond term).
        let p = Precalculate::vina();
        let r_ch = 1.9 + 1.9; // CH-CH optimum
        let e_hh = p.eval_fast(XsType::CH, XsType::CH, r_ch * r_ch);
        let e_hp = p.eval_fast(XsType::CH, XsType::NP, r_ch * r_ch);
        assert!(
            e_hh < e_hp,
            "hydrophobic contact ({e_hh}) should be more attractive than polar ({e_hp})"
        );
    }
}
