// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Affinity grid + trilinear interpolation, ported from AutoDock-Vina
// src/lib/{grid.h,grid.cpp,curl.h} (Apache-2.0).

//! A single scalar affinity grid over the search box, with trilinear
//! interpolation, an out-of-box slope penalty, and the soft `curl` cap —
//! the building block of the docking grid cache.
//!
//! One [`Grid`] stores, at each sample point, the energy a ligand atom of
//! a fixed XS type would feel from the *entire* receptor (precomputed by
//! [`crate::cache`]). Evaluating a ligand atom's interaction then costs a
//! single trilinear lookup instead of a loop over receptor atoms, and the
//! grid's out-of-bounds slope replaces the search-time `BoxPenalty`.
//!
//! [`Grid::evaluate`] returns `(energy, gradient)` where `gradient` is
//! `∂E/∂position` (matching upstream's `minus_forces`); the physics force
//! is its negation.

use crate::conf::Vec3;

/// Per-axis grid extent: the sampled interval `[begin, end]` divided into
/// `n_voxels` cells (so `n_voxels + 1` sample points).
#[derive(Clone, Copy, Debug)]
pub struct GridDim {
    /// Lower bound of the sampled interval.
    pub begin: f64,
    /// Upper bound of the sampled interval.
    pub end: f64,
    /// Number of voxels (cells); sample points along the axis is this + 1.
    pub n_voxels: usize,
}

impl GridDim {
    /// Width of the sampled interval.
    #[must_use]
    pub fn span(&self) -> f64 {
        self.end - self.begin
    }
}

/// A scalar 3-D affinity grid with trilinear interpolation.
#[derive(Clone, Debug)]
pub struct Grid {
    init: Vec3,        // world coords of sample (0,0,0)
    factor: Vec3,      // (n_samples-1) / range  — world→index scale
    factor_inv: Vec3,  // range / (n_samples-1)  — index→world scale
    dim_minus_1: Vec3, // (n_samples - 1) as f64 per axis
    n: [usize; 3],     // sample counts per axis (n_voxels + 1)
    data: Vec<f64>,    // row-major: data[(ix*n[1] + iy)*n[2] + iz]
}

impl Grid {
    /// Allocate a zeroed grid over the given per-axis dimensions.
    #[must_use]
    pub fn new(gd: [GridDim; 3]) -> Self {
        let n = [gd[0].n_voxels + 1, gd[1].n_voxels + 1, gd[2].n_voxels + 1];
        let mut factor = [0.0; 3];
        let mut factor_inv = [0.0; 3];
        let mut dim_minus_1 = [0.0; 3];
        for i in 0..3 {
            assert!(gd[i].span() > 0.0, "grid axis {i} has non-positive span");
            dim_minus_1[i] = (n[i] - 1) as f64;
            factor[i] = dim_minus_1[i] / gd[i].span();
            factor_inv[i] = 1.0 / factor[i];
        }
        Self {
            init: [gd[0].begin, gd[1].begin, gd[2].begin],
            factor,
            factor_inv,
            dim_minus_1,
            n,
            data: vec![0.0; n[0] * n[1] * n[2]],
        }
    }

    /// Sample counts per axis (`n_voxels + 1`).
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.n
    }

    #[inline]
    fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.n[1] + iy) * self.n[2] + iz
    }

    /// World coordinates of sample point `(ix, iy, iz)`.
    #[must_use]
    pub fn index_to_argument(&self, ix: usize, iy: usize, iz: usize) -> Vec3 {
        [
            self.init[0] + self.factor_inv[0] * ix as f64,
            self.init[1] + self.factor_inv[1] * iy as f64,
            self.init[2] + self.factor_inv[2] * iz as f64,
        ]
    }

    /// Set the value at sample point `(ix, iy, iz)`.
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, value: f64) {
        let idx = self.index(ix, iy, iz);
        self.data[idx] = value;
    }

    /// Value at sample point `(ix, iy, iz)`.
    #[must_use]
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.data[self.index(ix, iy, iz)]
    }

    /// Evaluate the grid at world position `location`, returning
    /// `(energy, gradient)` with `gradient = ∂E/∂position`.
    ///
    /// * `slope` — out-of-box penalty per Å (upstream's grid slope, ~1e6):
    ///   positions outside the sampled box clamp to the boundary and accrue
    ///   `slope · distance_outside`, with a constant restoring gradient.
    /// * `v` — soft cap for the `curl` helper (`f64::INFINITY` disables it;
    ///   1000.0 is the authentic Vina value).
    ///
    /// Faithful port of upstream `grid::evaluate_aux`.
    #[must_use]
    pub fn evaluate(&self, location: Vec3, slope: f64, v: f64) -> (f64, Vec3) {
        let mut s = [0.0_f64; 3];
        let mut miss = [0.0_f64; 3];
        let mut region = [0_i32; 3];
        let mut a = [0_usize; 3];

        for i in 0..3 {
            let si = (location[i] - self.init[i]) * self.factor[i];
            if si < 0.0 {
                miss[i] = -si;
                region[i] = -1;
                a[i] = 0;
                s[i] = 0.0;
            } else if si >= self.dim_minus_1[i] {
                miss[i] = si - self.dim_minus_1[i];
                region[i] = 1;
                a[i] = self.n[i] - 2; // dim - 2
                s[i] = 1.0;
            } else {
                region[i] = 0;
                a[i] = si as usize;
                s[i] = si - a[i] as f64;
            }
        }
        let penalty =
            slope * (miss[0] * self.factor_inv[0] + miss[1] * self.factor_inv[1] + miss[2] * self.factor_inv[2]);

        let (x0, y0, z0) = (a[0], a[1], a[2]);
        let (x1, y1, z1) = (x0 + 1, y0 + 1, z0 + 1);

        let f000 = self.get(x0, y0, z0);
        let f100 = self.get(x1, y0, z0);
        let f010 = self.get(x0, y1, z0);
        let f110 = self.get(x1, y1, z0);
        let f001 = self.get(x0, y0, z1);
        let f101 = self.get(x1, y0, z1);
        let f011 = self.get(x0, y1, z1);
        let f111 = self.get(x1, y1, z1);

        let (x, y, z) = (s[0], s[1], s[2]);
        let (mx, my, mz) = (1.0 - x, 1.0 - y, 1.0 - z);

        let mut f = f000 * mx * my * mz
            + f100 * x * my * mz
            + f010 * mx * y * mz
            + f110 * x * y * mz
            + f001 * mx * my * z
            + f101 * x * my * z
            + f011 * mx * y * z
            + f111 * x * y * z;

        // Gradient in index space.
        let mut g = [
            -f000 * my * mz + f100 * my * mz - f010 * y * mz + f110 * y * mz
                - f001 * my * z
                + f101 * my * z
                - f011 * y * z
                + f111 * y * z,
            -f000 * mx * mz - f100 * x * mz + f010 * mx * mz + f110 * x * mz
                - f001 * mx * z
                - f101 * x * z
                + f011 * mx * z
                + f111 * x * z,
            -f000 * mx * my - f100 * x * my - f010 * mx * y - f110 * x * y
                + f001 * mx * my
                + f101 * x * my
                + f011 * mx * y
                + f111 * x * y,
        ];

        curl_with_grad(&mut f, &mut g, v);

        let mut deriv = [0.0_f64; 3];
        for i in 0..3 {
            let g_here = if region[i] == 0 { g[i] } else { 0.0 };
            deriv[i] = self.factor[i] * g_here + slope * region[i] as f64;
        }
        (f + penalty, deriv)
    }
}

/// `curl` with a gradient, from upstream `curl.h`: soft-caps a positive
/// energy by `v/(v+e)` and scales the gradient by the square of that
/// factor. A non-finite `v` (or `e <= 0`) leaves both untouched.
#[inline]
fn curl_with_grad(e: &mut f64, grad: &mut Vec3, v: f64) {
    if *e > 0.0 && v.is_finite() {
        let tmp = if v < f64::EPSILON { 0.0 } else { v / (v + *e) };
        *e *= tmp;
        let s = tmp * tmp;
        grad[0] *= s;
        grad[1] *= s;
        grad[2] *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid spanning [0,4]^3 with 1 Å voxels (5 samples/axis), filled
    /// from a linear function — trilinear interpolation is *exact* for
    /// linear fields, so we can check values and gradients precisely.
    fn linear_grid(a: f64, b: f64, c: f64, d: f64) -> Grid {
        let gd = GridDim { begin: 0.0, end: 4.0, n_voxels: 4 };
        let mut g = Grid::new([gd, gd, gd]);
        for ix in 0..g.dims()[0] {
            for iy in 0..g.dims()[1] {
                for iz in 0..g.dims()[2] {
                    let p = g.index_to_argument(ix, iy, iz);
                    g.set(ix, iy, iz, a * p[0] + b * p[1] + c * p[2] + d);
                }
            }
        }
        g
    }

    #[test]
    fn trilinear_is_exact_for_linear_fields() {
        let g = linear_grid(0.5, -0.3, 0.2, 1.0);
        // Strictly-interior points: on the box boundary the grid switches to
        // the out-of-bounds region (gradient becomes the slope), so the
        // interior-gradient check only holds strictly inside.
        for &p in &[[1.3, 2.7, 0.4], [0.6, 3.4, 2.5], [3.9, 0.1, 3.3]] {
            let (e, grad) = g.evaluate(p, 1e6, f64::INFINITY);
            let expect = 0.5 * p[0] - 0.3 * p[1] + 0.2 * p[2] + 1.0;
            assert!((e - expect).abs() < 1e-9, "value {e} != {expect} at {p:?}");
            // Interior gradient equals the linear coefficients.
            assert!((grad[0] - 0.5).abs() < 1e-9);
            assert!((grad[1] + 0.3).abs() < 1e-9);
            assert!((grad[2] - 0.2).abs() < 1e-9);
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let g = linear_grid(0.4, 0.7, -0.5, 0.0);
        let p = [1.5, 2.2, 1.1];
        let (_e, grad) = g.evaluate(p, 1e6, f64::INFINITY);
        let h = 1e-6;
        for axis in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[axis] += h;
            pm[axis] -= h;
            let fd = (g.evaluate(pp, 1e6, f64::INFINITY).0 - g.evaluate(pm, 1e6, f64::INFINITY).0)
                / (2.0 * h);
            assert!((grad[axis] - fd).abs() < 1e-4, "axis {axis}: grad {} != fd {fd}", grad[axis]);
        }
    }

    #[test]
    fn out_of_box_applies_linear_slope_penalty() {
        // Constant-zero grid: in-box energy is 0; outside, energy = slope *
        // distance outside, with a constant restoring gradient of magnitude
        // `slope` along the violated axis.
        let gd = GridDim { begin: 0.0, end: 4.0, n_voxels: 4 };
        let g = Grid::new([gd, gd, gd]); // all zeros
        let slope = 100.0;

        let (e_in, grad_in) = g.evaluate([2.0, 2.0, 2.0], slope, f64::INFINITY);
        assert!(e_in.abs() < 1e-12 && grad_in.iter().all(|v| v.abs() < 1e-12));

        // 1.5 Å below the lower x bound.
        let (e_lo, grad_lo) = g.evaluate([-1.5, 2.0, 2.0], slope, f64::INFINITY);
        assert!((e_lo - slope * 1.5).abs() < 1e-9, "penalty {e_lo}");
        assert!((grad_lo[0] - (-slope)).abs() < 1e-9, "restoring grad {}", grad_lo[0]);

        // 0.5 Å above the upper z bound.
        let (e_hi, grad_hi) = g.evaluate([2.0, 2.0, 4.5], slope, f64::INFINITY);
        assert!((e_hi - slope * 0.5).abs() < 1e-9);
        assert!((grad_hi[2] - slope).abs() < 1e-9);
    }

    #[test]
    fn curl_softens_positive_energy() {
        // Uniform positive grid value E0: curl scales it by v/(v+E0).
        let gd = GridDim { begin: 0.0, end: 2.0, n_voxels: 2 };
        let mut g = Grid::new([gd, gd, gd]);
        let e0 = 50.0;
        for ix in 0..g.dims()[0] {
            for iy in 0..g.dims()[1] {
                for iz in 0..g.dims()[2] {
                    g.set(ix, iy, iz, e0);
                }
            }
        }
        let v = 1000.0;
        let (e, _) = g.evaluate([1.0, 1.0, 1.0], 1e6, v);
        assert!((e - e0 * (v / (v + e0))).abs() < 1e-9, "curl'd {e}");
    }
}
