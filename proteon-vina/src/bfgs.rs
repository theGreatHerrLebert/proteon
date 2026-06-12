// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/bfgs.h (Apache-2.0).
// Upstream author: Oleg Trott, Scripps Research Institute.

//! Quasi-Newton BFGS minimizer over [`Conf`] degrees of freedom.
//!
//! Direct port of upstream `bfgs.h`:
//! * Inverse-Hessian approximation `H` kept as a symmetric matrix,
//!   initialised to the identity. Re-scaled after the first step
//!   by `α · y·p / y·y`.
//! * Descent direction: `p = −H · g`. Armijo line search with
//!   backtracking (c₀=1e-4, ≤ 10 trials, multiplier 0.5).
//! * Rank-2 BFGS update on `H` using `s = α·p`, `y = g_new − g`.
//! * Convergence when `‖g‖ < 1e-5` or `max_steps` reached.
//! * Safety: if the final energy exceeds the starting energy (only
//!   possible if line search fails repeatedly), revert to the
//!   starting point.

use crate::conf::Conf;
use crate::gradient::ConfGrad;

/// Dense symmetric matrix stored flat in row-major upper triangle.
/// `n` entries per row; total `n·(n+1)/2` floats.
#[derive(Clone, Debug)]
struct SymMat {
    dim: usize,
    data: Vec<f64>,
}

impl SymMat {
    fn new(dim: usize, fill: f64) -> Self {
        Self { dim, data: vec![fill; dim * (dim + 1) / 2] }
    }
    /// Flat index into `data` for `(i, j)` with no ordering
    /// requirement (we swap internally).
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        // Upper-triangle row-major: offset = a*(2n − a + 1)/2 + (b − a).
        a * (2 * self.dim - a + 1) / 2 + (b - a)
    }
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.idx(i, j)]
    }
    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        let k = self.idx(i, j);
        self.data[k] += v;
    }
    fn set_diagonal(&mut self, x: f64) {
        for i in 0..self.dim {
            let k = self.idx(i, i);
            self.data[k] = x;
        }
    }
    fn zero_all(&mut self) {
        for v in &mut self.data {
            *v = 0.0;
        }
    }
}

/// Outcome of a BFGS run.
#[derive(Clone, Debug)]
pub struct BfgsOutcome {
    /// Energy at the final conformation.
    pub final_energy: f64,
    /// Energy at the starting conformation (for easy relative-gain
    /// reporting).
    pub initial_energy: f64,
    /// Number of energy evaluations (including line-search misses).
    pub n_evals: usize,
    /// Number of outer BFGS iterations actually performed.
    pub n_steps: usize,
    /// True if convergence hit (gradient norm < tolerance); false if
    /// max_steps exhausted.
    pub converged: bool,
}

/// Minimise `f` over a [`Conf`] starting from `x`. The energy
/// functor must return `(energy, gradient)` where the gradient is
/// `+∂E/∂DoF` — i.e. the NEGATIVE of the "force" returned by
/// [`crate::gradient::gradient_from_forces`]. Passing the raw
/// physics-force-based ConfGrad out of that helper would have BFGS
/// walk UPHILL, so we flip sign at the boundary: see the
/// `local_only` driver in [`crate::local_only`] for the idiomatic
/// wiring.
///
/// Returns the minimised energy; `x` is updated in place.
pub fn bfgs<F>(f: &mut F, x: &mut Conf, max_steps: usize) -> BfgsOutcome
where
    F: FnMut(&Conf) -> (f64, ConfGrad),
{
    let n = 6 + x.torsions.len();

    // Inverse-Hessian approximation (identity start).
    let mut h = SymMat::new(n, 0.0);
    h.set_diagonal(1.0);

    let mut evals = 0_usize;

    let (mut f0, mut g) = f(x);
    evals += 1;
    let f_orig = f0;
    let x_orig = x.clone();
    let g_orig = g.clone();

    let mut converged = false;
    let mut steps_taken = 0_usize;

    for step in 0..max_steps {
        steps_taken = step + 1;
        // Descent direction p = −H · g.
        let p = minus_h_times(&h, &g);
        // Line search (Armijo).
        let mut f1 = 0.0;
        let mut x_new = x.clone();
        let mut g_new = g.clone();
        let (alpha, line_evals) = line_search(f, x, &g, f0, &p, &mut x_new, &mut g_new, &mut f1);
        evals += line_evals;

        // s = α·p. Taken implicitly via x_new = x + α·p.
        let mut y = g_new.clone();
        y.subtract(&g);

        f0 = f1;
        *x = x_new;

        // Convergence test must use the gradient at the NEW x (g_new),
        // not the gradient at the previous x (g). Otherwise a step that
        // lands at a stationary point on the final permitted iteration
        // is reported as not-converged, and any extra budget burns one
        // useless line search at the minimum. Upstream Vina checks the
        // post-step gradient too (`bfgs.h::bfgs`).
        let gnorm = g_new.norm_sqr().sqrt();
        // Break for gradient below threshold OR NaN (hence the
        // partial_cmp + is-less-than-threshold test, matching the
        // `!(gnorm >= 1e-5)` idiom upstream uses).
        if !matches!(gnorm.partial_cmp(&1e-5), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) {
            converged = true;
            break;
        }
        g = g_new;

        if step == 0 {
            // Diagonal rescale recommended by Nocedal & Wright:
            // H₀ ← α · (yᵀp / yᵀy) · I.
            let yy = y.norm_sqr();
            if yy.abs() > f64::EPSILON {
                let yp = y.dot(&p);
                let scale = alpha * yp / yy;
                h.zero_all();
                h.set_diagonal(scale);
            }
        }

        let _ = bfgs_update(&mut h, &p, &y, alpha);
    }

    // Revert to starting point if line search never found an
    // improvement — matches upstream's safeguard at end of bfgs().
    // The `!(f0 <= f_orig)` idiom revert-on-NaN too.
    let worsened = !matches!(f0.partial_cmp(&f_orig), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal));
    let final_energy = if worsened {
        *x = x_orig;
        let _ = g_orig;
        f_orig
    } else {
        f0
    };

    BfgsOutcome {
        final_energy,
        initial_energy: f_orig,
        n_evals: evals,
        n_steps: steps_taken,
        converged,
    }
}

/// `out = −H · g`, implementing the descent-direction computation.
fn minus_h_times(h: &SymMat, g: &ConfGrad) -> ConfGrad {
    let n = h.dim;
    let mut p = ConfGrad::zero(g.torsions.len());
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += h.get(i, j) * g.get(j);
        }
        p.set(i, -s);
    }
    p
}

/// Armijo backtracking line search. Returns `(alpha, n_evals)`.
///
/// Direct port of `bfgs.h::line_search`:
/// * initial `α = 1`
/// * up to 10 trials halving `α`
/// * accept the first step that satisfies `f1 − f0 < c₀ · α · p·g`
fn line_search<F>(
    f: &mut F,
    x: &Conf,
    g: &ConfGrad,
    f0: f64,
    p: &ConfGrad,
    x_new: &mut Conf,
    g_new: &mut ConfGrad,
    f1: &mut f64,
) -> (f64, usize)
where
    F: FnMut(&Conf) -> (f64, ConfGrad),
{
    const C0: f64 = 1e-4;
    const MAX_TRIALS: usize = 10;
    const MULTIPLIER: f64 = 0.5;

    let pg = p.dot(g);
    let mut alpha = 1.0_f64;
    let mut evals = 0_usize;
    for trial in 0..MAX_TRIALS {
        *x_new = x.clone();
        x_new.increment(p, alpha);
        let (e, gn) = f(x_new);
        *f1 = e;
        *g_new = gn;
        evals += 1;
        if *f1 - f0 < C0 * alpha * pg {
            break;
        }
        // Halve alpha only when another trial will follow. Halving
        // after the last trial would return an alpha that no longer
        // matches the x_new just stored, corrupting the BFGS Hessian
        // initial-scaling and rank-2 update on the next iteration.
        // This is a deliberate deviation from upstream `bfgs.h`, which
        // halves unconditionally.
        if trial + 1 < MAX_TRIALS {
            alpha *= MULTIPLIER;
        }
    }
    (alpha, evals)
}

/// BFGS rank-2 update of `H`. Returns `false` without modifying
/// `H` if the curvature condition `α · y·p > 0` fails — matches
/// upstream's early return.
fn bfgs_update(h: &mut SymMat, p: &ConfGrad, y: &ConfGrad, alpha: f64) -> bool {
    let n = h.dim;
    let yp = y.dot(p);
    if alpha * yp < f64::EPSILON {
        return false;
    }
    // minus_hy = −H · y
    let minus_hy = {
        let mut out = ConfGrad::zero(p.torsions.len());
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += h.get(i, j) * y.get(j);
            }
            out.set(i, -s);
        }
        out
    };
    let yhy = -y.dot(&minus_hy);
    let r = 1.0 / (alpha * yp);
    for i in 0..n {
        for j in i..n {
            let term_a =
                alpha * r * (minus_hy.get(i) * p.get(j) + minus_hy.get(j) * p.get(i));
            let term_b = alpha * alpha * (r * r * yhy + r) * p.get(i) * p.get(j);
            h.add(i, j, term_a + term_b);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conf::{Conf, Quat};
    use crate::gradient::ConfGrad;

    /// Simple quadratic bowl in the three translation DoFs:
    /// `E(c) = ‖c − target‖²`, gradient `2 (c − target)`.
    fn make_quadratic_translation(target: [f64; 3]) -> impl FnMut(&Conf) -> (f64, ConfGrad) {
        move |c: &Conf| -> (f64, ConfGrad) {
            let d = [
                c.center[0] - target[0],
                c.center[1] - target[1],
                c.center[2] - target[2],
            ];
            let e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let grad = ConfGrad {
                center: [2.0 * d[0], 2.0 * d[1], 2.0 * d[2]],
                orientation: [0.0; 3],
                torsions: vec![],
            };
            (e, grad)
        }
    }

    #[test]
    fn bfgs_minimises_quadratic_translation_bowl() {
        let mut f = make_quadratic_translation([5.0, -3.0, 1.0]);
        let mut x = Conf {
            center: [0.0, 0.0, 0.0],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 50);
        assert!(out.converged);
        assert!(out.final_energy < 1e-8, "final E = {}", out.final_energy);
        for k in 0..3 {
            assert!((x.center[k] - [5.0, -3.0, 1.0][k]).abs() < 1e-4);
        }
    }

    #[test]
    fn bfgs_records_initial_and_final_energies_consistently() {
        let mut f = make_quadratic_translation([1.0, 1.0, 1.0]);
        let mut x = Conf {
            center: [10.0, 10.0, 10.0],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 50);
        // initial = ‖(10,10,10) − (1,1,1)‖² = 243
        assert!((out.initial_energy - 243.0).abs() < 1e-9);
        assert!(out.final_energy < out.initial_energy);
        assert!(out.final_energy < 1e-8);
    }

    /// Quadratic over torsion DoFs: E = Σᵢ (τᵢ − target_i)².
    fn make_quadratic_torsions(
        targets: Vec<f64>,
    ) -> impl FnMut(&Conf) -> (f64, ConfGrad) {
        move |c: &Conf| -> (f64, ConfGrad) {
            let mut e = 0.0_f64;
            let mut grad_t = Vec::with_capacity(targets.len());
            for (t, target) in c.torsions.iter().zip(targets.iter()) {
                let d = t - target;
                e += d * d;
                grad_t.push(2.0 * d);
            }
            let grad = ConfGrad {
                center: [0.0; 3],
                orientation: [0.0; 3],
                torsions: grad_t,
            };
            (e, grad)
        }
    }

    #[test]
    fn bfgs_minimises_torsion_quadratic() {
        let targets = vec![0.3, -0.5, 1.1];
        let mut f = make_quadratic_torsions(targets.clone());
        let mut x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![0.0, 0.0, 0.0],
        };
        let out = bfgs(&mut f, &mut x, 50);
        assert!(out.converged);
        for (t, target) in x.torsions.iter().zip(targets.iter()) {
            assert!((t - target).abs() < 1e-4);
        }
    }

    #[test]
    fn bfgs_at_minimum_takes_zero_steps_effectively() {
        // Start at the minimum: gradient is zero, BFGS converges
        // on the first gradient check.
        let mut f = make_quadratic_translation([0.0, 0.0, 0.0]);
        let mut x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 50);
        assert!(out.converged);
        assert_eq!(out.final_energy, 0.0);
    }

    #[test]
    fn bfgs_never_reports_an_energy_worse_than_the_start() {
        // Pathological flat energy (gradient reported but never drops).
        // Upstream only reverts when final > initial; on == it keeps
        // the drifted state. Either way, final_energy <= initial_energy.
        let mut f = |_: &Conf| -> (f64, ConfGrad) {
            (
                42.0,
                ConfGrad {
                    center: [1.0, 1.0, 1.0],
                    orientation: [0.0; 3],
                    torsions: vec![],
                },
            )
        };
        let mut x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 5);
        assert_eq!(out.initial_energy, 42.0);
        assert!(out.final_energy <= out.initial_energy);
    }

    #[test]
    fn bfgs_reports_convergence_on_step_that_lands_at_minimum() {
        // Quadratic with target [1,1,1] starting at the origin. The
        // first BFGS line search accepts alpha=0.5 (alpha=1 overshoots
        // and Armijo rejects it), x_new = [1,1,1] — exactly the
        // minimum. With only one permitted step, the convergence test
        // must read the gradient at the NEW x (≈0), not the gradient
        // at the previous x. This is the regression test for the
        // stale-gradient bug at the convergence check.
        let mut f = make_quadratic_translation([1.0, 1.0, 1.0]);
        let mut x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 1);
        assert_eq!(out.n_steps, 1);
        assert!(out.converged, "convergence at the new x must be reported on the same step");
        assert!(out.final_energy < 1e-9);
    }

    #[test]
    fn line_search_returned_alpha_matches_x_new_after_all_trials_fail() {
        // Pathological: the energy and gradient never change with
        // alpha, so the Armijo condition (Δf < C0·α·p·g, with p·g<0)
        // never holds for any positive alpha. After all 10 trials, the
        // returned alpha must equal the alpha at which x_new was last
        // evaluated — otherwise the BFGS rank-2 update and the initial
        // diagonal rescale see an alpha out of sync with the s = α·p
        // they assume, corrupting the inverse-Hessian state.
        use std::cell::Cell;

        let last_alpha = Cell::new(f64::NAN);
        let f0 = 5.0;
        let mut f = |c: &Conf| -> (f64, ConfGrad) {
            // p.center[0] = -1, x.center[0] = 0, so increment leaves
            // c.center[0] = -alpha. Recover alpha from the conf.
            last_alpha.set(-c.center[0]);
            (
                f0,
                ConfGrad {
                    center: [1.0, 0.0, 0.0],
                    orientation: [0.0; 3],
                    torsions: vec![],
                },
            )
        };
        let x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let g = ConfGrad {
            center: [1.0, 0.0, 0.0],
            orientation: [0.0; 3],
            torsions: vec![],
        };
        let p = ConfGrad {
            center: [-1.0, 0.0, 0.0],
            orientation: [0.0; 3],
            torsions: vec![],
        };
        let mut x_new = x.clone();
        let mut g_new = g.clone();
        let mut f1 = 0.0;
        let (returned_alpha, evals) =
            line_search(&mut f, &x, &g, f0, &p, &mut x_new, &mut g_new, &mut f1);
        assert_eq!(evals, 10, "all trials must run when Armijo never holds");
        let alpha_used_for_last_xnew = last_alpha.get();
        assert!(
            (returned_alpha - alpha_used_for_last_xnew).abs() < 1e-15,
            "alpha drift: returned={returned_alpha}, x_new at={alpha_used_for_last_xnew}"
        );
    }

    #[test]
    fn bfgs_reverts_when_final_energy_exceeds_initial() {
        // Contrived: report energy that INCREASES each call. BFGS's
        // line-search Armijo will fail (f1 > f0 + c·α·pg never) so
        // revert kicks in when the loop exits with f0 > f_orig. Tweak
        // the gradient to make line search accept bad steps.
        let mut counter = 0_usize;
        let mut f = move |_: &Conf| -> (f64, ConfGrad) {
            counter += 1;
            let e = 10.0 + counter as f64 * 0.01;
            (
                e,
                ConfGrad {
                    center: [-0.001, 0.0, 0.0], // small gradient so pg ≈ 0
                    orientation: [0.0; 3],
                    torsions: vec![],
                },
            )
        };
        let mut x = Conf {
            center: [0.0; 3],
            orientation: Quat::IDENTITY,
            torsions: vec![],
        };
        let out = bfgs(&mut f, &mut x, 10);
        // Revert: final should equal initial (never worse).
        assert!(out.final_energy <= out.initial_energy + 1e-12);
    }
}
