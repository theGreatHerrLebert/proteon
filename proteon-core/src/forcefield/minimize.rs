//! Energy minimization algorithms.
//!
//! Steepest descent with adaptive step size. Simple but robust
//! for hydrogen position optimization.

use super::energy::EnergyResult;
use super::nb_cache::NbCache;
use super::params::ForceField;
use super::topology::Topology;

/// How a minimization run terminated. Distinguishes genuine convergence from the
/// failure modes that the old `converged: bool` silently conflated — in particular a
/// line-search stall (the optimizer could not make progress) is **not** convergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimizeStatus {
    /// Max gradient fell below `gradient_tolerance`.
    ConvergedGradient,
    /// Energy plateaued (relative change below tolerance for `PLATEAU_PATIENCE`
    /// **accepted** steps) — a legitimate stop, distinct from a stall.
    ConvergedEnergy,
    /// Ran out of the step budget with the gradient still above tolerance.
    MaxSteps,
    /// The line search could not find an energy-decreasing step (the optimizer is
    /// stuck / the input is pathological). The caller should treat the result as
    /// **un-relaxed**, not converged.
    LineSearchFailed,
    /// A non-finite energy or force was encountered (NaN/Inf input or blow-up).
    NumericalFailure,
    /// Nothing ran (e.g. `max_steps == 0`, or no movable atoms).
    NotRun,
}

impl MinimizeStatus {
    /// Whether this status represents real convergence (gradient or energy).
    #[must_use]
    pub fn is_converged(self) -> bool {
        matches!(self, Self::ConvergedGradient | Self::ConvergedEnergy)
    }

    /// Stable lowercase tag for serialization across the FFI boundary.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConvergedGradient => "converged_gradient",
            Self::ConvergedEnergy => "converged_energy",
            Self::MaxSteps => "max_steps",
            Self::LineSearchFailed => "line_search_failed",
            Self::NumericalFailure => "numerical_failure",
            Self::NotRun => "not_run",
        }
    }
}

/// Result of energy minimization.
#[derive(Clone, Debug)]
pub struct MinimizeResult {
    /// Optimized coordinates
    pub coords: Vec<[f64; 3]>,
    /// Final energy breakdown
    pub energy: EnergyResult,
    /// Initial total energy
    pub initial_energy: f64,
    /// Number of outer iterations taken
    pub steps: usize,
    /// Number of line-search-accepted steps that actually moved the coordinates.
    /// Zero means the run was a no-op (the trap this field exists to expose).
    pub accepted_steps: usize,
    /// How the run terminated. The source of truth; `converged` is derived from it.
    pub status: MinimizeStatus,
    /// Whether minimization converged. Derived from [`status`](Self::status)
    /// (`status.is_converged()`); retained for backward compatibility.
    pub converged: bool,
}

impl MinimizeResult {
    /// Build a result, deriving `converged` from `status` so the two never disagree.
    fn new(
        coords: Vec<[f64; 3]>,
        energy: EnergyResult,
        initial_energy: f64,
        steps: usize,
        accepted_steps: usize,
        status: MinimizeStatus,
    ) -> Self {
        Self {
            coords,
            energy,
            initial_energy,
            steps,
            accepted_steps,
            status,
            converged: status.is_converged(),
        }
    }
}

/// Energy plateau fallback for the convergence check.
///
/// The pure `max_grad < tol` criterion is brittle near the boundary: structures
/// whose largest gradient hovers around `gradient_tolerance` end up burning the
/// step budget without ever crossing it, even though their energy has stopped
/// moving meaningfully. Some of those flips are also non-deterministic across
/// runs because parallel-reduction order can perturb the gradient norm by a
/// few ppm.
///
/// To make convergence robust, every minimizer in this module ALSO declares
/// success when the absolute energy change has stayed below
/// `PLATEAU_REL_TOL * max(|energy|, 1.0)` for `PLATEAU_PATIENCE` consecutive
/// iterations. This is the same energy-plateau pattern OpenMM and AMBER use
/// (though with slightly different parameters).
const PLATEAU_REL_TOL: f64 = 1.0e-6;
const PLATEAU_PATIENCE: usize = 5;
/// An energy plateau only counts as **convergence** if the gradient is also within
/// this factor of `gradient_tolerance`. The plateau fallback exists for structures
/// whose largest gradient *hovers around* the tolerance; a plateau with a gradient
/// far above it is a STALL (negligible progress at a non-minimum), not convergence —
/// reporting it as converged would recreate the silent no-op trap (codex review).
const PLATEAU_GRAD_FACTOR: f64 = 10.0;

/// Classify an energy-plateau stop given the current max gradient: real
/// [`ConvergedEnergy`](MinimizeStatus::ConvergedEnergy) only if the gradient is also
/// reasonably small; otherwise it is a stall
/// ([`LineSearchFailed`](MinimizeStatus::LineSearchFailed)).
fn plateau_status(max_grad: f64, gradient_tolerance: f64) -> MinimizeStatus {
    if max_grad < PLATEAU_GRAD_FACTOR * gradient_tolerance {
        MinimizeStatus::ConvergedEnergy
    } else {
        MinimizeStatus::LineSearchFailed
    }
}

/// Update the consecutive-tiny-change counter and report whether the energy
/// has plateaued. `prev_energy` is `None` on the first call (no comparison
/// possible yet). Returns the new counter value and whether convergence
/// should be declared.
fn check_energy_plateau(prev_energy: Option<f64>, energy: f64, counter: usize) -> (usize, bool) {
    let Some(prev) = prev_energy else {
        return (0, false);
    };
    let denom = energy.abs().max(1.0);
    if (prev - energy).abs() / denom < PLATEAU_REL_TOL {
        let new_counter = counter + 1;
        (new_counter, new_counter >= PLATEAU_PATIENCE)
    } else {
        (0, false)
    }
}

/// True iff every component of every vector is finite (no NaN/Inf).
///
/// The convergence test takes `max_force.max(per_atom_force)`, and `f64::max`
/// ignores NaN, so a non-finite force would be silently dropped and a blown-up
/// step could read as converged. Minimizers check this each step and bail with
/// `NumericalFailure` instead.
fn all_finite(v: &[[f64; 3]]) -> bool {
    v.iter()
        .all(|a| a[0].is_finite() && a[1].is_finite() && a[2].is_finite())
}

/// Minimize energy using steepest descent with line search.
///
/// # Arguments
/// * `coords` — Initial coordinates (modified in place)
/// * `topo` — Topology (bonds, angles, etc.)
/// * `params` — Force field parameters
/// * `max_steps` — Maximum iterations
/// * `gradient_tolerance` — Convergence criterion (kcal/mol/Å)
/// * `constrained` — Indices of atoms that should not move
pub fn steepest_descent(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
    constrained: &[bool],
) -> MinimizeResult {
    let n = coords.len();
    let mut pos: Vec<[f64; 3]> = coords.to_vec();
    let mut nbc = NbCache::new(&pos, topo, params);

    let initial_e = nbc.energy(&pos, topo, params);
    let initial_energy = initial_e.total;

    // Guard the inputs: a non-finite starting energy (NaN/Inf coords or a blown-up
    // field) cannot be optimized — report it instead of silently "not converging".
    if !initial_energy.is_finite() {
        return MinimizeResult::new(
            pos,
            initial_e,
            initial_energy,
            0,
            0,
            MinimizeStatus::NumericalFailure,
        );
    }
    if max_steps == 0 || constrained.iter().all(|&c| c) {
        return MinimizeResult::new(pos, initial_e, initial_energy, 0, 0, MinimizeStatus::NotRun);
    }

    let mut energy = initial_energy;
    let mut steps = 0;
    let mut accepted_steps = 0usize;
    let mut status = MinimizeStatus::MaxSteps;
    let mut plateau_counter = 0usize;
    // Plateau is fed ONLY accepted iterates (energy updates only on an accepted
    // line-search step), so a non-progressing run can never masquerade as converged.
    let mut plateau_prev: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        // True steepest descent: search direction is the raw force (= −gradient),
        // a single global vector — NOT the old per-atom unit step, which moved every
        // atom the same distance regardless of its force and was not a descent
        // direction on heterogeneous structures.
        let (e_res, forces) = nbc.energy_and_forces(&pos, topo, params);

        // A non-finite energy or force means the geometry blew up; the gradient
        // test would silently swallow a NaN (f64::max ignores it) and could
        // report convergence. Bail honestly instead.
        if !e_res.total.is_finite() || !all_finite(&forces) {
            status = MinimizeStatus::NumericalFailure;
            break;
        }

        let mut max_force = 0.0f64;
        for i in 0..n {
            if constrained[i] {
                continue;
            }
            let f2 = forces[i][0] * forces[i][0]
                + forces[i][1] * forces[i][1]
                + forces[i][2] * forces[i][2];
            max_force = max_force.max(f2.sqrt());
        }
        if max_force < gradient_tolerance {
            status = MinimizeStatus::ConvergedGradient;
            break;
        }

        // Energy-plateau convergence — only meaningful once we have at least one
        // accepted step to compare against, and only a real convergence if the
        // gradient is also small (otherwise a slow-progress run is a stall).
        let (new_counter, plateaued) = check_energy_plateau(plateau_prev, energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            status = plateau_status(max_force, gradient_tolerance);
            break;
        }
        plateau_prev = Some(energy);

        // direction = forces; grad·direction = (−forces)·forces = −|forces|² < 0.
        let grad_dot_dir = -dot3n(&forces, &forces, constrained);
        let (alpha, new_energy, new_pos) = line_search(
            &pos,
            &forces,
            grad_dot_dir,
            energy,
            topo,
            params,
            constrained,
            &mut nbc,
        );

        if alpha == 0.0 {
            // No Armijo-acceptable step exists along the steepest-descent direction —
            // genuinely stuck. Report it as a stall (NOT convergence). `pos` is
            // unchanged (line_search returns a fresh trial; we never committed it).
            status = MinimizeStatus::LineSearchFailed;
            break;
        }

        pos = new_pos;
        energy = new_energy;
        accepted_steps += 1;
        nbc.refresh(&pos, topo);
    }

    // Refresh the neighbor list against the FINAL accepted coords before the
    // reported energy: a failed line search leaves the cache built against a
    // rejected trial, which would otherwise make final_energy inconsistent with
    // the returned `pos` on NBL-sized systems.
    nbc.refresh(&pos, topo);
    let final_energy = nbc.energy(&pos, topo, params);
    MinimizeResult::new(
        pos,
        final_energy,
        initial_energy,
        steps,
        accepted_steps,
        status,
    )
}

// ---------------------------------------------------------------------------
// Conjugate gradient (Polak-Ribiere with automatic restarts)
// ---------------------------------------------------------------------------

/// Dot product of flat 3N force/gradient arrays.
fn dot3n(a: &[[f64; 3]], b: &[[f64; 3]], constrained: &[bool]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        if constrained[i] {
            continue;
        }
        sum += a[i][0] * b[i][0] + a[i][1] * b[i][1] + a[i][2] * b[i][2];
    }
    sum
}

/// Scale direction vector: out = a * x + b * y (only unconstrained atoms).
fn axpby(
    out: &mut [[f64; 3]],
    a: f64,
    x: &[[f64; 3]],
    b: f64,
    y: &[[f64; 3]],
    constrained: &[bool],
) {
    for i in 0..out.len() {
        if constrained[i] {
            out[i] = [0.0; 3];
        } else {
            out[i][0] = a * x[i][0] + b * y[i][0];
            out[i][1] = a * x[i][1] + b * y[i][1];
            out[i][2] = a * x[i][2] + b * y[i][2];
        }
    }
}

/// Backtracking line search with Armijo condition.
///
/// Finds step size α such that E(pos + α*dir) < E(pos) + c1*α*(grad·dir).
/// Returns (step_taken, new_energy) or (0.0, old_energy) on failure.
fn line_search<F: ForceField>(
    pos: &[[f64; 3]],
    direction: &[[f64; 3]],
    grad_dot_dir: f64,
    current_energy: f64,
    topo: &Topology,
    params: &F,
    constrained: &[bool],
    nbc: &mut NbCache,
) -> (f64, f64, Vec<[f64; 3]>) {
    let c1 = 1e-4; // Armijo parameter
    let n = pos.len();
    let mut trial = vec![[0.0; 3]; n];

    // Cap the initial alpha so a single trial displacement is modest. This bounds
    // the *incremental* move; it does NOT by itself keep the cached NBL valid,
    // because `pos` may already have drifted ~buffer/2 from the NBL's reference
    // across accepted steps (codex review). The per-trial `nbc.refresh` below is
    // what guarantees correctness — it rebuilds the list whenever a trial drifts
    // past the buffer, so no newly-interacting pair is ever missed.
    let max_disp = 0.8; // Å
    let mut max_d = 0.0_f64;
    for i in 0..n {
        if constrained[i] {
            continue;
        }
        let d2 = direction[i][0].powi(2) + direction[i][1].powi(2) + direction[i][2].powi(2);
        max_d = max_d.max(d2.sqrt());
    }
    let mut alpha = if max_d > 1e-12 {
        (max_disp / max_d).min(1.0)
    } else {
        1.0
    };

    for _ in 0..20 {
        // trial = pos + alpha * direction
        for i in 0..n {
            if constrained[i] {
                trial[i] = pos[i];
            } else {
                trial[i][0] = pos[i][0] + alpha * direction[i][0];
                trial[i][1] = pos[i][1] + alpha * direction[i][1];
                trial[i][2] = pos[i][2] + alpha * direction[i][2];
            }
        }

        // Validate the cached neighbor list against THIS trial before reading its
        // energy: `refresh` rebuilds iff the trial drifted past the buffer, so the
        // energy can never be computed from a stale list that omits a now-close
        // pair (a no-op for small systems with no NBL). Correctness over caching.
        nbc.refresh(&trial, topo);
        let e = nbc.energy(&trial, topo, params);

        // Armijo sufficient decrease condition
        if e.total <= current_energy + c1 * alpha * grad_dot_dir {
            return (alpha, e.total, trial);
        }

        alpha *= 0.5;
    }

    // Failed — return zero step
    (0.0, current_energy, pos.to_vec())
}

/// Minimize energy using conjugate gradient (Polak-Ribiere).
///
/// Much faster convergence than steepest descent for well-conditioned
/// systems. Automatically restarts to steepest descent every 3N iterations
/// or when the CG direction becomes a poor descent direction.
///
/// Reference: Polak & Ribiere (1969), Revue Francaise Informat. Recherche
/// Operationelle, 16, 35-43.
pub fn conjugate_gradient(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
    constrained: &[bool],
) -> MinimizeResult {
    let n = coords.len();
    let mut pos = coords.to_vec();
    let mut nbc = NbCache::new(&pos, topo, params);

    let initial_e = nbc.energy(&pos, topo, params);
    let initial_energy = initial_e.total;
    if !initial_energy.is_finite() {
        return MinimizeResult::new(
            pos,
            initial_e,
            initial_energy,
            0,
            0,
            MinimizeStatus::NumericalFailure,
        );
    }
    if max_steps == 0 || constrained.iter().all(|&c| c) {
        return MinimizeResult::new(pos, initial_e, initial_energy, 0, 0, MinimizeStatus::NotRun);
    }

    // First force evaluation
    let (_, forces) = nbc.energy_and_forces(&pos, topo, params);

    // Gradient = -force. We work with forces directly (descent direction).
    let mut old_forces = forces;

    // Initial direction = steepest descent (forces)
    let mut direction = vec![[0.0; 3]; n];
    for i in 0..n {
        if constrained[i] {
            direction[i] = [0.0; 3];
        } else {
            direction[i] = old_forces[i];
        }
    }

    let mut old_gtg = dot3n(&old_forces, &old_forces, constrained);
    let mut energy = initial_energy;
    let mut status = MinimizeStatus::MaxSteps;
    let mut steps = 0;
    let mut accepted_steps = 0usize;
    let restart_frequency = 3 * n;
    let mut plateau_counter = 0usize;
    let mut plateau_prev: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        // Non-finite forces would be swallowed by the max() convergence test.
        if !all_finite(&old_forces) {
            status = MinimizeStatus::NumericalFailure;
            break;
        }

        // Check convergence: max force magnitude
        let mut max_force = 0.0f64;
        for i in 0..n {
            if constrained[i] {
                continue;
            }
            let f2 = old_forces[i][0].powi(2) + old_forces[i][1].powi(2) + old_forces[i][2].powi(2);
            max_force = max_force.max(f2.sqrt());
        }
        if max_force < gradient_tolerance {
            status = MinimizeStatus::ConvergedGradient;
            break;
        }

        // Plateau fallback (accepted iterates only — `energy` updates on accept).
        let (new_counter, plateaued) = check_energy_plateau(plateau_prev, energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            status = plateau_status(max_force, gradient_tolerance);
            break;
        }
        plateau_prev = Some(energy);

        // Line search along direction
        // grad_dot_dir = -forces · direction (gradient = -force)
        let grad_dot_dir = -dot3n(&old_forces, &direction, constrained);

        // Direction must be a descent direction
        if grad_dot_dir >= 0.0 {
            // Restart: reset to steepest descent
            for i in 0..n {
                direction[i] = if constrained[i] {
                    [0.0; 3]
                } else {
                    old_forces[i]
                };
            }
            old_gtg = dot3n(&old_forces, &old_forces, constrained);
            continue;
        }

        let (alpha, new_energy, new_pos) = line_search(
            &pos,
            &direction,
            grad_dot_dir,
            energy,
            topo,
            params,
            constrained,
            &mut nbc,
        );

        if alpha == 0.0 {
            // Line search failed — at a minimum, or stuck. The latter is a stall,
            // not convergence; the gradient check above already caught a true minimum.
            status = MinimizeStatus::LineSearchFailed;
            break;
        }

        pos = new_pos;
        energy = new_energy;
        accepted_steps += 1;

        // Refresh the cached neighbor list if atoms drifted past the buffer.
        nbc.refresh(&pos, topo);

        // Compute new forces at the new position
        let (_, new_forces) = nbc.energy_and_forces(&pos, topo, params);

        let new_gtg = dot3n(&new_forces, &new_forces, constrained);

        // Polak-Ribiere beta: β = (g_new · (g_new - g_old)) / (g_old · g_old)
        // Using forces (negative gradient): β = (f_new · (f_new - f_old)) / (f_old · f_old)
        let mut beta = 0.0;
        if old_gtg > 1e-30 {
            let mut f_diff = vec![[0.0; 3]; n];
            for i in 0..n {
                f_diff[i][0] = new_forces[i][0] - old_forces[i][0];
                f_diff[i][1] = new_forces[i][1] - old_forces[i][1];
                f_diff[i][2] = new_forces[i][2] - old_forces[i][2];
            }
            beta = dot3n(&new_forces, &f_diff, constrained) / old_gtg;
        }

        // Polak-Ribiere with restart: β = max(β, 0)
        // Negative β means we should restart (Powell's criterion)
        beta = beta.max(0.0);

        // Periodic restart to steepest descent
        if step % restart_frequency == 0 {
            beta = 0.0;
        }

        // Update direction: d_new = f_new + β * d_old
        let old_dir = direction.clone();
        axpby(
            &mut direction,
            1.0,
            &new_forces,
            beta,
            &old_dir,
            constrained,
        );

        old_forces = new_forces;
        old_gtg = new_gtg;
    }

    // Refresh the neighbor list against the FINAL accepted coords before the
    // reported energy: a failed line search leaves the cache built against a
    // rejected trial, which would otherwise make final_energy inconsistent with
    // the returned `pos` on NBL-sized systems.
    nbc.refresh(&pos, topo);
    let final_energy = nbc.energy(&pos, topo, params);
    MinimizeResult::new(
        pos,
        final_energy,
        initial_energy,
        steps,
        accepted_steps,
        status,
    )
}

// ---------------------------------------------------------------------------
// L-BFGS (Limited-memory BFGS)
// ---------------------------------------------------------------------------

/// Minimize energy using L-BFGS (limited-memory BFGS).
///
/// Quasi-Newton method that approximates the inverse Hessian using the last
/// `m` gradient/position updates. Much faster convergence than CG for large
/// systems. Uses the two-loop recursion algorithm (Nocedal, 1980).
///
/// Reference: Jorge Nocedal, "Updating Quasi-Newton Matrices with Limited
/// Storage", Mathematics of Computation 35, 773-782 (1980).
pub fn lbfgs(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
    constrained: &[bool],
) -> MinimizeResult {
    let n = coords.len();
    let m = 10; // number of stored correction pairs (typical: 5-20)
    let mut pos = coords.to_vec();

    // Build a neighbor list once for large structures; reuse across iterations
    // and line search trials. This avoids rebuilding ~50×20 times per call on
    // the O(N²) nonbonded loop, which dominates cost above ~2K atoms.
    let mut nbc = NbCache::new(&pos, topo, params);

    let initial_e = nbc.energy(&pos, topo, params);
    let initial_energy = initial_e.total;
    if !initial_energy.is_finite() {
        return MinimizeResult::new(
            pos,
            initial_e,
            initial_energy,
            0,
            0,
            MinimizeStatus::NumericalFailure,
        );
    }
    if max_steps == 0 || constrained.iter().all(|&c| c) {
        return MinimizeResult::new(pos, initial_e, initial_energy, 0, 0, MinimizeStatus::NotRun);
    }

    let (_, forces) = nbc.energy_and_forces(&pos, topo, params);
    // Gradient = -force
    let mut grad = negate_forces(&forces, constrained);

    // Storage for correction pairs: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
    let mut s_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    let mut energy = initial_energy;
    let mut status = MinimizeStatus::MaxSteps;
    let mut steps = 0;
    let mut accepted_steps = 0usize;
    let mut plateau_counter = 0usize;
    let mut prev_energy: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        // Non-finite gradient would be swallowed by the max() convergence test.
        if !all_finite(&grad) {
            status = MinimizeStatus::NumericalFailure;
            break;
        }

        // Check convergence: max gradient magnitude
        let mut max_grad = 0.0f64;
        for i in 0..n {
            if constrained[i] {
                continue;
            }
            let g2 = grad[i][0].powi(2) + grad[i][1].powi(2) + grad[i][2].powi(2);
            max_grad = max_grad.max(g2.sqrt());
        }
        if max_grad < gradient_tolerance {
            status = MinimizeStatus::ConvergedGradient;
            break;
        }

        // Plateau fallback: a real convergence only if the gradient is also small;
        // a plateau with a still-large gradient is a stall, not convergence.
        let (new_counter, plateaued) = check_energy_plateau(prev_energy, energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            status = plateau_status(max_grad, gradient_tolerance);
            break;
        }
        prev_energy = Some(energy);

        // Two-loop recursion to compute search direction: d = -H_k * g_k
        let direction = lbfgs_two_loop(&grad, &s_hist, &y_hist, &rho_hist, constrained);

        // Line search (only if the direction is actually descent)
        let grad_dot_dir = dot3n_raw(&grad, &direction, constrained);
        let (alpha, new_energy, new_pos) = if grad_dot_dir < 0.0 {
            line_search(
                &pos,
                &direction,
                grad_dot_dir,
                energy,
                topo,
                params,
                constrained,
                &mut nbc,
            )
        } else {
            (0.0, energy, pos.clone())
        };

        // Either the direction wasn't descent, or the line search couldn't find an
        // Armijo-acceptable step. The LBFGS Hessian approximation has become
        // unreliable. Clear the history and try a recovery steepest-descent step via
        // the SAME Armijo line search — computed into a trial so we NEVER commit a
        // non-improving (or worse) position. If it can't make progress, stop and
        // leave `pos` at the last accepted iterate (transactional: no silent rollback
        // to higher energy).
        if alpha == 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            // Recovery direction = forces (= −grad), the steepest-descent direction.
            let sd_dir = negate_forces(&grad, constrained); // = forces
            let sd_gd = -dot3n_raw(&grad, &grad, constrained); // grad·(−grad) ≤ 0
            let (rec_alpha, rec_energy, rec_pos) = line_search(
                &pos,
                &sd_dir,
                sd_gd,
                energy,
                topo,
                params,
                constrained,
                &mut nbc,
            );
            if rec_alpha == 0.0 {
                // Genuinely stuck — pos unchanged (trial was separate), report stall.
                status = MinimizeStatus::LineSearchFailed;
                break;
            }
            pos = rec_pos;
            energy = rec_energy;
            accepted_steps += 1;
            nbc.refresh(&pos, topo);
            let (_, new_forces) = nbc.energy_and_forces(&pos, topo, params);
            grad = negate_forces(&new_forces, constrained);
            continue;
        }

        // Compute s_k = x_{k+1} - x_k
        let mut s_k = vec![[0.0; 3]; n];
        for i in 0..n {
            s_k[i][0] = new_pos[i][0] - pos[i][0];
            s_k[i][1] = new_pos[i][1] - pos[i][1];
            s_k[i][2] = new_pos[i][2] - pos[i][2];
        }

        pos = new_pos;
        energy = new_energy;
        accepted_steps += 1;

        // Refresh the cached neighbor list if atoms drifted past the buffer.
        nbc.refresh(&pos, topo);

        // New gradient
        let (_, new_forces) = nbc.energy_and_forces(&pos, topo, params);
        let new_grad = negate_forces(&new_forces, constrained);

        // y_k = g_{k+1} - g_k
        let mut y_k = vec![[0.0; 3]; n];
        for i in 0..n {
            y_k[i][0] = new_grad[i][0] - grad[i][0];
            y_k[i][1] = new_grad[i][1] - grad[i][1];
            y_k[i][2] = new_grad[i][2] - grad[i][2];
        }

        let sy = dot3n_raw(&s_k, &y_k, constrained);
        if sy > 1e-10 {
            // Store correction pair
            if s_hist.len() >= m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s_k);
            y_hist.push(y_k);
        }

        grad = new_grad;
    }

    // Refresh the neighbor list against the FINAL accepted coords before the
    // reported energy: a failed line search leaves the cache built against a
    // rejected trial, which would otherwise make final_energy inconsistent with
    // the returned `pos` on NBL-sized systems.
    nbc.refresh(&pos, topo);
    let final_energy = nbc.energy(&pos, topo, params);
    MinimizeResult::new(
        pos,
        final_energy,
        initial_energy,
        steps,
        accepted_steps,
        status,
    )
}

/// Two-loop recursion for L-BFGS search direction.
///
/// Returns d = -H_k * g where H_k is the L-BFGS approximation to the inverse Hessian.
fn lbfgs_two_loop(
    grad: &[[f64; 3]],
    s_hist: &[Vec<[f64; 3]>],
    y_hist: &[Vec<[f64; 3]>],
    rho_hist: &[f64],
    constrained: &[bool],
) -> Vec<[f64; 3]> {
    let n = grad.len();
    let k = s_hist.len();

    // q = g_k
    let mut q = grad.to_vec();

    // First loop (backward)
    let mut alpha_hist = vec![0.0; k];
    for i in (0..k).rev() {
        let alpha_i = rho_hist[i] * dot3n_raw(&s_hist[i], &q, constrained);
        alpha_hist[i] = alpha_i;
        // q = q - alpha_i * y_i
        for j in 0..n {
            if constrained[j] {
                continue;
            }
            q[j][0] -= alpha_i * y_hist[i][j][0];
            q[j][1] -= alpha_i * y_hist[i][j][1];
            q[j][2] -= alpha_i * y_hist[i][j][2];
        }
    }

    // Scale by initial Hessian approximation: H0 = (s_k · y_k) / (y_k · y_k) * I
    if k > 0 {
        let last = k - 1;
        let yy = dot3n_raw(&y_hist[last], &y_hist[last], constrained);
        let sy = dot3n_raw(&s_hist[last], &y_hist[last], constrained);
        if yy > 1e-30 {
            let gamma = sy / yy;
            for j in 0..n {
                if constrained[j] {
                    continue;
                }
                q[j][0] *= gamma;
                q[j][1] *= gamma;
                q[j][2] *= gamma;
            }
        }
    }

    // Second loop (forward)
    for i in 0..k {
        let beta = rho_hist[i] * dot3n_raw(&y_hist[i], &q, constrained);
        let diff = alpha_hist[i] - beta;
        for j in 0..n {
            if constrained[j] {
                continue;
            }
            q[j][0] += diff * s_hist[i][j][0];
            q[j][1] += diff * s_hist[i][j][1];
            q[j][2] += diff * s_hist[i][j][2];
        }
    }

    // Return -H*g (descent direction)
    for j in 0..n {
        q[j][0] = -q[j][0];
        q[j][1] = -q[j][1];
        q[j][2] = -q[j][2];
    }
    q
}

/// Negate forces to get gradient, zeroing constrained atoms.
fn negate_forces(forces: &[[f64; 3]], constrained: &[bool]) -> Vec<[f64; 3]> {
    forces
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if constrained[i] {
                [0.0; 3]
            } else {
                [-f[0], -f[1], -f[2]]
            }
        })
        .collect()
}

/// Raw dot product (no force negation).
fn dot3n_raw(a: &[[f64; 3]], b: &[[f64; 3]], constrained: &[bool]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        if constrained[i] {
            continue;
        }
        sum += a[i][0] * b[i][0] + a[i][1] * b[i][1] + a[i][2] * b[i][2];
    }
    sum
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Minimize only hydrogen positions (freeze all heavy atoms).
#[allow(dead_code)]
pub fn minimize_hydrogens(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
) -> MinimizeResult {
    let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();

    steepest_descent(
        coords,
        topo,
        params,
        max_steps,
        gradient_tolerance,
        &constrained,
    )
}

/// Minimize hydrogen positions using conjugate gradient.
#[allow(dead_code)]
pub fn minimize_hydrogens_cg(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
) -> MinimizeResult {
    let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();

    conjugate_gradient(
        coords,
        topo,
        params,
        max_steps,
        gradient_tolerance,
        &constrained,
    )
}

/// Minimize hydrogen positions using L-BFGS.
#[allow(dead_code)]
pub fn minimize_hydrogens_lbfgs(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
) -> MinimizeResult {
    let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();

    lbfgs(
        coords,
        topo,
        params,
        max_steps,
        gradient_tolerance,
        &constrained,
    )
}

#[cfg(test)]
mod plateau_tests {
    use super::*;

    #[test]
    fn first_call_never_plateaus() {
        let (counter, plateaued) = check_energy_plateau(None, -1000.0, 0);
        assert_eq!(counter, 0);
        assert!(!plateaued);
    }

    #[test]
    fn large_change_resets_counter() {
        // Energy moved by ~10% — well above any reasonable plateau tolerance.
        let (counter, plateaued) = check_energy_plateau(Some(-1000.0), -1100.0, 4);
        assert_eq!(counter, 0);
        assert!(!plateaued);
    }

    #[test]
    fn small_change_increments_counter() {
        // Energy moved by 1e-8 of magnitude — well below PLATEAU_REL_TOL=1e-6.
        let (counter, plateaued) = check_energy_plateau(Some(-1000.0), -1000.00001, 0);
        assert_eq!(counter, 1);
        assert!(!plateaued);
    }

    #[test]
    fn patience_threshold_triggers_convergence() {
        // After PLATEAU_PATIENCE-1 tiny-change steps, one more triggers it.
        let (counter, plateaued) =
            check_energy_plateau(Some(-1000.0), -1000.00001, PLATEAU_PATIENCE - 1);
        assert_eq!(counter, PLATEAU_PATIENCE);
        assert!(plateaued);
    }

    #[test]
    fn handles_near_zero_energy() {
        // Denominator is clamped to max(|e|, 1.0) so a near-zero energy
        // doesn't blow up the relative-change check.
        let (counter, plateaued) = check_energy_plateau(Some(0.0), 1e-10, 0);
        assert_eq!(counter, 1);
        assert!(!plateaued);
    }

    #[test]
    fn plateau_tol_is_strict_enough_to_distinguish_real_progress() {
        // 5e-6 relative change should NOT be flagged as plateau (above 1e-6).
        let (counter, plateaued) = check_energy_plateau(Some(-1000.0), -1000.005, 4);
        assert_eq!(counter, 0);
        assert!(!plateaued);
    }

    #[test]
    fn plateau_with_large_gradient_is_a_stall_not_convergence() {
        // codex P1: a plateau reached while the gradient is still far above tolerance
        // is a STALL — it must NOT be reported as ConvergedEnergy (that would set
        // relax_ok=True on a non-minimum and recreate the silent-quality failure).
        let tol = 0.1;
        // Gradient comfortably within the plateau gate -> real energy convergence.
        assert_eq!(
            plateau_status(0.5, tol),
            MinimizeStatus::ConvergedEnergy,
            "small gradient + plateau is genuine convergence"
        );
        // Gradient far above tolerance -> stall, reported as LineSearchFailed.
        assert_eq!(
            plateau_status(50.0, tol),
            MinimizeStatus::LineSearchFailed,
            "large gradient + plateau is a stall, not convergence"
        );
        // The status must not be is_converged() in the stall case.
        assert!(!plateau_status(50.0, tol).is_converged());
    }
}

/// Reliability guards for the minimizers (CPU). These are the tests whose absence
/// let two silent-correctness bugs ship: a steepest-descent no-op on clashing
/// structures (it returned `final == initial` with no error) and an LBFGS recovery
/// that could leave higher-energy coordinates in place.
#[cfg(test)]
mod reliability_tests {
    use super::*;
    use crate::add_hydrogens;
    use crate::forcefield::params::{amber96, AmberParams};
    use crate::forcefield::topology::build_topology;
    use std::path::PathBuf;

    /// Concrete signature shared by the three minimizers, so they collect into one array.
    type Runner = fn(&[[f64; 3]], &Topology, &AmberParams, usize, f64, &[bool]) -> MinimizeResult;
    const RUNNERS: [(&str, Runner); 3] = [
        ("sd", steepest_descent),
        ("cg", conjugate_gradient),
        ("lbfgs", lbfgs),
    ];

    /// Load crambin, add all-atom hydrogens (creates the clashes / high energy that
    /// triggered the original no-op), and build an AMBER96 topology + coords. Returns
    /// `None` if the fixture is absent (CI without test-pdbs) — caller skips.
    fn protonated_amber_system() -> Option<(Topology, Vec<[f64; 3]>)> {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../test-pdbs/1crn.pdb");
        if !p.exists() {
            eprintln!("reliability_tests: 1crn.pdb not found, skipping");
            return None;
        }
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(p.to_str().unwrap())
            .expect("failed to read 1crn");
        add_hydrogens::place_all_hydrogens(&mut pdb, false);
        let ff = amber96();
        let topo = build_topology(&pdb, &ff);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        Some((topo, coords))
    }

    fn max_disp(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| {
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
            })
            .fold(0.0_f64, f64::max)
    }

    /// THE guard: every minimizer must strictly lower the energy of a clashing
    /// structure, actually move atoms, and report honest progress — never the silent
    /// `final == initial` no-op the per-atom-unit-step SD used to produce.
    #[test]
    fn minimization_lowers_energy_and_reports_progress() {
        let Some((topo, coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let constrained = vec![false; coords.len()];
        // A clashing (protonated) structure sheds most of its energy in the first
        // handful of steps; 30 keeps the guard fast in debug CI while still proving a
        // strict decrease for all three optimizers.
        for (name, run) in RUNNERS {
            let r = run(&coords, &topo, &ff, 30, 0.1, &constrained);
            assert!(
                r.energy.total < r.initial_energy,
                "{name}: energy did not decrease ({} -> {}); the no-op trap",
                r.initial_energy,
                r.energy.total
            );
            assert!(r.accepted_steps > 0, "{name}: zero accepted steps (no-op)");
            assert!(
                max_disp(&r.coords, &coords) > 1e-6,
                "{name}: coordinates did not move"
            );
            assert!(
                !matches!(
                    r.status,
                    MinimizeStatus::NotRun | MinimizeStatus::NumericalFailure
                ),
                "{name}: unexpected status {:?}",
                r.status
            );
            assert_eq!(
                r.converged,
                r.status.is_converged(),
                "{name}: converged/status disagree"
            );
        }
    }

    /// A hard clash (overlapping non-bonded atoms → non-finite 1/r^12 energy and
    /// forces) must NEVER be reported as converged. Without the finiteness guards
    /// a NaN gradient slips through `max()` (which ignores NaN) and masquerades as
    /// `ConvergedGradient`; with them, every minimizer reports `NumericalFailure`.
    #[test]
    fn hard_clash_is_never_reported_as_converged() {
        let Some((topo, mut coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let n = coords.len();
        coords[n - 1] = coords[0]; // collapse a non-bonded pair onto one point
        let constrained = vec![false; n];
        for (name, run) in RUNNERS {
            let r = run(&coords, &topo, &ff, 30, 0.1, &constrained);
            assert!(
                !r.status.is_converged(),
                "{name}: a hard clash must not report convergence (got {:?})",
                r.status
            );
        }
    }

    #[test]
    fn all_finite_rejects_nan_and_inf() {
        assert!(super::all_finite(&[[0.0, 1.0, -2.0]]));
        assert!(!super::all_finite(&[[0.0, f64::NAN, 0.0]]));
        assert!(!super::all_finite(&[[f64::INFINITY, 0.0, 0.0]]));
    }

    /// Guards the LBFGS transactional fix: a minimizer must NEVER return a structure
    /// worse than its input (the old recovery path could commit a higher-energy step).
    #[test]
    fn minimizer_never_returns_worse_than_initial() {
        let Some((topo, coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let constrained = vec![false; coords.len()];
        for (_, run) in RUNNERS {
            // Few steps so the line search / recovery paths are exercised mid-run.
            let r = run(&coords, &topo, &ff, 5, 0.1, &constrained);
            assert!(
                r.energy.total <= r.initial_energy + 1e-9,
                "minimizer returned a worse structure: {} -> {}",
                r.initial_energy,
                r.energy.total
            );
        }
    }

    /// The energy-only path and the energy+forces path must agree on the total, or the
    /// line-search accept test compares against an inconsistent baseline.
    #[test]
    fn energy_only_and_energy_plus_forces_totals_agree() {
        let Some((topo, coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let mut nbc = NbCache::new(&coords, &topo, &ff);
        let e_only = nbc.energy(&coords, &topo, &ff).total;
        let e_force = nbc.energy_and_forces(&coords, &topo, &ff).0.total;
        assert!(
            (e_only - e_force).abs() <= 1e-6 * e_only.abs().max(1.0),
            "energy paths disagree: {e_only} vs {e_force}"
        );
    }

    #[test]
    fn max_steps_zero_is_not_run_and_unchanged() {
        let Some((topo, coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let constrained = vec![false; coords.len()];
        let r = lbfgs(&coords, &topo, &ff, 0, 0.1, &constrained);
        assert_eq!(r.status, MinimizeStatus::NotRun);
        assert_eq!(r.accepted_steps, 0);
        assert!(!r.converged);
        assert_eq!(r.coords, coords, "NotRun must leave coordinates untouched");
        assert_eq!(r.energy.total, r.initial_energy);
    }

    #[test]
    fn all_constrained_is_not_run() {
        let Some((topo, coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        let constrained = vec![true; coords.len()];
        for (_, run) in RUNNERS {
            let r = run(&coords, &topo, &ff, 50, 0.1, &constrained);
            assert_eq!(r.status, MinimizeStatus::NotRun);
            assert_eq!(r.coords, coords);
        }
    }

    #[test]
    fn non_finite_input_is_numerical_failure_not_panic() {
        let Some((topo, mut coords)) = protonated_amber_system() else {
            return;
        };
        let ff = amber96();
        coords[0][0] = f64::NAN;
        let constrained = vec![false; coords.len()];
        let r = lbfgs(&coords, &topo, &ff, 100, 0.1, &constrained);
        assert_eq!(r.status, MinimizeStatus::NumericalFailure);
        assert!(!r.converged);
    }
}

/// GPU parity test: LBFGS on a real structure via GPU must produce the same
/// final energy as the CPU path. Only compiled and run with `--features cuda`.
/// Skips silently if no GPU is detected at runtime (CI machines without GPU).
#[cfg(all(test, feature = "cuda"))]
mod gpu_parity_tests {
    use super::*;
    use crate::add_hydrogens;
    use crate::forcefield::params::charmm19_eef1;
    use crate::forcefield::topology::build_topology;
    use std::path::PathBuf;

    fn ake_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../test-pdbs/1ake.pdb");
        p
    }

    /// Test GPU energy+forces directly on crambin by constructing a
    /// GpuStructState manually (bypasses the 2000-atom NbCache threshold).
    /// Compares GPU energy to CPU energy on the same coordinates to 1e-4.
    #[test]
    fn gpu_energy_matches_cpu_on_crambin() {
        use super::super::gpu;

        let gpu_ctx = match gpu::GpuContext::try_global() {
            Some(ctx) => ctx,
            None => {
                eprintln!("gpu_parity_tests: no GPU detected, skipping");
                return;
            }
        };

        let path = ake_path();
        // Use crambin instead — always parses cleanly
        let mut crambin_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crambin_path.push("../test-pdbs/1crn.pdb");
        if !crambin_path.exists() {
            eprintln!("gpu_parity_tests: 1crn.pdb not found, skipping");
            return;
        }

        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(crambin_path.to_str().unwrap())
            .expect("failed to read 1crn");
        add_hydrogens::place_peptide_hydrogens(&mut pdb);

        let ff = charmm19_eef1();
        let topo = build_topology(&pdb, &ff);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();

        // CPU reference
        let nbl = super::super::neighbor_list::NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        let cpu_energy = super::super::energy::compute_energy_nbl(&coords, &topo, &ff, &nbl);

        // GPU: construct GpuStructState directly (bypasses NbCache threshold)
        let mut gpu_state = gpu::GpuStructState::new(gpu_ctx, &topo, &nbl, &ff)
            .expect("failed to create GPU state");
        let gpu_energy = gpu_state
            .energy(gpu_ctx, &coords_flat)
            .expect("GPU energy eval failed");

        let diff = (gpu_energy.total - cpu_energy.total).abs();
        let tol = 1e-4;
        assert!(
            diff < tol,
            "GPU energy ({:+.6}) vs CPU energy ({:+.6}): diff={:.2e} > tol={:.2e}",
            gpu_energy.total,
            cpu_energy.total,
            diff,
            tol,
        );

        eprintln!(
            "gpu_parity_tests: crambin GPU={:+.2} CPU={:+.2} diff={:.2e} PASS",
            gpu_energy.total, cpu_energy.total, diff,
        );
    }

    /// GPU OBC GB parity check: AMBER96+OBC2 on crambin, compared against
    /// CPU `compute_energy_nbl` on identical inputs. Exercises all four
    /// OBC CUDA kernels (born radii, rowwise energy+forces, chain
    /// transform, force spread).
    ///
    /// Tolerance 1e-3 kcal/mol total — slightly looser than the CHARMM
    /// crambin test (1e-4) because the OBC second-loop kernel uses
    /// atomicAdd for forces and therefore has non-deterministic FP
    /// summation order. The physics is identical to the CPU path, so any
    /// drift beyond rounding is a bug.
    #[test]
    fn gpu_obc_matches_cpu_on_crambin() {
        use super::super::gpu;
        use crate::forcefield::params::amber96_obc;

        let gpu_ctx = match gpu::GpuContext::try_global() {
            Some(ctx) => ctx,
            None => {
                eprintln!("gpu_parity_tests: no GPU detected, skipping OBC parity");
                return;
            }
        };

        let mut crambin_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crambin_path.push("../test-pdbs/1crn.pdb");
        if !crambin_path.exists() {
            eprintln!("gpu_parity_tests: 1crn.pdb not found, skipping OBC parity");
            return;
        }
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(crambin_path.to_str().unwrap())
            .expect("failed to read 1crn");
        add_hydrogens::place_peptide_hydrogens(&mut pdb);

        let ff = amber96_obc();
        let topo = build_topology(&pdb, &ff);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();

        let nbl = super::super::neighbor_list::NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        let (cpu_energy, cpu_forces) =
            super::super::energy::compute_energy_and_forces_nbl(&coords, &topo, &ff, &nbl);

        let mut gpu_state = gpu::GpuStructState::new(gpu_ctx, &topo, &nbl, &ff)
            .expect("failed to create GPU state");
        let (gpu_energy, gpu_forces) = gpu_state
            .energy_and_forces(gpu_ctx, &coords_flat)
            .expect("GPU OBC energy+forces eval failed");

        // Energy total within 1e-3 kcal/mol.
        let diff_total = (gpu_energy.total - cpu_energy.total).abs();
        assert!(
            diff_total < 1e-3,
            "OBC GPU total={:+.6} CPU total={:+.6} diff={:.2e}",
            gpu_energy.total,
            cpu_energy.total,
            diff_total,
        );
        // Solvation component too — isolates an OBC-only regression.
        let diff_solv = (gpu_energy.solvation - cpu_energy.solvation).abs();
        assert!(
            diff_solv < 1e-3,
            "OBC GPU solvation={:+.6} CPU solvation={:+.6} diff={:.2e}",
            gpu_energy.solvation,
            cpu_energy.solvation,
            diff_solv,
        );
        // Force max-component parity at 1e-3 kcal/mol/Å (atomicAdd-tolerant).
        let mut max_f_diff = 0.0_f64;
        for i in 0..cpu_forces.len() {
            for k in 0..3 {
                let d = (cpu_forces[i][k] - gpu_forces[i][k]).abs();
                if d > max_f_diff {
                    max_f_diff = d;
                }
            }
        }
        assert!(
            max_f_diff < 1e-3,
            "OBC GPU/CPU max force diff {:.2e} > 1e-3",
            max_f_diff
        );

        eprintln!(
            "gpu_parity_tests: OBC crambin GPU total={:+.2} CPU total={:+.2} diff_total={:.2e} \
             diff_solv={:.2e} max_force_diff={:.2e} PASS",
            gpu_energy.total, cpu_energy.total, diff_total, diff_solv, max_f_diff,
        );
    }

    /// Load crambin + place H, returning (topo, coords, coords_flat) for `ff`,
    /// or `None` if the GPU/fixture is unavailable.
    #[cfg(feature = "cuda")]
    fn crambin_topo<F: super::super::params::ForceField>(
        ff: &F,
    ) -> Option<(super::super::topology::Topology, Vec<[f64; 3]>, Vec<f64>)> {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../test-pdbs/1crn.pdb");
        if !p.exists() {
            return None;
        }
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(p.to_str().unwrap())
            .expect("failed to read 1crn");
        add_hydrogens::place_peptide_hydrogens(&mut pdb);
        let topo = build_topology(&pdb, ff);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
        Some((topo, coords, flat))
    }

    #[test]
    fn gpu_obc_cutoff_matches_cpu_on_crambin() {
        use super::super::gpu;
        use crate::forcefield::params::amber96_obc_cutoff;

        let gpu_ctx = match gpu::GpuContext::try_global() {
            Some(ctx) => ctx,
            None => {
                eprintln!("gpu_parity_tests: no GPU, skipping cutoff-OBC parity");
                return;
            }
        };

        // 8 Å cutoff genuinely truncates on crambin (~larger than 8 Å across),
        // so this exercises the GPU truncation + reaction-field shift, not just
        // the all-pairs path with everything in range.
        let mut ff = amber96_obc_cutoff();
        ff.cutoff_override = Some(8.0);
        let (topo, coords, coords_flat) = match crambin_topo(&ff) {
            Some(t) => t,
            None => return,
        };

        let nbl = super::super::neighbor_list::NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        // CPU cutoff path (builds the exclusion-free GB list internally at 8 Å).
        let (cpu_e, cpu_f) =
            super::super::energy::compute_energy_and_forces_nbl(&coords, &topo, &ff, &nbl);

        let mut gpu_state = gpu::GpuStructState::new(gpu_ctx, &topo, &nbl, &ff)
            .expect("GPU cutoff-OBC state build failed (guard removed?)");
        let (gpu_e, gpu_f) = gpu_state
            .energy_and_forces(gpu_ctx, &coords_flat)
            .expect("GPU cutoff-OBC eval failed");

        let diff_total = (gpu_e.total - cpu_e.total).abs();
        let diff_solv = (gpu_e.solvation - cpu_e.solvation).abs();
        assert!(diff_total < 1e-3, "cutoff-OBC total diff {diff_total:.2e}");
        assert!(
            diff_solv < 1e-3,
            "cutoff-OBC solvation diff {diff_solv:.2e}"
        );
        let mut max_f = 0.0_f64;
        for i in 0..cpu_f.len() {
            for k in 0..3 {
                max_f = max_f.max((cpu_f[i][k] - gpu_f[i][k]).abs());
            }
        }
        assert!(max_f < 1e-3, "cutoff-OBC max force diff {max_f:.2e}");
        eprintln!(
            "gpu_parity_tests: cutoff-OBC(8Å) crambin solv GPU={:+.2} CPU={:+.2} \
             diff_solv={:.2e} max_f={:.2e} PASS",
            gpu_e.solvation, cpu_e.solvation, diff_solv, max_f
        );
    }

    #[test]
    fn gpu_obc_cutoff_shift_is_forceless() {
        // The reaction-field shift is distance-independent ⇒ it changes the
        // energy but not the forces. With a cutoff LARGER than the box, every
        // pair is kept, so the cutoff path and NoCutoff path differ ONLY by the
        // shift: GPU forces must match (within atomicAdd noise) while the
        // solvation energies must differ by a clearly nonzero amount.
        use super::super::gpu;
        use crate::forcefield::params::{amber96_obc, amber96_obc_cutoff};

        let gpu_ctx = match gpu::GpuContext::try_global() {
            Some(ctx) => ctx,
            None => {
                eprintln!("gpu_parity_tests: no GPU, skipping forceless-shift");
                return;
            }
        };

        // Both force fields share the SAME nonbonded (LJ/Coulomb) cutoff so the
        // ONLY difference is the GB method (NoCutoff vs cutoff). The cutoff is
        // larger than the crambin diameter ⇒ the GB pair set is identical too,
        // leaving the reaction-field shift as the sole difference.
        let mut ff_nc = amber96_obc();
        ff_nc.cutoff_override = Some(1000.0);
        let (topo, coords, flat) = match crambin_topo(&ff_nc) {
            Some(t) => t,
            None => return,
        };
        let nbl = super::super::neighbor_list::NeighborList::build(
            &coords,
            ff_nc.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );

        let mut s_nc = gpu::GpuStructState::new(gpu_ctx, &topo, &nbl, &ff_nc).unwrap();
        let (e_nc, f_nc) = s_nc.energy_and_forces(gpu_ctx, &flat).unwrap();

        let mut ff_co = amber96_obc_cutoff();
        ff_co.cutoff_override = Some(1000.0);
        let mut s_co = gpu::GpuStructState::new(gpu_ctx, &topo, &nbl, &ff_co).unwrap();
        let (e_co, f_co) = s_co.energy_and_forces(gpu_ctx, &flat).unwrap();

        // Forces unchanged by the (forceless) shift.
        let mut max_f = 0.0_f64;
        for i in 0..f_nc.len() {
            for k in 0..3 {
                max_f = max_f.max((f_nc[i][k] - f_co[i][k]).abs());
            }
        }
        assert!(
            max_f < 1e-3,
            "forceless shift perturbed forces by {max_f:.2e}"
        );
        // But the energy IS shifted (and on crambin it's a sizable amount).
        let dsolv = (e_co.solvation - e_nc.solvation).abs();
        assert!(
            dsolv > 1.0,
            "expected a nonzero reaction-field shift, got {dsolv:.2e}"
        );
        eprintln!(
            "gpu_parity_tests: forceless-shift max_force_diff={max_f:.2e} solv_shift={dsolv:.2e} PASS"
        );
    }
}
