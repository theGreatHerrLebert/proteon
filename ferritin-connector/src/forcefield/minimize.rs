//! Energy minimization algorithms.
//!
//! Steepest descent with adaptive step size. Simple but robust
//! for hydrogen position optimization.

use super::energy::{
    compute_energy, compute_energy_and_forces, compute_energy_and_forces_nbl, compute_energy_nbl,
    EnergyResult,
};
use super::neighbor_list::NeighborList;
use super::params::ForceField;
use super::topology::Topology;

/// Threshold above which the minimizer uses a cached neighbor list for nonbonded
/// interactions. Matches `energy::NBL_AUTO_THRESHOLD` (2000 atoms). For small
/// structures the O(N²) loop is faster than building+querying a neighbor list.
const MIN_NBL_THRESHOLD: usize = 2000;

/// Lightweight cache that dispatches energy/force calls to the neighbor-list
/// accelerated path when a cached `NeighborList` is available.
///
/// The NBL is built once per minimizer call and reused across iterations and
/// line search trials. Call [`NbCache::refresh`] between iterations to rebuild
/// the list when atoms have moved beyond the buffer.
struct NbCache {
    nbl: Option<NeighborList>,
    cutoff: f64,
}

impl NbCache {
    fn new<F: ForceField>(coords: &[[f64; 3]], topo: &Topology, params: &F) -> Self {
        let cutoff = params.nonbonded_cutoff();
        let nbl = if coords.len() >= MIN_NBL_THRESHOLD {
            Some(NeighborList::build(
                coords,
                cutoff,
                &topo.excluded_pairs,
                &topo.pairs_14,
            ))
        } else {
            None
        };
        Self { nbl, cutoff }
    }

    /// Rebuild the neighbor list if any atom has drifted further than the buffer
    /// allows. Cheap no-op if not using NBL or if atoms haven't moved much.
    fn refresh(&mut self, coords: &[[f64; 3]], topo: &Topology) {
        if let Some(ref nbl) = self.nbl {
            if nbl.needs_rebuild(coords) {
                self.nbl = Some(NeighborList::build(
                    coords,
                    self.cutoff,
                    &topo.excluded_pairs,
                    &topo.pairs_14,
                ));
            }
        }
    }

    fn energy<F: ForceField>(
        &self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> EnergyResult {
        match &self.nbl {
            Some(nbl) => compute_energy_nbl(coords, topo, params, nbl),
            None => compute_energy(coords, topo, params),
        }
    }

    fn energy_and_forces<F: ForceField>(
        &self,
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &F,
    ) -> (EnergyResult, Vec<[f64; 3]>) {
        match &self.nbl {
            Some(nbl) => compute_energy_and_forces_nbl(coords, topo, params, nbl),
            None => compute_energy_and_forces(coords, topo, params),
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
    /// Number of steps taken
    pub steps: usize,
    /// Whether minimization converged
    pub converged: bool,
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

/// Update the consecutive-tiny-change counter and report whether the energy
/// has plateaued. `prev_energy` is `None` on the first call (no comparison
/// possible yet). Returns the new counter value and whether convergence
/// should be declared.
fn check_energy_plateau(
    prev_energy: Option<f64>,
    energy: f64,
    counter: usize,
) -> (usize, bool) {
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

    let mut step_size = 0.01; // initial step size in Å
    let mut prev_energy = initial_energy;
    let mut converged = false;
    let mut steps = 0;
    let mut plateau_counter = 0usize;
    let mut plateau_prev: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        let (energy_res, forces) = nbc.energy_and_forces(&pos, topo, params);
        let cur_energy = energy_res.total;

        // Compute max force magnitude
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

        // Check convergence
        if max_force < gradient_tolerance {
            converged = true;
            break;
        }

        // Plateau fallback
        let (new_counter, plateaued) =
            check_energy_plateau(plateau_prev, cur_energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            converged = true;
            break;
        }
        plateau_prev = Some(cur_energy);

        // Take step along gradient direction
        let mut new_pos = pos.clone();
        for i in 0..n {
            if constrained[i] {
                continue;
            }
            let f_mag = (forces[i][0] * forces[i][0]
                + forces[i][1] * forces[i][1]
                + forces[i][2] * forces[i][2])
            .sqrt();
            if f_mag < 1e-12 {
                continue;
            }
            // Normalize force direction, step by step_size
            let scale = step_size / f_mag;
            new_pos[i][0] += forces[i][0] * scale;
            new_pos[i][1] += forces[i][1] * scale;
            new_pos[i][2] += forces[i][2] * scale;
        }

        let new_energy = nbc.energy(&new_pos, topo, params);

        // Adaptive step size
        if new_energy.total < prev_energy {
            // Accept step, increase step size
            pos = new_pos;
            nbc.refresh(&pos, topo);
            prev_energy = new_energy.total;
            step_size *= 1.2;
            step_size = step_size.min(0.1); // cap at 0.1 Å
        } else {
            // Reject step, decrease step size
            step_size *= 0.5;
            if step_size < 1e-8 {
                break; // can't make progress
            }
        }
    }

    let final_energy = nbc.energy(&pos, topo, params);

    MinimizeResult {
        coords: pos,
        energy: final_energy,
        initial_energy,
        steps,
        converged,
    }
}

// ---------------------------------------------------------------------------
// Conjugate gradient (Polak-Ribiere with automatic restarts)
// ---------------------------------------------------------------------------

/// Dot product of flat 3N force/gradient arrays.
fn dot3n(a: &[[f64; 3]], b: &[[f64; 3]], constrained: &[bool]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        if constrained[i] { continue; }
        sum += a[i][0] * b[i][0] + a[i][1] * b[i][1] + a[i][2] * b[i][2];
    }
    sum
}

/// Scale direction vector: out = a * x + b * y (only unconstrained atoms).
fn axpby(
    out: &mut [[f64; 3]],
    a: f64, x: &[[f64; 3]],
    b: f64, y: &[[f64; 3]],
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
    nbc: &NbCache,
) -> (f64, f64, Vec<[f64; 3]>) {
    let c1 = 1e-4; // Armijo parameter
    let n = pos.len();
    let mut trial = vec![[0.0; 3]; n];

    // Cap the initial alpha so the maximum atom displacement stays inside the
    // NBL buffer region. Without this, alpha=1.0 with a large direction norm
    // (first LBFGS step before history accumulates) produces trial positions
    // many Å away — the cached NBL misses critical close pairs and returns
    // wrong energies. With the cap, every line search trial lands within the
    // cached NBL's valid region, so rebuilding mid-trial is unnecessary.
    let max_disp = 0.8; // Å — well inside the 2 Å NBL buffer
    let mut max_d = 0.0_f64;
    for i in 0..n {
        if constrained[i] { continue; }
        let d2 = direction[i][0].powi(2) + direction[i][1].powi(2) + direction[i][2].powi(2);
        max_d = max_d.max(d2.sqrt());
    }
    let mut alpha = if max_d > 1e-12 { (max_disp / max_d).min(1.0) } else { 1.0 };

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

        // Thanks to the alpha cap above, trials stay within the cached NBL's
        // valid region. No refresh needed here.
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
    let mut converged = false;
    let mut steps = 0;
    let restart_frequency = 3 * n;
    let mut plateau_counter = 0usize;
    let mut plateau_prev: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        // Check convergence: max force magnitude
        let mut max_force = 0.0f64;
        for i in 0..n {
            if constrained[i] { continue; }
            let f2 = old_forces[i][0].powi(2) + old_forces[i][1].powi(2) + old_forces[i][2].powi(2);
            max_force = max_force.max(f2.sqrt());
        }
        if max_force < gradient_tolerance {
            converged = true;
            break;
        }

        // Plateau fallback
        let (new_counter, plateaued) =
            check_energy_plateau(plateau_prev, energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            converged = true;
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
                direction[i] = if constrained[i] { [0.0; 3] } else { old_forces[i] };
            }
            old_gtg = dot3n(&old_forces, &old_forces, constrained);
            continue;
        }

        let (alpha, new_energy, new_pos) = line_search(
            &pos, &direction, grad_dot_dir, energy, topo, params, constrained, &nbc,
        );

        if alpha == 0.0 {
            // Line search failed — we're at a minimum (or stuck)
            break;
        }

        pos = new_pos;
        energy = new_energy;

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
        axpby(&mut direction, 1.0, &new_forces, beta, &old_dir, constrained);

        old_forces = new_forces;
        old_gtg = new_gtg;
    }

    let final_energy = nbc.energy(&pos, topo, params);

    MinimizeResult {
        coords: pos,
        energy: final_energy,
        initial_energy,
        steps,
        converged,
    }
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

    let (_, forces) = nbc.energy_and_forces(&pos, topo, params);
    // Gradient = -force
    let mut grad = negate_forces(&forces, constrained);

    // Storage for correction pairs: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
    let mut s_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    let mut energy = initial_energy;
    let mut converged = false;
    let mut steps = 0;
    let mut plateau_counter = 0usize;
    let mut prev_energy: Option<f64> = None;

    for step in 0..max_steps {
        steps = step + 1;

        // Check convergence: max gradient magnitude
        let mut max_grad = 0.0f64;
        for i in 0..n {
            if constrained[i] { continue; }
            let g2 = grad[i][0].powi(2) + grad[i][1].powi(2) + grad[i][2].powi(2);
            max_grad = max_grad.max(g2.sqrt());
        }
        if max_grad < gradient_tolerance {
            converged = true;
            break;
        }

        // Plateau fallback: declare convergence if energy has stopped moving
        // even though the gradient norm is still hovering above the threshold.
        let (new_counter, plateaued) =
            check_energy_plateau(prev_energy, energy, plateau_counter);
        plateau_counter = new_counter;
        if plateaued {
            converged = true;
            break;
        }
        prev_energy = Some(energy);

        // Two-loop recursion to compute search direction: d = -H_k * g_k
        let direction = lbfgs_two_loop(&grad, &s_hist, &y_hist, &rho_hist, constrained);

        // Line search (only if the direction is actually descent)
        let grad_dot_dir = dot3n_raw(&grad, &direction, constrained);
        let (alpha, new_energy, new_pos) = if grad_dot_dir < 0.0 {
            line_search(
                &pos, &direction, grad_dot_dir, energy, topo, params, constrained, &nbc,
            )
        } else {
            (0.0, energy, pos.clone())
        };

        // Either the direction wasn't descent, or the line search couldn't find
        // an Armijo-acceptable step. In both cases the LBFGS Hessian approximation
        // has become unreliable. Clear the history and take a small steepest-descent
        // step to recover. If even that doesn't make progress, we're stuck — stop.
        if alpha == 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let sd_step = 0.01;
            let prev_energy = energy;
            for i in 0..n {
                if constrained[i] { continue; }
                let g_mag = (grad[i][0].powi(2) + grad[i][1].powi(2) + grad[i][2].powi(2)).sqrt();
                if g_mag > 1e-12 {
                    let scale = sd_step / g_mag;
                    pos[i][0] -= grad[i][0] * scale;
                    pos[i][1] -= grad[i][1] * scale;
                    pos[i][2] -= grad[i][2] * scale;
                }
            }
            nbc.refresh(&pos, topo);
            let (new_e_res, new_forces) = nbc.energy_and_forces(&pos, topo, params);
            // If the SD recovery couldn't even reduce the energy, we're genuinely
            // stuck (saddle point / noisy region) — stop.
            if new_e_res.total >= prev_energy {
                break;
            }
            grad = negate_forces(&new_forces, constrained);
            energy = new_e_res.total;
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

    let final_energy = nbc.energy(&pos, topo, params);

    MinimizeResult {
        coords: pos,
        energy: final_energy,
        initial_energy,
        steps,
        converged,
    }
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
            if constrained[j] { continue; }
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
                if constrained[j] { continue; }
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
            if constrained[j] { continue; }
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
        if constrained[i] { continue; }
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
    let constrained: Vec<bool> = topo
        .atoms
        .iter()
        .map(|a| !a.is_hydrogen)
        .collect();

    steepest_descent(coords, topo, params, max_steps, gradient_tolerance, &constrained)
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
    let constrained: Vec<bool> = topo
        .atoms
        .iter()
        .map(|a| !a.is_hydrogen)
        .collect();

    conjugate_gradient(coords, topo, params, max_steps, gradient_tolerance, &constrained)
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
    let constrained: Vec<bool> = topo
        .atoms
        .iter()
        .map(|a| !a.is_hydrogen)
        .collect();

    lbfgs(coords, topo, params, max_steps, gradient_tolerance, &constrained)
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
        let (counter, plateaued) = check_energy_plateau(Some(-1000.0), -1000.000_01, 0);
        assert_eq!(counter, 1);
        assert!(!plateaued);
    }

    #[test]
    fn patience_threshold_triggers_convergence() {
        // After PLATEAU_PATIENCE-1 tiny-change steps, one more triggers it.
        let (counter, plateaued) =
            check_energy_plateau(Some(-1000.0), -1000.000_01, PLATEAU_PATIENCE - 1);
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
}
