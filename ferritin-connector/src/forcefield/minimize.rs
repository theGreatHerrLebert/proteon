//! Energy minimization algorithms.
//!
//! Steepest descent with adaptive step size. Simple but robust
//! for hydrogen position optimization.

use super::energy::{compute_energy, compute_energy_and_forces, EnergyResult};
use super::params::ForceField;
use super::topology::Topology;

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

    let initial_e = compute_energy(&pos, topo, params);
    let initial_energy = initial_e.total;

    let mut step_size = 0.01; // initial step size in Å
    let mut prev_energy = initial_energy;
    let mut converged = false;
    let mut steps = 0;

    for step in 0..max_steps {
        steps = step + 1;

        let (_energy, forces) = compute_energy_and_forces(&pos, topo, params);

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

        let new_energy = compute_energy(&new_pos, topo, params);

        // Adaptive step size
        if new_energy.total < prev_energy {
            // Accept step, increase step size
            pos = new_pos;
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

    let final_energy = compute_energy(&pos, topo, params);

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
fn line_search(
    pos: &[[f64; 3]],
    direction: &[[f64; 3]],
    grad_dot_dir: f64,
    current_energy: f64,
    topo: &Topology,
    params: &impl ForceField,
    constrained: &[bool],
) -> (f64, f64, Vec<[f64; 3]>) {
    let c1 = 1e-4; // Armijo parameter
    let mut alpha = 1.0;
    let n = pos.len();
    let mut trial = vec![[0.0; 3]; n];

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

        let e = compute_energy(&trial, topo, params);

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

    let initial_e = compute_energy(&pos, topo, params);
    let initial_energy = initial_e.total;

    // First force evaluation
    let (_, forces) = compute_energy_and_forces(&pos, topo, params);

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
            &pos, &direction, grad_dot_dir, energy, topo, params, constrained,
        );

        if alpha == 0.0 {
            // Line search failed — we're at a minimum (or stuck)
            break;
        }

        pos = new_pos;
        energy = new_energy;

        // Compute new forces at the new position
        let (_, new_forces) = compute_energy_and_forces(&pos, topo, params);

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

    let final_energy = compute_energy(&pos, topo, params);

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

    let initial_e = compute_energy(&pos, topo, params);
    let initial_energy = initial_e.total;

    let (_, forces) = compute_energy_and_forces(&pos, topo, params);
    // Gradient = -force
    let mut grad = negate_forces(&forces, constrained);

    // Storage for correction pairs: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
    let mut s_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<[f64; 3]>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    let mut energy = initial_energy;
    let mut converged = false;
    let mut steps = 0;

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

        // Two-loop recursion to compute search direction: d = -H_k * g_k
        let direction = lbfgs_two_loop(&grad, &s_hist, &y_hist, &rho_hist, constrained);

        // Line search
        let grad_dot_dir = dot3n_raw(&grad, &direction, constrained);
        if grad_dot_dir >= 0.0 {
            // Not a descent direction — restart (use steepest descent step)
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            // Take a small steepest descent step
            let sd_step = 0.01;
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
            let (_, new_forces) = compute_energy_and_forces(&pos, topo, params);
            grad = negate_forces(&new_forces, constrained);
            energy = compute_energy(&pos, topo, params).total;
            continue;
        }

        let (alpha, new_energy, new_pos) = line_search(
            &pos, &direction, grad_dot_dir, energy, topo, params, constrained,
        );

        if alpha == 0.0 {
            break; // line search failed
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

        // New gradient
        let (_, new_forces) = compute_energy_and_forces(&pos, topo, params);
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

    let final_energy = compute_energy(&pos, topo, params);

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
