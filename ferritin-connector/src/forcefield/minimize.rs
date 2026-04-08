//! Energy minimization algorithms.
//!
//! Steepest descent with adaptive step size. Simple but robust
//! for hydrogen position optimization.

use super::energy::{compute_energy, compute_energy_and_forces, EnergyResult};
use super::params::AmberParams;
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
    params: &AmberParams,
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
    params: &AmberParams,
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
    params: &AmberParams,
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
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Minimize only hydrogen positions (freeze all heavy atoms).
pub fn minimize_hydrogens(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &AmberParams,
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
pub fn minimize_hydrogens_cg(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &AmberParams,
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
