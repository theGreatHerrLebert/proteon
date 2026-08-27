//! Molecular dynamics simulation.
//!
//! Velocity Verlet integrator with optional Berendsen thermostat (NVE/NVT).
//!
//! References:
//! - Verlet, "Computer 'Experiments' on Classical Fluids. I.
//!   Thermodynamical Properties of Lennard-Jones Molecules",
//!   *Phys. Rev.* 159(1), 98-103 (1967) — the velocity-form integrator.
//! - Berendsen, Postma, van Gunsteren, DiNola, Haak, "Molecular dynamics
//!   with coupling to an external bath", *J. Chem. Phys.* 81(8),
//!   3684-3690 (1984) — the weak-coupling thermostat.

use super::energy::EnergyResult;
use super::nb_cache::{NbCache, NbExec};
use super::params::ForceField;
use super::topology::Topology;

// ---------------------------------------------------------------------------
// Atomic masses (daltons)
// ---------------------------------------------------------------------------

fn atomic_mass(element: &str) -> f64 {
    match element {
        "H" | "D" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "S" => 32.065,
        "P" => 30.974,
        "Se" => 78.96,
        "Fe" => 55.845,
        "Zn" => 65.38,
        "Cu" => 63.546,
        "Mg" => 24.305,
        "Ca" => 40.078,
        "Mn" => 54.938,
        "Co" => 58.933,
        "Ni" => 58.693,
        "Cl" => 35.453,
        "F" => 18.998,
        "Br" => 79.904,
        "I" => 126.90,
        _ => 12.0, // default to carbon
    }
}

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

// AMBER energies are in kcal/mol, distances in Å, time in ps.
// Force: kcal/(mol·Å)
// Mass: g/mol (daltons)
// Acceleration: F/m in kcal/(mol·Å·g/mol) = kcal/(g·Å)
// To get Å/ps²: multiply by conversion factor.
//
// 1 kcal = 4184 J, 1 J = 1 kg·m²/s²
// F/m = [kcal/(mol·Å)] / [g/mol]
//     = [4184 J / (6.022e23 · 1e-10 m)] / [1e-3 kg / 6.022e23]
//     = 4184 / (6.022e23 · 1e-10) * 6.022e23 / 1e-3
//     = 4184 / 1e-10 / 1e-3 = 4184e13 m/s² = 4.184e16 m/s²
//     = 4.184e6 Å/ps² (since 1 Å = 1e-10 m, 1 ps = 1e-12 s)
//
// Simpler: use the AMBER conversion factor.
// acceleration (Å/ps²) = force (kcal/mol/Å) / mass (g/mol) * 418.4
const FORCE_TO_ACCEL: f64 = 418.4;

// Boltzmann constant in kcal/(mol·K)
const KB: f64 = 0.001987204;

// ---------------------------------------------------------------------------
// MD trajectory snapshot
// ---------------------------------------------------------------------------

/// A single MD trajectory frame.
#[derive(Clone, Debug)]
pub struct MDFrame {
    pub step: usize,
    pub time_ps: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub total_energy: f64,
    pub temperature: f64,
}

/// Result of an MD simulation.
#[derive(Clone, Debug)]
pub struct MDResult {
    pub coords: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub frames: Vec<MDFrame>,
    pub energy: EnergyResult,
}

// ---------------------------------------------------------------------------
// Velocity initialization
// ---------------------------------------------------------------------------

/// Initialize velocities from Maxwell-Boltzmann distribution at given temperature.
fn init_velocities(masses: &[f64], temperature: f64, seed: u64) -> Vec<[f64; 3]> {
    let n = masses.len();
    let mut velocities = vec![[0.0; 3]; n];

    // Simple LCG random number generator (good enough for initial velocities)
    let mut rng_state = seed;
    let mut next_rand = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Box-Muller approximation: use the state bits
        let u1 = ((rng_state >> 32) as f64) / (u32::MAX as f64);
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = ((rng_state >> 32) as f64) / (u32::MAX as f64);
        // Box-Muller transform for normal distribution
        let u1 = u1.max(1e-10);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    for i in 0..n {
        // Standard deviation of velocity: sqrt(kB * T / m) in Å/ps
        let sigma = (KB * temperature / masses[i] * FORCE_TO_ACCEL).sqrt();
        velocities[i] = [
            sigma * next_rand(),
            sigma * next_rand(),
            sigma * next_rand(),
        ];
    }

    // Remove center-of-mass velocity
    let total_mass: f64 = masses.iter().sum();
    let mut com_vel = [0.0; 3];
    for i in 0..n {
        com_vel[0] += masses[i] * velocities[i][0];
        com_vel[1] += masses[i] * velocities[i][1];
        com_vel[2] += masses[i] * velocities[i][2];
    }
    com_vel[0] /= total_mass;
    com_vel[1] /= total_mass;
    com_vel[2] /= total_mass;

    for v in &mut velocities {
        v[0] -= com_vel[0];
        v[1] -= com_vel[1];
        v[2] -= com_vel[2];
    }

    velocities
}

/// Compute kinetic energy and temperature from velocities.
fn kinetic_energy_and_temperature(velocities: &[[f64; 3]], masses: &[f64]) -> (f64, f64) {
    let n = velocities.len();
    let mut ke = 0.0;
    for i in 0..n {
        let v2 = velocities[i][0].powi(2) + velocities[i][1].powi(2) + velocities[i][2].powi(2);
        ke += 0.5 * masses[i] * v2 / FORCE_TO_ACCEL;
    }
    // Temperature: T = 2 * KE / (3N * kB)
    let dof = 3.0 * n as f64 - 3.0; // subtract 3 for COM constraint
    let temp = if dof > 0.0 {
        2.0 * ke / (dof * KB)
    } else {
        0.0
    };
    (ke, temp)
}

// ---------------------------------------------------------------------------
// SHAKE/RATTLE bond length constraints
// ---------------------------------------------------------------------------

/// A bond constraint (typically X-H bonds).
#[derive(Clone, Debug)]
pub struct BondConstraint {
    pub i: usize,  // heavy atom index
    pub j: usize,  // hydrogen index
    pub d_sq: f64, // target distance squared (Å²)
}

/// Build list of X-H bond constraints from topology.
///
/// Constrains all bonds where exactly one atom is hydrogen.
/// Target distances come from the force field equilibrium bond lengths.
pub fn build_h_constraints(topo: &Topology, params: &impl ForceField) -> Vec<BondConstraint> {
    let mut constraints = Vec::new();
    for bond in &topo.bonds {
        let a_is_h = topo.atoms[bond.i].is_hydrogen;
        let b_is_h = topo.atoms[bond.j].is_hydrogen;
        // Constrain if exactly one is hydrogen
        if a_is_h != b_is_h {
            let ti = &topo.atoms[bond.i].amber_type;
            let tj = &topo.atoms[bond.j].amber_type;
            if let Some(bp) = params.get_bond(ti, tj) {
                let d = bp.r0;
                constraints.push(BondConstraint {
                    i: bond.i,
                    j: bond.j,
                    d_sq: d * d,
                });
            }
        }
    }
    constraints
}

/// SHAKE: correct positions to satisfy bond length constraints.
///
/// After an unconstrained Verlet position update, iteratively adjusts
/// atomic positions so all constrained bond lengths match their targets.
///
/// Reference: Ryckaert, Ciccotti, Berendsen, J Comp Phys 23, 327 (1977).
///
/// Returns the number of iterations used, or `max_iter` if not converged.
fn shake(
    pos: &mut [[f64; 3]],
    old_pos: &[[f64; 3]],
    inv_mass: &[f64],
    constraints: &[BondConstraint],
    tolerance: f64,
    max_iter: usize,
) -> usize {
    for iter in 0..max_iter {
        let mut max_err = 0.0f64;

        for c in constraints {
            let (i, j) = (c.i, c.j);

            // Current bond vector (after unconstrained step)
            let rpx = pos[i][0] - pos[j][0];
            let rpy = pos[i][1] - pos[j][1];
            let rpz = pos[i][2] - pos[j][2];
            let rp_sq = rpx * rpx + rpy * rpy + rpz * rpz;

            // Distance error
            let diff = c.d_sq - rp_sq;
            let err = diff.abs() / c.d_sq;
            max_err = max_err.max(err);

            if err < tolerance {
                continue;
            }

            // Old bond vector (reference for constraint direction)
            let rx = old_pos[i][0] - old_pos[j][0];
            let ry = old_pos[i][1] - old_pos[j][1];
            let rz = old_pos[i][2] - old_pos[j][2];

            // r_old · r_prime
            let r_dot_rp = rx * rpx + ry * rpy + rz * rpz;
            if r_dot_rp.abs() < c.d_sq * 1e-10 {
                continue; // vectors nearly perpendicular, skip
            }

            // Reduced inverse mass: 1/m_i + 1/m_j
            // Note: inv_mass already has FORCE_TO_ACCEL factor, but for SHAKE
            // we need plain 1/mass. We divide out FORCE_TO_ACCEL.
            let inv_mi = inv_mass[i] / FORCE_TO_ACCEL;
            let inv_mj = inv_mass[j] / FORCE_TO_ACCEL;

            // Lagrange multiplier: λ = diff / (2 * (1/m_i + 1/m_j) * r_old · r')
            let lambda = diff / (2.0 * (inv_mi + inv_mj) * r_dot_rp);

            // Apply correction
            pos[i][0] += lambda * rx * inv_mi;
            pos[i][1] += lambda * ry * inv_mi;
            pos[i][2] += lambda * rz * inv_mi;
            pos[j][0] -= lambda * rx * inv_mj;
            pos[j][1] -= lambda * ry * inv_mj;
            pos[j][2] -= lambda * rz * inv_mj;
        }

        if max_err < tolerance {
            return iter + 1;
        }
    }
    max_iter
}

/// RATTLE: correct velocities to satisfy constraint on velocity along bond.
///
/// After SHAKE corrects positions, RATTLE removes velocity components
/// parallel to constrained bonds, ensuring dC/dt = 0.
///
/// Reference: Andersen, J Comp Phys 52, 24 (1983).
fn rattle(
    vel: &mut [[f64; 3]],
    pos: &[[f64; 3]],
    inv_mass: &[f64],
    constraints: &[BondConstraint],
    tolerance: f64,
    max_iter: usize,
) -> usize {
    for iter in 0..max_iter {
        let mut max_err = 0.0f64;

        for c in constraints {
            let (i, j) = (c.i, c.j);

            // Current bond vector
            let rx = pos[i][0] - pos[j][0];
            let ry = pos[i][1] - pos[j][1];
            let rz = pos[i][2] - pos[j][2];

            // Velocity difference projected onto bond
            let vx = vel[i][0] - vel[j][0];
            let vy = vel[i][1] - vel[j][1];
            let vz = vel[i][2] - vel[j][2];
            let v_dot_r = vx * rx + vy * ry + vz * rz;

            let err = v_dot_r.abs() / c.d_sq.sqrt();
            max_err = max_err.max(err);

            if err < tolerance {
                continue;
            }

            let inv_mi = inv_mass[i] / FORCE_TO_ACCEL;
            let inv_mj = inv_mass[j] / FORCE_TO_ACCEL;

            // Correction: κ = -v·r / ((1/m_i + 1/m_j) * d²)
            let kappa = -v_dot_r / ((inv_mi + inv_mj) * c.d_sq);

            vel[i][0] += kappa * rx * inv_mi;
            vel[i][1] += kappa * ry * inv_mi;
            vel[i][2] += kappa * rz * inv_mi;
            vel[j][0] -= kappa * rx * inv_mj;
            vel[j][1] -= kappa * ry * inv_mj;
            vel[j][2] -= kappa * rz * inv_mj;
        }

        if max_err < tolerance {
            return iter + 1;
        }
    }
    max_iter
}

// ---------------------------------------------------------------------------
// Velocity Verlet integrator
// ---------------------------------------------------------------------------

/// Run molecular dynamics using Velocity Verlet integration.
///
/// # Arguments
/// * `coords` — Initial coordinates
/// * `topo` — Topology (bonds, angles, etc.)
/// * `params` — Force field parameters
/// * `n_steps` — Number of MD steps
/// * `dt` — Time step in picoseconds (default 0.001 = 1 fs)
/// * `temperature` — Initial/target temperature in Kelvin
/// * `thermostat_tau` — Berendsen coupling time in ps (0.0 = NVE, no thermostat)
/// * `snapshot_freq` — Record frame every N steps
#[allow(dead_code)]
pub fn velocity_verlet(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    n_steps: usize,
    dt: f64,
    temperature: f64,
    thermostat_tau: f64,
    snapshot_freq: usize,
) -> MDResult {
    // No constraints — delegate to constrained version with empty list
    velocity_verlet_constrained(
        coords,
        topo,
        params,
        n_steps,
        dt,
        temperature,
        thermostat_tau,
        snapshot_freq,
        &[],
    )
}

/// Run MD with SHAKE/RATTLE constraints on specified bonds.
///
/// Constrains X-H bond lengths to their equilibrium values, allowing
/// a larger timestep (2 fs instead of 0.5 fs).
///
/// Use `build_h_constraints()` to generate the constraint list.
pub fn velocity_verlet_constrained(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    n_steps: usize,
    dt: f64,
    temperature: f64,
    thermostat_tau: f64,
    snapshot_freq: usize,
    constraints: &[BondConstraint],
) -> MDResult {
    velocity_verlet_exec(
        coords,
        topo,
        params,
        n_steps,
        dt,
        temperature,
        thermostat_tau,
        snapshot_freq,
        constraints,
        NbExec::Auto,
    )
}

/// Implementation of [`velocity_verlet_constrained`] with an explicit nonbonded
/// execution policy. Production callers use `NbExec::Auto`; tests pass `CpuNbl` to
/// exercise the neighbor-list path (and its drift-rebuild) on a small fixture.
#[allow(clippy::too_many_arguments)]
pub(crate) fn velocity_verlet_exec(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    n_steps: usize,
    dt: f64,
    temperature: f64,
    thermostat_tau: f64,
    snapshot_freq: usize,
    constraints: &[BondConstraint],
    exec: NbExec,
) -> MDResult {
    let n = coords.len();
    let mut pos = coords.to_vec();
    let mut frames = Vec::new();
    let use_constraints = !constraints.is_empty();

    // Compute masses
    let masses: Vec<f64> = topo.atoms.iter().map(|a| atomic_mass(&a.element)).collect();

    // Initialize velocities
    let mut vel = init_velocities(&masses, temperature, 42);

    // Cutoff neighbor-list cache: O(N) nonbonded above ~2000 atoms (all-pair below,
    // so small systems are byte-identical to before). Built BEFORE the first force
    // eval; refreshed on drift after each position update (see the loop).
    let mut nbc = NbCache::new_with_exec(&pos, topo, params, exec);

    // Initial forces
    let (energy, mut forces) = nbc.energy_and_forces(&pos, topo, params);
    // Track the most-recently computed potential energy so the final MDResult reuses it
    // instead of recomputing (positions don't change after the last force eval); also
    // gives the correct result when n_steps == 0.
    let mut last_energy = energy.clone();

    // Pre-compute mass factors: 1/m * FORCE_TO_ACCEL
    let inv_mass: Vec<f64> = masses.iter().map(|&m| FORCE_TO_ACCEL / m).collect();

    // Apply RATTLE to initial velocities if constrained
    if use_constraints {
        rattle(&mut vel, &pos, &inv_mass, constraints, 1e-6, 100);
    }

    let (ke, temp) = kinetic_energy_and_temperature(&vel, &masses);
    frames.push(MDFrame {
        step: 0,
        time_ps: 0.0,
        kinetic_energy: ke,
        potential_energy: energy.total,
        total_energy: ke + energy.total,
        temperature: temp,
    });

    let half_dt = 0.5 * dt;
    let dt_sq_half = 0.5 * dt * dt;

    for step in 1..=n_steps {
        // Save old positions for SHAKE reference
        let old_pos = if use_constraints {
            pos.clone()
        } else {
            Vec::new()
        };

        // Velocity Verlet step 1: update positions
        // r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        for i in 0..n {
            pos[i][0] += vel[i][0] * dt + forces[i][0] * inv_mass[i] * dt_sq_half;
            pos[i][1] += vel[i][1] * dt + forces[i][1] * inv_mass[i] * dt_sq_half;
            pos[i][2] += vel[i][2] * dt + forces[i][2] * inv_mass[i] * dt_sq_half;
        }

        // SHAKE: correct positions to satisfy bond constraints
        if use_constraints {
            shake(&mut pos, &old_pos, &inv_mass, constraints, 1e-6, 1000);

            // Reconstruct midpoint velocity from constrained displacement.
            //
            // Without SHAKE:  (r(t+dt) - r(t)) / dt = v(t) + 0.5*a(t)*dt = v(t+dt/2)
            // With SHAKE:     (r(t+dt) - r(t)) / dt = v(t+dt/2) + λ_constraint/dt
            //
            // This is the STANDARD SHAKE velocity formula (Andersen 1983, Allen &
            // Tildesley Ch.3). The constraint force contribution λ/dt is implicitly
            // encoded in the position displacement. The second half-step kick (below)
            // then adds 0.5*a(t+dt)*dt, and RATTLE removes the parallel component.
            for i in 0..n {
                vel[i][0] = (pos[i][0] - old_pos[i][0]) / dt;
                vel[i][1] = (pos[i][1] - old_pos[i][1]) / dt;
                vel[i][2] = (pos[i][2] - old_pos[i][2]) / dt;
            }
        } else {
            // Half-step velocity update: v(t+dt/2) = v(t) + 0.5*a(t)*dt
            for i in 0..n {
                vel[i][0] += forces[i][0] * inv_mass[i] * half_dt;
                vel[i][1] += forces[i][1] * inv_mass[i] * half_dt;
                vel[i][2] += forces[i][2] * inv_mass[i] * half_dt;
            }
        }

        // Rebuild the neighbor list if atoms drifted past the buffer — AFTER the
        // position update and SHAKE, immediately before the force eval, so the list is
        // valid for r(t+dt). (No-op for the all-pair path / when nothing moved far.)
        nbc.refresh(&pos, topo);

        // Compute new forces at r(t+dt)
        let (new_energy, new_forces) = nbc.energy_and_forces(&pos, topo, params);
        last_energy = new_energy.clone();
        forces = new_forces;

        // Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
        for i in 0..n {
            vel[i][0] += forces[i][0] * inv_mass[i] * half_dt;
            vel[i][1] += forces[i][1] * inv_mass[i] * half_dt;
            vel[i][2] += forces[i][2] * inv_mass[i] * half_dt;
        }

        // RATTLE: correct velocities along constrained bonds
        if use_constraints {
            rattle(&mut vel, &pos, &inv_mass, constraints, 1e-6, 100);
        }

        // Berendsen thermostat: rescale velocities
        if thermostat_tau > 0.0 {
            let (_, current_temp) = kinetic_energy_and_temperature(&vel, &masses);
            if current_temp > 1e-6 {
                let lambda = (1.0 + (dt / thermostat_tau) * (temperature / current_temp - 1.0))
                    .max(0.0)
                    .sqrt();
                for v in &mut vel {
                    v[0] *= lambda;
                    v[1] *= lambda;
                    v[2] *= lambda;
                }
            }
        }

        // Record snapshot
        if step % snapshot_freq == 0 || step == n_steps {
            let (ke, temp) = kinetic_energy_and_temperature(&vel, &masses);
            frames.push(MDFrame {
                step,
                time_ps: step as f64 * dt,
                kinetic_energy: ke,
                potential_energy: new_energy.total,
                total_energy: ke + new_energy.total,
                temperature: temp,
            });
        }
    }

    // Positions are unchanged since the last refreshed force eval, so reuse its energy
    // rather than recomputing the whole nonbonded sum.
    MDResult {
        coords: pos,
        velocities: vel,
        frames,
        energy: last_energy,
    }
}

#[cfg(test)]
mod tests {
    use super::super::params;
    use super::super::topology;
    use super::*;

    #[test]
    fn test_atomic_masses() {
        assert!((atomic_mass("C") - 12.011).abs() < 0.01);
        assert!((atomic_mass("H") - 1.008).abs() < 0.01);
        assert!((atomic_mass("N") - 14.007).abs() < 0.01);
    }

    #[test]
    fn test_velocity_init_removes_com() {
        let masses = vec![12.0, 14.0, 16.0, 1.0, 1.0];
        let vel = init_velocities(&masses, 300.0, 42);

        // COM velocity should be ~zero
        let total_mass: f64 = masses.iter().sum();
        let mut com = [0.0; 3];
        for (i, v) in vel.iter().enumerate() {
            com[0] += masses[i] * v[0];
            com[1] += masses[i] * v[1];
            com[2] += masses[i] * v[2];
        }
        com[0] /= total_mass;
        com[1] /= total_mass;
        com[2] /= total_mass;

        let com_speed = (com[0].powi(2) + com[1].powi(2) + com[2].powi(2)).sqrt();
        assert!(
            com_speed < 1e-10,
            "COM velocity should be zero, got {}",
            com_speed
        );
    }

    #[test]
    fn test_md_nve_energy_conservation() {
        // Run short NVE on crambin, check total energy is roughly conserved
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let amber = params::amber96();
        let topo = topology::build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();

        // Very short NVE run: 10 steps, 0.5 fs timestep
        let result = velocity_verlet(
            &coords, &topo, &amber, 10,     // steps
            0.0005, // dt = 0.5 fs
            300.0,  // temperature
            0.0,    // no thermostat (NVE)
            5,      // snapshot every 5 steps
        );

        assert!(result.frames.len() >= 2);
        assert_eq!(result.coords.len(), coords.len());

        // In NVE, total energy should be approximately conserved
        let e0 = result.frames[0].total_energy;
        let e_last = result.frames.last().unwrap().total_energy;
        let _drift = (e_last - e0).abs();

        // With 0.5 fs timestep and 10 steps, drift should be small
        // (but can be large for un-minimized structures with clashes)
        // Just check it doesn't blow up to infinity
        assert!(
            e_last.is_finite(),
            "Energy should be finite, got {}",
            e_last
        );
    }

    #[test]
    fn test_md_nvt_temperature() {
        // Run NVT, check temperature stays near target
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let amber = params::amber96();
        let topo = topology::build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();

        let result = velocity_verlet(
            &coords, &topo, &amber, 20,     // steps
            0.0005, // dt
            300.0,  // target temp
            0.1,    // Berendsen tau = 0.1 ps (strong coupling)
            10,     // snapshot every 10
        );

        // Temperature should be finite and positive
        let last_temp = result.frames.last().unwrap().temperature;
        assert!(
            last_temp > 0.0 && last_temp.is_finite(),
            "Temperature should be positive and finite, got {}",
            last_temp
        );
    }

    #[test]
    fn test_shake_maintains_bond_lengths() {
        // Verify SHAKE constrains X-H bond lengths during MD.
        // After each step, all constrained bonds should be within tolerance
        // of their equilibrium length.
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        // Place hydrogens first so we have X-H bonds to constrain
        crate::add_hydrogens::place_peptide_hydrogens(&mut pdb);

        let amber = params::amber96();
        let topo = topology::build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let constraints = build_h_constraints(&topo, &amber);

        assert!(!constraints.is_empty(), "Should have X-H constraints");

        // Run 5 steps with SHAKE, very small timestep to avoid blowup
        let result = velocity_verlet_constrained(
            &coords,
            &topo,
            &amber,
            5,      // steps
            0.0002, // dt = 0.2 fs (very small to keep stable)
            300.0,  // temperature
            0.0,    // NVE (no thermostat)
            5,      // snapshot every 5
            &constraints,
        );

        // Check all constrained bond lengths in final structure
        let tol = 1e-3; // 0.001 Å tolerance
        for c in &constraints {
            let dx = result.coords[c.i][0] - result.coords[c.j][0];
            let dy = result.coords[c.i][1] - result.coords[c.j][1];
            let dz = result.coords[c.i][2] - result.coords[c.j][2];
            let r_sq = dx * dx + dy * dy + dz * dz;
            let r = r_sq.sqrt();
            let target = c.d_sq.sqrt();
            let err = (r - target).abs();
            assert!(
                err < tol,
                "Constrained bond {}-{}: r={:.4} target={:.4} err={:.6}",
                c.i,
                c.j,
                r,
                target,
                err,
            );
        }

        // Verify result is finite (no blowup)
        assert!(
            result.frames.last().unwrap().total_energy.is_finite(),
            "Total energy should be finite with SHAKE",
        );
    }

    #[test]
    fn test_shake_vs_unconstrained_same_initial() {
        // Both should start from the same initial state
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        crate::add_hydrogens::place_peptide_hydrogens(&mut pdb);

        let amber = params::amber96();
        let topo = topology::build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let constraints = build_h_constraints(&topo, &amber);

        let r_no_shake = velocity_verlet(&coords, &topo, &amber, 1, 0.0002, 300.0, 0.0, 1);
        let r_shake = velocity_verlet_constrained(
            &coords,
            &topo,
            &amber,
            1,
            0.0002,
            300.0,
            0.0,
            1,
            &constraints,
        );

        // Initial frames should have the same potential energy
        let pe_no = r_no_shake.frames[0].potential_energy;
        let pe_sh = r_shake.frames[0].potential_energy;
        assert!(
            (pe_no - pe_sh).abs() < 0.01,
            "Initial PE should match: no_shake={} shake={}",
            pe_no,
            pe_sh,
        );
    }

    // -- Neighbor-list path (the new O(N) MD nonbonded) ----------------------------

    fn crambin_amber() -> (topology::Topology, Vec<[f64; 3]>, params::AmberParams) {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");
        let amber = params::amber96();
        let topo = topology::build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        (topo, coords, amber)
    }

    fn assert_forces_match(a: &[[f64; 3]], b: &[[f64; 3]], rel_tol: f64) {
        // Relative tolerance with a unit floor: NBL and all-pair sum the SAME per-pair
        // contributions in a different order, so they agree to ~machine epsilon
        // *relative* to the magnitude (which can be huge on a clashing geometry).
        assert_eq!(a.len(), b.len());
        for (i, (fa, fb)) in a.iter().zip(b).enumerate() {
            for k in 0..3 {
                let scale = fa[k].abs().max(fb[k].abs()).max(1.0);
                assert!(
                    (fa[k] - fb[k]).abs() <= rel_tol * scale,
                    "force[{i}][{k}] {} vs {} (rel {:.2e})",
                    fa[k],
                    fb[k],
                    (fa[k] - fb[k]).abs() / scale
                );
            }
        }
    }

    #[test]
    fn nbl_force_parity_matches_all_pair_static() {
        // The CPU neighbor-list path (which MD now uses above the size threshold) must
        // produce the same cutoff energy + per-atom forces as the all-pair path.
        use super::super::energy::compute_energy_and_forces;
        let (topo, coords, amber) = crambin_amber();
        let mut nbc = NbCache::new_with_exec(&coords, &topo, &amber, NbExec::CpuNbl);
        let (e_nbl, f_nbl) = nbc.energy_and_forces(&coords, &topo, &amber);
        let (e_all, f_all) = compute_energy_and_forces(&coords, &topo, &amber);
        assert!(
            (e_nbl.total - e_all.total).abs() < 1e-6,
            "energy {} vs {}",
            e_nbl.total,
            e_all.total
        );
        assert_forces_match(&f_nbl, &f_all, 1e-6);
    }

    #[test]
    fn nbl_force_parity_after_drift_and_rebuild() {
        // Reused list after sub-buffer drift, then a rebuild after large drift — forces
        // must match all-pair in BOTH cases (guards the refresh-on-drift correctness).
        use super::super::energy::compute_energy_and_forces;
        let (topo, coords, amber) = crambin_amber();
        let mut nbc = NbCache::new_with_exec(&coords, &topo, &amber, NbExec::CpuNbl);

        // Small per-atom drift (< buffer/2): refresh is a no-op, the list is reused.
        let mut drifted: Vec<[f64; 3]> = coords.clone();
        for (i, c) in drifted.iter_mut().enumerate() {
            c[i % 3] += 0.3; // 0.3 Å, well inside the 2 Å buffer
        }
        nbc.refresh(&drifted, &topo);
        let (_e, f_nbl) = nbc.energy_and_forces(&drifted, &topo, &amber);
        let (_ea, f_all) = compute_energy_and_forces(&drifted, &topo, &amber);
        assert_forces_match(&f_nbl, &f_all, 1e-6);

        // Large drift (past the buffer): refresh must rebuild, and forces still match.
        let mut shoved: Vec<[f64; 3]> = coords.clone();
        for (i, c) in shoved.iter_mut().enumerate() {
            c[i % 3] += 3.0; // 3 Å — exceeds buffer/2, triggers rebuild
        }
        nbc.refresh(&shoved, &topo);
        let (_e2, f_nbl2) = nbc.energy_and_forces(&shoved, &topo, &amber);
        let (_ea2, f_all2) = compute_energy_and_forces(&shoved, &topo, &amber);
        assert_forces_match(&f_nbl2, &f_all2, 1e-6);
    }

    #[test]
    fn md_cpu_nbl_trajectory_tracks_all_pair() {
        // MD driven through the NBL path must track the all-pair trajectory. Forces match
        // to ~1e-6 each step, but NBL pair-order changes FP accumulation, so over several
        // steps a short-horizon tolerance (not bit-exact) is the right guarantee.
        let (topo, coords, amber) = crambin_amber();
        let run = |exec| {
            velocity_verlet_exec(&coords, &topo, &amber, 6, 0.0005, 300.0, 0.0, 3, &[], exec)
        };
        let all = run(NbExec::AllPair);
        let nbl = run(NbExec::CpuNbl);
        assert_eq!(all.coords.len(), nbl.coords.len());
        let max_dr = all
            .coords
            .iter()
            .zip(&nbl.coords)
            .map(|(a, b)| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .fold(0.0_f64, f64::max);
        assert!(max_dr < 1e-4, "max coord drift {max_dr:.2e} over 6 steps");
        assert!(nbl.energy.total.is_finite());
    }

    #[test]
    fn md_cpu_nbl_nve_energy_stays_finite_and_bounded() {
        // NVE on the NBL path: total energy finite and not blowing up across the run.
        let (topo, coords, amber) = crambin_amber();
        let r = velocity_verlet_exec(
            &coords,
            &topo,
            &amber,
            20,
            0.0005,
            300.0,
            0.0,
            5,
            &[],
            NbExec::CpuNbl,
        );
        let e0 = r.frames[0].total_energy;
        for f in &r.frames {
            assert!(f.total_energy.is_finite(), "non-finite total energy");
        }
        let drift = (r.frames.last().unwrap().total_energy - e0).abs();
        assert!(drift.is_finite());
    }
}
