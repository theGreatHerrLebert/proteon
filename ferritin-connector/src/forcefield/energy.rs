//! Energy and gradient computation for the AMBER force field.
//!
//! Computes total energy and per-atom forces (negative gradients)
//! for bond stretching, angle bending, torsions, improper torsions,
//! and nonbonded interactions with cubic switching functions.

use super::neighbor_list::NeighborList;
use super::params::ForceField;
use super::topology::Topology;

/// Energy breakdown by component.
#[derive(Clone, Debug, Default)]
pub struct EnergyResult {
    pub bond_stretch: f64,
    pub angle_bend: f64,
    pub torsion: f64,
    pub improper_torsion: f64,
    pub vdw: f64,
    pub electrostatic: f64,
    pub solvation: f64,
    pub total: f64,
}

// ---------------------------------------------------------------------------
// Cubic switching function (Brooks et al., J. Comput. Chem., 4:191, 1983)
// ---------------------------------------------------------------------------

/// Precomputed switching function parameters.
struct CubicSwitch {
    sq_cutoff: f64,
    sq_cuton: f64,
    inv_range_cubed: f64, // 1 / (sq_cutoff - sq_cuton)^3
}

impl CubicSwitch {
    fn new(cutoff: f64, cuton: f64) -> Self {
        let sq_cutoff = cutoff * cutoff;
        let sq_cuton = cuton * cuton;
        let range = sq_cutoff - sq_cuton;
        let inv_range_cubed = if range > 0.0 { 1.0 / (range * range * range) } else { 0.0 };
        Self { sq_cutoff, sq_cuton, inv_range_cubed }
    }

    /// Returns (switch_value, d_switch / d_r2) for a given squared distance.
    #[inline]
    fn eval(&self, r2: f64) -> (f64, f64) {
        if r2 >= self.sq_cutoff {
            return (0.0, 0.0);
        }
        if r2 <= self.sq_cuton {
            return (1.0, 0.0);
        }
        let diff_off = self.sq_cutoff - r2;
        let diff_on = self.sq_cuton - r2;
        let sw = diff_off * diff_off
            * (self.sq_cutoff + 2.0 * r2 - 3.0 * self.sq_cuton)
            * self.inv_range_cubed;
        // d(sw)/d(r²) = 12 * (sq_cutoff - r²)(sq_cuton - r²) / (sq_cutoff - sq_cuton)³
        // But we need the sign: derivative of sw w.r.t. r², which is negative in the transition.
        // Full derivative: 12 * diff_off * diff_on * inv_range_cubed (this is negative since diff_on < 0)
        let dsw = 12.0 * diff_off * diff_on * self.inv_range_cubed;
        (sw, dsw)
    }
}

/// Compute total energy of the system.
///
/// If `distance_dependent_dielectric` is true, the Coulomb term uses ε = r
/// (divides electrostatic energy by an additional factor of r), providing a
/// simple implicit solvation surrogate.
/// Default atom count above which neighbor list is used automatically.
pub const NBL_AUTO_THRESHOLD: usize = 2000;

pub fn compute_energy(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
) -> EnergyResult {
    compute_energy_auto(coords, topo, params, NBL_AUTO_THRESHOLD)
}

/// Compute energy with configurable neighbor list threshold.
///
/// `nbl_threshold`: use neighbor list when n_atoms > threshold.
/// Set to 0 to always use neighbor list, usize::MAX to never use it.
pub fn compute_energy_auto(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    nbl_threshold: usize,
) -> EnergyResult {
    if coords.len() > nbl_threshold {
        let nbl = NeighborList::build(coords, 15.0, &topo.excluded_pairs, &topo.pairs_14);
        compute_energy_nbl(coords, topo, params, &nbl)
    } else {
        compute_energy_impl(coords, topo, params, false)
    }
}

/// Compute energy with optional distance-dependent dielectric.
pub fn compute_energy_dd(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    distance_dependent_dielectric: bool,
) -> EnergyResult {
    compute_energy_impl(coords, topo, params, distance_dependent_dielectric)
}

fn compute_energy_impl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    distance_dependent_dielectric: bool,
) -> EnergyResult {
    let mut result = EnergyResult::default();

    // --- Bond stretching: E = k * (r - r0)² ---
    for bond in &topo.bonds {
        let ti = &topo.atoms[bond.i].amber_type;
        let tj = &topo.atoms[bond.j].amber_type;
        if let Some(bp) = params.get_bond(ti, tj) {
            let r = dist(&coords[bond.i], &coords[bond.j]);
            let dr = r - bp.r0;
            result.bond_stretch += bp.k * dr * dr;
        }
    }

    // --- Angle bending: E = k * (θ - θ0)² ---
    for angle in &topo.angles {
        let ti = &topo.atoms[angle.i].amber_type;
        let tj = &topo.atoms[angle.j].amber_type;
        let tk = &topo.atoms[angle.k].amber_type;
        if let Some(ap) = params.get_angle(ti, tj, tk) {
            let theta = compute_angle(&coords[angle.i], &coords[angle.j], &coords[angle.k]);
            let dtheta = theta - ap.theta0;
            result.angle_bend += ap.k * dtheta * dtheta;
        }
    }

    // --- Proper torsions: E = (V/div) * (1 + cos(f*φ - φ0)) ---
    for torsion in &topo.torsions {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;
        if let Some(terms) = params.get_torsion(ti, tj, tk, tl) {
            let phi = compute_dihedral(
                &coords[torsion.i],
                &coords[torsion.j],
                &coords[torsion.k],
                &coords[torsion.l],
            );
            for term in terms {
                result.torsion += (term.v / term.div) * (1.0 + (term.f * phi - term.phi0).cos());
            }
        }
    }

    // --- Improper torsions ---
    for torsion in &topo.improper_torsions {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;
        if let Some(terms) = params.get_improper_torsion(ti, tj, tk, tl) {
            let phi = compute_dihedral(
                &coords[torsion.i],
                &coords[torsion.j],
                &coords[torsion.k],
                &coords[torsion.l],
            );
            for term in terms {
                result.improper_torsion +=
                    (term.v / term.div) * (1.0 + (term.f * phi - term.phi0).cos());
            }
        }
    }

    // --- Nonbonded: LJ 12-6 + Coulomb with switching ---
    let n = coords.len();
    let cutoff = 15.0;
    let cuton = 13.0;
    let sw = CubicSwitch::new(cutoff, cuton);
    let coulomb_factor = 332.0; // kcal/mol * Å / e²

    for i in 0..n {
        for j in (i + 1)..n {
            let pair = (i, j);
            if topo.excluded_pairs.contains(&pair) {
                continue;
            }

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;

            if r2 > sw.sq_cutoff || r2 < 0.01 {
                continue;
            }

            let (switch_val, _) = sw.eval(r2);
            let is_14 = topo.pairs_14.contains(&pair);
            let scale_vdw = if is_14 { 1.0 / params.scnb() } else { 1.0 };
            let scale_es = if is_14 { 1.0 / params.scee() } else { 1.0 };

            let r = r2.sqrt();

            // LJ 12-6
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            if let (Some(lj_i), Some(lj_j)) = (params.get_lj(ti), params.get_lj(tj)) {
                let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                let rmin = lj_i.r + lj_j.r;
                if eps > 1e-10 && rmin > 1e-10 {
                    let sr6 = (rmin / r).powi(6);
                    result.vdw += switch_val * scale_vdw * eps * (sr6 * sr6 - 2.0 * sr6);
                }
            }

            // Coulomb (with optional distance-dependent dielectric ε = r)
            let qi = topo.atoms[i].charge;
            let qj = topo.atoms[j].charge;
            if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
                let denom = if distance_dependent_dielectric { r * r } else { r };
                result.electrostatic += switch_val * scale_es * coulomb_factor * qi * qj / denom;
            }
        }
    }

    // --- EEF1 solvation (if enabled) ---
    if params.has_eef1() {
        eef1_energy(coords, topo, params, &mut result.solvation);
    }

    result.total = result.bond_stretch
        + result.angle_bend
        + result.torsion
        + result.improper_torsion
        + result.vdw
        + result.electrostatic
        + result.solvation;
    result
}

/// Compute energy and forces (negative gradient) for all atoms.
///
/// Returns (energy, forces) where forces[i] = -dE/dr_i.
pub fn compute_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
) -> (EnergyResult, Vec<[f64; 3]>) {
    compute_energy_and_forces_auto(coords, topo, params, NBL_AUTO_THRESHOLD)
}

/// Compute energy and forces with configurable neighbor list threshold.
pub fn compute_energy_and_forces_auto(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    nbl_threshold: usize,
) -> (EnergyResult, Vec<[f64; 3]>) {
    if coords.len() > nbl_threshold {
        let nbl = NeighborList::build(coords, 15.0, &topo.excluded_pairs, &topo.pairs_14);
        compute_energy_and_forces_nbl(coords, topo, params, &nbl)
    } else {
        compute_energy_and_forces_impl(coords, topo, params, false)
    }
}

/// Compute energy and forces with optional distance-dependent dielectric.
pub fn compute_energy_and_forces_dd(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    distance_dependent_dielectric: bool,
) -> (EnergyResult, Vec<[f64; 3]>) {
    compute_energy_and_forces_impl(coords, topo, params, distance_dependent_dielectric)
}

fn compute_energy_and_forces_impl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    distance_dependent_dielectric: bool,
) -> (EnergyResult, Vec<[f64; 3]>) {
    let n = coords.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut result = EnergyResult::default();

    // --- Bond stretching ---
    for bond in &topo.bonds {
        let ti = &topo.atoms[bond.i].amber_type;
        let tj = &topo.atoms[bond.j].amber_type;
        if let Some(bp) = params.get_bond(ti, tj) {
            let (i, j) = (bond.i, bond.j);
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 { continue; }

            let dr = r - bp.r0;
            result.bond_stretch += bp.k * dr * dr;

            // Force = -dE/dr = -2k(r-r0) * unit_vector
            let f_mag = -2.0 * bp.k * dr / r;
            let fx = f_mag * dx;
            let fy = f_mag * dy;
            let fz = f_mag * dz;

            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[i][2] -= fz;
            forces[j][0] += fx;
            forces[j][1] += fy;
            forces[j][2] += fz;
        }
    }

    // --- Angle bending ---
    for angle in &topo.angles {
        let ti = &topo.atoms[angle.i].amber_type;
        let tj = &topo.atoms[angle.j].amber_type;
        let tk = &topo.atoms[angle.k].amber_type;
        if let Some(ap) = params.get_angle(ti, tj, tk) {
            let (i, j, k) = (angle.i, angle.j, angle.k);
            let theta = compute_angle(&coords[i], &coords[j], &coords[k]);
            let dtheta = theta - ap.theta0;
            result.angle_bend += ap.k * dtheta * dtheta;

            // Compute angle gradient using cross-product method
            let rji = sub(&coords[i], &coords[j]);
            let rjk = sub(&coords[k], &coords[j]);
            let rji_len = norm(&rji);
            let rjk_len = norm(&rjk);
            if rji_len < 1e-10 || rjk_len < 1e-10 { continue; }

            let cos_theta = dot(&rji, &rjk) / (rji_len * rjk_len);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(1e-12).sqrt();

            let dv = -2.0 * ap.k * dtheta / sin_theta;

            // dθ/dr_i
            let fi = [
                dv * (rjk[0] / (rji_len * rjk_len) - cos_theta * rji[0] / (rji_len * rji_len)),
                dv * (rjk[1] / (rji_len * rjk_len) - cos_theta * rji[1] / (rji_len * rji_len)),
                dv * (rjk[2] / (rji_len * rjk_len) - cos_theta * rji[2] / (rji_len * rji_len)),
            ];
            // dθ/dr_k
            let fk = [
                dv * (rji[0] / (rji_len * rjk_len) - cos_theta * rjk[0] / (rjk_len * rjk_len)),
                dv * (rji[1] / (rji_len * rjk_len) - cos_theta * rjk[1] / (rjk_len * rjk_len)),
                dv * (rji[2] / (rji_len * rjk_len) - cos_theta * rjk[2] / (rjk_len * rjk_len)),
            ];

            for d in 0..3 {
                forces[i][d] += fi[d];
                forces[k][d] += fk[d];
                forces[j][d] -= fi[d] + fk[d];
            }
        }
    }

    // --- Proper torsion energy + gradient ---
    torsion_energy_and_forces(
        &topo.torsions,
        coords,
        topo,
        params,
        &mut result.torsion,
        &mut forces,
        false,
    );

    // --- Improper torsion energy + gradient ---
    torsion_energy_and_forces(
        &topo.improper_torsions,
        coords,
        topo,
        params,
        &mut result.improper_torsion,
        &mut forces,
        true,
    );

    // --- Nonbonded: LJ 12-6 + Coulomb with switching + gradients ---
    let cutoff = 15.0;
    let cuton = 13.0;
    let sw = CubicSwitch::new(cutoff, cuton);
    let coulomb_factor = 332.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let pair = (i, j);
            if topo.excluded_pairs.contains(&pair) {
                continue;
            }

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;

            if r2 > sw.sq_cutoff || r2 < 0.01 {
                continue;
            }

            let (switch_val, dsw_dr2) = sw.eval(r2);
            let is_14 = topo.pairs_14.contains(&pair);
            let scale_vdw = if is_14 { 1.0 / params.scnb() } else { 1.0 };
            let scale_es = if is_14 { 1.0 / params.scee() } else { 1.0 };

            let r = r2.sqrt();
            let inv_r = 1.0 / r;

            // LJ 12-6: E = eps * [(rmin/r)^12 - 2*(rmin/r)^6]
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            if let (Some(lj_i), Some(lj_j)) = (params.get_lj(ti), params.get_lj(tj)) {
                let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                let rmin = lj_i.r + lj_j.r;
                if eps > 1e-10 && rmin > 1e-10 {
                    let sr = rmin * inv_r;
                    let sr6 = sr.powi(6);
                    let sr12 = sr6 * sr6;
                    let e_lj = scale_vdw * eps * (sr12 - 2.0 * sr6);
                    result.vdw += switch_val * e_lj;

                    // d(sw*E)/dr = sw * dE/dr + E * dsw/dr
                    // dsw/dr = dsw/d(r²) * d(r²)/dr = dsw_dr2 * 2r
                    let de_dr = scale_vdw * eps * (-12.0 * sr12 + 12.0 * sr6) * inv_r;
                    let total_de_dr = switch_val * de_dr + e_lj * dsw_dr2 * 2.0 * r;
                    let fx = total_de_dr * dx * inv_r;
                    let fy = total_de_dr * dy * inv_r;
                    let fz = total_de_dr * dz * inv_r;

                    forces[i][0] -= fx;
                    forces[i][1] -= fy;
                    forces[i][2] -= fz;
                    forces[j][0] += fx;
                    forces[j][1] += fy;
                    forces[j][2] += fz;
                }
            }

            // Coulomb: E = k*q1*q2/r (or k*q1*q2/r² with distance-dependent dielectric)
            let qi = topo.atoms[i].charge;
            let qj = topo.atoms[j].charge;
            if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
                let (e_es, de_dr) = if distance_dependent_dielectric {
                    // ε = r: E = k*q1*q2/r², dE/dr = -2*k*q1*q2/r³
                    let e = scale_es * coulomb_factor * qi * qj * inv_r * inv_r;
                    (e, -2.0 * e * inv_r)
                } else {
                    let e = scale_es * coulomb_factor * qi * qj * inv_r;
                    (e, -e * inv_r)
                };
                result.electrostatic += switch_val * e_es;

                let total_de_dr = switch_val * de_dr + e_es * dsw_dr2 * 2.0 * r;
                let fx = total_de_dr * dx * inv_r;
                let fy = total_de_dr * dy * inv_r;
                let fz = total_de_dr * dz * inv_r;

                forces[i][0] -= fx;
                forces[i][1] -= fy;
                forces[i][2] -= fz;
                forces[j][0] += fx;
                forces[j][1] += fy;
                forces[j][2] += fz;
            }
        }
    }

    // --- EEF1 solvation (if enabled) ---
    if params.has_eef1() {
        eef1_energy_and_forces(coords, topo, params, &mut result.solvation, &mut forces);
    }

    result.total = result.bond_stretch
        + result.angle_bend
        + result.torsion
        + result.improper_torsion
        + result.vdw
        + result.electrostatic
        + result.solvation;
    (result, forces)
}

// ---------------------------------------------------------------------------
// Neighbor-list accelerated energy + forces
// ---------------------------------------------------------------------------

/// Compute energy (no forces) using a prebuilt neighbor list.
fn compute_energy_nbl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    nbl: &NeighborList,
) -> EnergyResult {
    // Use the forces version and discard forces — the energy is the same
    let (result, _) = compute_energy_and_forces_nbl(coords, topo, params, nbl);
    result
}

/// Compute energy and forces using a prebuilt neighbor list for O(N) nonbonded.
///
/// The bonded terms (bonds, angles, torsions) are the same as the O(N²) version.
/// Only the nonbonded loop uses the neighbor list.
pub fn compute_energy_and_forces_nbl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    nbl: &NeighborList,
) -> (EnergyResult, Vec<[f64; 3]>) {
    let n = coords.len();
    let mut forces = vec![[0.0f64; 3]; n];
    let mut result = EnergyResult::default();

    // --- Bonded terms (same as full version) ---
    bonded_energy_and_forces(coords, topo, params, &mut result, &mut forces);

    // --- Torsions ---
    torsion_energy_and_forces(
        &topo.torsions, coords, topo, params, &mut result.torsion, &mut forces, false,
    );
    torsion_energy_and_forces(
        &topo.improper_torsions, coords, topo, params, &mut result.improper_torsion, &mut forces, true,
    );

    // --- Nonbonded via neighbor list ---
    let cutoff = 15.0;
    let cuton = 13.0;
    let sw = CubicSwitch::new(cutoff, cuton);
    let coulomb_factor = 332.0;

    for pair in &nbl.pairs {
        let (i, j) = (pair.i, pair.j);
        let dx = coords[i][0] - coords[j][0];
        let dy = coords[i][1] - coords[j][1];
        let dz = coords[i][2] - coords[j][2];
        let r2 = dx * dx + dy * dy + dz * dz;

        if r2 > sw.sq_cutoff || r2 < 0.01 {
            continue;
        }

        let (switch_val, dsw_dr2) = sw.eval(r2);
        let scale_vdw = if pair.is_14 { 1.0 / params.scnb() } else { 1.0 };
        let scale_es = if pair.is_14 { 1.0 / params.scee() } else { 1.0 };

        let r = r2.sqrt();
        let inv_r = 1.0 / r;

        // LJ
        let ti = &topo.atoms[i].amber_type;
        let tj = &topo.atoms[j].amber_type;
        if let (Some(lj_i), Some(lj_j)) = (params.get_lj(ti), params.get_lj(tj)) {
            let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
            let rmin = lj_i.r + lj_j.r;
            if eps > 1e-10 && rmin > 1e-10 {
                let sr = rmin * inv_r;
                let sr6 = sr.powi(6);
                let sr12 = sr6 * sr6;
                let e_lj = scale_vdw * eps * (sr12 - 2.0 * sr6);
                result.vdw += switch_val * e_lj;

                let de_dr = scale_vdw * eps * (-12.0 * sr12 + 12.0 * sr6) * inv_r;
                let total_de_dr = switch_val * de_dr + e_lj * dsw_dr2 * 2.0 * r;
                let fx = total_de_dr * dx * inv_r;
                let fy = total_de_dr * dy * inv_r;
                let fz = total_de_dr * dz * inv_r;

                forces[i][0] -= fx;
                forces[i][1] -= fy;
                forces[i][2] -= fz;
                forces[j][0] += fx;
                forces[j][1] += fy;
                forces[j][2] += fz;
            }
        }

        // Coulomb
        let qi = topo.atoms[i].charge;
        let qj = topo.atoms[j].charge;
        if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
            let e_es = scale_es * coulomb_factor * qi * qj * inv_r;
            result.electrostatic += switch_val * e_es;

            let de_dr = -e_es * inv_r;
            let total_de_dr = switch_val * de_dr + e_es * dsw_dr2 * 2.0 * r;
            let fx = total_de_dr * dx * inv_r;
            let fy = total_de_dr * dy * inv_r;
            let fz = total_de_dr * dz * inv_r;

            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[i][2] -= fz;
            forces[j][0] += fx;
            forces[j][1] += fy;
            forces[j][2] += fz;
        }
    }

    result.total = result.bond_stretch
        + result.angle_bend
        + result.torsion
        + result.improper_torsion
        + result.vdw
        + result.electrostatic;
    (result, forces)
}

/// Compute only bonded energy terms + forces (bonds, angles).
/// Used internally to avoid code duplication between full and neighbor-list paths.
fn bonded_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    result: &mut EnergyResult,
    forces: &mut [[f64; 3]],
) {
    // Bond stretching
    for bond in &topo.bonds {
        let ti = &topo.atoms[bond.i].amber_type;
        let tj = &topo.atoms[bond.j].amber_type;
        if let Some(bp) = params.get_bond(ti, tj) {
            let (i, j) = (bond.i, bond.j);
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 { continue; }
            let dr = r - bp.r0;
            result.bond_stretch += bp.k * dr * dr;
            let f_mag = -2.0 * bp.k * dr / r;
            let fx = f_mag * dx;
            let fy = f_mag * dy;
            let fz = f_mag * dz;
            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[i][2] -= fz;
            forces[j][0] += fx;
            forces[j][1] += fy;
            forces[j][2] += fz;
        }
    }

    // Angle bending
    for angle in &topo.angles {
        let ti = &topo.atoms[angle.i].amber_type;
        let tj = &topo.atoms[angle.j].amber_type;
        let tk = &topo.atoms[angle.k].amber_type;
        if let Some(ap) = params.get_angle(ti, tj, tk) {
            let (i, j, k) = (angle.i, angle.j, angle.k);
            let theta = compute_angle(&coords[i], &coords[j], &coords[k]);
            let dtheta = theta - ap.theta0;
            result.angle_bend += ap.k * dtheta * dtheta;

            let rji = sub(&coords[i], &coords[j]);
            let rjk = sub(&coords[k], &coords[j]);
            let rji_len = norm(&rji);
            let rjk_len = norm(&rjk);
            if rji_len < 1e-10 || rjk_len < 1e-10 { continue; }

            let cos_theta = dot(&rji, &rjk) / (rji_len * rjk_len);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(1e-12).sqrt();
            let dv = -2.0 * ap.k * dtheta / sin_theta;

            let fi = [
                dv * (rjk[0] / (rji_len * rjk_len) - cos_theta * rji[0] / (rji_len * rji_len)),
                dv * (rjk[1] / (rji_len * rjk_len) - cos_theta * rji[1] / (rji_len * rji_len)),
                dv * (rjk[2] / (rji_len * rjk_len) - cos_theta * rji[2] / (rji_len * rji_len)),
            ];
            let fk = [
                dv * (rji[0] / (rji_len * rjk_len) - cos_theta * rjk[0] / (rjk_len * rjk_len)),
                dv * (rji[1] / (rji_len * rjk_len) - cos_theta * rjk[1] / (rjk_len * rjk_len)),
                dv * (rji[2] / (rji_len * rjk_len) - cos_theta * rjk[2] / (rjk_len * rjk_len)),
            ];
            for d in 0..3 {
                forces[i][d] += fi[d];
                forces[k][d] += fk[d];
                forces[j][d] -= fi[d] + fk[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EEF1 implicit solvation (Lazaridis & Karplus, Proteins 35:133, 1999)
//
// E_solv = Σ_i ΔG_ref_i                                   (self-solvation)
//        + Σ_{i<j} f(r_ij)                                (pair exclusion)
//
// f(r_ij) = -0.5 * V_j * ΔG_free_i * exp(-(r - R_min_i)²/σ_i²) / (σ_i * π√π * r²)
//         + -0.5 * V_i * ΔG_free_j * exp(-(r - R_min_j)²/σ_j²) / (σ_j * π√π * r²)
// ---------------------------------------------------------------------------

const PI_SQRT_PI: f64 = 5.568_327_996_831_708; // π * √π

/// EEF1 solvation energy (energy only, no forces).
fn eef1_energy(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    solvation: &mut f64,
) {
    let n = coords.len();
    let cutoff_sq = 9.0 * 9.0; // 9 Å cutoff for solvation

    // Self-solvation: Σ ΔG_ref
    for atom in &topo.atoms {
        if atom.is_hydrogen { continue; }
        if let Some(eef) = params.get_eef1(&atom.amber_type) {
            *solvation += eef.dg_ref;
        }
    }

    // Pair exclusion
    for i in 0..n {
        if topo.atoms[i].is_hydrogen { continue; }
        let eef_i = match params.get_eef1(&topo.atoms[i].amber_type) {
            Some(e) => e,
            None => continue,
        };

        for j in (i + 1)..n {
            if topo.atoms[j].is_hydrogen { continue; }
            let eef_j = match params.get_eef1(&topo.atoms[j].amber_type) {
                Some(e) => e,
                None => continue,
            };

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > cutoff_sq || r2 < 0.01 { continue; }

            let r = r2.sqrt();

            // Contribution from atom j excluding atom i's solvation
            if eef_i.dg_free.abs() > 1e-10 && eef_j.volume > 1e-10 {
                let dr = (r - eef_i.r_min) / eef_i.sigma;
                *solvation += -0.5 * eef_j.volume * eef_i.dg_free
                    * (-dr * dr).exp() / (eef_i.sigma * PI_SQRT_PI * r2);
            }

            // Contribution from atom i excluding atom j's solvation
            if eef_j.dg_free.abs() > 1e-10 && eef_i.volume > 1e-10 {
                let dr = (r - eef_j.r_min) / eef_j.sigma;
                *solvation += -0.5 * eef_i.volume * eef_j.dg_free
                    * (-dr * dr).exp() / (eef_j.sigma * PI_SQRT_PI * r2);
            }
        }
    }
}

/// EEF1 solvation energy + forces.
fn eef1_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    solvation: &mut f64,
    forces: &mut [[f64; 3]],
) {
    let n = coords.len();
    let cutoff_sq = 9.0 * 9.0;

    // Self-solvation (no force contribution — constant per atom)
    for atom in &topo.atoms {
        if atom.is_hydrogen { continue; }
        if let Some(eef) = params.get_eef1(&atom.amber_type) {
            *solvation += eef.dg_ref;
        }
    }

    // Pair exclusion + forces
    for i in 0..n {
        if topo.atoms[i].is_hydrogen { continue; }
        let eef_i = match params.get_eef1(&topo.atoms[i].amber_type) {
            Some(e) => e,
            None => continue,
        };

        for j in (i + 1)..n {
            if topo.atoms[j].is_hydrogen { continue; }
            let eef_j = match params.get_eef1(&topo.atoms[j].amber_type) {
                Some(e) => e,
                None => continue,
            };

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > cutoff_sq || r2 < 0.01 { continue; }

            let r = r2.sqrt();
            let inv_r = 1.0 / r;
            let mut de_dr_total = 0.0;

            // i's solvation excluded by j
            if eef_i.dg_free.abs() > 1e-10 && eef_j.volume > 1e-10 {
                let dr_i = (r - eef_i.r_min) / eef_i.sigma;
                let exp_i = (-dr_i * dr_i).exp();
                let norm_i = eef_i.sigma * PI_SQRT_PI;
                let e_i = -0.5 * eef_j.volume * eef_i.dg_free * exp_i / (norm_i * r2);
                *solvation += e_i;

                // dE/dr = E * (-2*dr/(σ*r) - 2/r) = E * (-2/(σ*r)) * (dr + σ/1) ...
                // More carefully: E = C * exp(-dr²) / r²
                // dE/dr = C * [-2*dr/(σ) * exp(-dr²) / r² + exp(-dr²) * (-2/r³)]
                //       = E * [-2*dr/σ + (-2/r)] = E * (-2) * (dr/σ + 1/r)
                // Wait, dr = (r - R_min) / σ, so d(dr)/dr = 1/σ
                // d(exp(-dr²))/dr = -2*dr * (1/σ) * exp(-dr²)
                // d(1/r²)/dr = -2/r³
                // dE/dr = C * [d(exp(-dr²))/dr * 1/r² + exp(-dr²) * d(1/r²)/dr]
                //       = C * [-2*dr/σ * exp(-dr²)/r² + exp(-dr²) * (-2/r³)]
                //       = C * exp(-dr²) * [-2*dr/(σ*r²) - 2/r³]
                //       = E * [-2*dr/σ - 2/r] (simplifying with E = C*exp/r²)
                // Hmm wait: E = C * exp(-dr²) / r², so E * r² = C * exp(-dr²)
                // Then: dE/dr = C * exp(-dr²) * [-2dr/(σ*r²) - 2/r³]
                //             = (E * r²) * [-2dr/(σ*r²) - 2/r³]
                //             = E * [-2*dr/σ - 2*r²/r³]
                //             = E * [-2*dr/σ - 2/r]
                de_dr_total += e_i * (-2.0 * dr_i / eef_i.sigma - 2.0 * inv_r);
            }

            // j's solvation excluded by i
            if eef_j.dg_free.abs() > 1e-10 && eef_i.volume > 1e-10 {
                let dr_j = (r - eef_j.r_min) / eef_j.sigma;
                let exp_j = (-dr_j * dr_j).exp();
                let norm_j = eef_j.sigma * PI_SQRT_PI;
                let e_j = -0.5 * eef_i.volume * eef_j.dg_free * exp_j / (norm_j * r2);
                *solvation += e_j;

                de_dr_total += e_j * (-2.0 * dr_j / eef_j.sigma - 2.0 * inv_r);
            }

            // Apply force along r_ij
            let fx = de_dr_total * dx * inv_r;
            let fy = de_dr_total * dy * inv_r;
            let fz = de_dr_total * dz * inv_r;

            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[i][2] -= fz;
            forces[j][0] += fx;
            forces[j][1] += fy;
            forces[j][2] += fz;
        }
    }
}

// ---------------------------------------------------------------------------
// Torsion energy + gradient (shared by proper and improper)
//
// Uses the BALL/BiochemicalAlgorithms.jl cross-product formula:
//   dEdt =  (dE/dϕ / (|n1|² * |b2|)) * cross(n1, b2)
//   dEdu = -(dE/dϕ / (|n2|² * |b2|)) * cross(n2, b2)
//   F1 = cross(dEdt, b2)
//   F2 = cross(r13, dEdt) + cross(dEdu, r34)
//   F3 = cross(r21, dEdt) + cross(r24, dEdu)
//   F4 = cross(dEdu, b2)
// ---------------------------------------------------------------------------

fn torsion_energy_and_forces(
    torsion_list: &[super::topology::Torsion],
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    energy_accum: &mut f64,
    forces: &mut [[f64; 3]],
    is_improper: bool,
) {
    for torsion in torsion_list {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;

        let terms = if is_improper {
            params.get_improper_torsion(ti, tj, tk, tl)
        } else {
            params.get_torsion(ti, tj, tk, tl)
        };
        let terms = match terms {
            Some(t) => t,
            None => continue,
        };

        let (ai, aj, ak, al) = (torsion.i, torsion.j, torsion.k, torsion.l);
        let p0 = &coords[ai];
        let p1 = &coords[aj];
        let p2 = &coords[ak];
        let p3 = &coords[al];

        // Vectors along the torsion
        let a21 = sub(p0, p1); // p0 - p1
        let a23 = sub(p2, p1); // p2 - p1
        let a34 = sub(p3, p2); // p3 - p2

        let n1 = cross(&a23, &a21); // cross(b2, b1)
        let n2 = cross(&a23, &a34); // cross(b2, b3)

        let len_n1 = norm(&n1);
        let len_n2 = norm(&n2);
        if len_n1 < 1e-10 || len_n2 < 1e-10 {
            continue;
        }

        let cos_phi = (dot(&n1, &n2) / (len_n1 * len_n2)).clamp(-1.0, 1.0);
        let phi = cos_phi.acos();

        // Compute energy
        let mut e_torsion = 0.0;
        let mut de_dphi = 0.0;
        for term in terms {
            e_torsion += (term.v / term.div) * (1.0 + (term.f * phi - term.phi0).cos());
            de_dphi += -(term.v / term.div) * term.f * (term.f * phi - term.phi0).sin();
        }
        *energy_accum += e_torsion;

        // Determine sign of phi using direction = dot(cross(n1, n2), a23)
        let n1xn2 = cross(&n1, &n2);
        let direction = dot(&n1xn2, &a23);
        if direction > 0.0 {
            de_dphi = -de_dphi;
        }

        // Force distribution (BALL formula)
        let len_a23 = norm(&a23);
        if len_a23 < 1e-10 { continue; }

        let n1_cross_a23 = cross(&n1, &a23);
        let n2_cross_a23 = cross(&n2, &a23);

        let scale_t = de_dphi / (len_n1 * len_n1 * len_a23);
        let scale_u = -de_dphi / (len_n2 * len_n2 * len_a23);

        let dedt = [
            scale_t * n1_cross_a23[0],
            scale_t * n1_cross_a23[1],
            scale_t * n1_cross_a23[2],
        ];
        let dedu = [
            scale_u * n2_cross_a23[0],
            scale_u * n2_cross_a23[1],
            scale_u * n2_cross_a23[2],
        ];

        // F1 = cross(dEdt, a23)
        let f1 = cross(&dedt, &a23);
        // F4 = cross(dEdu, a23)
        let f4 = cross(&dedu, &a23);

        let a13 = sub(p2, p0); // p2 - p0
        let a24 = sub(p3, p1); // p3 - p1

        // F2 = cross(a13, dEdt) + cross(dEdu, a34)
        let f2a = cross(&a13, &dedt);
        let f2b = cross(&dedu, &a34);

        // F3 = cross(a21, dEdt) + cross(a24, dEdu)
        let f3a = cross(&a21, &dedt);
        let f3b = cross(&a24, &dedu);

        for d in 0..3 {
            forces[ai][d] += f1[d];
            forces[aj][d] += f2a[d] + f2b[d];
            forces[ak][d] += f3a[d] + f3b[d];
            forces[al][d] += f4[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[inline]
fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn compute_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let ba = sub(a, b);
    let bc = sub(c, b);
    let cos_theta = dot(&ba, &bc) / (norm(&ba) * norm(&bc));
    cos_theta.clamp(-1.0, 1.0).acos()
}

fn compute_dihedral(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
    let b1 = sub(p1, p0);
    let b2 = sub(p2, p1);
    let b3 = sub(p3, p2);

    let n1 = cross(&b1, &b2);
    let n2 = cross(&b2, &b3);

    let n1_len = norm(&n1);
    let n2_len = norm(&n2);
    if n1_len < 1e-10 || n2_len < 1e-10 {
        return 0.0;
    }

    let x = dot(&n1, &n2) / (n1_len * n2_len);
    let b2_len = norm(&b2);
    let m1 = cross(&n1, &[b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len]);
    let y = dot(&m1, &n2) / (norm(&m1).max(1e-10) * n2_len);

    y.atan2(x)
}
