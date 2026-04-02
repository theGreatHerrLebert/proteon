//! Energy and gradient computation for the AMBER force field.
//!
//! Computes total energy and per-atom forces (negative gradients)
//! for bond stretching, angle bending, torsions, and nonbonded interactions.

use super::params::AmberParams;
use super::topology::Topology;

/// Energy breakdown by component.
#[derive(Clone, Debug, Default)]
pub struct EnergyResult {
    pub bond_stretch: f64,
    pub angle_bend: f64,
    pub torsion: f64,
    pub vdw: f64,
    pub electrostatic: f64,
    pub total: f64,
}

/// Compute total energy of the system.
pub fn compute_energy(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &AmberParams,
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

    // --- Torsions: E = (V/div) * (1 + cos(f*φ - φ0)) ---
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

    // --- Nonbonded: LJ 12-6 + Coulomb ---
    let n = coords.len();
    let cutoff_sq = 15.0 * 15.0; // 15 Å cutoff
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

            if r2 > cutoff_sq || r2 < 0.01 {
                continue;
            }

            let is_14 = topo.pairs_14.contains(&pair);
            let scale_vdw = if is_14 { 1.0 / params.scnb } else { 1.0 };
            let scale_es = if is_14 { 1.0 / params.scee } else { 1.0 };

            let r = r2.sqrt();

            // LJ 12-6
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            if let (Some(lj_i), Some(lj_j)) = (params.get_lj(ti), params.get_lj(tj)) {
                let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                let rmin = lj_i.r + lj_j.r;
                if eps > 1e-10 && rmin > 1e-10 {
                    let sr6 = (rmin / r).powi(6);
                    result.vdw += scale_vdw * eps * (sr6 * sr6 - 2.0 * sr6);
                }
            }

            // Coulomb
            let qi = topo.atoms[i].charge;
            let qj = topo.atoms[j].charge;
            if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
                result.electrostatic += scale_es * coulomb_factor * qi * qj / r;
            }
        }
    }

    result.total =
        result.bond_stretch + result.angle_bend + result.torsion + result.vdw + result.electrostatic;
    result
}

/// Compute energy and forces (negative gradient) for all atoms.
///
/// Returns (energy, forces) where forces[i] = -dE/dr_i.
pub fn compute_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &AmberParams,
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

    // --- Nonbonded: LJ 12-6 + Coulomb with gradients ---
    let cutoff_sq = 15.0 * 15.0;
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

            if r2 > cutoff_sq || r2 < 0.01 {
                continue;
            }

            let is_14 = topo.pairs_14.contains(&pair);
            let scale_vdw = if is_14 { 1.0 / params.scnb } else { 1.0 };
            let scale_es = if is_14 { 1.0 / params.scee } else { 1.0 };

            let r = r2.sqrt();
            let inv_r = 1.0 / r;

            // LJ 12-6: E = eps * [(rmin/r)^12 - 2*(rmin/r)^6]
            // dE/dr = eps * [-12*rmin^12/r^13 + 12*rmin^6/r^7]
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            if let (Some(lj_i), Some(lj_j)) = (params.get_lj(ti), params.get_lj(tj)) {
                let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                let rmin = lj_i.r + lj_j.r;
                if eps > 1e-10 && rmin > 1e-10 {
                    let sr = rmin * inv_r;
                    let sr6 = sr.powi(6);
                    let sr12 = sr6 * sr6;
                    result.vdw += scale_vdw * eps * (sr12 - 2.0 * sr6);

                    // Force: F = -dE/dr * (r_vec / r)
                    let de_dr = scale_vdw * eps * (-12.0 * sr12 + 12.0 * sr6) * inv_r;
                    let fx = de_dr * dx * inv_r;
                    let fy = de_dr * dy * inv_r;
                    let fz = de_dr * dz * inv_r;

                    forces[i][0] -= fx;
                    forces[i][1] -= fy;
                    forces[i][2] -= fz;
                    forces[j][0] += fx;
                    forces[j][1] += fy;
                    forces[j][2] += fz;
                }
            }

            // Coulomb: E = k*q1*q2/r, dE/dr = -k*q1*q2/r²
            let qi = topo.atoms[i].charge;
            let qj = topo.atoms[j].charge;
            if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
                let e_es = scale_es * coulomb_factor * qi * qj * inv_r;
                result.electrostatic += e_es;

                let de_dr = -e_es * inv_r; // -k*q1*q2/r²
                let fx = de_dr * dx * inv_r;
                let fy = de_dr * dy * inv_r;
                let fz = de_dr * dz * inv_r;

                forces[i][0] -= fx;
                forces[i][1] -= fy;
                forces[i][2] -= fz;
                forces[j][0] += fx;
                forces[j][1] += fy;
                forces[j][2] += fz;
            }
        }
    }

    // --- Torsion energy (energy only, gradient is complex) ---
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

    result.total =
        result.bond_stretch + result.angle_bend + result.torsion + result.vdw + result.electrostatic;
    (result, forces)
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
