//! Energy and gradient computation for the AMBER and CHARMM force fields.
//!
//! Computes total energy and per-atom forces (negative gradients)
//! for bond stretching, angle bending, torsions, improper torsions,
//! and nonbonded interactions with cubic switching functions.
//!
//! References (functional-form sources; parameter citations are in
//! `params.rs`):
//! - Cornell et al., *J. Am. Chem. Soc.* 117, 5179-5197 (1995) — AMBER
//!   energy-function conventions.
//! - Brooks et al., "CHARMM: A program for macromolecular energy,
//!   minimization, and dynamics calculations", *J. Comput. Chem.* 4(2),
//!   187-217 (1983) — CHARMM energy-function conventions.
//! - Lazaridis & Karplus, *Proteins* 35, 133-152 (1999) — EEF1 solvation
//!   Gaussian-integral form evaluated alongside the bonded terms.

use super::neighbor_list::NeighborList;
use super::params::{ForceField, LJParam};
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
        let inv_range_cubed = if range > 0.0 {
            1.0 / (range * range * range)
        } else {
            0.0
        };
        Self {
            sq_cutoff,
            sq_cuton,
            inv_range_cubed,
        }
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
        let sw = diff_off
            * diff_off
            * (self.sq_cutoff + 2.0 * r2 - 3.0 * self.sq_cuton)
            * self.inv_range_cubed;
        // sw(r²) = (a - r²)² (a + 2r² - 3b) / (a - b)³  where a=sq_cutoff, b=sq_cuton
        // d(sw)/d(r²) = 6 (a - r²)(b - r²) / (a - b)³
        // (Negative in the transition since b - r² < 0 < a - r².)
        let dsw = 6.0 * diff_off * diff_on * self.inv_range_cubed;
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
        let nbl = NeighborList::build(
            coords,
            params.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        compute_energy_nbl(coords, topo, params, &nbl)
    } else {
        compute_energy_impl(
            coords,
            topo,
            params,
            params.uses_distance_dependent_dielectric(),
        )
    }
}

/// Compute energy with optional distance-dependent dielectric.
#[allow(dead_code)]
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
    // Try cosine path first (AMBER); if unavailable, try harmonic
    // (CHARMM). The two parameter tables are mutually exclusive
    // per-FF — AMBER doesn't ship harmonic impropers, CHARMM doesn't
    // ship cosine ones — so the dispatch is unambiguous in practice.
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
            continue;
        }
        // CHARMM harmonic improper: central atom is at slot K, but
        // BALL's convention puts central at slot 1. Use the same
        // unsigned phi = acos(cos_phi) BALL uses, so finite-difference
        // gradients of this energy match the analytical force in
        // `harmonic_improper_energy_and_forces`. See that function
        // for the full derivation.
        if let Some(terms) = params.get_harmonic_improper(tk, ti, tj, tl) {
            let p1 = &coords[torsion.k]; // central
            let p2 = &coords[torsion.i];
            let p3 = &coords[torsion.j];
            let p4 = &coords[torsion.l];
            let ba = sub(p2, p1);
            let bc = sub(p2, p3);
            let bd = sub(p2, p4);
            let bcxba = cross(&bc, &ba);
            let bcxbd = cross(&bc, &bd);
            let len_bcxba = norm(&bcxba);
            let len_bcxbd = norm(&bcxbd);
            if len_bcxba < 1e-10 || len_bcxbd < 1e-10 {
                continue;
            }
            let cos_phi = (bcxba[0] * bcxbd[0] + bcxba[1] * bcxbd[1] + bcxba[2] * bcxbd[2])
                / (len_bcxba * len_bcxbd);
            let phi = cos_phi.clamp(-1.0, 1.0).acos();
            for term in terms {
                let dphi = phi - term.phi0;
                result.improper_torsion += term.k * dphi * dphi;
            }
        }
    }

    // --- Nonbonded: LJ 12-6 + Coulomb with switching ---
    let n = coords.len();
    let cutoff = params.nonbonded_cutoff();
    let cuton = params.switching_on();
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
            let scale_es = if is_14 { 1.0 / params.scee() } else { 1.0 };

            let r = r2.sqrt();

            // LJ 12-6 — for 1-4 pairs prefer the FF's special [LennardJones14]
            // table when it has one (CHARMM convention: separate (R, eps) per
            // 1-4 atom-type, no scnb scaling). Fall back to regular LJ scaled
            // by 1/scnb (AMBER convention) when the table is absent.
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            let lj_pair: Option<(&LJParam, &LJParam, f64)> = if is_14 {
                match (params.get_lj_14(ti), params.get_lj_14(tj)) {
                    (Some(a), Some(b)) => Some((a, b, 1.0)),
                    _ => params
                        .get_lj(ti)
                        .zip(params.get_lj(tj))
                        .map(|(a, b)| (a, b, 1.0 / params.scnb())),
                }
            } else {
                params
                    .get_lj(ti)
                    .zip(params.get_lj(tj))
                    .map(|(a, b)| (a, b, 1.0))
            };
            // CHARMM aromatic-ring para-diagonal exclusion: zero LJ
            // but keep electrostatic. See `Topology::lj_excluded_pairs`.
            let lj_excluded = topo.lj_excluded_pairs.contains(&pair);
            if !lj_excluded {
                if let Some((lj_i, lj_j, scale_vdw)) = lj_pair {
                    let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                    let rmin = lj_i.r + lj_j.r;
                    if eps > 1e-10 && rmin > 1e-10 {
                        let sr6 = (rmin / r).powi(6);
                        result.vdw += switch_val * scale_vdw * eps * (sr6 * sr6 - 2.0 * sr6);
                    }
                }
            }

            // Coulomb (with optional distance-dependent dielectric ε = r)
            let qi = topo.atoms[i].charge;
            let qj = topo.atoms[j].charge;
            if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
                let denom = if distance_dependent_dielectric {
                    r * r
                } else {
                    r
                };
                result.electrostatic += switch_val * scale_es * coulomb_factor * qi * qj / denom;
            }
        }
    }

    // --- EEF1 solvation (if enabled) ---
    if params.has_eef1() {
        eef1_energy(coords, topo, params, &mut result.solvation);
    }

    // --- OBC GB implicit solvent (if enabled) ---
    // IMPORTANT: OpenMM's GBSAOBCForce is hardcoded to `ObcTypeII` in
    // ReferenceCalcGBSAOBCForceKernel::initialize — i.e. regardless of
    // whether the XML is named `amber96_obc.xml` or `charmm36_obc2.xml`,
    // the α/β/γ applied are OBC2 (α=1.0, β=0.8, γ=4.85). The XML only
    // carries per-atom (radius, scale) pairs. So for oracle parity we
    // always use OBC2 here. OBC1 is kept on `ObcGbParams` as a
    // programmatic option but is NOT the AMBER default.
    if params.has_obc_gb() {
        let obc = super::gb_obc::ObcGbParams::obc2();
        super::gb_obc::gb_obc_energy(coords, topo, params, &obc, &mut result.solvation);
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
        let nbl = NeighborList::build(
            coords,
            params.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        compute_energy_and_forces_nbl(coords, topo, params, &nbl)
    } else {
        compute_energy_and_forces_impl(
            coords,
            topo,
            params,
            params.uses_distance_dependent_dielectric(),
        )
    }
}

/// Compute energy and forces with optional distance-dependent dielectric.
#[allow(dead_code)]
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
            if r < 1e-10 {
                continue;
            }

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
            if rji_len < 1e-10 || rjk_len < 1e-10 {
                continue;
            }

            let cos_theta = dot(&rji, &rjk) / (rji_len * rjk_len);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(1e-12).sqrt();

            // F_i = -dE/dr_i = +2k(θ-θ₀)/sin(θ) × [rjk/(|rji||rjk|) - cos(θ)·rji/|rji|²]
            // Positive sign: the code below ADDS fi to forces[i], and fi must BE F_i.
            let dv = 2.0 * ap.k * dtheta / sin_theta;

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
    //
    // The cosine path (AMBER) and harmonic path (CHARMM) are mutually
    // exclusive at the per-FF level — AMBER's parameter table holds
    // only cosine entries, CHARMM's holds only harmonic entries — but
    // the topology can carry impropers under either label, so we run
    // both compute paths and let the parameter dispatch decide
    // per-torsion. `torsion_energy_and_forces(is_improper=true)` will
    // skip torsions whose atom-type quadruple isn't in the cosine
    // table; `harmonic_improper_energy_and_forces` skips ones not in
    // the harmonic table.
    torsion_energy_and_forces(
        &topo.improper_torsions,
        coords,
        topo,
        params,
        &mut result.improper_torsion,
        &mut forces,
        true,
    );
    harmonic_improper_energy_and_forces(
        &topo.improper_torsions,
        coords,
        topo,
        params,
        &mut result.improper_torsion,
        &mut forces,
    );

    // --- Nonbonded: LJ 12-6 + Coulomb with switching + gradients ---
    let cutoff = params.nonbonded_cutoff();
    let cuton = params.switching_on();
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
            let scale_es = if is_14 { 1.0 / params.scee() } else { 1.0 };

            let r = r2.sqrt();
            let inv_r = 1.0 / r;

            // LJ 12-6: E = eps * [(rmin/r)^12 - 2*(rmin/r)^6]
            // For 1-4 pairs prefer the FF's [LennardJones14] table when present
            // (CHARMM); fall back to regular LJ scaled by 1/scnb (AMBER).
            let ti = &topo.atoms[i].amber_type;
            let tj = &topo.atoms[j].amber_type;
            let lj_pair: Option<(&LJParam, &LJParam, f64)> = if is_14 {
                match (params.get_lj_14(ti), params.get_lj_14(tj)) {
                    (Some(a), Some(b)) => Some((a, b, 1.0)),
                    _ => params
                        .get_lj(ti)
                        .zip(params.get_lj(tj))
                        .map(|(a, b)| (a, b, 1.0 / params.scnb())),
                }
            } else {
                params
                    .get_lj(ti)
                    .zip(params.get_lj(tj))
                    .map(|(a, b)| (a, b, 1.0))
            };
            // CHARMM aromatic-ring para-diagonal exclusion: zero LJ
            // (energy + force) but keep electrostatic.
            let lj_excluded = topo.lj_excluded_pairs.contains(&pair);
            if !lj_excluded {
                if let Some((lj_i, lj_j, scale_vdw)) = lj_pair {
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

    // --- OBC GB implicit solvent (if enabled) ---
    if params.has_obc_gb() {
        let obc = super::gb_obc::ObcGbParams::obc2();
        super::gb_obc::gb_obc_energy_and_forces(
            coords,
            topo,
            params,
            &obc,
            &mut result.solvation,
            &mut forces,
        );
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
pub fn compute_energy_nbl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    nbl: &NeighborList,
) -> EnergyResult {
    // Use the forces version and discard forces — the energy is the
    // same. The CHARMM harmonic improper energy is added inside
    // compute_energy_and_forces_nbl now, so no extra step is needed
    // here.
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
        &topo.torsions,
        coords,
        topo,
        params,
        &mut result.torsion,
        &mut forces,
        false,
    );
    torsion_energy_and_forces(
        &topo.improper_torsions,
        coords,
        topo,
        params,
        &mut result.improper_torsion,
        &mut forces,
        true,
    );
    harmonic_improper_energy_and_forces(
        &topo.improper_torsions,
        coords,
        topo,
        params,
        &mut result.improper_torsion,
        &mut forces,
    );

    // --- Nonbonded via neighbor list ---
    let cutoff = params.nonbonded_cutoff();
    let cuton = params.switching_on();
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
        let is_14 = pair.is_14;
        let scale_es = if is_14 { 1.0 / params.scee() } else { 1.0 };

        let r = r2.sqrt();
        let inv_r = 1.0 / r;

        // LJ — for 1-4 pairs prefer the FF's [LennardJones14] table when
        // present (CHARMM); fall back to regular LJ scaled by 1/scnb (AMBER).
        let ti = &topo.atoms[i].amber_type;
        let tj = &topo.atoms[j].amber_type;
        let lj_pair: Option<(&LJParam, &LJParam, f64)> = if is_14 {
            match (params.get_lj_14(ti), params.get_lj_14(tj)) {
                (Some(a), Some(b)) => Some((a, b, 1.0)),
                _ => params
                    .get_lj(ti)
                    .zip(params.get_lj(tj))
                    .map(|(a, b)| (a, b, 1.0 / params.scnb())),
            }
        } else {
            params
                .get_lj(ti)
                .zip(params.get_lj(tj))
                .map(|(a, b)| (a, b, 1.0))
        };
        // CHARMM aromatic-ring para-diagonal exclusion: zero LJ but
        // keep electrostatic. See `Topology::lj_excluded_pairs`.
        let lj_excluded = topo.lj_excluded_pairs.contains(&(i.min(j), i.max(j)));
        if !lj_excluded {
            if let Some((lj_i, lj_j, scale_vdw)) = lj_pair {
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
        }

        // Coulomb (with optional distance-dependent dielectric ε = r,
        // CHARMM19+EEF1 convention).
        let qi = topo.atoms[i].charge;
        let qj = topo.atoms[j].charge;
        if qi.abs() > 1e-10 && qj.abs() > 1e-10 {
            let dist_dep = params.uses_distance_dependent_dielectric();
            let (e_es, de_dr) = if dist_dep {
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

    // --- EEF1 solvation (if enabled) ---
    //
    // The neighbor-list path previously omitted EEF1 entirely — both the
    // energy contribution and the force contribution — leaving
    // `result.solvation` at its default 0.0 for any structure that crossed
    // NBL_AUTO_THRESHOLD (2000 atoms). This silently disabled implicit
    // solvation for every large structure in batch_prepare / minimize, and
    // left solvation out of the `total` sum too. Fix both: call the
    // energy+forces EEF1 kernel and include solvation in the total.
    //
    // Use the NBL-accelerated EEF1 kernel: iterates the neighbor-list
    // pairs (O(N·k) where k is avg neighbors within 15 Å) instead of
    // the O(N²) all-pairs loop. EEF1's 9 Å cutoff is a subset of the
    // NBL's 15 Å cutoff so all relevant pairs are covered. On a 1907-
    // atom structure this cuts EEF1 from ~50 ms to ~6 ms per eval;
    // on 30k atoms it's 75× faster, which makes LBFGS heavy-atom
    // minimization on large proteins tractable.
    if params.has_eef1() {
        eef1_energy_and_forces_nbl(
            coords,
            topo,
            params,
            &mut result.solvation,
            &mut forces,
            nbl,
        );
    }

    // --- OBC GB implicit solvent (if enabled) ---
    // NBL path uses the all-pair GB for now (the only real performance
    // concern is on systems >>2000 atoms; a neighbor-list GB path is a
    // Phase C follow-up). The math is identical, so cross-path parity is
    // automatic for this step.
    if params.has_obc_gb() {
        let obc = super::gb_obc::ObcGbParams::obc2();
        super::gb_obc::gb_obc_energy_and_forces(
            coords,
            topo,
            params,
            &obc,
            &mut result.solvation,
            &mut forces,
        );
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
            if r < 1e-10 {
                continue;
            }
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
            if rji_len < 1e-10 || rjk_len < 1e-10 {
                continue;
            }

            let cos_theta = dot(&rji, &rjk) / (rji_len * rjk_len);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(1e-12).sqrt();
            // F_i = -dE/dr_i = +2k(θ-θ₀)/sin(θ) × [bracket]
            // (See the matching comment in compute_energy_and_forces_impl.)
            let dv = 2.0 * ap.k * dtheta / sin_theta;

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
        if atom.is_hydrogen {
            continue;
        }
        if let Some(eef) = params.get_eef1(&atom.amber_type) {
            *solvation += eef.dg_ref;
        }
    }

    // Pair exclusion term (Gaussian shielding of water by neighbor atoms).
    // 1-2 and 1-3 bonded partners must be skipped: they sit at the Gaussian
    // peak (r ≈ r_min) and would contribute spurious O(10²) kcal/mol per
    // pair, inflating total solvation to wrong sign. BALL's CHARMM EEF1
    // implementation (charmmNonBonded.C) skips them via the LJ pair list
    // which excludes 1-2/1-3 at construction; we do it by checking
    // topo.excluded_pairs directly. 1-4 pairs ARE kept (BALL keeps them too,
    // unscaled for EEF1).
    for i in 0..n {
        if topo.atoms[i].is_hydrogen {
            continue;
        }
        let eef_i = match params.get_eef1(&topo.atoms[i].amber_type) {
            Some(e) => e,
            None => continue,
        };

        for j in (i + 1)..n {
            if topo.atoms[j].is_hydrogen {
                continue;
            }
            if topo.excluded_pairs.contains(&(i, j)) {
                continue;
            }
            let eef_j = match params.get_eef1(&topo.atoms[j].amber_type) {
                Some(e) => e,
                None => continue,
            };

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > cutoff_sq || r2 < 0.01 {
                continue;
            }

            let r = r2.sqrt();

            // Contribution from atom j excluding atom i's solvation
            if eef_i.dg_free.abs() > 1e-10 && eef_j.volume > 1e-10 {
                let dr = (r - eef_i.r_min) / eef_i.sigma;
                *solvation += -0.5 * eef_j.volume * eef_i.dg_free * (-dr * dr).exp()
                    / (eef_i.sigma * PI_SQRT_PI * r2);
            }

            // Contribution from atom i excluding atom j's solvation
            if eef_j.dg_free.abs() > 1e-10 && eef_i.volume > 1e-10 {
                let dr = (r - eef_j.r_min) / eef_j.sigma;
                *solvation += -0.5 * eef_i.volume * eef_j.dg_free * (-dr * dr).exp()
                    / (eef_j.sigma * PI_SQRT_PI * r2);
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
        if atom.is_hydrogen {
            continue;
        }
        if let Some(eef) = params.get_eef1(&atom.amber_type) {
            *solvation += eef.dg_ref;
        }
    }

    // Pair exclusion + forces. Skip 1-2 and 1-3 bonded partners — see the
    // comment in eef1_energy() above for rationale.
    for i in 0..n {
        if topo.atoms[i].is_hydrogen {
            continue;
        }
        let eef_i = match params.get_eef1(&topo.atoms[i].amber_type) {
            Some(e) => e,
            None => continue,
        };

        for j in (i + 1)..n {
            if topo.atoms[j].is_hydrogen {
                continue;
            }
            if topo.excluded_pairs.contains(&(i, j)) {
                continue;
            }
            let eef_j = match params.get_eef1(&topo.atoms[j].amber_type) {
                Some(e) => e,
                None => continue,
            };

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > cutoff_sq || r2 < 0.01 {
                continue;
            }

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

/// NBL-accelerated EEF1 solvation energy + forces.
///
/// Same physics as `eef1_energy_and_forces` but iterates the neighbor-list
/// pairs instead of the O(N²) all-pairs loop. The NBL is built with a 15 Å
/// cutoff (+ 2 Å buffer), which is a superset of EEF1's 9 Å pair cutoff —
/// so we just add an inner `r2 > 81.0` check and get the correct result.
///
/// **Speedup**: on a 1907-atom structure, the O(N²) loop checks 1.82M pairs;
/// the NBL loop checks ~200k. On 30k atoms: 450M vs ~6M (75×). Profiled on
/// monster3 2026-04-12: per-LBFGS-step cost dropped from 116 ms to ~72 ms
/// on the 1907-atom 7yv3 benchmark.
///
/// The NBL already excludes 1-2 and 1-3 pairs (they were filtered at
/// `NeighborList::build` time via `excluded_pairs`), so the explicit
/// `excluded_pairs.contains` check from the O(N²) version is unnecessary.
/// 1-4 pairs are kept (flagged as `is_14` but still present in the list),
/// matching the BALL convention.
fn eef1_energy_and_forces_nbl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    solvation: &mut f64,
    forces: &mut [[f64; 3]],
    nbl: &NeighborList,
) {
    let cutoff_sq = 9.0 * 9.0;

    // Self-solvation: Σ ΔG_ref (constant per atom, no force contribution)
    for atom in &topo.atoms {
        if atom.is_hydrogen {
            continue;
        }
        if let Some(eef) = params.get_eef1(&atom.amber_type) {
            *solvation += eef.dg_ref;
        }
    }

    // Pair exclusion + forces — iterate NBL pairs instead of all i<j.
    for pair in &nbl.pairs {
        let (i, j) = (pair.i, pair.j);
        if topo.atoms[i].is_hydrogen {
            continue;
        }
        if topo.atoms[j].is_hydrogen {
            continue;
        }

        let eef_i = match params.get_eef1(&topo.atoms[i].amber_type) {
            Some(e) => e,
            None => continue,
        };
        let eef_j = match params.get_eef1(&topo.atoms[j].amber_type) {
            Some(e) => e,
            None => continue,
        };

        let dx = coords[i][0] - coords[j][0];
        let dy = coords[i][1] - coords[j][1];
        let dz = coords[i][2] - coords[j][2];
        let r2 = dx * dx + dy * dy + dz * dz;
        // EEF1 cutoff (9 Å) is tighter than NBL cutoff (15 Å).
        if r2 > cutoff_sq || r2 < 0.01 {
            continue;
        }

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

/// Harmonic improper-torsion energy + forces: `E = k * (phi - phi0)²`.
///
/// **Currently NOT called from the active energy pipeline.** The
/// no-gradient path (`compute_energy_impl`) computes the harmonic
/// energy directly using `abs(phi_signed)` to match CHARMM's
/// unsigned-angle convention. This function lands as scaffolding
/// for the eventual gradient-path wiring once the analytical force
/// formula is corrected to match the unsigned-angle energy (the
/// Bekker derivative below assumes a signed dihedral).
///
/// CHARMM-only — used for the harmonic impropers shipped in the
/// 7-column `[ImproperTorsions]` section. AMBER's impropers are
/// cosine-series and handled by `torsion_energy_and_forces` with
/// `is_improper=true`; the trait default for `get_harmonic_improper`
/// returns `None`, so this loop is a no-op on AMBER force fields.
///
/// The dihedral is measured with the central atom at slot 1
/// (CHARMM convention) — see the comment inside the loop body for
/// the index re-bind that aligns proteon's slot-3-central storage
/// with CHARMM's slot-1-central measurement. Without the re-bind
/// the function produces a geometrically different angle and gives
/// ~10⁴× over-count on crambin.
///
/// Force formula: `dE/dphi = 2*k*(phi - phi0)`. The dihedral-to-
/// Cartesian transformation is shared with `torsion_energy_and_forces`
/// so the two paths stay numerically consistent.
///
/// Used by CHARMM, where impropers are stored as `(k, phi0)` pairs in
/// `[ImproperTorsions]` (7-column INI format) and computed harmonic.
/// AMBER's impropers are cosine-series (handled by
/// `torsion_energy_and_forces` with `is_improper=true`); the trait
/// default for `get_harmonic_improper` returns `None`, so this loop
/// is a no-op on AMBER.
///
/// CHARMM harmonic improper torsion: `E = K * (phi - phi0)²` where
/// `phi = acos(cos_phi) ∈ [0, π]` is the unsigned dihedral angle
/// between the (central, n1, n2) plane and the (n1, n2, n3) plane.
///
/// This is a port of BALL's `CharmmImproperTorsion::updateForces`
/// (`charmmImproperTorsion.C:452-563`), using the cosphi-based force
/// chain rather than proteon's standard Bekker chain. The cosphi chain
/// avoids the signed-vs-unsigned dihedral mismatch that broke the
/// gradient FD test in earlier wirings.
///
/// Energy uses the same `phi = acos(cos_phi)` so finite-difference
/// gradients of energy match the analytical force exactly.
///
/// Atom convention (matching BALL's `atom1`-as-central):
/// - `atom1` = central atom (proteon's `torsion.k`)
/// - `atom2`, `atom3`, `atom4` = the 3 atoms bonded to the central one
///
/// Force formula (BALL):
/// - `factor = -2 K (phi - phi0) / sin(phi)`
/// - `force_atom1 += factor * dcosphi/dba`
/// - `force_atom2 -= factor * (dcosphi/dba + dcosphi/dbc + dcosphi/dbd)`
/// - `force_atom3 += factor * dcosphi/dbc`
/// - `force_atom4 += factor * dcosphi/dbd`
///
/// where `ba = atom2 - atom1`, `bc = atom2 - atom3`, `bd = atom2 - atom4`.
/// At `sin(phi) = 0` (planar at `phi=0` or `phi=π`) the force is
/// numerically singular; BALL skips those cases and so do we — the
/// energy contribution is well-defined but the per-atom force has a
/// removable singularity that's easier to skip than to L'Hopital
/// through.
fn harmonic_improper_energy_and_forces(
    torsion_list: &[super::topology::Torsion],
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    energy_accum: &mut f64,
    forces: &mut [[f64; 3]],
) {
    for torsion in torsion_list {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;

        // CHARMM harmonic table keys central at slot 1; proteon's
        // Torsion convention stores central at slot K.
        let terms = match params.get_harmonic_improper(tk, ti, tj, tl) {
            Some(t) => t,
            None => continue,
        };

        // BALL: atom1=central, atoms 2/3/4 are the bonded neighbors.
        // proteon: torsion.k = central, torsion.i / torsion.j / torsion.l
        // are the neighbors.
        let (a1, a2, a3, a4) = (torsion.k, torsion.i, torsion.j, torsion.l);
        let p1 = &coords[a1];
        let p2 = &coords[a2];
        let p3 = &coords[a3];
        let p4 = &coords[a4];

        // BALL's vectors (note the order: all are atom2 - atomX).
        let ba = sub(p2, p1);
        let bc = sub(p2, p3);
        let bd = sub(p2, p4);

        let bcxba = cross(&bc, &ba);
        let bcxbd = cross(&bc, &bd);
        let len_bcxba = norm(&bcxba);
        let len_bcxbd = norm(&bcxbd);
        if len_bcxba < 1e-10 || len_bcxbd < 1e-10 {
            continue;
        }

        let inv_len_bcxba = 1.0 / len_bcxba;
        let inv_len_bcxbd = 1.0 / len_bcxbd;

        // cos(phi) is the dot product of the two normalized plane
        // normals.
        let cos_phi = (bcxba[0] * bcxbd[0] + bcxba[1] * bcxbd[1] + bcxba[2] * bcxbd[2])
            * inv_len_bcxba
            * inv_len_bcxbd;
        let cos_phi = cos_phi.clamp(-1.0, 1.0);
        let phi = cos_phi.acos();

        // Sum harmonic terms (CHARMM typically has one).
        let mut e_imp = 0.0;
        let mut sum_2k_dphi = 0.0;
        for term in terms {
            let dphi = phi - term.phi0;
            e_imp += term.k * dphi * dphi;
            sum_2k_dphi += 2.0 * term.k * dphi;
        }
        *energy_accum += e_imp;

        // Force: skip the sin(phi) ≈ 0 singularity (planar at phi=0
        // or phi=π). In normal protein geometries this is a measure-
        // zero set; BALL skips the same case.
        let sin_phi_sq = 1.0 - cos_phi * cos_phi;
        if sin_phi_sq <= 1e-12 {
            continue;
        }
        let sin_phi = sin_phi_sq.sqrt();
        let factor = -sum_2k_dphi / sin_phi;

        // dcosphi/dba, dcosphi/dbc, dcosphi/dbd (BALL's formulas).
        let inv_lba2 = inv_len_bcxba * inv_len_bcxba;
        let inv_lbd2 = inv_len_bcxbd * inv_len_bcxbd;
        let inv_lba_lbd = inv_len_bcxba * inv_len_bcxbd;
        let bcba = bc[0] * ba[0] + bc[1] * ba[1] + bc[2] * ba[2];
        let bcbd = bc[0] * bd[0] + bc[1] * bd[1] + bc[2] * bd[2];
        let babd = ba[0] * bd[0] + ba[1] * bd[1] + ba[2] * bd[2];
        let ba2 = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
        let bc2 = bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2];
        let bd2 = bd[0] * bd[0] + bd[1] * bd[1] + bd[2] * bd[2];

        let mut dcos_dba = [0.0f64; 3];
        let mut dcos_dbc = [0.0f64; 3];
        let mut dcos_dbd = [0.0f64; 3];
        for d in 0..3 {
            dcos_dbc[d] = cos_phi
                * (inv_lbd2 * (bcbd * bd[d] - bd2 * bc[d])
                    + inv_lba2 * (bcba * ba[d] - ba2 * bc[d]))
                + inv_lba_lbd * (2.0 * babd * bc[d] - bcba * bd[d] - bcbd * ba[d]);
            dcos_dbd[d] = cos_phi * inv_lbd2 * (bcbd * bc[d] - bc2 * bd[d])
                + inv_lba_lbd * (bc2 * ba[d] - bcba * bc[d]);
            dcos_dba[d] = cos_phi * inv_lba2 * (bcba * bc[d] - bc2 * ba[d])
                + inv_lba_lbd * (bc2 * bd[d] - bcbd * bc[d]);
        }

        for d in 0..3 {
            let f1d = factor * dcos_dba[d];
            let f3d = factor * dcos_dbc[d];
            let f4d = factor * dcos_dbd[d];
            let f2d = -(f1d + f3d + f4d);
            forces[a1][d] += f1d;
            forces[a2][d] += f2d;
            forces[a3][d] += f3d;
            forces[a4][d] += f4d;
        }
    }
}

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
        if len_a23 < 1e-10 {
            continue;
        }

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

#[cfg(test)]
mod gradient_tests {
    //! Regression tests that verify analytical forces match numerical gradients.
    //!
    //! These tests would have caught the angle sign-flip and CubicSwitch factor-of-2
    //! bugs that silently broke the minimizer. Any future refactor of the force
    //! computation should keep these passing.
    use super::*;
    use crate::add_hydrogens;
    use crate::forcefield::params::{amber96, amber96_obc};
    use crate::forcefield::topology::build_topology;
    use pdbtbx;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    fn crambin_path() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../test-pdbs/1crn.pdb");
        p
    }

    /// Load crambin with backbone H added once per test process.
    fn crambin_with_h() -> &'static pdbtbx::PDB {
        static CACHE: OnceLock<pdbtbx::PDB> = OnceLock::new();
        CACHE.get_or_init(|| {
            let (mut pdb, _) = pdbtbx::ReadOptions::default()
                .set_level(pdbtbx::StrictnessLevel::Loose)
                .read(crambin_path().to_str().unwrap())
                .expect("failed to read 1crn.pdb");
            add_hydrogens::place_peptide_hydrogens(&mut pdb);
            pdb
        })
    }

    fn collect_coords(pdb: &pdbtbx::PDB) -> Vec<[f64; 3]> {
        let mut coords = Vec::new();
        let first_model = pdb.models().next().expect("crambin has no models");
        for chain in first_model.chains() {
            for residue in chain.residues() {
                for atom in crate::altloc::residue_atoms_primary(residue) {
                    let (x, y, z) = atom.pos();
                    coords.push([x, y, z]);
                }
            }
        }
        coords
    }

    /// For a set of atom indices, verify analytical gradient (= -force) matches
    /// the finite-difference gradient of the energy on every Cartesian component.
    fn check_gradient_on_atoms(indices: &[usize]) {
        let pdb = crambin_with_h();
        let ff = amber96();
        let topo = build_topology(pdb, &ff);
        let coords = collect_coords(pdb);
        let (_, forces) = compute_energy_and_forces_impl(&coords, &topo, &ff, false);

        let eps = 1e-5;
        for &idx in indices {
            for d in 0..3 {
                let mut shifted = coords.clone();
                shifted[idx][d] += eps;
                let ep = compute_energy_impl(&shifted, &topo, &ff, false).total;
                shifted[idx][d] -= 2.0 * eps;
                let em = compute_energy_impl(&shifted, &topo, &ff, false).total;
                let num_grad = (ep - em) / (2.0 * eps);
                let ana_grad = -forces[idx][d]; // force = -gradient

                let axis = ['x', 'y', 'z'][d];
                let tol = 1e-3 + 1e-3 * ana_grad.abs();
                let diff = (num_grad - ana_grad).abs();
                assert!(
                    diff < tol,
                    "Gradient mismatch on atom {} axis {}: analytical={:.6}, numerical={:.6}, diff={:.2e}, tol={:.2e}",
                    idx, axis, ana_grad, num_grad, diff, tol
                );
            }
        }
    }

    #[test]
    fn gradient_matches_numerical_on_hydrogens() {
        // Check the first 5 backbone hydrogen atoms — this is what the minimizer
        // actually moves during prepare(). Catches angle sign-flip and switch bugs.
        let pdb = crambin_with_h();
        let ff = amber96();
        let topo = build_topology(pdb, &ff);
        let h_indices: Vec<usize> = topo
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_hydrogen { Some(i) } else { None })
            .take(5)
            .collect();
        assert!(!h_indices.is_empty(), "no hydrogens found in topology");
        check_gradient_on_atoms(&h_indices);
    }

    #[test]
    fn gradient_matches_numerical_on_heavy_atoms() {
        // Check a few heavy atoms too — these exercise different force terms
        // (bonds, more angles, more torsions involving backbone/sidechain).
        check_gradient_on_atoms(&[0, 4, 10, 25, 50]);
    }

    /// Same gradient check but with the AMBER96+OBC GB force field. This is
    /// the Phase B force parity guard: it catches any sign error in the
    /// OBC force port (first-loop direct pair force vs second-loop Born-
    /// radii-chain back-propagation through the HCT integrand).
    fn check_amber96_obc_gradient_on_atoms(indices: &[usize]) {
        let pdb = crambin_with_h();
        let ff = amber96_obc();
        let topo = build_topology(pdb, &ff);
        let coords = collect_coords(pdb);
        let (_, forces) = compute_energy_and_forces_impl(&coords, &topo, &ff, false);

        let eps = 1e-5;
        for &idx in indices {
            for d in 0..3 {
                let mut shifted = coords.clone();
                shifted[idx][d] += eps;
                let ep = compute_energy_impl(&shifted, &topo, &ff, false).total;
                shifted[idx][d] -= 2.0 * eps;
                let em = compute_energy_impl(&shifted, &topo, &ff, false).total;
                let num_grad = (ep - em) / (2.0 * eps);
                let ana_grad = -forces[idx][d];

                let axis = ['x', 'y', 'z'][d];
                // GB adds long-range coupling so absolute gradient magnitudes
                // are larger than AMBER96 vacuum; keep the same 0.1% relative
                // tolerance as the base gradient test.
                let tol = 1e-3 + 1e-3 * ana_grad.abs();
                let diff = (num_grad - ana_grad).abs();
                assert!(
                    diff < tol,
                    "AMBER96+OBC gradient mismatch on atom {} axis {}: analytical={:.6}, numerical={:.6}, diff={:.2e}, tol={:.2e}",
                    idx, axis, ana_grad, num_grad, diff, tol
                );
            }
        }
    }

    #[test]
    fn amber96_obc_gradient_matches_numerical_on_hydrogens() {
        let pdb = crambin_with_h();
        let ff = amber96_obc();
        let topo = build_topology(pdb, &ff);
        let h_indices: Vec<usize> = topo
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_hydrogen { Some(i) } else { None })
            .take(5)
            .collect();
        assert!(!h_indices.is_empty(), "no hydrogens found in topology");
        check_amber96_obc_gradient_on_atoms(&h_indices);
    }

    #[test]
    fn amber96_obc_gradient_matches_numerical_on_heavy_atoms() {
        check_amber96_obc_gradient_on_atoms(&[0, 4, 10, 25, 50]);
    }

    /// Same gradient check but going through the neighbor-list (NBL) code path.
    /// The minimizer uses this path for structures above NBL_AUTO_THRESHOLD,
    /// and it has its own independent implementation of the nonbonded loop
    /// (see `compute_energy_and_forces_nbl`). A sign or derivative bug in the
    /// NBL variant that doesn't exist in the O(N²) variant would not be
    /// caught by the other gradient tests above.
    fn check_nbl_gradient_on_atoms(indices: &[usize]) {
        let pdb = crambin_with_h();
        let ff = amber96();
        let topo = build_topology(pdb, &ff);
        let coords = collect_coords(pdb);
        let nbl = NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );
        let (_, forces) = compute_energy_and_forces_nbl(&coords, &topo, &ff, &nbl);

        let eps = 1e-5;
        for &idx in indices {
            for d in 0..3 {
                let mut shifted = coords.clone();
                shifted[idx][d] += eps;
                let ep = compute_energy_nbl(&shifted, &topo, &ff, &nbl).total;
                shifted[idx][d] -= 2.0 * eps;
                let em = compute_energy_nbl(&shifted, &topo, &ff, &nbl).total;
                let num_grad = (ep - em) / (2.0 * eps);
                let ana_grad = -forces[idx][d];

                let axis = ['x', 'y', 'z'][d];
                let tol = 1e-3 + 1e-3 * ana_grad.abs();
                let diff = (num_grad - ana_grad).abs();
                assert!(
                    diff < tol,
                    "NBL gradient mismatch on atom {} axis {}: analytical={:.6}, numerical={:.6}, diff={:.2e}, tol={:.2e}",
                    idx, axis, ana_grad, num_grad, diff, tol
                );
            }
        }
    }

    #[test]
    fn nbl_gradient_matches_numerical_on_hydrogens() {
        let pdb = crambin_with_h();
        let ff = amber96();
        let topo = build_topology(pdb, &ff);
        let h_indices: Vec<usize> = topo
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_hydrogen { Some(i) } else { None })
            .take(5)
            .collect();
        assert!(!h_indices.is_empty(), "no hydrogens found in topology");
        check_nbl_gradient_on_atoms(&h_indices);
    }

    #[test]
    fn nbl_gradient_matches_numerical_on_heavy_atoms() {
        check_nbl_gradient_on_atoms(&[0, 4, 10, 25, 50]);
    }

    /// Shared parity-check helper: compute_energy_impl (O(N²) exact) and
    /// compute_energy_nbl (neighbor-list) must return identical values for
    /// EVERY component — including `solvation` and `total` — on a structure
    /// small enough that the 15 Å neighbor list captures all interactions.
    ///
    /// Parametrized over the force field so both AMBER96 and CHARMM19+EEF1
    /// exercise this assertion. The CHARMM case is specifically what would
    /// have caught the 2026-04-11 EEF1 bugs:
    ///   1. eef1_energy() missing 1-2/1-3 exclusions (same on both paths)
    ///   2. compute_energy_and_forces_nbl() silently skipping EEF1 entirely,
    ///      leaving solvation=0 and total missing the solvation term.
    ///
    /// Bug #1 corrupts both paths by the same amount so a naive "exact vs
    /// nbl" comparison on `solvation` wouldn't necessarily flag it, BUT
    /// bug #2 leaves the NBL solvation at 0 while the exact path computes
    /// it — an O(10³) kcal/mol divergence that this test catches trivially.
    fn assert_nbl_matches_exact<F: ForceField>(ff: &F, ff_name: &str) {
        let pdb = crambin_with_h();
        let topo = build_topology(pdb, ff);
        let coords = collect_coords(pdb);
        let nbl = NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );

        let e_exact =
            compute_energy_impl(&coords, &topo, ff, ff.uses_distance_dependent_dielectric());
        let e_nbl = compute_energy_nbl(&coords, &topo, ff, &nbl);

        // Component-by-component — if any one drifts, we want to know
        // which one. Matching all components to 1e-6 is stricter than
        // matching only `total`, because offsetting component errors
        // can hide inside an unchanged total.
        let tol = 1e-6;
        let components = [
            ("bond_stretch", e_exact.bond_stretch, e_nbl.bond_stretch),
            ("angle_bend", e_exact.angle_bend, e_nbl.angle_bend),
            ("torsion", e_exact.torsion, e_nbl.torsion),
            (
                "improper_torsion",
                e_exact.improper_torsion,
                e_nbl.improper_torsion,
            ),
            ("vdw", e_exact.vdw, e_nbl.vdw),
            ("electrostatic", e_exact.electrostatic, e_nbl.electrostatic),
            ("solvation", e_exact.solvation, e_nbl.solvation),
            ("total", e_exact.total, e_nbl.total),
        ];
        for (name, exact, nbl) in components {
            assert!(
                (exact - nbl).abs() < tol,
                "[{}] {}: exact={:.9} nbl={:.9} diff={:.2e}",
                ff_name,
                name,
                exact,
                nbl,
                (exact - nbl).abs()
            );
        }
    }

    #[test]
    fn nbl_energy_matches_exact_energy_amber96() {
        // Original parity test — no EEF1 contribution (AMBER96 sets it to zero).
        assert_nbl_matches_exact(&amber96(), "amber96");
    }

    #[test]
    fn nbl_energy_matches_exact_energy_charmm19_eef1() {
        // Regression guard for the 2026-04-11 EEF1 bugs:
        //   * pre-fix: exact path produced buggy-but-nonzero solvation (missing
        //     1-2/1-3 exclusions), NBL path produced exactly 0 (never called
        //     EEF1 at all) → huge divergence, this test would have failed
        //     loudly and pointed straight at the gap in compute_energy_and_forces_nbl.
        //   * post-fix: both paths call eef1_energy_and_forces with proper
        //     exclusion filtering and must agree to 1e-6 kcal/mol.
        use crate::forcefield::params::charmm19_eef1;
        assert_nbl_matches_exact(&charmm19_eef1(), "charmm19_eef1");
    }

    #[test]
    fn nbl_energy_matches_exact_energy_amber96_obc() {
        // Parity guard for the OBC GB implicit-solvent dispatch. Today
        // both paths invoke the same all-pair `gb_obc_energy_and_forces`,
        // so this is trivially identical — but the test protects a
        // future NBL specialization (cutoff-aware HCT integral or GPU
        // port) from silently drifting from the exact reference.
        assert_nbl_matches_exact(&amber96_obc(), "amber96_obc");
    }

    /// Forces-path analogue of [`assert_nbl_matches_exact`]:
    /// `compute_energy_and_forces_impl` vs `compute_energy_and_forces_nbl`
    /// must return identical per-atom forces AND identical energy
    /// components. Parametrized over the force field so any accelerated
    /// code path (NBL, eventual GPU, SIMD) that specializes only the
    /// forces branch can't silently drift from the exact reference.
    fn assert_nbl_matches_exact_forces<F: ForceField>(ff: &F, ff_name: &str) {
        let pdb = crambin_with_h();
        let topo = build_topology(pdb, ff);
        let coords = collect_coords(pdb);
        let nbl = NeighborList::build(
            &coords,
            ff.nonbonded_cutoff(),
            &topo.excluded_pairs,
            &topo.pairs_14,
        );

        let (e_exact, f_exact) = compute_energy_and_forces_impl(
            &coords,
            &topo,
            ff,
            ff.uses_distance_dependent_dielectric(),
        );
        let (e_nbl, f_nbl) = compute_energy_and_forces_nbl(&coords, &topo, ff, &nbl);

        // Energy components first (same scaffolding as the energy-only guard).
        let tol_e = 1e-6;
        for (name, exact, nbl) in [
            ("bond_stretch", e_exact.bond_stretch, e_nbl.bond_stretch),
            ("angle_bend", e_exact.angle_bend, e_nbl.angle_bend),
            ("torsion", e_exact.torsion, e_nbl.torsion),
            (
                "improper_torsion",
                e_exact.improper_torsion,
                e_nbl.improper_torsion,
            ),
            ("vdw", e_exact.vdw, e_nbl.vdw),
            ("electrostatic", e_exact.electrostatic, e_nbl.electrostatic),
            ("solvation", e_exact.solvation, e_nbl.solvation),
            ("total", e_exact.total, e_nbl.total),
        ] {
            assert!(
                (exact - nbl).abs() < tol_e,
                "[{ff_name}] forces-path energy {name}: exact={exact:.9} nbl={nbl:.9} diff={:.2e}",
                (exact - nbl).abs()
            );
        }

        // Per-atom force parity. The nonbonded+solvation contributions are
        // the ones that actually differ between paths — if an NBL
        // specialization misses a pair or applies an inconsistent cutoff,
        // this surfaces it immediately.
        assert_eq!(f_exact.len(), f_nbl.len(), "force vector length mismatch");
        let tol_f = 1e-6;
        let mut max_diff = 0.0_f64;
        let mut worst: (usize, usize) = (0, 0);
        for i in 0..f_exact.len() {
            for k in 0..3 {
                let d = (f_exact[i][k] - f_nbl[i][k]).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i, k);
                }
            }
        }
        assert!(
            max_diff < tol_f,
            "[{ff_name}] max force mismatch {:.2e} > {:.0e} at atom {} axis {} \
             (exact={:.6} nbl={:.6})",
            max_diff,
            tol_f,
            worst.0,
            worst.1,
            f_exact[worst.0][worst.1],
            f_nbl[worst.0][worst.1]
        );
    }

    #[test]
    fn nbl_forces_match_exact_forces_amber96() {
        assert_nbl_matches_exact_forces(&amber96(), "amber96");
    }

    #[test]
    fn nbl_forces_match_exact_forces_charmm19_eef1() {
        use crate::forcefield::params::charmm19_eef1;
        assert_nbl_matches_exact_forces(&charmm19_eef1(), "charmm19_eef1");
    }

    #[test]
    fn nbl_forces_match_exact_forces_amber96_obc() {
        assert_nbl_matches_exact_forces(&amber96_obc(), "amber96_obc");
    }

    #[test]
    fn cubic_switch_derivative_matches_numerical() {
        // The derivative of the cubic switching function must match a numerical
        // finite-difference — this is where the factor-of-2 bug lived.
        let sw = CubicSwitch::new(15.0, 13.0);
        let eps = 1e-7;
        // Test points across the switching region (sq_cuton=169 to sq_cutoff=225).
        for &r2 in &[170.0_f64, 180.0, 195.0, 210.0, 220.0] {
            let (_, dsw_analytical) = sw.eval(r2);
            let (sw_plus, _) = sw.eval(r2 + eps);
            let (sw_minus, _) = sw.eval(r2 - eps);
            let dsw_numerical = (sw_plus - sw_minus) / (2.0 * eps);
            let diff = (dsw_analytical - dsw_numerical).abs();
            assert!(
                diff < 1e-5,
                "CubicSwitch dsw/dr² mismatch at r²={}: analytical={:.6e}, numerical={:.6e}",
                r2,
                dsw_analytical,
                dsw_numerical
            );
        }
    }

    // ------------------------------------------------------------------
    // Physical symmetry tests
    //
    // These would catch any class of bug where a position-absolute term
    // sneaks into the energy function. None of the existing tests cover
    // this; numerical gradient tests catch wrong derivatives, parity
    // tests catch NBL-vs-exact divergence, but neither catches
    // "electrostatic energy depends on the absolute origin of the
    // coordinate system."
    //
    // The tests are parametrized over force field via a generic helper
    // so adding a new FF (amber14, charmm36, ...) means appending to
    // the list of #[test] wrappers at the bottom — not rewriting the
    // check logic.
    // ------------------------------------------------------------------

    /// Apply a rigid translation to every coordinate: `coords[i] += delta`.
    fn translate(coords: &[[f64; 3]], delta: [f64; 3]) -> Vec<[f64; 3]> {
        coords
            .iter()
            .map(|c| [c[0] + delta[0], c[1] + delta[1], c[2] + delta[2]])
            .collect()
    }

    /// Apply a rigid rotation about the origin. Axis is Z, angle in radians.
    /// (Simple case — a single axis is enough to catch rotation-dependence
    /// bugs; we don't need full Euler-angle coverage for a regression test.)
    fn rotate_z(coords: &[[f64; 3]], theta: f64) -> Vec<[f64; 3]> {
        let c = theta.cos();
        let s = theta.sin();
        coords
            .iter()
            .map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
            .collect()
    }

    /// Core symmetry check: energy at `coords` must equal energy at
    /// `transformed_coords` to `tol` on every component. Generic over the
    /// force field so both AMBER and CHARMM share the same assertion.
    fn assert_symmetry<F: ForceField>(
        ff: &F,
        ff_name: &str,
        transform_name: &str,
        original: &[[f64; 3]],
        transformed: &[[f64; 3]],
        topo: &Topology,
        tol: f64,
    ) {
        let e_orig = compute_energy_impl(original, topo, ff, false);
        let e_xform = compute_energy_impl(transformed, topo, ff, false);

        let components = [
            ("bond_stretch", e_orig.bond_stretch, e_xform.bond_stretch),
            ("angle_bend", e_orig.angle_bend, e_xform.angle_bend),
            ("torsion", e_orig.torsion, e_xform.torsion),
            (
                "improper_torsion",
                e_orig.improper_torsion,
                e_xform.improper_torsion,
            ),
            ("vdw", e_orig.vdw, e_xform.vdw),
            ("electrostatic", e_orig.electrostatic, e_xform.electrostatic),
            ("solvation", e_orig.solvation, e_xform.solvation),
            ("total", e_orig.total, e_xform.total),
        ];
        for (name, orig, xform) in components {
            let diff = (orig - xform).abs();
            // Relative tolerance on large values (electrostatic on a clashy
            // raw PDB can be O(10⁵) kcal/mol; forcing strict 1e-6 there is
            // pointless). Absolute floor catches any near-zero component.
            let rel_tol = tol.max(1e-10 * orig.abs());
            assert!(
                diff < rel_tol,
                "[{}/{}] {}: orig={:.9} xform={:.9} diff={:.3e} tol={:.3e}",
                ff_name,
                transform_name,
                name,
                orig,
                xform,
                diff,
                rel_tol
            );
        }
    }

    fn check_translation_invariance<F: ForceField>(ff: &F, ff_name: &str) {
        let pdb = crambin_with_h();
        let topo = build_topology(pdb, ff);
        let coords = collect_coords(pdb);

        // A few translations: small, medium, and large (to make sure the
        // energy is invariant far from the origin too — catches bugs where
        // numerical error accumulates with coordinate magnitude).
        for &delta in &[[0.5, 0.0, 0.0], [10.0, 20.0, 30.0], [1000.0, -500.0, 250.0]] {
            let translated = translate(&coords, delta);
            assert_symmetry(
                ff,
                ff_name,
                &format!("translate{:?}", delta),
                &coords,
                &translated,
                &topo,
                1e-6,
            );
        }
    }

    fn check_rotation_invariance<F: ForceField>(ff: &F, ff_name: &str) {
        let pdb = crambin_with_h();
        let topo = build_topology(pdb, ff);
        let coords = collect_coords(pdb);

        // A few rotation angles. π/3 and π/2 catch sign bugs; π/7 catches
        // any bug where the check happens to pass on "nice" angles.
        for &theta in &[
            std::f64::consts::PI / 7.0,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI / 2.0,
            std::f64::consts::PI * 0.9,
        ] {
            let rotated = rotate_z(&coords, theta);
            // Rotation invariance has a slightly higher numerical floor
            // than translation — rotating multiplies every coordinate by
            // a 3×3 matrix, so FP error scales with |coord|. A rigid-body
            // rotation on coords ~10 Å changes FP ordering in dist() and
            // dot() calls, which gives O(1e-9) drift on large components.
            assert_symmetry(
                ff,
                ff_name,
                &format!("rotate_z({:.3})", theta),
                &coords,
                &rotated,
                &topo,
                1e-4,
            );
        }
    }

    fn check_rigid_body_forces_sum_to_zero<F: ForceField>(ff: &F, ff_name: &str) {
        // Newton's third law applied to an isolated system: the sum of
        // forces on all atoms must be zero (translational symmetry of the
        // Lagrangian → momentum conservation). Any nonzero total force is
        // a spurious "body force" — equivalent to an external field leaking
        // into the supposedly-internal energy function.
        let pdb = crambin_with_h();
        let topo = build_topology(pdb, ff);
        let coords = collect_coords(pdb);
        let (_, forces) = compute_energy_and_forces_impl(
            &coords,
            &topo,
            ff,
            ff.uses_distance_dependent_dielectric(),
        );

        let mut sum = [0.0_f64; 3];
        for f in &forces {
            sum[0] += f[0];
            sum[1] += f[1];
            sum[2] += f[2];
        }
        let mag = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        // Tolerance: 1e-6 kcal/mol/Å in any component is below the noise
        // floor of the force kernel. Large LJ interactions on raw PDBs
        // produce per-atom forces of O(10⁴), and the sum over 600 atoms
        // FP-accumulates to ~1e-9 * 10⁴ * 600 ≈ 1e-2 — so we need a
        // relative tolerance. 1e-8 × max-|f| is enough.
        let max_f = forces.iter().fold(0.0_f64, |m, f| {
            let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            m.max(n)
        });
        let tol = (1e-6_f64).max(1e-8 * max_f);
        assert!(
            mag < tol,
            "[{}] Σforces = ({:.3e}, {:.3e}, {:.3e}), |Σ|={:.3e} > tol={:.3e}. \
             max per-atom force magnitude = {:.3e}",
            ff_name,
            sum[0],
            sum[1],
            sum[2],
            mag,
            tol,
            max_f
        );
    }

    #[test]
    fn translation_invariance_amber96() {
        check_translation_invariance(&amber96(), "amber96");
    }

    #[test]
    fn translation_invariance_charmm19_eef1() {
        use crate::forcefield::params::charmm19_eef1;
        check_translation_invariance(&charmm19_eef1(), "charmm19_eef1");
    }

    #[test]
    fn rotation_invariance_amber96() {
        check_rotation_invariance(&amber96(), "amber96");
    }

    #[test]
    fn rotation_invariance_charmm19_eef1() {
        use crate::forcefield::params::charmm19_eef1;
        check_rotation_invariance(&charmm19_eef1(), "charmm19_eef1");
    }

    #[test]
    fn rigid_body_forces_sum_to_zero_amber96() {
        check_rigid_body_forces_sum_to_zero(&amber96(), "amber96");
    }

    #[test]
    fn rigid_body_forces_sum_to_zero_charmm19_eef1() {
        use crate::forcefield::params::charmm19_eef1;
        check_rigid_body_forces_sum_to_zero(&charmm19_eef1(), "charmm19_eef1");
    }

    // ------------------------------------------------------------------
    // Extended numerical gradient tests — cover CHARMM19+EEF1 too.
    // The existing gradient_matches_numerical_on_hydrogens /
    // gradient_matches_numerical_on_heavy_atoms tests only run AMBER96;
    // CHARMM had no gradient-consistency check until now. A sign error
    // in eef1_energy_and_forces (e.g. wrong force accumulation direction)
    // wouldn't be caught by the numerical-gradient tests on AMBER since
    // AMBER never calls EEF1.
    // ------------------------------------------------------------------

    fn check_gradient_on_atoms_charmm(indices: &[usize]) {
        use crate::forcefield::params::charmm19_eef1;
        let pdb = crambin_with_h();
        let ff = charmm19_eef1();
        let topo = build_topology(pdb, &ff);
        let coords = collect_coords(pdb);
        let (_, forces) = compute_energy_and_forces_impl(&coords, &topo, &ff, false);

        let eps = 1e-5;
        for &idx in indices {
            for d in 0..3 {
                let mut shifted = coords.clone();
                shifted[idx][d] += eps;
                let ep = compute_energy_impl(&shifted, &topo, &ff, false).total;
                shifted[idx][d] -= 2.0 * eps;
                let em = compute_energy_impl(&shifted, &topo, &ff, false).total;
                let num_grad = (ep - em) / (2.0 * eps);
                let ana_grad = -forces[idx][d];

                let axis = ['x', 'y', 'z'][d];
                let tol = 1e-3 + 1e-3 * ana_grad.abs();
                let diff = (num_grad - ana_grad).abs();
                assert!(
                    diff < tol,
                    "[charmm19_eef1] Gradient mismatch on atom {} axis {}: analytical={:.6}, numerical={:.6}, diff={:.2e}, tol={:.2e}",
                    idx, axis, ana_grad, num_grad, diff, tol
                );
            }
        }
    }

    #[test]
    fn charmm19_eef1_gradient_matches_numerical_on_hydrogens() {
        use crate::forcefield::params::charmm19_eef1;
        let pdb = crambin_with_h();
        let ff = charmm19_eef1();
        let topo = build_topology(pdb, &ff);
        let h_indices: Vec<usize> = topo
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_hydrogen { Some(i) } else { None })
            .take(5)
            .collect();
        assert!(!h_indices.is_empty(), "no hydrogens found in topology");
        check_gradient_on_atoms_charmm(&h_indices);
    }

    #[test]
    fn charmm19_eef1_gradient_matches_numerical_on_heavy_atoms() {
        check_gradient_on_atoms_charmm(&[0, 4, 10, 25, 50]);
    }
}
