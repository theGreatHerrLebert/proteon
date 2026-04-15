//! OBC generalized Born implicit solvent (Onufriev-Bashford-Case 2004).
//!
//! This is the Phase A scaffolding. Contract + structure + parameter hooks
//! are in place; the pair-integral math, Born-radii rescaling, and forces
//! are marked `todo!()` pending a dedicated math session against:
//!   - Onufriev, Bashford, Case 2004 (GBOBC paper)
//!   - OpenMM reference: platforms/reference/src/SimTKReference/ReferenceObc.cpp
//!
//! Rationale for the skeleton-first split: OBC-GB forces (chain rule through
//! per-pair Born-radii dependence) are where sign errors slip in. Landing the
//! contract + a failing oracle test first forces the implementation session
//! to converge against a verified spec rather than "looks right" heuristics.
//!
//! # Energy terms
//!
//! - **Self term:** `E_self = -(τ/2) · Σ_i q_i² / R_eff_i`
//! - **Pair term:** `E_pair = -τ · Σ_{i<j} q_i·q_j / f_GB(r_ij, R_i, R_j)`
//!   where `f_GB = √(r² + R_i·R_j · exp(-r²/(4·R_i·R_j)))`
//! - `τ = (1/ε_in - 1/ε_out)`; for AMBER96+OBC the standard choice is
//!   `ε_in = 1.0`, `ε_out = 78.5`.
//!
//! # Effective Born radius (OBC1 rescaling)
//!
//! 1. Compute the HCT pair integral `I_i = Σ_j H(r_ij, R_i, S_j·R_j)`
//! 2. `psi_i = I_i · ρ'_i` where `ρ'_i = ρ_i - offset` (offset = 0.09 Å)
//! 3. `R_eff_i = 1 / (1/ρ'_i - tanh(α·ψ - β·ψ² + γ·ψ³) / ρ_i)`
//!    with OBC1: (α, β, γ) = (0.8, 0.0, 2.909125)
//!    with OBC2: (α, β, γ) = (1.0, 0.8, 4.85)
//!
//! OBC1 matches OpenMM's `amber96_obc.xml` default and is the oracle target
//! for AMBER96. OBC2 is used by `charmm36_obc2.xml` and is the second-phase
//! target.
//!
//! # Follow-ups (explicit non-goals for Phase A)
//!
//! - GPU port (Phase A ships CPU only).
//! - Surface-area (ACE) correction term — usually omitted in OBC but some
//!   amber_obc variants include it. Decide when matching the oracle.
//! - Neighbor-list path: Phase A has the signature but returns the all-pair
//!   result, so correctness is identical until Phase B's math lands.

use crate::forcefield::params::ForceField;
use crate::forcefield::topology::Topology;

/// Per-atom OBC parameters: intrinsic Bondi-style radius + the Hawkins-
/// Cramer-Truhlar "scale factor" used to tune the pair integral.
///
/// Standard AMBER96+OBC defaults (OpenMM `amber96_obc.xml`):
///   H: radius=1.2 Å, scale=0.85
///   C: radius=1.7 Å, scale=0.72
///   N: radius=1.55 Å, scale=0.79
///   O: radius=1.5 Å, scale=0.85
///   S: radius=1.8 Å, scale=0.96
///   P: radius=1.85 Å, scale=0.86
///
/// Other atom types fall back to the element-based defaults above. Phase B
/// loads these from `amber96_obc.ini`.
#[derive(Clone, Copy, Debug)]
pub struct ObcAtomParams {
    /// Intrinsic atomic radius (Å) — Bondi-style, used as ρ_i in the GB
    /// integral.
    pub radius: f64,
    /// HCT overlap scale factor (dimensionless). Tunes how strongly atom j
    /// contributes to atom i's Born-radius integral.
    pub scale: f64,
}

/// Global OBC GB parameters. Same constants for every atom in the system;
/// per-atom parameters live in [`ObcAtomParams`].
#[derive(Clone, Copy, Debug)]
pub struct ObcGbParams {
    /// Inner (solute) dielectric. Standard: 1.0.
    pub dielectric_in: f64,
    /// Outer (solvent) dielectric. Standard: 78.5 for water.
    pub dielectric_out: f64,
    /// Dielectric offset ρ' = ρ - offset. Standard: 0.09 Å.
    pub offset: f64,
    /// OBC rescaling α parameter.
    pub alpha: f64,
    /// OBC rescaling β parameter.
    pub beta: f64,
    /// OBC rescaling γ parameter.
    pub gamma: f64,
    /// Whether to include the self-term in the total energy. Kept optional
    /// so the pair-vs-total decomposition can be unit-tested separately.
    pub include_self_term: bool,
}

impl ObcGbParams {
    /// OBC1 parameters — matches OpenMM `amber96_obc.xml`, our oracle target
    /// for AMBER96+OBC.
    pub fn obc1() -> Self {
        Self {
            dielectric_in: 1.0,
            dielectric_out: 78.5,
            offset: 0.09,
            alpha: 0.8,
            beta: 0.0,
            gamma: 2.909125,
            include_self_term: true,
        }
    }

    /// OBC2 parameters — matches OpenMM `charmm36_obc2.xml`, needed later
    /// if we want apples-to-apples CHARMM36 comparison.
    pub fn obc2() -> Self {
        Self {
            dielectric_in: 1.0,
            dielectric_out: 78.5,
            offset: 0.09,
            alpha: 1.0,
            beta: 0.8,
            gamma: 4.85,
            include_self_term: true,
        }
    }

    /// Kirkwood-Onsager prefactor τ = (1/ε_in − 1/ε_out).
    pub fn tau(&self) -> f64 {
        1.0 / self.dielectric_in - 1.0 / self.dielectric_out
    }
}

/// Compute effective Born radii for every atom.
///
/// Port of OpenMM's `ReferenceObc::computeBornRadii`
/// (`platforms/reference/src/SimTKReference/ReferenceObc.cpp`).
/// Variable names preserved (`offset_radius_i`, `radius_i_inverse`,
/// `scaled_radius_j`, `l_ij`, `u_ij`) so the port is line-traceable.
///
/// Steps, per OBC:
///   1. For each atom `i`, sum the HCT pair integral `I_i` over all `j != i`,
///      where each term is a piecewise function of r_ij, `offsetRadiusI`,
///      and `scaledRadiusJ = S_j * (ρ_j - offset)`.
///   2. Rescale: `ψ = 0.5 · offsetRadiusI · I_i`; then
///      `R_eff_i = 1 / (1/offsetRadiusI − tanh(α·ψ − β·ψ² + γ·ψ³) / ρ_i)`.
///
/// Notes that differ from a naive reading of the OBC paper:
/// - `scaledRadiusJ` uses the dielectric-**offset** radius of j, not the
///   intrinsic radius (OpenMM convention; follow it so the oracle matches).
/// - The "i inside j's scaled sphere" branch adds a `2·(radiusIInverse − l_ij)`
///   correction where `radiusIInverse = 1/offsetRadiusI`.
///
/// # Panics
/// Panics if any atom's AMBER class is missing from the OBC parameter table
/// (indicates a mis-configured parameter set — an invariant violation we want
/// to surface loudly rather than silently mis-score).
pub(crate) fn compute_born_radii(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
) -> Vec<f64> {
    let n = coords.len();
    assert_eq!(
        n,
        topo.atoms.len(),
        "compute_born_radii: coords/topology length mismatch ({} vs {})",
        n,
        topo.atoms.len()
    );

    // Preload per-atom (intrinsic radius, HCT scale) so the inner loop is
    // table-free. Missing params are a hard invariant violation.
    let per_atom: Vec<(f64, f64)> = topo
        .atoms
        .iter()
        .map(|a| {
            let p = params.get_obc_gb(&a.amber_type).unwrap_or_else(|| {
                panic!(
                    "compute_born_radii: no OBC params for AMBER class '{}' \
                     (atom index {}); is amber96_obc.ini loaded?",
                    a.amber_type, 0
                )
            });
            (p.radius, p.scale)
        })
        .collect();

    let mut born_radii = vec![0.0_f64; n];

    for i in 0..n {
        let (radius_i, _) = per_atom[i];
        let offset_radius_i = radius_i - obc.offset;
        let radius_i_inverse = 1.0 / offset_radius_i;

        let mut sum = 0.0_f64;
        for j in 0..n {
            if i == j {
                continue;
            }
            let (radius_j, scale_j) = per_atom[j];
            let offset_radius_j = radius_j - obc.offset;
            let scaled_radius_j = scale_j * offset_radius_j;

            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let r_scaled_radius_j = r + scaled_radius_j;

            // Atom j's scaled sphere doesn't reach i's offset sphere.
            if offset_radius_i >= r_scaled_radius_j {
                continue;
            }

            let r_inverse = 1.0 / r;
            let diff = (r - scaled_radius_j).abs();
            let l_ij_denom = if offset_radius_i > diff { offset_radius_i } else { diff };
            let l_ij = 1.0 / l_ij_denom;
            let u_ij = 1.0 / r_scaled_radius_j;

            let l_ij2 = l_ij * l_ij;
            let u_ij2 = u_ij * u_ij;
            let ratio = (u_ij / l_ij).ln();
            let mut term = l_ij - u_ij
                + 0.25 * r * (u_ij2 - l_ij2)
                + 0.5 * r_inverse * ratio
                + 0.25 * scaled_radius_j * scaled_radius_j * r_inverse * (l_ij2 - u_ij2);

            // Atom i lies entirely inside atom j's scaled sphere.
            if offset_radius_i < (scaled_radius_j - r) {
                term += 2.0 * (radius_i_inverse - l_ij);
            }

            sum += term;
        }

        sum *= 0.5 * offset_radius_i;
        let sum2 = sum * sum;
        let sum3 = sum * sum2;
        let tanh_arg = obc.alpha * sum - obc.beta * sum2 + obc.gamma * sum3;
        let tanh_sum = tanh_arg.tanh();
        born_radii[i] = 1.0 / (1.0 / offset_radius_i - tanh_sum / radius_i);
    }

    born_radii
}

/// OBC GB solvation energy (energy only, no forces).
///
/// Mirrors the EEF1 entry-point signature so `energy.rs` can dispatch to
/// this the same way. Phase B fills in the self + pair terms.
#[allow(dead_code)]
pub(crate) fn gb_obc_energy(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
    solvation: &mut f64,
) {
    let _ = (coords, topo, params, obc);
    // TODO(Phase B): E_self + E_pair via f_GB; add to *solvation.
    let _ = solvation;
}

/// OBC GB solvation energy + forces (all-pair).
#[allow(dead_code)]
pub(crate) fn gb_obc_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
    solvation: &mut f64,
    forces: &mut [[f64; 3]],
) {
    let _ = (coords, topo, params, obc);
    // TODO(Phase B): analytical derivatives of E_pair + E_self through the
    // Born-radii chain rule. This is the most error-prone part — port
    // against OpenMM's ReferenceObc.cpp and verify via finite-difference
    // before calling done.
    let _ = (solvation, forces);
}

/// OBC GB solvation energy + forces (neighbor-list).
///
/// Phase A stub just delegates to the all-pair path so that the NBL
/// code path can be exercised end-to-end without duplicating placeholder
/// math. Phase B / the cross-path parity test will make this a real
/// cutoff-aware implementation.
#[allow(dead_code)]
pub(crate) fn gb_obc_energy_and_forces_nbl(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
    solvation: &mut f64,
    forces: &mut [[f64; 3]],
) {
    gb_obc_energy_and_forces(coords, topo, params, obc, solvation, forces);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::params::amber96_obc;
    use crate::forcefield::topology::{FFAtom, Topology};
    use std::collections::HashSet;

    /// Minimal Topology carrying only the fields `compute_born_radii` reads
    /// (the atoms vec). Everything bonded/nonbonded-related is empty.
    fn topo_of(atoms: Vec<FFAtom>) -> Topology {
        Topology {
            atoms,
            bonds: vec![],
            angles: vec![],
            torsions: vec![],
            improper_torsions: vec![],
            excluded_pairs: HashSet::new(),
            pairs_14: HashSet::new(),
            unassigned_atoms: vec![],
        }
    }

    fn ff_atom(amber_type: &str, element: &str, pos: [f64; 3]) -> FFAtom {
        FFAtom {
            pos,
            amber_type: amber_type.to_string(),
            charge: 0.0,
            residue_name: "XXX".into(),
            atom_name: "X".into(),
            element: element.into(),
            residue_idx: 0,
            is_hydrogen: element == "H",
        }
    }

    #[test]
    fn isolated_atom_born_radius_equals_offset_radius() {
        // Single atom -> sum == 0 -> tanh(0) == 0 -> R_eff = offsetRadius = rho - offset.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0]];
        let topo = topo_of(vec![ff_atom("CT", "C", coords[0])]);
        let br = compute_born_radii(&coords, &topo, &params, &obc);
        let ct = params.get_obc_gb("CT").unwrap();
        let expected = ct.radius - obc.offset; // 1.9 - 0.09 = 1.81
        assert_eq!(br.len(), 1);
        assert!(
            (br[0] - expected).abs() < 1e-12,
            "isolated CT: got {}, expected {}",
            br[0],
            expected
        );
    }

    #[test]
    fn far_separation_recovers_isolated_radii() {
        // Two CT atoms 100 Å apart should each be within 1% of the
        // offset-radius limit (sum ~ 0 at long range).
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom("CT", "C", coords[0]),
            ff_atom("CT", "C", coords[1]),
        ]);
        let br = compute_born_radii(&coords, &topo, &params, &obc);
        let expected = params.get_obc_gb("CT").unwrap().radius - obc.offset;
        for r in &br {
            let rel = (r - expected).abs() / expected;
            assert!(rel < 0.01, "far CT: R_eff = {}, expected ~ {}", r, expected);
        }
    }

    #[test]
    fn close_pair_descreens_both_atoms() {
        // Two CT atoms at 2 Å (van-der-Waals-ish contact): each atom's
        // effective Born radius must be meaningfully larger than the
        // isolated-atom limit (offsetRadius), i.e. the pair integral is
        // generating nonzero descreening. Also: symmetric.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom("CT", "C", coords[0]),
            ff_atom("CT", "C", coords[1]),
        ]);
        let br = compute_born_radii(&coords, &topo, &params, &obc);
        let offset_r = params.get_obc_gb("CT").unwrap().radius - obc.offset; // 1.81
        // Symmetric (same species + same distance both directions).
        assert!((br[0] - br[1]).abs() < 1e-12, "asymmetric: {:?}", br);
        // Meaningful descreening -> R_eff strictly larger than offset radius.
        assert!(
            br[0] > offset_r + 0.01,
            "expected descreening, got R_eff={} vs offset={}",
            br[0],
            offset_r
        );
        // Sanity upper bound: R_eff should stay finite and reasonable
        // (<< 100 Å). The OBC tanh cap keeps it bounded.
        assert!(br[0].is_finite() && br[0] < 100.0, "R_eff out of range: {}", br[0]);
    }

    #[test]
    #[should_panic(expected = "no OBC params for AMBER class")]
    fn panics_on_missing_amber_class() {
        // amber96_obc knows standard AMBER classes; a bogus type must
        // trip the invariant-violation panic rather than silently
        // producing nonsense radii.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0]];
        let topo = topo_of(vec![ff_atom("BOGUS", "C", coords[0])]);
        let _ = compute_born_radii(&coords, &topo, &params, &obc);
    }

    #[test]
    fn obc1_matches_amber96_obc_xml_defaults() {
        let p = ObcGbParams::obc1();
        assert!((p.alpha - 0.8).abs() < 1e-12);
        assert!((p.beta - 0.0).abs() < 1e-12);
        assert!((p.gamma - 2.909125).abs() < 1e-12);
        assert!((p.offset - 0.09).abs() < 1e-12);
        assert!(p.include_self_term);
    }

    #[test]
    fn obc2_matches_charmm36_obc2_defaults() {
        let p = ObcGbParams::obc2();
        assert!((p.alpha - 1.0).abs() < 1e-12);
        assert!((p.beta - 0.8).abs() < 1e-12);
        assert!((p.gamma - 4.85).abs() < 1e-12);
    }

    #[test]
    fn tau_matches_standard_water_dielectrics() {
        let p = ObcGbParams::obc1();
        let expected = 1.0 / 1.0 - 1.0 / 78.5;
        assert!((p.tau() - expected).abs() < 1e-12);
    }
}
