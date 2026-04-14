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
/// Phase B will populate: per-atom HCT pair integral → ψ = I · ρ' → OBC tanh
/// rescaling → R_eff.
///
/// Current stub returns `vec![1.0; coords.len()]` so downstream code has a
/// stable-shape placeholder to exercise interfaces against; it does NOT
/// produce correct physics.
#[allow(dead_code)]
pub(crate) fn compute_born_radii(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
) -> Vec<f64> {
    let _ = (coords, topo, params, obc);
    // TODO(Phase B): HCT integral + OBC1/2 tanh rescaling.
    vec![1.0; coords.len()]
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
