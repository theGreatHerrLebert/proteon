//! OBC generalized Born implicit solvent (Onufriev-Bashford-Case 2004).
//!
//! References:
//! - Onufriev, Bashford, Case, "Exploring protein native states and
//!   large-scale conformational changes with a modified generalized
//!   Born model", *Proteins* 55(2), 383-394 (2004) — GB-OBC model.
//! - Hawkins, Cramer, Truhlar, "Parametrized models of aqueous free
//!   energies of solvation based on pairwise descreening of solute
//!   atomic charges from a dielectric medium", *J. Phys. Chem.* 100(51),
//!   19824-19839 (1996) — HCT pair-integral form.
//!
//! Derived from OpenMM (MIT; © Stanford University and the Authors),
//! `platforms/reference/src/SimTKReference/ReferenceObc.cpp`, for the
//! Born-radii rescaling, pair-integral assembly, and force chain-rule
//! implementation. See `THIRD_PARTY_NOTICES.md` for the upstream license.
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
    compute_born_radii_with_chain(coords, topo, params, obc).0
}

/// Same math as [`compute_born_radii`] but also returns `obc_chain`, the
/// per-atom derivative factor needed for force back-propagation through
/// the Born-radii chain rule.
///
/// `obc_chain[i] = (1 − tanh²) · offsetRadiusI · (α − 2β·ψ + 3γ·ψ²) / ρ_i`
///
/// In the force pass, OpenMM multiplies `bornForces[i]` (i.e. ∂E/∂R_eff_i)
/// by `R_eff_i² · obc_chain[i]` before spreading it through the HCT
/// integrand derivative. Cache it here to avoid recomputing on the force
/// side.
pub(crate) fn compute_born_radii_with_chain(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
) -> (Vec<f64>, Vec<f64>) {
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
    let mut obc_chain = vec![0.0_f64; n];

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
            let r2 = dx * dx + dy * dy + dz * dz;
            // Skip coincident / near-coincident atom pairs. The HCT
            // integrand below divides by r and log(u_ij/l_ij), both of
            // which diverge as r→0. Matches the r²<0.01 policy used
            // elsewhere in nonbonded terms (energy.rs:200, :424, :820).
            // Malformed inputs or mid-minimization overlaps would
            // otherwise produce NaN Born radii that poison everything
            // downstream.
            if r2 < 0.01 {
                continue;
            }
            let r = r2.sqrt();
            let r_scaled_radius_j = r + scaled_radius_j;

            // Atom j's scaled sphere doesn't reach i's offset sphere.
            if offset_radius_i >= r_scaled_radius_j {
                continue;
            }

            let r_inverse = 1.0 / r;
            let diff = (r - scaled_radius_j).abs();
            let l_ij_denom = if offset_radius_i > diff {
                offset_radius_i
            } else {
                diff
            };
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

        // Chain-rule factor used by gb_obc_energy_and_forces. Follows
        // OpenMM's identical two-line form so the force port is
        // line-traceable.
        let chain = offset_radius_i * (obc.alpha - 2.0 * obc.beta * sum + 3.0 * obc.gamma * sum2);
        obc_chain[i] = (1.0 - tanh_sum * tanh_sum) * chain / radius_i;
    }

    (born_radii, obc_chain)
}

/// OBC GB solvation energy (energy only, no forces).
///
/// Implements the canonical Still/OBC polar-solvation energy:
///
/// ```text
/// G_pol = -½ · τ · k_C · Σ_{i,j}  q_i · q_j / f_GB(r_ij, R_eff_i, R_eff_j)
/// ```
///
/// where
/// - τ = 1/ε_in − 1/ε_out (positive for water),
/// - k_C is the Coulomb constant (332.0 kcal·Å/(mol·e²) in proteon's units),
/// - f_GB(r, a, b) = √(r² + a·b · exp(−r²/(4·a·b))),
/// - f_GB(0, R, R) = R (so the diagonal terms are the self-energies
///   `q_i² / R_eff_i` with a ½ factor out front).
///
/// The sum runs over every ordered (i,j) pair including `i==j`. The result
/// is added to `*solvation` in **kcal/mol** (same unit convention as the
/// rest of `energy.rs`; conversion to kJ/mol happens in the Python wrapper).
///
/// Phase B leaves `include_self_term = false` on `ObcGbParams` as a
/// diagnostic hook: when set, the self term is dropped so the pair
/// contribution can be unit-tested in isolation.
pub(crate) fn gb_obc_energy(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
    solvation: &mut f64,
) {
    if coords.is_empty() {
        return;
    }
    let n = coords.len();
    assert_eq!(
        n,
        topo.atoms.len(),
        "gb_obc_energy: coords/topology length mismatch ({} vs {})",
        n,
        topo.atoms.len()
    );

    let born = compute_born_radii(coords, topo, params, obc);
    // Coulomb constant consistent with the rest of energy.rs
    // (kcal·Å / (mol·e²)). τ = 1/ε_in − 1/ε_out.
    const K_COULOMB_KCAL: f64 = 332.0;
    let tau = obc.tau();
    let charges: Vec<f64> = topo.atoms.iter().map(|a| a.charge).collect();

    // Upper-triangular accumulation of Σ_{i<=j} (counting diagonal once
    // and off-diagonal once with an explicit 2× factor to replay the
    // full double sum). Matches OpenMM's ReferenceObc loop shape.
    let mut sum = 0.0_f64;

    // Self terms (i==j): f_GB(0, R, R) = R.
    let mut sum_self = 0.0_f64;
    if obc.include_self_term {
        for i in 0..n {
            let r_eff = born[i];
            if r_eff > 0.0 {
                sum_self += charges[i] * charges[i] / r_eff;
            }
        }
    }

    // Off-diagonal terms (i < j), multiplied by 2 to cover both (i,j) and (j,i).
    let mut sum_pair = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let alpha2 = born[i] * born[j];
            if alpha2 <= 0.0 {
                continue;
            }
            let d_ij = r2 / (4.0 * alpha2);
            let f_gb = (r2 + alpha2 * (-d_ij).exp()).sqrt();
            sum_pair += 2.0 * charges[i] * charges[j] / f_gb;
        }
    }

    sum += sum_self + sum_pair;
    let contribution = -0.5 * tau * K_COULOMB_KCAL * sum;
    if std::env::var("PROTEON_OBC_DEBUG").is_ok() {
        let br_min = born.iter().copied().fold(f64::INFINITY, f64::min);
        let br_max = born.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let br_mean: f64 = born.iter().sum::<f64>() / born.len() as f64;
        eprintln!(
            "[obc] n={} self_sum={:.4} pair_sum={:.4} (e²/Å)  \
             G_self={:.3} G_pair={:.3} kcal/mol  \
             R_eff min/mean/max = {:.3}/{:.3}/{:.3} Å",
            n,
            sum_self,
            sum_pair,
            -0.5 * tau * K_COULOMB_KCAL * sum_self,
            -0.5 * tau * K_COULOMB_KCAL * sum_pair,
            br_min,
            br_mean,
            br_max,
        );
    }
    *solvation += contribution;
}

/// OBC GB solvation energy + forces (all-pair).
///
/// Port of OpenMM's `ReferenceObc::computeBornEnergyForces`
/// (`platforms/reference/src/SimTKReference/ReferenceObc.cpp`). Follows
/// the same three-phase structure:
///
///   1. Build Born radii + obcChain via `compute_born_radii_with_chain`.
///   2. Upper-triangular loop: accumulate energy + direct pair forces
///      (`dGpol/dr` at fixed R_eff) + bornForces[i] += ∂Gpol/∂(R_i·R_j)·R_j
///      and symmetric for j. For i==j the 0.5 prefactor folds in the
///      self-energy contribution without double-counting.
///   3. Transform: `bornForces[i] *= R_eff_i² · obcChain[i]`.
///   4. Full double loop: spread bornForces through the HCT integrand
///      derivative `t3` to get the indirect per-atom force on j due to
///      atom i's Born radius changing when j moves.
///
/// Sign convention: `forces[i]` accumulates the physical force
/// (F = −∇E) consistent with the rest of `energy.rs`. Energy is added
/// to `*solvation` in kcal/mol.
///
/// Finite-difference verified against `gb_obc_energy` in the unit tests
/// on simple 2–6 atom systems.
pub(crate) fn gb_obc_energy_and_forces(
    coords: &[[f64; 3]],
    topo: &Topology,
    params: &impl ForceField,
    obc: &ObcGbParams,
    solvation: &mut f64,
    forces: &mut [[f64; 3]],
) {
    let n = coords.len();
    if n == 0 {
        return;
    }
    assert_eq!(
        n,
        topo.atoms.len(),
        "gb_obc_energy_and_forces: coords/topology length mismatch ({} vs {})",
        n,
        topo.atoms.len()
    );
    assert_eq!(
        forces.len(),
        n,
        "forces buffer length {} != atoms {}",
        forces.len(),
        n
    );

    let (born, obc_chain) = compute_born_radii_with_chain(coords, topo, params, obc);

    // Per-atom (intrinsic radius, HCT scale) for the second loop's HCT
    // integrand rebuild.
    let per_atom: Vec<(f64, f64)> = topo
        .atoms
        .iter()
        .map(|a| {
            let p = params
                .get_obc_gb(&a.amber_type)
                .expect("OBC params checked in born radii");
            (p.radius, p.scale)
        })
        .collect();

    // Pre-factor for the Still-formula loop. OpenMM literally uses
    // `preFactor = 2·electricConstant·τ` but OpenMM's reported `obcEnergy`
    // ends up matching a factor of 2× Still's `-½·τ·k·Σ_{ij} q_i·q_j/f_GB`
    // — which is the implementation detail that produces the observed
    // numerical values. Proteon's [`gb_obc_energy`] stays in Still's
    // form (oracle-matched at ≤5% on crambin), so the force loop needs
    // `pre_factor = -τ·k_C` (half of OpenMM's) to accumulate the SAME
    // energy via the upper-triangular-with-0.5-diagonal loop structure:
    //   Σ_i [0.5·pre·q_i²/R_i] + Σ_{i<j} [pre·q_i·q_j/f_ij]
    //   = (-τ·k_C/2)·Σ q²/R + (-τ·k_C)·Σ_{i<j} q·q/f
    //   = -½·τ·k_C·[Σ q²/R + 2·Σ_{i<j} q·q/f]    (matches gb_obc_energy).
    // Direct and bornForces-chain gradients then fall out of the same
    // pre_factor automatically.
    const K_COULOMB_KCAL: f64 = 332.0;
    let pre_factor = -K_COULOMB_KCAL * obc.tau();

    let charges: Vec<f64> = topo.atoms.iter().map(|a| a.charge).collect();

    let mut obc_energy = 0.0_f64;
    let mut born_forces = vec![0.0_f64; n];

    // --- First loop: energy + direct-pair forces + bornForces accumulator.
    for i in 0..n {
        let pqi = pre_factor * charges[i];
        for j in i..n {
            // Skip the self term when disabled, so the forces path
            // mirrors the `include_self_term` semantics of the energy
            // path. Without this, the two APIs disagree on the same
            // `ObcGbParams` — which makes oracle debugging painful when
            // someone flips the toggle to isolate the pair contribution.
            if i == j && !obc.include_self_term {
                continue;
            }

            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;

            let alpha2 = born[i] * born[j];
            // If ever zero, the pair contributes nothing (guard against
            // degenerate Born radii; in practice both > 0).
            if alpha2 <= 0.0 {
                continue;
            }
            let d_ij = r2 / (4.0 * alpha2);
            let exp_term = (-d_ij).exp();
            let denom2 = r2 + alpha2 * exp_term;
            let denom = denom2.sqrt();

            let gpol = pqi * charges[j] / denom;
            // Note: at r=0 with i != j we have denom = sqrt(alpha2) > 0
            // (both Born radii strictly positive), so the energy and
            // force expressions below are finite even for coincident
            // atoms. The explicit r² < 0.01 skip is reserved for the
            // HCT integrand in the second loop which genuinely has a
            // 1/r singularity.
            let d_gpol_dr = -gpol * (1.0 - 0.25 * exp_term) / denom2;
            let d_gpol_dalpha2 = -0.5 * gpol * exp_term * (1.0 + d_ij) / denom2;

            let mut energy = gpol;
            if i != j {
                // Direct pair gradient (R_eff held fixed): F_i = +r·dGpol_dr.
                // Sign note: OpenMM does inputForces[i] += deltaR*dGpol_dr
                // where deltaR = r_j − r_i. Our `forces` array accumulates
                // the physical force (−∇E), so the same sign works here.
                let fx = dx * d_gpol_dr;
                let fy = dy * d_gpol_dr;
                let fz = dz * d_gpol_dr;
                forces[i][0] += fx;
                forces[i][1] += fy;
                forces[i][2] += fz;
                forces[j][0] -= fx;
                forces[j][1] -= fy;
                forces[j][2] -= fz;
                born_forces[j] += d_gpol_dalpha2 * born[i];
            } else {
                energy *= 0.5;
            }
            obc_energy += energy;
            born_forces[i] += d_gpol_dalpha2 * born[j];
        }
    }

    // --- Transform bornForces through dR_eff/dI chain factor.
    for i in 0..n {
        born_forces[i] *= born[i] * born[i] * obc_chain[i];
    }

    // --- Second loop: spread bornForces through the HCT integrand
    // derivative. For every ordered pair (i, j), j != i, the derivative
    // of I_i wrt r_ij (with L_ij, U_ij treated as constants — their
    // contributions cancel analytically, see OpenMM comment).
    for i in 0..n {
        let (radius_i, _) = per_atom[i];
        let offset_radius_i = radius_i - obc.offset;
        for j in 0..n {
            if i == j {
                continue;
            }
            let (radius_j, scale_j) = per_atom[j];
            let offset_radius_j = radius_j - obc.offset;
            let scaled_radius_j = scale_j * offset_radius_j;
            let scaled_radius_j2 = scaled_radius_j * scaled_radius_j;

            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            // Same r²<0.01 guard as compute_born_radii_with_chain —
            // the integrand derivative t3 below uses 1/r and 1/r² and
            // would NaN on a coincident pair.
            if r2 < 0.01 {
                continue;
            }
            let r = r2.sqrt();
            let r_scaled_radius_j = r + scaled_radius_j;

            if offset_radius_i >= r_scaled_radius_j {
                continue;
            }

            let diff = (r - scaled_radius_j).abs();
            let l_ij_denom = if offset_radius_i > diff {
                offset_radius_i
            } else {
                diff
            };
            let l_ij = 1.0 / l_ij_denom;
            let u_ij = 1.0 / r_scaled_radius_j;
            let l_ij2 = l_ij * l_ij;
            let u_ij2 = u_ij * u_ij;

            let r_inverse = 1.0 / r;
            let r2_inverse = r_inverse * r_inverse;

            let t3 = 0.125 * (1.0 + scaled_radius_j2 * r2_inverse) * (l_ij2 - u_ij2)
                + 0.25 * (u_ij / l_ij).ln() * r2_inverse;
            let de = born_forces[i] * t3 * r_inverse;

            let fx = dx * de;
            let fy = dy * de;
            let fz = dz * de;
            // OpenMM: inputForces[i] -= deltaR·de; inputForces[j] += deltaR·de.
            // proteon's `forces` is the same (-∇E) convention, so signs
            // carry over unchanged.
            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[i][2] -= fz;
            forces[j][0] += fx;
            forces[j][1] += fy;
            forces[j][2] += fz;
        }
    }

    *solvation += obc_energy;
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
            lj_excluded_pairs: HashSet::new(),
            unassigned_atoms: vec![],
        }
    }

    fn ff_atom(amber_type: &str, element: &str, pos: [f64; 3]) -> FFAtom {
        ff_atom_q(amber_type, element, pos, 0.0)
    }

    fn ff_atom_q(amber_type: &str, element: &str, pos: [f64; 3], charge: f64) -> FFAtom {
        FFAtom {
            pos,
            amber_type: amber_type.to_string(),
            charge,
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
        assert!(
            br[0].is_finite() && br[0] < 100.0,
            "R_eff out of range: {}",
            br[0]
        );
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
    fn single_ion_matches_born_formula() {
        // For a single charged atom, G_Born = -½·τ·k_C·q²/R_eff.
        // Here R_eff = offsetRadius = ρ - 0.09 since sum = 0 → tanh = 0.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let charge = 1.0_f64;
        let coords = vec![[0.0, 0.0, 0.0]];
        let topo = topo_of(vec![ff_atom_q("CT", "C", coords[0], charge)]);
        let r_eff = params.get_obc_gb("CT").unwrap().radius - obc.offset; // 1.81
        let expected = -0.5 * obc.tau() * 332.0 * charge * charge / r_eff;

        let mut solvation = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut solvation);
        assert!(
            (solvation - expected).abs() < 1e-9,
            "single ion: got {}, expected {}",
            solvation,
            expected
        );
        // Sanity: solvation energy must be negative (favorable) for τ > 0.
        assert!(solvation < 0.0);
    }

    #[test]
    fn zero_charges_give_zero_solvation() {
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom("CT", "C", coords[0]),
            ff_atom("CT", "C", coords[1]),
        ]);
        let mut solvation = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut solvation);
        assert!(solvation.abs() < 1e-12, "expected 0, got {}", solvation);
    }

    #[test]
    fn opposite_charges_at_distance_sum_is_self_minus_pair() {
        // Two opposite charges far apart: pair term → ε · q·(−q) / r (Coulomb-like),
        // self terms → two Born-ion self-energies. Compare to hand formula.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom_q("CT", "C", coords[0], 1.0),
            ff_atom_q("CT", "C", coords[1], -1.0),
        ]);
        let r = 50.0;
        let r_eff = params.get_obc_gb("CT").unwrap().radius - obc.offset;
        // At 50 Å separation, born radii are ≈ r_eff (far-field regime),
        // so this is a tight bound.
        let alpha2 = r_eff * r_eff;
        let d = r * r / (4.0 * alpha2);
        let f_gb = (r * r + alpha2 * (-d).exp()).sqrt();
        let self_pair = 2.0 * 1.0 * 1.0 / r_eff; // q² / R, summed for i and j (both q²=1)
        let cross_pair = -(2.0 * 1.0) / f_gb; // 2·q_i·q_j / f_GB
        let expected = -0.5 * obc.tau() * 332.0 * (self_pair + cross_pair);

        let mut solvation = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut solvation);
        let rel = (solvation - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel < 0.01,
            "expected ~{}, got {} ({}% off)",
            expected,
            solvation,
            rel * 100.0
        );
    }

    #[test]
    fn include_self_term_toggle_drops_self_energy() {
        // With include_self_term = false, a single-atom system has
        // zero solvation (no pair partner, no self term).
        let params = amber96_obc();
        let mut obc = ObcGbParams::obc1();
        obc.include_self_term = false;
        let coords = vec![[0.0, 0.0, 0.0]];
        let topo = topo_of(vec![ff_atom_q("CT", "C", coords[0], 1.0)]);
        let mut solvation = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut solvation);
        assert!(
            solvation.abs() < 1e-12,
            "expected 0 (self off), got {}",
            solvation
        );
    }

    /// Central-differences estimate of -∂E/∂r_i (the force on atom i) by
    /// perturbing each coordinate by ±h.
    fn fd_forces(
        coords: &[[f64; 3]],
        topo: &Topology,
        params: &impl ForceField,
        obc: &ObcGbParams,
        h: f64,
    ) -> Vec<[f64; 3]> {
        let n = coords.len();
        let mut out = vec![[0.0; 3]; n];
        let mut local = coords.to_vec();
        for i in 0..n {
            for k in 0..3 {
                let saved = local[i][k];
                local[i][k] = saved + h;
                let mut e_plus = 0.0;
                gb_obc_energy(&local, topo, params, obc, &mut e_plus);
                local[i][k] = saved - h;
                let mut e_minus = 0.0;
                gb_obc_energy(&local, topo, params, obc, &mut e_minus);
                local[i][k] = saved;
                out[i][k] = -(e_plus - e_minus) / (2.0 * h); // F = -dE/dx
            }
        }
        out
    }

    fn max_force_err(analytic: &[[f64; 3]], numeric: &[[f64; 3]]) -> f64 {
        let mut err = 0.0_f64;
        for (a, n) in analytic.iter().zip(numeric.iter()) {
            for k in 0..3 {
                err = err.max((a[k] - n[k]).abs());
            }
        }
        err
    }

    #[test]
    fn forces_match_finite_difference_opposite_charges() {
        // Two opposite charges on CT type at 3 Å. Strong coupling, should
        // exercise both direct pair force and Born-radii chain rule.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom_q("CT", "C", coords[0], 1.0),
            ff_atom_q("CT", "C", coords[1], -1.0),
        ]);

        let mut e = 0.0;
        let mut f = vec![[0.0; 3]; 2];
        gb_obc_energy_and_forces(&coords, &topo, &params, &obc, &mut e, &mut f);

        // Energy from the dedicated path must match.
        let mut e_ref = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut e_ref);
        assert!(
            (e - e_ref).abs() < 1e-6,
            "energy path disagree: forces-path={}, energy-path={}",
            e,
            e_ref
        );

        let fd = fd_forces(&coords, &topo, &params, &obc, 1e-5);
        let err = max_force_err(&f, &fd);
        assert!(
            err < 1e-4,
            "max force error {:.3e} exceeds 1e-4\nanalytic={:?}\nnumeric={:?}",
            err,
            f,
            fd
        );
    }

    #[test]
    fn forces_match_finite_difference_six_atom_mixed() {
        // 6 atoms, mixed AMBER classes, mixed charges, in a non-symmetric
        // geometry so the chain-rule forces don't accidentally cancel.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.55, 0.2, 0.1],
            [2.8, 1.3, -0.4],
            [0.1, 2.0, 1.1],
            [3.5, 0.0, 0.5],
            [1.2, -1.5, 0.7],
        ];
        let atoms = vec![
            ff_atom_q("CT", "C", coords[0], 0.3),
            ff_atom_q("N", "N", coords[1], -0.4),
            ff_atom_q("CT", "C", coords[2], 0.1),
            ff_atom_q("O", "O", coords[3], -0.5),
            ff_atom_q("H", "H", coords[4], 0.25),
            ff_atom_q("HC", "H", coords[5], 0.15),
        ];
        let topo = topo_of(atoms);

        let mut e = 0.0;
        let mut f = vec![[0.0; 3]; 6];
        gb_obc_energy_and_forces(&coords, &topo, &params, &obc, &mut e, &mut f);

        let mut e_ref = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut e_ref);
        assert!(
            (e - e_ref).abs() < 1e-6,
            "energy mismatch: {} vs {}",
            e,
            e_ref
        );

        let fd = fd_forces(&coords, &topo, &params, &obc, 1e-5);
        let err = max_force_err(&f, &fd);
        assert!(err < 1e-3, "max force error {:.3e} exceeds 1e-3", err);
    }

    #[test]
    fn include_self_term_toggle_also_affects_forces_path() {
        // When include_self_term is flipped, the forces-path energy must
        // track the energy-path's output exactly. Without the toggle in
        // the forces path, the two APIs disagree on the same params.
        let params = amber96_obc();
        let mut obc = ObcGbParams::obc1();
        obc.include_self_term = false;
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom_q("CT", "C", coords[0], 1.0),
            ff_atom_q("CT", "C", coords[1], -1.0),
        ]);

        let mut e_energy = 0.0;
        gb_obc_energy(&coords, &topo, &params, &obc, &mut e_energy);

        let mut e_forces = 0.0;
        let mut f = vec![[0.0; 3]; 2];
        gb_obc_energy_and_forces(&coords, &topo, &params, &obc, &mut e_forces, &mut f);

        assert!(
            (e_energy - e_forces).abs() < 1e-9,
            "self-term toggle mismatch: energy-path={}, forces-path={}",
            e_energy,
            e_forces
        );
        // Also: with self off and pair on, FD must still match analytical.
        let fd = fd_forces(&coords, &topo, &params, &obc, 1e-5);
        let err = max_force_err(&f, &fd);
        assert!(err < 1e-4, "self-off FD err {:.2e}", err);
    }

    #[test]
    fn zero_separation_pair_does_not_blow_up() {
        // Two distinct atoms at the same coordinate would normally
        // produce NaN Born radii and NaN forces via 1/r. The r²<0.01
        // guard must keep everything finite even in this pathological
        // configuration.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let topo = topo_of(vec![
            ff_atom_q("CT", "C", coords[0], 1.0),
            ff_atom_q("CT", "C", coords[1], -1.0),
        ]);

        // Born radii should be finite (falls back to the isolated-atom
        // limit when the pair is skipped).
        let (born, _) = compute_born_radii_with_chain(&coords, &topo, &params, &obc);
        for b in &born {
            assert!(b.is_finite() && *b > 0.0, "non-finite born radius: {}", b);
        }

        let mut e = 0.0;
        let mut f = vec![[0.0; 3]; 2];
        gb_obc_energy_and_forces(&coords, &topo, &params, &obc, &mut e, &mut f);
        assert!(e.is_finite(), "non-finite energy: {}", e);
        for atom_f in &f {
            for k in 0..3 {
                assert!(
                    atom_f[k].is_finite(),
                    "non-finite force component: {}",
                    atom_f[k]
                );
            }
        }
    }

    #[test]
    fn newtons_third_law_holds() {
        // Net force on an isolated system must sum to zero: Σ F_i = 0.
        let params = amber96_obc();
        let obc = ObcGbParams::obc1();
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [1.0, 1.5, -0.5]];
        let topo = topo_of(vec![
            ff_atom_q("CT", "C", coords[0], 0.5),
            ff_atom_q("N", "N", coords[1], -0.4),
            ff_atom_q("H", "H", coords[2], 0.3),
        ]);
        let mut e = 0.0;
        let mut f = vec![[0.0; 3]; 3];
        gb_obc_energy_and_forces(&coords, &topo, &params, &obc, &mut e, &mut f);
        let sum: [f64; 3] = f.iter().fold([0.0; 3], |acc, v| {
            [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]]
        });
        for k in 0..3 {
            assert!(
                sum[k].abs() < 1e-9,
                "net force component {} = {:e}",
                k,
                sum[k]
            );
        }
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
