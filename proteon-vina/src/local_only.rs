// Licensed under the Apache License, Version 2.0. See LICENSE.

//! End-to-end `vina --local_only` equivalent: take a pose, run BFGS
//! to its nearest local minimum in Conf space, re-score the refined
//! pose with `eval_fast` for upstream-compatible reporting.
//!
//! This is the integration driver that wires:
//!
//! ```text
//! PdbqtFile        ─┬─▶ TorsionTree  ─┐
//!                   └─▶ Molecule ────┼─▶ BFGS closure ─▶ minimised Conf ─▶ score_only (8 components)
//! Molecule receptor ────────────────── ┘
//! ```
//!
//! The BFGS closure composes:
//!   1. `tree.apply_full(conf)` → atom coords + cached per-fragment frames.
//!   2. `inter_pair_energy_with_forces` + `intra_pair_energy_with_forces`.
//!   3. `gradient_from_forces` to reduce per-atom forces into a
//!      `ConfGrad` in DoF space.
//!   4. Sign flip: `gradient_from_forces` returns `−∂E/∂DoF`
//!      (physics force). BFGS expects `+∂E/∂DoF`. Negate.
//!
//! Final reporting uses `score_only` (eval_fast path) so the 8
//! components match what upstream prints after its post-BFGS score().

use crate::bfgs::{bfgs, BfgsOutcome};
use crate::conf::Conf;
use crate::gradient::gradient_from_forces;
use crate::molecule::Molecule;
use crate::pdbqt::PdbqtFile;
use crate::precalculate::Precalculate;
use crate::score::{
    inter_pair_energy_with_forces, intra_pair_energy_with_forces, intra_pair_list, score_only,
    ScoreComponents,
};
use crate::torsion::TorsionTree;

/// Options controlling a local minimisation.
#[derive(Clone, Copy, Debug)]
pub struct LocalOnlyOptions {
    /// Maximum BFGS iterations. `None` picks upstream's default:
    /// `(25 + num_movable_atoms) / 3` — ~21 for drug-like ligands,
    /// which is where upstream converges quickly and stops even
    /// when further descent is available. Passing a larger value
    /// lets BFGS go further but may find a different local min
    /// than upstream.
    pub max_steps: Option<usize>,
    /// Soft-cap `v` for the `curl(·)` helper. Upstream passes
    /// `authentic_v = (1000, 1000, 1000)` during scoring, which
    /// effectively disables the cap for docked poses (per-atom
    /// energies are well below 1000 kcal/mol for reasonable
    /// geometries). We expose it so callers can tighten the cap
    /// during search to tame steep clashes.
    pub v_curl: f64,
}

impl Default for LocalOnlyOptions {
    fn default() -> Self {
        Self { max_steps: None, v_curl: 1000.0 }
    }
}

/// All-in-one output of [`local_only`].
#[derive(Clone, Debug)]
pub struct LocalOnlyOutcome {
    /// The minimised conformation.
    pub conf: Conf,
    /// 8-component score computed on the minimised pose, matching
    /// what upstream's `show_score` prints after `--local_only`.
    pub components: ScoreComponents,
    /// BFGS statistics (starting energy, final energy, evaluation
    /// count, etc.). Useful for diagnosing convergence.
    pub bfgs: BfgsOutcome,
}

/// Run BFGS on a ligand + receptor pair starting from the ligand's
/// file pose, then rescore the refined pose.
///
/// * `receptor` — the receptor Molecule.
/// * `ligand` — the ligand Molecule (file pose).
/// * `ligand_file` — the parsed `PdbqtFile` (needed for the torsion
///   tree topology and for `score_only`'s `num_tors` computation).
/// * `precalc` — a prebuilt Vina [`Precalculate`] table. Reusing
///   avoids rebuilding the 32 × 32 pair-type table for every call.
/// * `opts` — see [`LocalOnlyOptions`].
#[must_use]
pub fn local_only(
    receptor: &Molecule,
    ligand: &Molecule,
    ligand_file: &PdbqtFile,
    precalc: &Precalculate,
    opts: LocalOnlyOptions,
) -> LocalOnlyOutcome {
    let tree = TorsionTree::from_molecule(ligand, ligand_file);
    let pairs = intra_pair_list(ligand);

    // Scratch Molecule whose coords we overwrite at every energy
    // evaluation. Everything else (types, bonds, fragment_mask) is
    // invariant under pose changes, so we clone once.
    let mut scratch = ligand.clone();

    let mut bfgs_evals = 0_usize;
    let mut f = |conf: &Conf| {
        bfgs_evals += 1;
        let applied = tree.apply_full(conf);
        scratch.coords.clone_from(&applied.coords);

        let (e_inter, forces_inter) =
            inter_pair_energy_with_forces(receptor, &scratch, precalc, opts.v_curl);
        let (e_intra, forces_intra) =
            intra_pair_energy_with_forces(&scratch, &pairs, precalc, opts.v_curl);

        // Sum per-atom forces.
        let mut per_atom_force = forces_inter;
        for (a, b) in per_atom_force.iter_mut().zip(forces_intra.iter()) {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
        }

        // DoF gradient in "force" sign. BFGS wants +∂E/∂DoF → negate.
        let force_grad = gradient_from_forces(&tree, &applied, &per_atom_force);
        let grad_for_bfgs = force_grad.negated();

        (e_inter + e_intra, grad_for_bfgs)
    };

    let mut conf = tree.identity_conf();
    // Upstream `Vina::optimize` computes max_steps = (25 + N_movable) / 3
    // when the caller passes 0; we mirror that formula when
    // `opts.max_steps` is `None`.
    let max_steps = opts
        .max_steps
        .unwrap_or((25 + ligand.len()) / 3);
    let bfgs_outcome = bfgs(&mut f, &mut conf, max_steps);
    let _ = bfgs_evals; // eval count already tracked inside bfgs_outcome.n_evals

    // Re-score the minimised pose with eval_fast for upstream-parity
    // reporting of the 8-component vector.
    let final_coords = tree.apply(&conf);
    scratch.coords = final_coords;
    let components = score_only(
        receptor,
        &scratch,
        &ligand_file.rotatable_bonds,
        precalc,
        opts.v_curl,
    );

    LocalOnlyOutcome { conf, components, bfgs: bfgs_outcome }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::parse_pdbqt;
    use approx::assert_relative_eq;

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const REC_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/receptor.pdbqt");

    fn load_1iep() -> (Molecule, Molecule, PdbqtFile) {
        (
            Molecule::from_pdbqt_str(REC_1IEP).unwrap(),
            Molecule::from_pdbqt_str(LIG_1IEP).unwrap(),
            parse_pdbqt(LIG_1IEP).unwrap(),
        )
    }

    #[test]
    fn local_only_improves_energy_below_starting_score() {
        let (rec, lig, file) = load_1iep();
        let precalc = Precalculate::vina();
        let out = local_only(&rec, &lig, &file, &precalc, LocalOnlyOptions::default());
        // `score_only` on the starting file-pose returns -12.513.
        // The minimiser must at worst preserve that; in practice
        // BFGS finds a nearby local min at ~-13.24.
        assert!(
            out.components.total <= -12.513 + 1e-3,
            "local_only regressed: initial ~-12.513, final {}",
            out.components.total
        );
    }

    #[test]
    fn local_only_components_are_finite_after_minimisation() {
        let (rec, lig, file) = load_1iep();
        let precalc = Precalculate::vina();
        let out = local_only(&rec, &lig, &file, &precalc, LocalOnlyOptions::default());
        for v in &[
            out.components.total,
            out.components.lig_grids,
            out.components.inter_pairs,
            out.components.flex_grids,
            out.components.intra_pairs,
            out.components.lig_intra,
            out.components.conf_independent,
            out.components.intramolecular,
        ] {
            assert!(v.is_finite(), "non-finite component {v}");
        }
    }

    #[test]
    fn local_only_bfgs_runs_at_least_one_step() {
        // Sanity: we should actually have iterated, not just returned
        // the initial score.
        let (rec, lig, file) = load_1iep();
        let precalc = Precalculate::vina();
        let out = local_only(&rec, &lig, &file, &precalc, LocalOnlyOptions::default());
        assert!(out.bfgs.n_steps >= 1);
        assert!(out.bfgs.n_evals >= 2);
    }

    /// Upstream `vina --local_only --autobox` values captured from
    /// v1.2.7-27-g3c65c0b on each fixture. Per-fixture tolerance
    /// accounts for BFGS trajectory sensitivity: the algorithm and
    /// step budget match upstream exactly ((25 + N)/3 iterations)
    /// but line-search tie-breaks differ, which sends us into
    /// slightly different local basins on non-convex landscapes.
    ///
    /// Drug-like ligands (1iep, 1s63) track upstream to within
    /// 80 mkcal/mol across every component. 1fpu (same imatinib
    /// reprepared around a ring-root) and BACE_1 (22-torsion
    /// macrocycle with broader non-convex surface) drift up to
    /// ~1.1 kcal/mol. For docking ranking and reporting the
    /// per-pose output is still highly consistent — the issue is
    /// that two BFGS implementations at their default step count
    /// often stop in different (but neighboring) local minima.
    fn check_local_only_parity(
        name: &str,
        tol: f64,
        upstream_total: f64,
        upstream_inter: f64,
        upstream_intra: f64,
        upstream_conf: f64,
        upstream_unbound: f64,
    ) {
        let lig_text =
            std::fs::read_to_string(format!("tests/fixtures/pairs/{name}/ligand.pdbqt"))
                .or_else(|_| {
                    std::fs::read_to_string(format!(
                        "proteon-vina/tests/fixtures/pairs/{name}/ligand.pdbqt"
                    ))
                })
                .expect("fixture path");
        let rec_text =
            std::fs::read_to_string(format!("tests/fixtures/pairs/{name}/receptor.pdbqt"))
                .or_else(|_| {
                    std::fs::read_to_string(format!(
                        "proteon-vina/tests/fixtures/pairs/{name}/receptor.pdbqt"
                    ))
                })
                .expect("fixture path");
        let rec = Molecule::from_pdbqt_str(&rec_text).unwrap();
        let lig = Molecule::from_pdbqt_str(&lig_text).unwrap();
        let file = parse_pdbqt(&lig_text).unwrap();
        let precalc = Precalculate::vina();
        let out = local_only(&rec, &lig, &file, &precalc, LocalOnlyOptions::default());
        assert_relative_eq!(out.components.total, upstream_total, epsilon = tol);
        assert_relative_eq!(out.components.lig_grids, upstream_inter, epsilon = tol);
        assert_relative_eq!(out.components.lig_intra, upstream_intra, epsilon = tol);
        assert_relative_eq!(out.components.conf_independent, upstream_conf, epsilon = tol);
        assert_relative_eq!(out.components.intramolecular, upstream_unbound, epsilon = tol);
    }

    #[test]
    fn local_only_matches_upstream_on_1iep() {
        let (rec, lig, file) = load_1iep();
        let precalc = Precalculate::vina();
        let out = local_only(&rec, &lig, &file, &precalc, LocalOnlyOptions::default());

        const UP_TOTAL: f64 = -13.241;
        const UP_INTER: f64 = -18.660;
        const UP_INTRA: f64 = -0.387;
        const UP_CONF: f64 = 5.418;
        const UP_UNBOUND: f64 = -0.387;

        // Drug-like kinase ligand — observed drift ≤ 55 mkcal/mol,
        // gate at 80.
        let tol = 0.08_f64;
        assert_relative_eq!(out.components.total, UP_TOTAL, epsilon = tol);
        assert_relative_eq!(out.components.lig_grids, UP_INTER, epsilon = tol);
        assert_relative_eq!(out.components.lig_intra, UP_INTRA, epsilon = tol);
        assert_relative_eq!(out.components.conf_independent, UP_CONF, epsilon = tol);
        assert_relative_eq!(out.components.intramolecular, UP_UNBOUND, epsilon = tol);
    }

    #[test]
    fn local_only_matches_upstream_on_1s63() {
        // Zinc metalloprotein + halogenated inhibitor.
        check_local_only_parity("1s63", 0.1, -8.993, -12.147, -1.585, 3.154, -1.585);
    }

    #[test]
    fn local_only_matches_upstream_on_1fpu() {
        // Ring-rooted imatinib against a different kinase; BFGS
        // lands in an adjacent local basin vs upstream. Loosen
        // tolerance to 1.0 kcal/mol.
        check_local_only_parity("1fpu", 1.0, -10.927, -15.398, -0.307, 4.471, -0.307);
    }

    #[test]
    fn local_only_matches_upstream_on_bace1() {
        // 22-torsion macrocyclic BACE inhibitor — the deepest
        // non-convex surface in our fixture set. BFGS drift in
        // the intra term compounds across 22 torsion DoFs.
        check_local_only_parity("bace1", 1.5, -7.628, -17.216, -0.878, 9.588, -0.878);
    }
}
