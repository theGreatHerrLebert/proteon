//! Shared pure-Rust structure-preparation pipeline:
//! reconstruct missing atoms → place hydrogens → energy-minimize.
//!
//! This is the orchestration the PyO3 `batch_prepare` binding ran inline; it is
//! hoisted here so the `proteon prepare` / `protonate` / `minimize` CLI commands
//! drive the EXACT same pipeline (and the production 50K battle test and the CLI
//! cannot diverge). `py_add_hydrogens::batch_prepare` and the CLI both build a
//! [`PrepareOptions`] and call [`prepare_structure`]; the three CLI verbs are
//! presets over the one pipeline (protonate = place-H only, minimize = minimize
//! only, prepare = the full thing).

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::add_hydrogens;
use crate::forcefield::{
    minimize,
    params::{self, ForceField},
    topology,
};

/// Knobs for [`prepare_structure`]. [`Default`] mirrors the Python
/// `batch_prepare` signature (reconstruct, hydrogens="all", minimize via lbfgs
/// 500 steps, strip pre-existing H, FF-aware heavy-atom constraint).
#[derive(Clone, Debug)]
pub struct PrepareOptions {
    pub reconstruct: bool,
    /// "backbone" | "general" | "none" | "all".
    pub hydrogens: String,
    pub include_water: bool,
    pub minimize: bool,
    /// "sd" | "cg" | "lbfgs".
    pub minimize_method: String,
    pub minimize_steps: usize,
    pub gradient_tolerance: f64,
    pub strip_hydrogens: bool,
    /// Freeze heavy atoms during minimization (move only H). `None` =
    /// FF-aware: AMBER96 freezes heavy atoms, CHARMM19+EEF1 relaxes them
    /// (united-atom inflated carbon radii need to settle).
    pub constrain_heavy: Option<bool>,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            reconstruct: true,
            hydrogens: "all".to_string(),
            include_water: false,
            minimize: true,
            minimize_method: "lbfgs".to_string(),
            minimize_steps: 500,
            gradient_tolerance: 0.1,
            strip_hydrogens: true,
            constrain_heavy: None,
        }
    }
}

/// Per-structure preparation outcome. Field names match the Python
/// `batch_prepare` result dict so the PyO3 wrapper maps them verbatim.
#[derive(Clone, Debug, Default)]
pub struct PrepareReport {
    pub reconstructed: usize,
    pub h_added: usize,
    pub h_skipped: usize,
    pub n_unassigned: usize,
    pub skipped_no_protein: bool,
    pub init_e: f64,
    pub final_e: f64,
    pub bond_stretch: f64,
    pub angle_bend: f64,
    pub torsion: f64,
    pub improper_torsion: f64,
    pub vdw: f64,
    pub electrostatic: f64,
    pub solvation: f64,
    pub steps: usize,
    pub converged: bool,
    /// Whether the minimization branch actually ran (vs skipped: no H, or
    /// minimize=false, or skipped_no_protein).
    pub minimized: bool,
}

/// Force fields the preparation pipeline supports (`amber96_obc` is not a
/// preparation FF — it only changes the nonbonded solvent term at energy time).
pub const PREPARE_FORCE_FIELDS: &[&str] =
    &["amber", "amber96", "charmm", "charmm19", "charmm19_eef1"];

/// Whether `ff` is a force field [`prepare_from_pdb`] can prepare under.
pub fn is_prepare_force_field(ff: &str) -> bool {
    PREPARE_FORCE_FIELDS.contains(&ff)
}

/// Run the preparation pipeline on `pdb` in place under a concrete force field.
///
/// Mirrors the body the PyO3 `batch_prepare` ran per structure: optional
/// strip-H → optional reconstruct → place hydrogens (polar-only under a
/// united-atom EEF1 force field) → build topology → not-a-protein heuristic →
/// optional minimize (heavy atoms frozen per `constrain_heavy`) → write coords
/// back.
pub fn prepare_structure<P: ForceField>(
    pdb: &mut pdbtbx::PDB,
    opts: &PrepareOptions,
    ff: &P,
) -> PrepareReport {
    let mut out = PrepareReport::default();

    if opts.strip_hydrogens {
        add_hydrogens::strip_hydrogens(pdb);
    }

    out.reconstructed = if opts.reconstruct {
        crate::reconstruct::reconstruct_fragments(pdb).added
    } else {
        0
    };

    // Under a polar-H united-atom force field (CHARMM19+EEF1) only place
    // hydrogens bonded to N/O/S; non-polar C-H are absorbed into united carbon
    // types and must not be placed.
    let polar_only = ff.has_eef1();
    let (h_added, h_skipped) = match opts.hydrogens.as_str() {
        "backbone" => {
            let r = add_hydrogens::place_peptide_hydrogens(pdb);
            (r.added, r.skipped)
        }
        "general" => {
            let r = add_hydrogens::place_general_hydrogens(pdb, opts.include_water);
            (r.added, r.skipped)
        }
        "all" => {
            let r = add_hydrogens::place_all_hydrogens(pdb, polar_only);
            (r.added, r.skipped)
        }
        _ => (0, 0), // "none" and unknown
    };
    out.h_added = h_added;
    out.h_skipped = h_skipped;

    // Build topology once; n_unassigned depends only on residue/atom names, so
    // it is invariant under the coordinate changes minimization makes.
    let topo = topology::build_topology(pdb, ff);
    out.n_unassigned = topo.unassigned_atoms.len();

    // Not-a-protein heuristic: if >50% of NON-WATER atoms have no FF type
    // (nucleic acids, ligand-only entries, exotic residues), skip minimization
    // and flag it. Waters are excluded from numerator and denominator (they are
    // expected to be unassigned under a protein-only FF but don't mean "give
    // up").
    let non_water_total = topo
        .atoms
        .iter()
        .filter(|a| !add_hydrogens::is_water_residue(&a.residue_name))
        .count();
    let non_water_unassigned = topo
        .unassigned_atoms
        .iter()
        .filter(|s| !add_hydrogens::is_water_residue(s.split(':').next().unwrap_or("")))
        .count();
    out.skipped_no_protein = non_water_total > 0 && non_water_unassigned * 2 > non_water_total;

    let has_any_h = crate::altloc::pdb_atoms_primary(pdb).any(|a| {
        a.element()
            .is_some_and(|e| e.symbol() == "H" || e.symbol() == "D")
    });

    if !out.skipped_no_protein && opts.minimize && (h_added > 0 || has_any_h) {
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        // FF-aware default: AMBER96 freezes heavy atoms, CHARMM19+EEF1 relaxes
        // them. `!ff.has_eef1()` reproduces the per-FF default the PyO3 caller
        // set explicitly (amber → true, charmm → false).
        let constrain_heavy = opts.constrain_heavy.unwrap_or(!ff.has_eef1());
        let constrained: Vec<bool> = if constrain_heavy {
            topo.atoms.iter().map(|a| !a.is_hydrogen).collect()
        } else {
            vec![false; topo.atoms.len()]
        };
        let result = match opts.minimize_method.as_str() {
            "cg" => minimize::conjugate_gradient(
                &coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
            "lbfgs" => minimize::lbfgs(
                &coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
            _ => minimize::steepest_descent(
                &coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
        };
        apply_coords_to_pdb(pdb, &result.coords, ff);
        out.init_e = result.initial_energy;
        out.final_e = result.energy.total;
        out.bond_stretch = result.energy.bond_stretch;
        out.angle_bend = result.energy.angle_bend;
        out.torsion = result.energy.torsion;
        out.improper_torsion = result.energy.improper_torsion;
        out.vdw = result.energy.vdw;
        out.electrostatic = result.energy.electrostatic;
        out.solvation = result.energy.solvation;
        out.steps = result.steps;
        out.converged = result.converged;
        out.minimized = true;
    }

    out
}

/// Prepare `pdb` in place, dispatching on a force-field string. Returns `Err`
/// for an unknown FF or if the pipeline panics (so both the PyO3 wrapper and
/// the CLI get a `Result` instead of an uncatchable panic; on the CLI side a
/// failure isolates to that file).
pub fn prepare_from_pdb(
    pdb: &mut pdbtbx::PDB,
    ff: &str,
    opts: &PrepareOptions,
) -> Result<PrepareReport, String> {
    let computed = catch_unwind(AssertUnwindSafe(|| -> Result<PrepareReport, String> {
        match ff {
            "amber" | "amber96" => Ok(prepare_structure(pdb, opts, &params::amber96())),
            "charmm" | "charmm19" | "charmm19_eef1" => {
                Ok(prepare_structure(pdb, opts, &params::charmm19_eef1()))
            }
            _ => Err(format!(
                "Unknown force field '{ff}'. Use 'charmm19_eef1' or 'amber96'."
            )),
        }
    }));
    match computed {
        Ok(result) => result,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "internal panic with no message".to_string());
            Err(format!(
                "structure preparation failed on this input (usually an \
                 unparameterized residue or atom): {detail}"
            ))
        }
    }
}

/// Apply minimized coordinates back to `pdb`, walking chains → residues →
/// primary conformer → atoms in the same order and with the same
/// `should_include_atom` filter as `build_topology`, so the flat `coords` array
/// stays aligned. Panics if the array length and atom count disagree.
pub fn apply_coords_to_pdb<F: ForceField + ?Sized>(
    pdb: &mut pdbtbx::PDB,
    coords: &[[f64; 3]],
    params: &F,
) {
    let mut idx = 0;
    let first_model = match pdb.models_mut().next() {
        Some(m) => m,
        None => return,
    };
    for chain in first_model.chains_mut() {
        for residue in chain.residues_mut() {
            let res_name = residue.name().unwrap_or("UNK").to_string();

            let primary_alt: Option<Option<String>> = {
                let blank = residue
                    .conformers()
                    .find(|c| c.alternative_location().is_none());
                let a = residue
                    .conformers()
                    .find(|c| c.alternative_location() == Some("A"));
                blank
                    .or(a)
                    .or_else(|| residue.conformers().next())
                    .map(|c| c.alternative_location().map(str::to_string))
            };
            let Some(target_alt) = primary_alt else {
                continue;
            };

            for conformer in residue.conformers_mut() {
                let matches = match (conformer.alternative_location(), target_alt.as_deref()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };
                if !matches {
                    continue;
                }
                for atom in conformer.atoms_mut() {
                    let atom_name = atom.name().trim().to_string();
                    let element = atom
                        .element()
                        .map(|e| e.symbol().to_string())
                        .unwrap_or_else(|| "C".to_string());
                    if !topology::should_include_atom(
                        &res_name, &atom_name, &element, params, &res_name,
                    ) {
                        continue;
                    }
                    assert!(
                        idx < coords.len(),
                        "apply_coords_to_pdb: coord array too short ({} coords, atom index {})",
                        coords.len(),
                        idx,
                    );
                    atom.set_pos((coords[idx][0], coords[idx][1], coords[idx][2]))
                        .expect("apply_coords_to_pdb: invalid coordinates (NaN/Inf)");
                    idx += 1;
                }
                break;
            }
        }
    }
    assert_eq!(
        idx,
        coords.len(),
        "apply_coords_to_pdb: coord array length ({}) != atom count ({})",
        coords.len(),
        idx,
    );
}
