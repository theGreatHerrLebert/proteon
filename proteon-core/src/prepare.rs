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

/// Standard PDB residue names for nucleic-acid (DNA/RNA) monomers, plus the
/// legacy 3-letter spellings. Used to keep untyped nucleic-acid atoms out of
/// the soft "cofactor/ligand" bucket — a nucleic-acid strand is a polymer the
/// FF should cover, not a small het-group, so an untyped one is a hard
/// `incomplete_ff` defect, not `READY_WITH_LIGANDS`.
fn is_nucleic_acid_residue(name: &str) -> bool {
    matches!(
        name,
        // DNA
        "DA" | "DC" | "DG" | "DT" | "DU" | "DI"
        // RNA
        | "A" | "C" | "G" | "U" | "I"
        // legacy 3-letter
        | "ADE" | "CYT" | "GUA" | "THY" | "URA" | "DGN"
    )
}

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
    /// Untyped atoms EXCLUDING water (the raw `n_unassigned` counts waters, which
    /// are always untyped under a protein-only FF). Zero iff every protein and
    /// het atom got a force-field type — the basis for the strict `fully_typed`
    /// gate on the Python side.
    pub n_unassigned_nonwater: usize,
    pub skipped_no_protein: bool,
    /// A size-significant chunk of untyped atoms are in a POLYMER chain — a
    /// standard amino-acid OR nucleic-acid residue (>10 AND >2% of non-water):
    /// a macromolecule the FF should cover is missing params, so its
    /// topology/energy is partially wrong. A hard defect. Computed here (not in
    /// the Python verdict) because the raw `n_unassigned` counts waters and
    /// het-groups too — only the Rust side has the residue-classified counts.
    /// Distinct from `skipped_no_protein` (>50%, not a protein at all) and from
    /// `untyped_cofactors` (untyped small het-groups on an otherwise
    /// well-covered protein, which is NOT a defect).
    pub incomplete_ff: bool,
    /// The protein chain is well covered, but there are untyped NON-WATER,
    /// NON-AMINO-ACID atoms (heme, other cofactors, ligands, ions, modified
    /// residues). These contribute nothing to the force field, but the protein
    /// is still usable — this drives the soft `READY_WITH_LIGANDS` tier rather
    /// than a hard failure. Mutually exclusive with `incomplete_ff` /
    /// `skipped_no_protein` (those take precedence).
    pub untyped_cofactors: bool,
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
    /// Optimizer termination status (`MinimizeStatus::as_str`), e.g.
    /// `"converged_gradient"` / `"line_search_failed"`. Empty when minimization
    /// did not run; lets the supervision layer distinguish a real relax from a
    /// stall instead of trusting a bare `converged` bool.
    pub minimizer_status: String,
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
        let r = crate::reconstruct::reconstruct_fragments(pdb);
        // reconstruct_fragments also adds template hydrogens, but the
        // force-field-aware placer below owns H placement — so when we cleaned H
        // up front, strip the template H again to keep the output heavy-only +
        // FF-consistent (no non-polar C-H leaking under a polar-H force field).
        if opts.strip_hydrogens {
            add_hydrogens::strip_hydrogens(pdb);
        }
        r.heavy_added
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
    out.n_unassigned_nonwater = non_water_unassigned;
    out.skipped_no_protein = non_water_total > 0 && non_water_unassigned * 2 > non_water_total;

    // Classify the non-water unassigned atoms by residue. There are three
    // buckets that matter for the verdict:
    //   * MACROMOLECULAR (protein or nucleic-acid residues) — a polymer chain
    //     the FF should cover but doesn't. A real defect: those atoms enter the
    //     topology with fallback types/zero charge, so energy/minimization is
    //     partial. Drives the HARD `incomplete_ff`.
    //   * HET-GROUP (heme, other cofactors, ligands, ions, modified residues) —
    //     small groups the protein-only FF simply doesn't parameterise. The
    //     protein itself is still usable. Drives the SOFT `untyped_cofactors`.
    // We build the set of residue names pdbtbx classifies as amino acids, plus a
    // static nucleic-acid name set, then bucket each unassigned "RESNAME:atom"
    // string by its residue name.
    // Mirror topology::build_topology exactly: first model only, residue name
    // via `Residue::name().unwrap_or("UNK")` (the same string used as the
    // "RESNAME" prefix of each unassigned_atoms entry), amino-acid test via the
    // first conformer. This keeps the bucketing keys identical to the strings
    // we are bucketing.
    let aa_residue_names: std::collections::HashSet<String> = pdb
        .models()
        .next()
        .into_iter()
        .flat_map(|m| m.chains())
        .flat_map(|c| c.residues())
        .filter(|r| r.conformers().next().is_some_and(|c| c.is_amino_acid()))
        .map(|r| r.name().unwrap_or("UNK").to_string())
        .collect();
    // Untyped atoms in a polymer chain (protein OR nucleic acid). pdbtbx has no
    // nucleic-acid classifier, so nucleic acids are matched by residue name.
    let unassigned_macromol = topo
        .unassigned_atoms
        .iter()
        .filter(|s| {
            let rn = s.split(':').next().unwrap_or("");
            !add_hydrogens::is_water_residue(rn)
                && (aa_residue_names.contains(rn) || is_nucleic_acid_residue(rn))
        })
        .count();
    // Het-group untyped atoms = non-water unassigned that are NOT in a polymer
    // chain (i.e. cofactors / ligands / ions / modified residues).
    let unassigned_cofactor = non_water_unassigned.saturating_sub(unassigned_macromol);

    // HARD: a polymer chain is under-covered. Size-aware: >10 untyped polymer
    // atoms AND >2% of non-water. Both bounds matter — 11 unassigned in a
    // 5000-atom protein is negligible, but 11 in a small peptide is not. This
    // catches protein-chain gaps AND protein–nucleic-acid complexes where the
    // nucleic acid is a sub-50% (so not `skipped_no_protein`) untyped component.
    out.incomplete_ff = !out.skipped_no_protein
        && unassigned_macromol > 10
        && unassigned_macromol * 50 > non_water_total;
    // SOFT: protein well covered, but untyped het-groups are present (cofactors,
    // ligands, ions, modified residues). Usable, not a defect — only set when
    // neither hard condition fired.
    out.untyped_cofactors =
        !out.skipped_no_protein && !out.incomplete_ff && unassigned_cofactor > 0;

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
        out.minimizer_status = result.status.as_str().to_string();
        // Honest even when this branch was entered but the optimizer did no work
        // (e.g. minimize_steps=0 or every atom constrained -> NotRun): `minimized`
        // must reflect that the optimizer actually ran, not just that we tried.
        out.minimized = result.status != minimize::MinimizeStatus::NotRun;
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

#[cfg(test)]
mod tests {
    use super::is_nucleic_acid_residue;

    #[test]
    fn nucleic_acid_residue_names() {
        // DNA / RNA monomers are nucleic acids.
        for n in ["DA", "DC", "DG", "DT", "DU", "DI", "A", "C", "G", "U", "I"] {
            assert!(is_nucleic_acid_residue(n), "{n} should be nucleic acid");
        }
        // Legacy 3-letter spellings.
        for n in ["ADE", "CYT", "GUA", "THY", "URA"] {
            assert!(is_nucleic_acid_residue(n), "{n} should be nucleic acid");
        }
        // Amino acids, het-groups, ions and water are NOT nucleic acids.
        for n in ["ALA", "GLY", "HEM", "ATP", "NA", "ZN", "SO4", "HOH", "UNK"] {
            assert!(
                !is_nucleic_acid_residue(n),
                "{n} should NOT be nucleic acid"
            );
        }
    }
}
