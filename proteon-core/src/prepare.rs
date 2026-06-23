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

/// Canonical amino-acid residue names (the 20) plus protonation / tautomer /
/// disulfide naming variants that are STILL standard residues. Used to flag
/// genuinely NON-standard (modified) residues in a protein chain.
fn is_canonical_amino_acid(name: &str) -> bool {
    matches!(
        name,
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "GLY" | "HIS"
            | "ILE" | "LEU" | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR"
            | "TRP" | "TYR" | "VAL"
        // protonation / tautomer / disulfide variants of the standard 20
            | "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP" | "CYX" | "CYM"
            | "ASH" | "GLH" | "LYN" | "TYM"
    )
}

/// Common modified amino-acid residue codes (PTMs, oxidations, selenomethionine,
/// the 21st/22nd amino acids) that occupy a polymer position but are not
/// standard tokens — a residue-identity / typing hazard for labels.
fn is_modified_amino_acid(name: &str) -> bool {
    matches!(
        name,
        "MSE"
            | "SEP"
            | "TPO"
            | "PTR"
            | "CSO"
            | "CSD"
            | "KCX"
            | "MLY"
            | "M3L"
            | "MLZ"
            | "HYP"
            | "PCA"
            | "CME"
            | "OCS"
            | "SAC"
            | "LLP"
            | "CAS"
            | "SEC"
            | "PYL"
            | "FME"
            | "AYA"
            | "NEP"
            | "SCY"
            | "CSS"
            | "CGU"
    )
}

/// Whether `symbol` is a metal element (coordination chemistry the protein-only
/// force field does not model). Case-insensitive — PDB columns are upper-case,
/// pdbtbx canonicalises to mixed case.
fn is_metal_element(symbol: &str) -> bool {
    matches!(
        symbol.to_ascii_uppercase().as_str(),
        "NA" | "K"
            | "LI"
            | "RB"
            | "CS"
            | "MG"
            | "CA"
            | "SR"
            | "BA"
            | "MN"
            | "FE"
            | "CO"
            | "NI"
            | "CU"
            | "ZN"
            | "CD"
            | "HG"
            | "MO"
            | "W"
            | "V"
            | "CR"
            | "PT"
            | "AU"
            | "AG"
            | "PD"
            | "RU"
            | "RH"
            | "AL"
            | "PB"
    )
}

/// Maximum C(i)–N(i+1) peptide-bond distance (Å) for two consecutive amino-acid
/// residues to count as connected. A real peptide bond is ~1.33 Å; beyond this
/// the backbone is broken (missing residues / a physical gap), which creates a
/// FALSE sequential edge in graph / sequence-indexed labels.
const PEPTIDE_GAP_MAX: f64 = 2.5;

/// Position of the named atom in a residue's primary conformer.
fn residue_atom_pos(residue: &pdbtbx::Residue, name: &str) -> Option<[f64; 3]> {
    crate::altloc::residue_atoms_primary(residue)
        .find(|a| a.name().trim() == name)
        .map(|a| {
            let (x, y, z) = a.pos();
            [x, y, z]
        })
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// CA handedness via the signed volume of (N-CA, C-CA, CB-CA). L-amino acids —
/// the overwhelming norm — give a POSITIVE triple product with this atom order;
/// a negative value is a D-amino acid or a chirality modeling error (sign
/// calibrated on the all-L crambin: 0 outliers). GLY (no CB) has no chirality.
fn ca_chirality_is_d(n: [f64; 3], ca: [f64; 3], c: [f64; 3], cb: [f64; 3]) -> bool {
    let v1 = [n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]];
    let v2 = [c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]];
    let v3 = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
    // (v1 x v2) . v3
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    cross[0] * v3[0] + cross[1] * v3[1] + cross[2] * v3[2] < 0.0
}

/// Count backbone-geometry label hazards in model 0: `(n_chain_gaps,
/// n_chirality_outliers)`. Chain gaps are broken peptide bonds between
/// consecutive amino acids; chirality outliers are non-L CA centres.
fn scan_geometry_hazards(pdb: &pdbtbx::PDB) -> (usize, usize, Vec<usize>) {
    let mut n_gaps = 0;
    let mut chirality_residues: Vec<usize> = Vec::new();
    let Some(model) = pdb.models().next() else {
        return (0, 0, Vec::new());
    };
    // `res_idx` increments for EVERY residue in chain->residue order, identical to
    // the topology's `res_idx` convention, so the recorded chirality-outlier
    // indices align with `clash_residue_indices` and the export's per-chain
    // re-walk (the global all-model-0-residue namespace).
    let mut res_idx = 0usize;
    for chain in model.chains() {
        let mut prev_c: Option<[f64; 3]> = None;
        let mut prev_was_aa = false;
        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if prev_was_aa && is_aa {
                if let (Some(c), Some(n)) = (prev_c, residue_atom_pos(residue, "N")) {
                    if dist3(c, n) > PEPTIDE_GAP_MAX {
                        n_gaps += 1;
                    }
                }
            }
            if is_aa && residue.name() != Some("GLY") {
                if let (Some(n), Some(ca), Some(c), Some(cb)) = (
                    residue_atom_pos(residue, "N"),
                    residue_atom_pos(residue, "CA"),
                    residue_atom_pos(residue, "C"),
                    residue_atom_pos(residue, "CB"),
                ) {
                    if ca_chirality_is_d(n, ca, c, cb) {
                        chirality_residues.push(res_idx);
                    }
                }
            }
            prev_c = residue_atom_pos(residue, "C");
            prev_was_aa = is_aa;
            res_idx += 1;
        }
    }
    (n_gaps, chirality_residues.len(), chirality_residues)
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
            // Max per-atom force target (kcal/mol/Å). 1.0 is the achievable band
            // for the default heavy-atom relaxation of a crystal structure:
            // L-BFGS plateaus with a few strained atoms keeping the max force in
            // 0.1–1.0, so a tighter 0.1 NEVER converges (it just burns the step
            // budget at the same fold — measured CA-RMSD 0.59 Å @0.1 vs 0.53 Å
            // @1.0, energy within 0.3%, converged 0% -> 93% on 30 proteins).
            gradient_tolerance: 1.0,
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
    /// Whether the minimizer was allowed to move HEAVY atoms (vs H-only). False
    /// for the default H-only preparation (heavy atoms frozen — the structure
    /// keeps its experimental coordinates and `final_e` is NOT a heavy-atom
    /// energy minimum, it still carries crystal strain). True only when
    /// `minimized` AND `constrain_heavy` was effectively false. Lets energy/MD
    /// callers tell an equilibrated structure from a faithfully-preserved one.
    pub heavy_relaxed: bool,
    /// Optimizer termination status (`MinimizeStatus::as_str`), e.g.
    /// `"converged_gradient"` / `"line_search_failed"`. Empty when minimization
    /// did not run; lets the supervision layer distinguish a real relax from a
    /// stall instead of trusting a bare `converged` bool.
    pub minimizer_status: String,
    // --- label-safety hazards (for geometric-DL supervision) ---
    /// Heavy-atom steric clashes on the final geometry (see [`crate::clash`]).
    /// Silent label-poison otherwise: H-only minimization cannot relax a
    /// deposited or reconstruction-induced clash away.
    pub n_heavy_clashes: usize,
    /// Heavy (non-hydrogen) atoms considered by the clash scan — the denominator
    /// for the MolProbity-style clashscore (`1000 * n_heavy_clashes / this`).
    pub n_heavy_atoms: usize,
    /// Worst single heavy-atom overlap depth in Å (0.0 when clash-free). A large
    /// value is a catastrophic local defect a size-normalized clashscore hides.
    pub max_heavy_overlap: f64,
    /// `residue_idx` (0-based over ALL model-0 residues, chain→residue order) of
    /// every residue participating in a heavy-atom clash — for per-residue
    /// masking of clash-corrupted coordinate labels.
    pub clash_residue_indices: Vec<usize>,
    /// True if the clash count is APPROXIMATE because the topology used the
    /// distance-inferred bond fallback for un-templated residues (ligands /
    /// non-standard); intra-residue clashes there cannot be told from bonds.
    pub clash_count_inferred: bool,
    /// Number of models in the input. Only model 0 is prepared; `> 1` (e.g. an
    /// NMR ensemble) means a silent model choice was made.
    pub n_models: usize,
    /// True if any residue carries alternate locations (a conformer was silently
    /// chosen — an arbitrary label decision).
    pub has_altlocs: bool,
    /// True if any residue carries a PDB insertion code (residue-identity /
    /// numbering hazard for `(chain, resnum)`-keyed labels).
    pub has_insertion_codes: bool,
    /// HEAVY atoms MISSING from standard residues on the FINAL structure (vs
    /// fragment templates). With reconstruction off (the supervision default) an
    /// incomplete residue otherwise has no signal — a partial coordinate label.
    /// Zero when reconstruction filled everything (those atoms are then flagged
    /// as reconstructed instead).
    pub n_missing_heavy_atoms: usize,
    /// A non-standard / modified amino-acid residue is present (selenomethionine,
    /// a PTM, the 21st/22nd amino acids) — a residue-identity / typing hazard.
    pub has_nonstandard_residues: bool,
    /// A metal atom is present — coordination chemistry the protein-only force
    /// field does not model (an energy-label hazard).
    pub has_metals: bool,
    /// Broken peptide bonds between consecutive amino acids (C(i)–N(i+1) beyond
    /// PEPTIDE_GAP_MAX) — missing residues / physical breaks that create FALSE
    /// sequential adjacency in graph / sequence-indexed labels.
    pub n_chain_gaps: usize,
    /// CA centres with non-L (D) chirality — a D-amino acid or a modeling error;
    /// a coordinate-geometry anomaly a standard L-protein pipeline should see.
    pub n_chirality_outliers: usize,
    /// `residue_idx` (0-based over ALL model-0 residues, chain→residue order) of
    /// every non-L (D) chirality CA centre — for per-residue masking, aligned to
    /// `clash_residue_indices` and the supervision residue order.
    pub chirality_residue_indices: Vec<usize>,
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

    // Coords in topology order; updated to the minimized geometry if
    // minimization runs. Used for the final-geometry clash count below.
    let mut final_coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();

    if !out.skipped_no_protein && opts.minimize && (h_added > 0 || has_any_h) {
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
                &final_coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
            "lbfgs" => minimize::lbfgs(
                &final_coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
            _ => minimize::steepest_descent(
                &final_coords,
                &topo,
                ff,
                opts.minimize_steps,
                opts.gradient_tolerance,
                &constrained,
            ),
        };
        apply_coords_to_pdb(pdb, &result.coords, ff);
        final_coords = result.coords;
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
        // Did heavy atoms actually relax? Only when the optimizer ran with heavy
        // atoms free. Drives the honest "is this an equilibrated structure or a
        // faithfully-preserved one" signal on the report.
        out.heavy_relaxed = out.minimized && !constrain_heavy;
    }

    // --- label-safety hazards ---
    // Heavy-atom clashes on the FINAL geometry (post-minimization if it ran).
    // H-only minimization cannot relax a deposited or reconstruction-induced
    // clash away, so without this the corruption is silent.
    let clash = crate::clash::clash_stats(&final_coords, &topo);
    out.n_heavy_clashes = clash.n_clashes;
    out.n_heavy_atoms = clash.n_heavy_atoms;
    out.max_heavy_overlap = clash.max_overlap;
    out.clash_residue_indices = clash.clash_residues;
    // Approximate whenever ANY un-templated residue was present: the clash count
    // skips every pair touching one (ligands / non-standard / single-atom metals),
    // so its contacts are excluded. Sourced from `inferred_residues` (non-empty),
    // not `inferred_bonds` — a lone metal ion has no inferred bond but its
    // coordination contacts are still excluded (codex).
    out.clash_count_inferred = !topo.inferred_residues.is_empty();
    // Structure-level hazards (model count, altlocs, insertion codes) from the
    // input — silent label decisions otherwise (only model 0 / primary conformer
    // is prepared).
    let sh = scan_structure_hazards(pdb);
    out.n_models = sh.n_models;
    out.has_altlocs = sh.has_altlocs;
    out.has_insertion_codes = sh.has_insertion_codes;
    out.has_nonstandard_residues = sh.has_nonstandard_residues;
    out.has_metals = sh.has_metals;
    let (n_chain_gaps, n_chirality_outliers, chirality_residue_indices) =
        scan_geometry_hazards(pdb);
    out.n_chain_gaps = n_chain_gaps;
    out.n_chirality_outliers = n_chirality_outliers;
    out.chirality_residue_indices = chirality_residue_indices;
    // Missing heavy atoms on the FINAL structure: nonzero only when residues are
    // incomplete AND reconstruction did not fill them (the supervision default).
    out.n_missing_heavy_atoms = crate::reconstruct::count_missing_heavy_atoms(pdb);

    out
}

/// Scan model 0 of `pdb` for label-safety structure hazards: returns
/// `(n_models, has_altlocs, has_insertion_codes)`. Only the first model and the
/// primary conformer are prepared, so `n_models > 1` and `has_altlocs` mark
/// silent selection decisions; insertion codes are a residue-identity hazard for
/// sequence-indexed labels.
struct StructureHazards {
    n_models: usize,
    has_altlocs: bool,
    has_insertion_codes: bool,
    /// A residue is a non-standard / modified amino acid (selenomethionine, a
    /// PTM, the 21st/22nd amino acids) — a residue-identity / typing hazard.
    has_nonstandard_residues: bool,
    /// A metal atom is present — coordination chemistry the protein-only force
    /// field does not model (an energy-label hazard).
    has_metals: bool,
}

fn scan_structure_hazards(pdb: &pdbtbx::PDB) -> StructureHazards {
    let mut out = StructureHazards {
        n_models: pdb.model_count(),
        has_altlocs: false,
        has_insertion_codes: false,
        has_nonstandard_residues: false,
        has_metals: false,
    };
    if let Some(model) = pdb.models().next() {
        for chain in model.chains() {
            for residue in chain.residues() {
                if residue.insertion_code().is_some() {
                    out.has_insertion_codes = true;
                }
                if residue.conformer_count() > 1
                    || residue
                        .conformers()
                        .any(|c| c.alternative_location().is_some())
                {
                    out.has_altlocs = true;
                }
                let name = residue.name().unwrap_or("");
                let is_aa = residue
                    .conformers()
                    .next()
                    .is_some_and(|c| c.is_amino_acid());
                // Non-standard if pdbtbx calls it an amino acid but it isn't a
                // canonical one, OR it is a known modified-residue code (which
                // pdbtbx may not classify as an amino acid at all, e.g. MSE).
                if (is_aa && !is_canonical_amino_acid(name)) || is_modified_amino_acid(name) {
                    out.has_nonstandard_residues = true;
                }
                if crate::altloc::residue_atoms_primary(residue)
                    .any(|a| a.element().is_some_and(|e| is_metal_element(e.symbol())))
                {
                    out.has_metals = true;
                }
            }
        }
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
    use super::{
        is_canonical_amino_acid, is_metal_element, is_modified_amino_acid, is_nucleic_acid_residue,
    };

    #[test]
    fn canonical_vs_modified_amino_acids() {
        for n in ["ALA", "GLY", "HIS", "TRP", "VAL"] {
            assert!(is_canonical_amino_acid(n), "{n} is canonical");
            assert!(!is_modified_amino_acid(n));
        }
        // Protonation / tautomer variants are still canonical (not "modified").
        for n in ["HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN"] {
            assert!(is_canonical_amino_acid(n), "{n} is a canonical variant");
        }
        // Modified residues / PTMs / Sec/Pyl are NOT canonical but ARE modified.
        for n in [
            "MSE", "SEP", "TPO", "PTR", "CSO", "MLY", "HYP", "PCA", "SEC", "PYL",
        ] {
            assert!(!is_canonical_amino_acid(n), "{n} is not canonical");
            assert!(is_modified_amino_acid(n), "{n} is a modified residue");
        }
    }

    #[test]
    fn ca_chirality_flips_under_mirror() {
        // The detector must be chirality-sensitive: mirroring one substituent
        // flips the handedness verdict. (The L/D sign itself is calibrated on the
        // all-L crambin in the Python integration test -> 0 outliers.)
        use super::ca_chirality_is_d;
        let n = [1.0, 1.0, 0.0];
        let ca = [0.0, 0.0, 0.0];
        let c = [1.0, -1.0, 0.0];
        let cb = [0.0, 0.0, 1.0];
        let d1 = ca_chirality_is_d(n, ca, c, cb);
        let d2 = ca_chirality_is_d(n, ca, c, [0.0, 0.0, -1.0]); // mirror CB
        assert_ne!(d1, d2, "mirroring CB must flip the chirality verdict");
    }

    #[test]
    fn metal_elements() {
        for s in ["Zn", "FE", "Mg", "ca", "MN", "Cu", "NA", "K", "Co", "Ni"] {
            assert!(is_metal_element(s), "{s} is a metal");
        }
        // C/N/O/S/H/P and the selenium of MSE are not metals.
        for s in ["C", "N", "O", "S", "H", "P", "Se", "Cl"] {
            assert!(!is_metal_element(s), "{s} is not a metal");
        }
    }

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
