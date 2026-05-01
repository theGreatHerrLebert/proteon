//! Force field parameters and the `ForceField` trait.
//!
//! Provides a common interface for AMBER96 and CHARMM19+EEF1 force fields.
//! All energy/topology/MD functions are generic over `impl ForceField`.
//!
//! References:
//! - Neria, Fischer, & Karplus, "Simulation of activation free energies in
//!   molecular systems", *J. Chem. Phys.* 105(5), 1902-1921 (1996) — the
//!   CHARMM19 parameter set.
//! - Lazaridis & Karplus, "Effective energy function for proteins in
//!   solution", *Proteins* 35(2), 133-152 (1999) — the EEF1 implicit-
//!   solvation model paired with CHARMM19 here.
//! - Cornell et al., "A Second Generation Force Field for the Simulation
//!   of Proteins, Nucleic Acids, and Organic Molecules", *J. Am. Chem.
//!   Soc.* 117(19), 5179-5197 (1995) — the AMBER94/96 parameter family.

use std::collections::{HashMap, HashSet};

/// Atom type assignment: residue_name:atom_name → (type, charge)
#[derive(Clone, Debug)]
pub struct AtomTypeEntry {
    pub amber_type: String, // force field atom type (works for both AMBER and CHARMM)
    pub charge: f64,        // in elementary charge units
}

/// EEF1 implicit solvation parameters per atom type.
#[derive(Clone, Debug)]
pub struct EEF1Param {
    pub volume: f64,  // van der Waals volume (ų)
    pub dg_ref: f64,  // reference solvation free energy (kcal/mol)
    pub dg_free: f64, // solvation free energy for Gaussian exclusion (kcal/mol)
    pub sigma: f64,   // Gaussian width (Å)
    pub r_min: f64,   // minimum interaction radius (Å)
}

/// Common interface for force field parameter lookup.
///
/// Implemented by `AmberParams` and `CharmmParams`. All energy computation,
/// topology building, minimization, and MD functions are generic over this trait.
pub trait ForceField: Send + Sync {
    fn get_atom_type(&self, residue: &str, atom: &str) -> Option<&AtomTypeEntry>;
    fn get_bond(&self, type_a: &str, type_b: &str) -> Option<&BondParam>;
    fn get_angle(&self, type_a: &str, type_b: &str, type_c: &str) -> Option<&AngleParam>;
    fn get_torsion(&self, a: &str, b: &str, c: &str, d: &str) -> Option<&Vec<TorsionTerm>>;
    fn get_improper_torsion(&self, a: &str, b: &str, c: &str, d: &str)
        -> Option<&Vec<TorsionTerm>>;
    fn is_improper_center(&self, residue: &str, atom: &str) -> bool;
    fn get_lj(&self, atype: &str) -> Option<&LJParam>;
    fn scee(&self) -> f64;
    fn scnb(&self) -> f64;
    /// EEF1 solvation parameters (None for force fields without implicit solvent).
    fn get_eef1(&self, _atype: &str) -> Option<&EEF1Param> {
        None
    }
    /// Whether this force field has EEF1 solvation enabled.
    fn has_eef1(&self) -> bool {
        false
    }

    /// OBC GB per-atom parameters (None for force fields without OBC).
    /// Phase A scaffolding — Phase B will populate AmberParams.
    fn get_obc_gb(&self, _atype: &str) -> Option<&crate::forcefield::gb_obc::ObcAtomParams> {
        None
    }
    /// Whether this force field has OBC GB solvation enabled.
    fn has_obc_gb(&self) -> bool {
        false
    }

    /// Nonbonded interaction cutoff (Å). Pairs beyond this distance are
    /// ignored in the LJ + Coulomb loops and the NBL builder.
    ///
    /// Default: 15.0 Å (conservative, no long-range treatment). Override
    /// for force fields parameterized with shorter cutoffs: CHARMM19+EEF1
    /// uses 9.0 Å per BALL's `@CTOFNB=9.0` (the EEF1 solvation damping
    /// makes the energy less sensitive to truncation), giving a ~4.6×
    /// reduction in NBL pairs vs 15 Å.
    fn nonbonded_cutoff(&self) -> f64 {
        15.0
    }

    /// Distance at which the switching function begins to taper interactions.
    /// Must be ≤ `nonbonded_cutoff()`. Interactions are full-strength below
    /// this distance, smoothly tapered between cuton and cutoff, and zero
    /// above cutoff.
    ///
    /// Default: 13.0 Å (consistent with the 15 Å cutoff). CHARMM19+EEF1
    /// overrides to 7.0 Å per BALL's `@CTONNB=7.0`.
    fn switching_on(&self) -> f64 {
        13.0
    }
}

/// Bond stretch parameters: E = k * (r - r0)²
#[derive(Clone, Debug)]
pub struct BondParam {
    pub k: f64,  // kcal/mol/Å²
    pub r0: f64, // Å
}

/// Angle bend parameters: E = k * (θ - θ0)²
#[derive(Clone, Debug)]
pub struct AngleParam {
    pub k: f64,      // kcal/mol/rad²
    pub theta0: f64, // radians
}

/// Single torsion term: E = (V/div) * (1 + cos(f*φ - φ0))
#[derive(Clone, Debug)]
pub struct TorsionTerm {
    pub v: f64,    // barrier height (kcal/mol)
    pub phi0: f64, // phase (radians)
    pub f: f64,    // periodicity
    pub div: f64,  // divisor
}

/// Lennard-Jones parameters per atom type
#[derive(Clone, Debug)]
pub struct LJParam {
    pub r: f64,       // van der Waals radius (Å)
    pub epsilon: f64, // well depth (kcal/mol)
}

/// Complete AMBER force field parameter set.
#[derive(Clone, Debug)]
pub struct AmberParams {
    /// Residue:Atom → (type, charge)
    pub atom_types: HashMap<String, AtomTypeEntry>,
    /// Wildcard (*:Atom) entries
    pub wildcard_types: HashMap<String, AtomTypeEntry>,
    /// (type_i, type_j) → BondParam (sorted key)
    pub bonds: HashMap<(String, String), BondParam>,
    /// (type_i, type_j, type_k) → AngleParam (sorted outer key)
    pub angles: HashMap<(String, String, String), AngleParam>,
    /// (type_i, type_j, type_k, type_l) → Vec<TorsionTerm>
    pub torsions: HashMap<(String, String, String, String), Vec<TorsionTerm>>,
    /// Improper torsion parameters (separate from proper)
    pub improper_torsions: HashMap<(String, String, String, String), Vec<TorsionTerm>>,
    /// Atoms that should have improper torsions (e.g., "ALA:N", "ALA:C")
    pub residue_impropers: HashSet<String>,
    /// type → LJParam
    pub lj: HashMap<String, LJParam>,
    /// 1-4 electrostatic scaling factor
    pub scee: f64,
    /// 1-4 vdW scaling factor
    pub scnb: f64,
    /// Optional runtime override for the nonbonded cutoff (Å). When
    /// `None`, the trait default (15 Å for AMBER96) is used. Override
    /// is intended for oracle-style cross-tool comparisons where both
    /// tools must see the same pair set — e.g. setting this to 1e6
    /// effectively disables the cutoff so proteon matches an OpenMM
    /// `NoCutoff` oracle.
    pub cutoff_override: Option<f64>,
    /// Optional runtime override for the switching-on distance (Å).
    /// If `cutoff_override` is set but this is not, switching is disabled
    /// by setting the on-distance to `cutoff_override - 1e-9`.
    pub switching_on_override: Option<f64>,
    /// OBC GB per-atom parameters keyed by AMBER class name (e.g. "CT",
    /// "N", "HO"). Empty unless populated via [`AmberParams::with_obc_ini`]
    /// or [`amber96_obc`]; when empty, `has_obc_gb()` returns false and
    /// the GB solvation term is skipped.
    pub obc_gb: HashMap<String, crate::forcefield::gb_obc::ObcAtomParams>,
}

fn sorted_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn sorted_triple(a: &str, b: &str, c: &str) -> (String, String, String) {
    if a <= c {
        (a.to_string(), b.to_string(), c.to_string())
    } else {
        (c.to_string(), b.to_string(), a.to_string())
    }
}

impl AmberParams {
    /// Parse AMBER parameters from INI file content.
    pub fn from_ini(content: &str) -> Self {
        let mut params = AmberParams {
            atom_types: HashMap::new(),
            wildcard_types: HashMap::new(),
            bonds: HashMap::new(),
            angles: HashMap::new(),
            torsions: HashMap::new(),
            improper_torsions: HashMap::new(),
            residue_impropers: HashSet::new(),
            lj: HashMap::new(),
            scee: 1.2,
            scnb: 2.0,
            cutoff_override: None,
            switching_on_override: None,
            obc_gb: HashMap::new(),
        };

        let mut section = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            // Section header
            if line.starts_with('[') {
                section = line
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                continue;
            }

            // Properties
            if line.starts_with('@') {
                if let Some(rest) = line.strip_prefix("@SCEE=") {
                    if let Ok(v) = rest.parse::<f64>() {
                        params.scee = v;
                    }
                }
                continue;
            }

            // Header lines
            if line.starts_with("ver:") || line.starts_with("key:") {
                continue;
            }

            // Parse data lines by section
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }

            match section.as_str() {
                "QuadraticBondStretch"
                    // ver I J k r0 comment
                    if fields.len() >= 5 => {
                        let key = sorted_pair(fields[1], fields[2]);
                        if let (Ok(k), Ok(r0)) =
                            (fields[3].parse::<f64>(), fields[4].parse::<f64>())
                        {
                            params.bonds.insert(key, BondParam { k, r0 });
                        }
                    }
                "QuadraticAngleBend"
                    // ver I J K k theta0 comment
                    if fields.len() >= 6 => {
                        let key = sorted_triple(fields[1], fields[2], fields[3]);
                        if let (Ok(k), Ok(theta0_deg)) =
                            (fields[4].parse::<f64>(), fields[5].parse::<f64>())
                        {
                            params.angles.insert(
                                key,
                                AngleParam {
                                    k,
                                    theta0: theta0_deg.to_radians(),
                                },
                            );
                        }
                    }
                "Torsions"
                    // ver I J K L N div V phi0 f comment
                    if fields.len() >= 10 => {
                        let key = (
                            fields[1].to_string(),
                            fields[2].to_string(),
                            fields[3].to_string(),
                            fields[4].to_string(),
                        );
                        if let (Ok(div), Ok(v), Ok(phi0_deg), Ok(f)) = (
                            fields[6].parse::<f64>(),
                            fields[7].parse::<f64>(),
                            fields[8].parse::<f64>(),
                            fields[9].parse::<f64>(),
                        ) {
                            let term = TorsionTerm {
                                v,
                                phi0: phi0_deg.to_radians(),
                                f,
                                div: div.max(1.0),
                            };
                            params.torsions.entry(key).or_default().push(term);
                        }
                    }
                "ImproperTorsions"
                    // Same format as Torsions but stored separately
                    if fields.len() >= 10 => {
                        let key = (
                            fields[1].to_string(),
                            fields[2].to_string(),
                            fields[3].to_string(),
                            fields[4].to_string(),
                        );
                        if let (Ok(div), Ok(v), Ok(phi0_deg), Ok(f)) = (
                            fields[6].parse::<f64>(),
                            fields[7].parse::<f64>(),
                            fields[8].parse::<f64>(),
                            fields[9].parse::<f64>(),
                        ) {
                            let term = TorsionTerm {
                                v,
                                phi0: phi0_deg.to_radians(),
                                f,
                                div: div.max(1.0),
                            };
                            params.improper_torsions.entry(key).or_default().push(term);
                        }
                    }
                "ResidueImproperTorsions"
                    // Single column: residue:atom names
                    if !fields.is_empty() => {
                        let name = fields[0].trim().to_string();
                        if name.contains(':') {
                            params.residue_impropers.insert(name);
                        }
                    }
                "LennardJones"
                    // ver I R epsilon comment
                    if fields.len() >= 4 => {
                        if let (Ok(r), Ok(eps)) =
                            (fields[2].parse::<f64>(), fields[3].parse::<f64>())
                        {
                            params
                                .lj
                                .insert(fields[1].to_string(), LJParam { r, epsilon: eps });
                        }
                    }
                "ChargesAndTypeNames"
                    // ver name q type
                    if fields.len() >= 4 => {
                        let name = fields[1]; // e.g., "ALA:N"
                        if let Ok(q) = fields[2].parse::<f64>() {
                            let atype = fields[3].to_string();
                            let entry = AtomTypeEntry {
                                amber_type: atype,
                                charge: q,
                            };
                            if let Some(atom_name) = name.strip_prefix("*:") {
                                params.wildcard_types.insert(atom_name.to_string(), entry);
                            } else {
                                params.atom_types.insert(name.to_string(), entry);
                            }
                        }
                    }
                _ => {}
            }
        }

        params
    }

    /// Look up atom type and charge for a given residue:atom pair.
    pub fn get_atom_type(&self, residue: &str, atom: &str) -> Option<&AtomTypeEntry> {
        let key = format!("{residue}:{atom}");
        self.atom_types
            .get(&key)
            .or_else(|| self.wildcard_types.get(atom))
    }

    /// Look up bond parameters for two atom types.
    pub fn get_bond(&self, type_a: &str, type_b: &str) -> Option<&BondParam> {
        let key = sorted_pair(type_a, type_b);
        self.bonds.get(&key)
    }

    /// Look up angle parameters.
    pub fn get_angle(&self, type_a: &str, type_b: &str, type_c: &str) -> Option<&AngleParam> {
        let key = sorted_triple(type_a, type_b, type_c);
        self.angles.get(&key)
    }

    /// Look up torsion parameters using BALL's 9-pattern fallback order.
    ///
    /// Tries: exact, reverse, single-wildcard ends, double-wildcard, then
    /// double-wildcard on inner atoms. Matches BiochemicalAlgorithms.jl
    /// `_try_assign_torsion!` pattern.
    pub fn get_torsion(
        &self,
        type_a: &str,
        type_b: &str,
        type_c: &str,
        type_d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        let (a, b, c, d) = (
            type_a.to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        let w = "*".to_string();
        // BALL fallback order (9 patterns):
        self.torsions
            .get(&(a.clone(), b.clone(), c.clone(), d.clone())) // exact
            .or_else(|| {
                self.torsions
                    .get(&(d.clone(), c.clone(), b.clone(), a.clone()))
            }) // reverse
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), b.clone(), c.clone(), d.clone()))
            }) // *-b-c-d
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), c.clone(), b.clone(), a.clone()))
            }) // *-c-b-a
            .or_else(|| {
                self.torsions
                    .get(&(a.clone(), b.clone(), c.clone(), w.clone()))
            }) // a-b-c-*
            .or_else(|| {
                self.torsions
                    .get(&(d.clone(), c.clone(), b.clone(), w.clone()))
            }) // d-c-b-*
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), b.clone(), c.clone(), w.clone()))
            }) // *-b-c-*
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), c.clone(), b.clone(), w.clone()))
            }) // *-c-b-*
    }

    /// Look up improper torsion parameters.
    ///
    /// AMBER's improper-torsion templates commonly use double wildcards in
    /// the first two positions to match "any two substituents around a
    /// given center with a specific terminal atom" — e.g. `* * N H`
    /// enforces amide-plane planarity for every backbone N regardless of
    /// what else the N is attached to. Without the double-wildcard
    /// fallback, proteon silently failed to find these entries, leading
    /// to only ~15 impropers on a 46-residue protein (should be ~92).
    /// The diagnostic that surfaced this: 2026-04-13 OpenMM oracle on
    /// crambin showed 113 "omm-only" torsions all of the form
    /// `Cprev-CA-N-H` and `CA-Nnext-C-O` — both amide-plane impropers.
    ///
    /// Fallback order (7 patterns):
    ///   exact → reverse → single-wildcard outer → double-wildcard outer.
    pub fn get_improper_torsion(
        &self,
        type_a: &str,
        type_b: &str,
        type_c: &str,
        type_d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        let (a, b, c, d) = (
            type_a.to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        let w = "*".to_string();
        self.improper_torsions
            .get(&(a.clone(), b.clone(), c.clone(), d.clone()))
            .or_else(|| {
                self.improper_torsions
                    .get(&(d.clone(), c.clone(), b.clone(), a.clone()))
            })
            .or_else(|| {
                self.improper_torsions
                    .get(&(w.clone(), b.clone(), c.clone(), d.clone()))
            })
            .or_else(|| {
                self.improper_torsions
                    .get(&(a.clone(), b.clone(), c.clone(), w.clone()))
            })
            // Double-wildcard outer (AMBER amide-plane pattern).
            .or_else(|| {
                self.improper_torsions
                    .get(&(w.clone(), w.clone(), c.clone(), d.clone()))
            })
            .or_else(|| {
                self.improper_torsions
                    .get(&(d.clone(), c.clone(), w.clone(), w.clone()))
            })
        // `d-c-*-*` reverse is equivalent to `*-*-c-d` on an already-
        // canonical stored form; include both to handle either store.
    }

    // kept for backward compatibility — old 3-pattern version removed
    #[allow(dead_code)]
    fn _get_improper_torsion_old(
        &self,
        type_a: &str,
        type_b: &str,
        type_c: &str,
        type_d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        let key = (
            type_a.to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        if let Some(terms) = self.improper_torsions.get(&key) {
            return Some(terms);
        }
        let key = (
            "*".to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        if let Some(terms) = self.improper_torsions.get(&key) {
            return Some(terms);
        }
        let key = (
            "*".to_string(),
            "*".to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        self.improper_torsions.get(&key)
    }

    /// Check if an atom should have improper torsions.
    pub fn is_improper_center(&self, residue: &str, atom: &str) -> bool {
        let key = format!("{residue}:{atom}");
        self.residue_impropers.contains(&key)
    }

    /// Look up LJ parameters for an atom type.
    pub fn get_lj(&self, atype: &str) -> Option<&LJParam> {
        self.lj.get(atype)
    }
}

impl AmberParams {
    /// Layer OBC GB per-atom parameters onto an existing parameter set by
    /// parsing the `[OBCSolvation]` section of the supplied INI content.
    ///
    /// Each data row is `ver class radius scale` with radius in Å. Rows
    /// with a class that is already present overwrite the prior entry.
    /// Lines outside `[OBCSolvation]` and metadata lines (starting with
    /// `@`, `ver:`, `key:`, `value:`, `;`) are ignored.
    #[must_use]
    pub fn with_obc_ini(mut self, content: &str) -> Self {
        use crate::forcefield::gb_obc::ObcAtomParams;
        let mut section = String::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with(';')
                || line.starts_with('@')
                || line.starts_with("ver:")
                || line.starts_with("key:")
                || line.starts_with("value:")
            {
                continue;
            }
            if line.starts_with('[') {
                section = line
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                continue;
            }
            if section != "OBCSolvation" {
                continue;
            }
            // Strip trailing "; comment" before splitting.
            let payload = line.split(';').next().unwrap_or(line).trim();
            let fields: Vec<&str> = payload.split_whitespace().collect();
            if fields.len() < 4 {
                continue;
            }
            let class = fields[1].to_string();
            if let (Ok(radius), Ok(scale)) = (fields[2].parse::<f64>(), fields[3].parse::<f64>()) {
                self.obc_gb.insert(class, ObcAtomParams { radius, scale });
            }
        }
        self
    }
}

impl ForceField for AmberParams {
    fn get_atom_type(&self, residue: &str, atom: &str) -> Option<&AtomTypeEntry> {
        self.get_atom_type(residue, atom)
    }
    fn get_bond(&self, type_a: &str, type_b: &str) -> Option<&BondParam> {
        self.get_bond(type_a, type_b)
    }
    fn get_angle(&self, type_a: &str, type_b: &str, type_c: &str) -> Option<&AngleParam> {
        self.get_angle(type_a, type_b, type_c)
    }
    fn get_torsion(&self, a: &str, b: &str, c: &str, d: &str) -> Option<&Vec<TorsionTerm>> {
        self.get_torsion(a, b, c, d)
    }
    fn get_improper_torsion(
        &self,
        a: &str,
        b: &str,
        c: &str,
        d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        self.get_improper_torsion(a, b, c, d)
    }
    fn is_improper_center(&self, residue: &str, atom: &str) -> bool {
        self.is_improper_center(residue, atom)
    }
    fn get_lj(&self, atype: &str) -> Option<&LJParam> {
        self.get_lj(atype)
    }
    fn scee(&self) -> f64 {
        self.scee
    }
    fn scnb(&self) -> f64 {
        self.scnb
    }
    fn nonbonded_cutoff(&self) -> f64 {
        self.cutoff_override.unwrap_or(15.0)
    }
    fn switching_on(&self) -> f64 {
        self.switching_on_override
            .or_else(|| self.cutoff_override.map(|c| c - 1e-9))
            .unwrap_or(13.0)
    }
    fn get_obc_gb(&self, atype: &str) -> Option<&crate::forcefield::gb_obc::ObcAtomParams> {
        self.obc_gb.get(atype)
    }
    fn has_obc_gb(&self) -> bool {
        !self.obc_gb.is_empty()
    }
}

/// Load the embedded AMBER96 parameter set.
pub fn amber96() -> AmberParams {
    AmberParams::from_ini(include_str!("../../data/amber96.ini"))
}

/// Load AMBER96 with OBC GB per-atom parameters layered on — i.e. the
/// equivalent of OpenMM's `ForceField("amber96.xml", "amber96_obc.xml")`.
pub fn amber96_obc() -> AmberParams {
    amber96().with_obc_ini(include_str!("../../data/amber96_obc.ini"))
}

// ---------------------------------------------------------------------------
// CHARMM19 + EEF1 force field
// ---------------------------------------------------------------------------

/// CHARMM19 force field parameters with EEF1 implicit solvation.
#[derive(Clone, Debug)]
pub struct CharmmParams {
    /// Same bonded/nonbonded parameter storage as AMBER
    pub atom_types: HashMap<String, AtomTypeEntry>,
    pub wildcard_types: HashMap<String, AtomTypeEntry>,
    pub bonds: HashMap<(String, String), BondParam>,
    pub angles: HashMap<(String, String, String), AngleParam>,
    pub torsions: HashMap<(String, String, String, String), Vec<TorsionTerm>>,
    pub improper_torsions: HashMap<(String, String, String, String), Vec<TorsionTerm>>,
    pub residue_impropers: HashSet<String>,
    pub lj: HashMap<String, LJParam>,
    pub scee: f64,
    pub scnb: f64,
    /// EEF1 solvation parameters per atom type
    pub eef1: HashMap<String, EEF1Param>,
    /// Optional runtime override for the nonbonded cutoff (Å). When
    /// `None`, the canonical CHARMM19+EEF1 cutoff (9 Å, from BALL's
    /// `param19_eef1.ini` `@CTOFNB=9.0`) is used. Override is intended
    /// for oracle-style cross-tool comparisons where both tools must
    /// see the same pair set — e.g. setting this to 1e6 effectively
    /// disables the cutoff so proteon matches a BALL `nonbonded_cutoff
    /// =1e6` reference. EEF1 solvation makes CHARMM's energy less
    /// sensitive to long-range truncation than AMBER, so the production
    /// 9 Å default is safe; the override is purely an oracle knob.
    pub cutoff_override: Option<f64>,
    /// Optional runtime override for the switching-on distance (Å).
    /// If `cutoff_override` is set but this is not, switching is
    /// disabled by setting the on-distance to `cutoff_override - 1e-9`,
    /// matching the AMBER side's convention.
    pub switching_on_override: Option<f64>,
}

impl CharmmParams {
    /// Parse CHARMM parameters from BALL INI file content.
    pub fn from_ini(content: &str) -> Self {
        // Reuse the AMBER INI parser for bonded/nonbonded terms (same format)
        let mut amber = AmberParams::from_ini(content);
        let mut eef1 = HashMap::new();

        // Parse EEF1 solvation section + CHARMM-specific @-directives
        // (notably @E14FAC, the 1-4 electrostatic scaling).
        let mut section = String::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with(';')
                || line.starts_with("ver:")
                || line.starts_with("key:")
                || line.starts_with("value:")
            {
                continue;
            }
            // CHARMM ships @E14FAC=0.4 as the 1-4 electrostatic
            // *multiplier* (E_14 = q_i*q_j/r * E14FAC). proteon's energy
            // code uses an AMBER-style *divisor* via `scee`
            // (scale_es = 1/scee). Translate: scee = 1/E14FAC.
            // Without this, proteon falls back to the AMBER scee=1.2
            // default and over-scales 1-4 Coulomb by ~2× on CHARMM
            // systems. AMBER files don't have @E14FAC so this branch is
            // CHARMM-specific in practice.
            if let Some(rest) = line.strip_prefix("@E14FAC=") {
                if let Ok(v) = rest.parse::<f64>() {
                    if v > 1e-10 {
                        amber.scee = 1.0 / v;
                    }
                }
                continue;
            }
            if line.starts_with('@') {
                continue;
            }
            if line.starts_with('[') {
                section = line
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                continue;
            }
            if section == "EEF1Solvation" {
                let fields: Vec<&str> = line.split_whitespace().collect();
                // format: ver type V dG_ref dG_free dH_ref Cp_ref sig_w R_min
                if fields.len() >= 9 {
                    let atype = fields[1].to_string();
                    if let (Ok(v), Ok(dg_ref), Ok(dg_free), Ok(sigma), Ok(r_min)) = (
                        fields[2].parse::<f64>(),
                        fields[3].parse::<f64>(),
                        fields[4].parse::<f64>(),
                        fields[7].parse::<f64>(),
                        fields[8].parse::<f64>(),
                    ) {
                        // Skip hydrogen types (volume = 0 and dG = 0)
                        if v.abs() > 1e-10 || dg_ref.abs() > 1e-10 || dg_free.abs() > 1e-10 {
                            eef1.insert(
                                atype,
                                EEF1Param {
                                    volume: v,
                                    dg_ref,
                                    dg_free,
                                    sigma,
                                    r_min,
                                },
                            );
                        }
                    }
                }
            }
        }

        CharmmParams {
            atom_types: amber.atom_types,
            wildcard_types: amber.wildcard_types,
            bonds: amber.bonds,
            angles: amber.angles,
            torsions: amber.torsions,
            improper_torsions: amber.improper_torsions,
            residue_impropers: amber.residue_impropers,
            lj: amber.lj,
            scee: amber.scee,
            scnb: amber.scnb,
            eef1,
            cutoff_override: None,
            switching_on_override: None,
        }
    }
}

impl ForceField for CharmmParams {
    fn get_atom_type(&self, residue: &str, atom: &str) -> Option<&AtomTypeEntry> {
        let key = format!("{residue}:{atom}");
        self.atom_types
            .get(&key)
            .or_else(|| self.wildcard_types.get(atom))
    }
    fn get_bond(&self, type_a: &str, type_b: &str) -> Option<&BondParam> {
        let key = sorted_pair(type_a, type_b);
        self.bonds.get(&key)
    }
    fn get_angle(&self, type_a: &str, type_b: &str, type_c: &str) -> Option<&AngleParam> {
        let key = sorted_triple(type_a, type_b, type_c);
        self.angles.get(&key)
    }
    fn get_torsion(
        &self,
        type_a: &str,
        type_b: &str,
        type_c: &str,
        type_d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        let (a, b, c, d) = (
            type_a.to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        let w = "*".to_string();
        self.torsions
            .get(&(a.clone(), b.clone(), c.clone(), d.clone()))
            .or_else(|| {
                self.torsions
                    .get(&(d.clone(), c.clone(), b.clone(), a.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), b.clone(), c.clone(), d.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), c.clone(), b.clone(), a.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(a.clone(), b.clone(), c.clone(), w.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(d.clone(), c.clone(), b.clone(), w.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), b.clone(), c.clone(), w.clone()))
            })
            .or_else(|| {
                self.torsions
                    .get(&(w.clone(), c.clone(), b.clone(), w.clone()))
            })
    }
    fn get_improper_torsion(
        &self,
        type_a: &str,
        type_b: &str,
        type_c: &str,
        type_d: &str,
    ) -> Option<&Vec<TorsionTerm>> {
        let (a, b, c, d) = (
            type_a.to_string(),
            type_b.to_string(),
            type_c.to_string(),
            type_d.to_string(),
        );
        let w = "*".to_string();
        self.improper_torsions
            .get(&(a.clone(), b.clone(), c.clone(), d.clone()))
            .or_else(|| {
                self.improper_torsions
                    .get(&(d.clone(), c.clone(), b.clone(), a.clone()))
            })
            .or_else(|| {
                self.improper_torsions
                    .get(&(w.clone(), b.clone(), c.clone(), d.clone()))
            })
            .or_else(|| {
                self.improper_torsions
                    .get(&(a.clone(), b.clone(), c.clone(), w.clone()))
            })
    }
    fn is_improper_center(&self, residue: &str, atom: &str) -> bool {
        let key = format!("{residue}:{atom}");
        self.residue_impropers.contains(&key)
    }
    fn get_lj(&self, atype: &str) -> Option<&LJParam> {
        self.lj.get(atype)
    }
    fn scee(&self) -> f64 {
        self.scee
    }
    fn scnb(&self) -> f64 {
        self.scnb
    }
    fn get_eef1(&self, atype: &str) -> Option<&EEF1Param> {
        self.eef1.get(atype)
    }
    fn has_eef1(&self) -> bool {
        !self.eef1.is_empty()
    }
    /// CHARMM19+EEF1 canonical cutoff from BALL's param19_eef1.ini:
    /// `@CTOFNB=9.0`. EEF1 solvation makes the energy less sensitive
    /// to long-range truncation, so the short cutoff is safe and gives
    /// ~4.6× fewer NBL pairs vs the default 15 Å. Override available
    /// for oracle-style comparisons (see `cutoff_override` doc).
    fn nonbonded_cutoff(&self) -> f64 {
        self.cutoff_override.unwrap_or(9.0)
    }
    /// CHARMM19 switching on from BALL: `@CTONNB=7.0`. When the cutoff
    /// is overridden but switching-on is not, the on-distance is set
    /// to `cutoff_override - 1e-9` so switching is effectively
    /// disabled — matches the AMBER side's convention and the typical
    /// oracle case (e.g. `nonbonded_cutoff=1e6` → no switching).
    fn switching_on(&self) -> f64 {
        self.switching_on_override
            .or_else(|| self.cutoff_override.map(|c| c - 1e-9))
            .unwrap_or(7.0)
    }
}

/// Load the embedded CHARMM19 + EEF1 parameter set.
pub fn charmm19_eef1() -> CharmmParams {
    CharmmParams::from_ini(include_str!("../../data/charmm19_eef1.ini"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_amber96() {
        let p = amber96();
        assert!(!p.bonds.is_empty());
        assert!(!p.angles.is_empty());
        assert!(!p.torsions.is_empty());
        assert!(!p.lj.is_empty());
        assert!(!p.atom_types.is_empty());
        assert!((p.scee - 1.2).abs() < 1e-6);
        // amber96 alone must not carry OBC params.
        assert!(p.obc_gb.is_empty());
        assert!(!p.has_obc_gb());
    }

    #[test]
    fn test_load_amber96_obc() {
        let p = amber96_obc();
        assert!(p.has_obc_gb());
        // 36 AMBER classes were extracted from OpenMM amber96.xml + amber96_obc.xml.
        assert_eq!(
            p.obc_gb.len(),
            36,
            "expected 36 OBC classes, got {}",
            p.obc_gb.len()
        );

        // Spot-check four classes against the OpenMM XML (nm -> A):
        // CT (sp3 C): radius=0.19 nm=1.9 A, scale=0.72
        let ct = p.get_obc_gb("CT").expect("CT missing");
        assert!((ct.radius - 1.9).abs() < 1e-9, "CT radius = {}", ct.radius);
        assert!((ct.scale - 0.72).abs() < 1e-9);
        // N (sp2 N): radius=0.1706 nm=1.706 A, scale=0.79
        let n = p.get_obc_gb("N").expect("N missing");
        assert!((n.radius - 1.706).abs() < 1e-9);
        assert!((n.scale - 0.79).abs() < 1e-9);
        // HO (hydroxyl H): radius=0.105 nm=1.05 A, scale=0.85
        let ho = p.get_obc_gb("HO").expect("HO missing");
        assert!((ho.radius - 1.05).abs() < 1e-9);
        assert!((ho.scale - 0.85).abs() < 1e-9);
        // S (sulfur): radius=0.1775 nm=1.775 A, scale=0.96
        let s = p.get_obc_gb("S").expect("S missing");
        assert!((s.radius - 1.775).abs() < 1e-9);
        assert!((s.scale - 0.96).abs() < 1e-9);
    }

    #[test]
    fn test_amber96_obc_preserves_base_params() {
        // Layering OBC must not disturb bonded / LJ / atom-type tables.
        let base = amber96();
        let obc = amber96_obc();
        assert_eq!(obc.bonds.len(), base.bonds.len());
        assert_eq!(obc.angles.len(), base.angles.len());
        assert_eq!(obc.torsions.len(), base.torsions.len());
        assert_eq!(obc.lj.len(), base.lj.len());
        assert_eq!(obc.atom_types.len(), base.atom_types.len());
        assert!((obc.scee - base.scee).abs() < 1e-12);
    }

    #[test]
    fn test_atom_type_lookup() {
        let p = amber96();
        let entry = p.get_atom_type("ALA", "CA").unwrap();
        assert_eq!(entry.amber_type, "CT");
        assert!((entry.charge - 0.0337).abs() < 0.001);
    }

    #[test]
    fn test_bond_lookup() {
        let p = amber96();
        let b = p.get_bond("CT", "CT").unwrap();
        assert!((b.k - 310.0).abs() < 1.0);
        assert!((b.r0 - 1.526).abs() < 0.01);
    }

    #[test]
    fn test_lj_lookup() {
        let p = amber96();
        let lj = p.get_lj("CT").unwrap();
        assert!((lj.r - 1.908).abs() < 0.01);
        assert!((lj.epsilon - 0.1094).abs() < 0.01);
    }

    // ------------------------------------------------------------------
    // CHARMM19+EEF1 diagnostic test for the 2026-04-10 sign bug.
    //
    // Tier-2 weak oracle (commit 8f979f4) and the diagnostic in
    // validation/sota_comparison/diagnose_charmm_eef1.py established
    // that proteon's CHARMM19+EEF1 returns +537 kJ/mol solvation on
    // 1crn while the canonical Σ dg_ref read from charmm19_eef1.ini
    // gives -2748 kJ/mol with all 326 of 327 heavy atoms hitting
    // the parameter table. n_unassigned_atoms=0 at runtime. So the
    // bug must live somewhere between the .ini parser and the eef1
    // hashmap lookup at runtime.
    //
    // This test is a pure parameter-table introspection: it loads
    // charmm19_eef1() and dumps the relevant atom_types entries plus
    // the eef1 entries they should resolve to. The eprintln output
    // (run with `-- --nocapture`) will show exactly what the parser
    // produces, which localizes the bug in <30 seconds without
    // needing to drag in pdbtbx or build_topology.
    // ------------------------------------------------------------------
    #[test]
    fn test_charmm19_eef1_charges_and_types_dump() {
        let p = charmm19_eef1();
        eprintln!(
            "\n=== CharmmParams::atom_types size: {} ===",
            p.atom_types.len()
        );
        eprintln!("=== CharmmParams::eef1 size:       {} ===", p.eef1.len());

        // Sample lookups for ALA backbone — should be NH1, CH1E, C, O.
        for (res, atom, expected_type) in &[
            ("ALA", "N", "NH1"),
            ("ALA", "CA", "CH1E"),
            ("ALA", "C", "C"),
            ("ALA", "O", "O"),
            ("ALA", "CB", "CH3E"),
            ("GLY", "N", "NH1"),
            ("GLY", "CA", "CH2E"),
            ("GLY", "C", "C"),
            ("GLY", "O", "O"),
            ("PRO", "N", "N"), // proline N has no H, type "N" not "NH1"
            ("THR", "OG1", "OH1"),
            ("ASP", "OD1", "OC"),
            ("ARG", "NH1", "NC2"), // guanidinium
        ] {
            let key = format!("{res}:{atom}");
            match p.get_atom_type(res, atom) {
                Some(entry) => {
                    let match_marker = if entry.amber_type == *expected_type {
                        "✓"
                    } else {
                        "✗"
                    };
                    eprintln!(
                        "  [{}] {:<10} -> amber_type={:<6} charge={:>+8.4} (expected {})",
                        match_marker, key, entry.amber_type, entry.charge, expected_type,
                    );
                }
                None => {
                    eprintln!("  [✗] {:<10} -> NONE (not in atom_types)", key);
                }
            }
        }

        // Now look up the EEF1 dg_ref for the canonical types and
        // print them. Anything missing here means the eef1 hashmap
        // wasn't populated correctly by the parser.
        eprintln!("\n=== EEF1 dg_ref lookups (kcal/mol) ===");
        for ctype in &[
            "NH1", "NH2", "NH3", "NC2", "N", "NR", "NP", "C", "CR", "CH1E", "CH2E", "CH3E", "CR1E",
            "CT", "CM", "O", "OC", "OH1", "OM", "S", "SH1E",
        ] {
            match p.eef1.get(*ctype) {
                Some(eef) => eprintln!(
                    "  {:<6} dg_ref={:>+8.3} dg_free={:>+8.3} vol={:>7.3}",
                    ctype, eef.dg_ref, eef.dg_free, eef.volume,
                ),
                None => eprintln!("  {:<6} <NOT IN eef1 hashmap>", ctype),
            }
        }

        // The big claim: ALA:N → NH1 → dg_ref = -5.95
        let ala_n = p
            .get_atom_type("ALA", "N")
            .expect("ALA:N must exist in CHARMM atom_types");
        let nh1 = p
            .eef1
            .get(&ala_n.amber_type)
            .expect("ALA:N's amber_type must exist in EEF1 hashmap");
        eprintln!(
            "\nFINAL: ALA:N -> {} -> dg_ref={} kcal/mol (expect -5.95)",
            ala_n.amber_type, nh1.dg_ref,
        );
        assert_eq!(
            ala_n.amber_type, "NH1",
            "atom_types parser stored wrong type for ALA:N"
        );
        assert!(
            (nh1.dg_ref - (-5.95)).abs() < 0.01,
            "EEF1 parser stored wrong dg_ref for NH1: got {}",
            nh1.dg_ref
        );
    }

    // ------------------------------------------------------------------
    // Deeper diagnostic: actually call build_topology() on 1crn and
    // walk every heavy atom's runtime amber_type. Compute the
    // canonical Σ dg_ref two ways:
    //   (a) via params.get_eef1(&atom.amber_type) — what eef1_energy()
    //       does internally
    //   (b) via params.eef1.get(&atom.amber_type) — direct hashmap
    //       lookup, same result if the trait method is correct
    // If (a) and (b) both give -657 kcal/mol but proteon's compute_energy
    // returns +128 kcal/mol on the same input, the bug is in the
    // accumulation OR in something later.
    // ------------------------------------------------------------------
    #[test]
    fn test_charmm19_eef1_runtime_topology_dump() {
        use crate::forcefield::topology::build_topology;
        use std::collections::HashMap;

        // Find 1crn relative to the workspace root.
        let candidates = [
            "../test-pdbs/1crn.pdb",
            "test-pdbs/1crn.pdb",
            "../../test-pdbs/1crn.pdb",
        ];
        let pdb_path = candidates.iter().find(|p| std::path::Path::new(p).exists());
        let pdb_path = if let Some(p) = pdb_path {
            *p
        } else {
            eprintln!("SKIP: 1crn.pdb not found in expected locations");
            return;
        };

        let (pdb, _errors) = pdbtbx::open(pdb_path).expect("failed to load 1crn");
        let p = charmm19_eef1();
        let topo = build_topology(&pdb, &p);

        eprintln!("\n=== 1crn topology dump ===");
        eprintln!("  total atoms: {}", topo.atoms.len());
        eprintln!(
            "  unassigned:  {} ({:?})",
            topo.unassigned_atoms.len(),
            &topo.unassigned_atoms[..topo.unassigned_atoms.len().min(10)],
        );

        let n_heavy = topo.atoms.iter().filter(|a| !a.is_hydrogen).count();
        eprintln!("  heavy atoms: {}", n_heavy);

        // Walk every heavy atom and accumulate dg_ref two ways.
        let mut sum_via_trait = 0.0_f64;
        let mut sum_via_direct = 0.0_f64;
        let mut hits_trait = 0_usize;
        let mut hits_direct = 0_usize;
        let mut type_counts: HashMap<String, (usize, f64)> = HashMap::new();
        let mut sample_dump = Vec::new();

        for (i, a) in topo.atoms.iter().enumerate() {
            if a.is_hydrogen {
                continue;
            }

            // Method (a): trait method
            if let Some(eef) = p.get_eef1(&a.amber_type) {
                sum_via_trait += eef.dg_ref;
                hits_trait += 1;
            }
            // Method (b): direct hashmap
            if let Some(eef) = p.eef1.get(&a.amber_type) {
                sum_via_direct += eef.dg_ref;
                hits_direct += 1;
            }

            let entry = type_counts.entry(a.amber_type.clone()).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += p.eef1.get(&a.amber_type).map(|e| e.dg_ref).unwrap_or(0.0);

            if i < 12 {
                sample_dump.push(format!(
                    "  atom[{:3}] {}:{} type={} dg_ref={:?}",
                    i,
                    a.residue_name,
                    a.atom_name,
                    a.amber_type,
                    p.eef1.get(&a.amber_type).map(|e| e.dg_ref),
                ));
            }
        }

        eprintln!("\n  sample atoms (first 12):");
        for line in &sample_dump {
            eprintln!("{}", line);
        }

        eprintln!("\n  hits_via_trait:  {}/{} heavy", hits_trait, n_heavy);
        eprintln!("  hits_via_direct: {}/{} heavy", hits_direct, n_heavy);
        eprintln!(
            "  Σ dg_ref via trait : {:>+10.3} kcal/mol = {:>+10.3} kJ/mol",
            sum_via_trait,
            sum_via_trait * 4.184
        );
        eprintln!(
            "  Σ dg_ref via direct: {:>+10.3} kcal/mol = {:>+10.3} kJ/mol",
            sum_via_direct,
            sum_via_direct * 4.184
        );

        // Per-type breakdown sorted by absolute contribution.
        let mut sorted_types: Vec<_> = type_counts.iter().collect();
        sorted_types.sort_by(|a, b| b.1 .1.abs().partial_cmp(&a.1 .1.abs()).unwrap());
        eprintln!("\n  Per-type breakdown (kcal/mol total):");
        for (t, (n, total)) in sorted_types.iter().take(20) {
            eprintln!("    {:<8} count={:>4}  total={:>+10.3}", t, n, total);
        }

        // Now actually call eef1_energy via energy::compute_energy and
        // see what it produces. We have to do this through the public
        // API because eef1_energy is private.
        use crate::forcefield::energy::compute_energy;
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let result = compute_energy(&coords, &topo, &p);
        eprintln!("\n  eef1_energy via compute_energy:");
        eprintln!(
            "    solvation: {:>+10.3} kJ/mol  (canonical via direct sum: {:>+10.3} kJ/mol)",
            result.solvation,
            sum_via_direct * 4.184
        );
        eprintln!(
            "    diff (compute - direct): {:>+10.3} kJ/mol",
            result.solvation - sum_via_direct * 4.184
        );

        // The crucial assertion: compute_energy's solvation should equal
        // sum_via_direct * 4.184 plus the pair correction (which we can't
        // easily compute here). For 1crn, the pair correction is small
        // relative to the self-solvation, so sign should match.
        if sum_via_direct < 0.0 && result.solvation > 0.0 {
            eprintln!(
                "\n  ⚠ MISMATCH: direct Σ dg_ref is negative but compute_energy reports positive!"
            );
            eprintln!(
                "  EEF1 pair correction is +{:.1} kcal/mol, dominating self-solvation.",
                result.solvation - sum_via_direct
            );
            eprintln!("  Either the formula is wrong (verified against BALL — it's not)");
            eprintln!("  OR proteon is summing more than just self+pair (also unlikely)");
            eprintln!("  OR the canonical L&K answer for 1crn is genuinely positive.");
        }

        // Dump all components and the suspect LJ + bond lookups
        eprintln!("\n  All energy components on 1crn (raw, no H, no minimize):");
        eprintln!(
            "    bond_stretch    = {:>+12.3} kcal/mol",
            result.bond_stretch
        );
        eprintln!(
            "    angle_bend      = {:>+12.3} kcal/mol",
            result.angle_bend
        );
        eprintln!("    torsion         = {:>+12.3} kcal/mol", result.torsion);
        eprintln!(
            "    improper_torsion= {:>+12.3} kcal/mol",
            result.improper_torsion
        );
        eprintln!("    vdw             = {:>+12.3} kcal/mol", result.vdw);
        eprintln!(
            "    electrostatic   = {:>+12.3} kcal/mol",
            result.electrostatic
        );
        eprintln!("    solvation       = {:>+12.3} kcal/mol", result.solvation);
        eprintln!("    total           = {:>+12.3} kcal/mol", result.total);

        // Suspect: are the LJ params populated for CHARMM atom types?
        eprintln!("\n  LJ parameter lookups for CHARMM atom types:");
        for ctype in &[
            "NH1", "CH1E", "CH2E", "CH3E", "C", "CR", "CT", "O", "OC", "OH1", "S",
        ] {
            match p.get_lj(ctype) {
                Some(lj) => eprintln!("    {:<6} r={:>7.4} eps={:>7.4}", ctype, lj.r, lj.epsilon),
                None => eprintln!("    {:<6} <NOT IN lj hashmap>", ctype),
            }
        }

        // And the bond / angle params
        eprintln!("\n  Bond parameter lookups for CHARMM atom-type pairs:");
        for (a, b) in &[
            ("CH1E", "C"),
            ("C", "O"),
            ("C", "NH1"),
            ("CH1E", "CH3E"),
            ("CH1E", "NH1"),
        ] {
            match p.get_bond(a, b) {
                Some(bp) => eprintln!("    {:<6}-{:<6}  k={:>7.2} r0={:>6.4}", a, b, bp.k, bp.r0),
                None => eprintln!("    {:<6}-{:<6}  <NOT IN bonds hashmap>", a, b),
            }
        }

        // -----------------------------------------------------------------
        // Decisive test: re-run the EEF1 pair loop inline and classify each
        // pair as 1-2 / 1-3 (excluded_pairs), 1-4 (pairs_14), or "other".
        // Sum the pair-correction contribution from each class so we can
        // see exactly where the +716 kcal/mol inflation lives.
        //
        // Hypothesis: BALL's EEF1 loop runs over the pre-filtered LJ pair
        // list (no 1-2, no 1-3). proteon's eef1_energy() runs over ALL
        // i<j pairs within 9 Å with NO exclusion check, so it double-counts
        // bonded partners that are at near-peak Gaussian distance.
        // -----------------------------------------------------------------
        const PI_SQRT_PI: f64 = 5.568_327_996_831_708;
        let cutoff_sq = 9.0_f64 * 9.0;

        let mut pair_12_13 = 0.0_f64;
        let mut pair_14 = 0.0_f64;
        let mut pair_other = 0.0_f64;
        let mut n_12_13 = 0_usize;
        let mut n_14 = 0_usize;
        let mut n_other = 0_usize;
        let coords_vec: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();

        for ii in 0..coords_vec.len() {
            if topo.atoms[ii].is_hydrogen {
                continue;
            }
            let eef_i = match p.get_eef1(&topo.atoms[ii].amber_type) {
                Some(e) => e,
                None => continue,
            };
            for jj in (ii + 1)..coords_vec.len() {
                if topo.atoms[jj].is_hydrogen {
                    continue;
                }
                let eef_j = match p.get_eef1(&topo.atoms[jj].amber_type) {
                    Some(e) => e,
                    None => continue,
                };

                let dx = coords_vec[ii][0] - coords_vec[jj][0];
                let dy = coords_vec[ii][1] - coords_vec[jj][1];
                let dz = coords_vec[ii][2] - coords_vec[jj][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 > cutoff_sq || r2 < 0.01 {
                    continue;
                }
                let r = r2.sqrt();

                let mut contrib = 0.0_f64;
                if eef_i.dg_free.abs() > 1e-10 && eef_j.volume > 1e-10 {
                    let dr = (r - eef_i.r_min) / eef_i.sigma;
                    contrib += -0.5 * eef_j.volume * eef_i.dg_free * (-dr * dr).exp()
                        / (eef_i.sigma * PI_SQRT_PI * r2);
                }
                if eef_j.dg_free.abs() > 1e-10 && eef_i.volume > 1e-10 {
                    let dr = (r - eef_j.r_min) / eef_j.sigma;
                    contrib += -0.5 * eef_i.volume * eef_j.dg_free * (-dr * dr).exp()
                        / (eef_j.sigma * PI_SQRT_PI * r2);
                }

                let pair = (ii, jj); // already ii < jj, matches excluded_pairs keying
                if topo.excluded_pairs.contains(&pair) {
                    pair_12_13 += contrib;
                    n_12_13 += 1;
                } else if topo.pairs_14.contains(&pair) {
                    pair_14 += contrib;
                    n_14 += 1;
                } else {
                    pair_other += contrib;
                    n_other += 1;
                }
            }
        }

        let total_inline = pair_12_13 + pair_14 + pair_other;
        eprintln!("\n  === EEF1 pair-correction breakdown (1crn, raw) ===");
        eprintln!(
            "    excluded_pairs size (1-2 + 1-3): {}",
            topo.excluded_pairs.len()
        );
        eprintln!(
            "    pairs_14 size:                   {}",
            topo.pairs_14.len()
        );
        eprintln!();
        eprintln!("    class     |     n_pairs |     contribution (kcal/mol)");
        eprintln!("    ----------+-------------+---------------------------");
        eprintln!("    1-2 + 1-3 | {:>11} | {:>+14.3}", n_12_13, pair_12_13);
        eprintln!("    1-4       | {:>11} | {:>+14.3}", n_14, pair_14);
        eprintln!("    other     | {:>11} | {:>+14.3}", n_other, pair_other);
        eprintln!("    ----------+-------------+---------------------------");
        eprintln!(
            "    total     | {:>11} | {:>+14.3}",
            n_12_13 + n_14 + n_other,
            total_inline
        );
        eprintln!();
        eprintln!(
            "    self-solvation:              {:>+10.3} kcal/mol",
            sum_via_direct
        );
        eprintln!(
            "    pair correction (inline):    {:>+10.3} kcal/mol",
            total_inline
        );
        eprintln!(
            "    → total (inline):            {:>+10.3} kcal/mol",
            sum_via_direct + total_inline
        );
        eprintln!(
            "    if 1-2+1-3 excluded:         {:>+10.3} kcal/mol",
            sum_via_direct + pair_14 + pair_other
        );
        eprintln!(
            "    if 1-2+1-3 AND 1-4 excluded: {:>+10.3} kcal/mol",
            sum_via_direct + pair_other
        );
    }
}
