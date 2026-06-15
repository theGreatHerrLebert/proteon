//! Pure-Rust force-field orchestration shared by the PyO3 energy bindings
//! (`py_forcefield::compute_energy`) and the `proteon energy` CLI.
//!
//! The ff-string → params → topology → energy pipeline lived inside the PyO3
//! wrapper, so a CLI would have had to re-implement it and could silently
//! drift. Hoisting it here gives both callers one source of truth: the wrapper
//! builds a Python dict from [`EnergyReport`], the CLI formats it as TSV/JSON,
//! and neither owns the physics.

use std::panic::{catch_unwind, AssertUnwindSafe};

use super::energy::{self, EnergyResult};
use super::params::{self, ForceField};
use super::topology;

/// Energy components plus the topology-shape diagnostics the Python dict
/// exposes (atom/bond/angle/... counts), so cross-tool oracle tooling and the
/// CLI see the same fields.
#[derive(Clone, Debug)]
pub struct EnergyReport {
    pub energy: EnergyResult,
    pub n_unassigned_atoms: usize,
    pub n_topo_atoms: usize,
    pub n_bonds: usize,
    pub n_angles: usize,
    pub n_torsions: usize,
    pub n_impropers: usize,
    pub n_excluded_pairs: usize,
    pub n_14_pairs: usize,
}

/// Canonical list of accepted force-field strings (including aliases). Kept
/// next to [`energy_from_pdb`] so the CLI's up-front validation and the
/// computation agree on exactly one vocabulary.
pub const KNOWN_FORCE_FIELDS: &[&str] = &[
    "charmm",
    "charmm19",
    "charmm19_eef1",
    "amber",
    "amber96",
    "amber96_obc",
    "amber96+obc",
    "amber96_obc2",
    "amber96_obc_cutoff",
    "amber96+obc_cutoff",
];

/// Whether `ff` names a force field [`energy_from_pdb`] can evaluate.
pub fn is_known_force_field(ff: &str) -> bool {
    KNOWN_FORCE_FIELDS.contains(&ff)
}

fn run<P: ForceField>(pdb: &pdbtbx::PDB, params: &P, nbl_threshold: Option<usize>) -> EnergyReport {
    let topo = topology::build_topology(pdb, params);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let energy = match nbl_threshold {
        Some(t) => energy::compute_energy_auto(&coords, &topo, params, t),
        None => energy::compute_energy(&coords, &topo, params),
    };
    EnergyReport {
        energy,
        n_unassigned_atoms: topo.unassigned_atoms.len(),
        n_topo_atoms: topo.atoms.len(),
        n_bonds: topo.bonds.len(),
        n_angles: topo.angles.len(),
        n_torsions: topo.torsions.len(),
        n_impropers: topo.improper_torsions.len(),
        n_excluded_pairs: topo.excluded_pairs.len(),
        n_14_pairs: topo.pairs_14.len(),
    }
}

/// Compute force-field energy from a structure.
///
/// `ff` is one of [`KNOWN_FORCE_FIELDS`]. `nbl_threshold` overrides the
/// neighbor-list atom-count cutoff (None = library default); `nonbonded_cutoff`
/// overrides the nonbonded cutoff distance. Returns `Err` for an unknown force
/// field or if the computation panics — force-field internals assert hard
/// invariants (e.g. every atom's class has a parameter), and a structure with
/// an unparameterized residue/atom trips one. Catching it here means both the
/// Python wrapper (→ `ValueError`) and the CLI (→ per-file error, run
/// continues) get an ordinary `Result` instead of an uncatchable panic.
pub fn energy_from_pdb(
    pdb: &pdbtbx::PDB,
    ff: &str,
    nbl_threshold: Option<usize>,
    nonbonded_cutoff: Option<f64>,
) -> Result<EnergyReport, String> {
    let computed = catch_unwind(AssertUnwindSafe(|| -> Result<EnergyReport, String> {
        match ff {
            "charmm" | "charmm19" | "charmm19_eef1" => {
                let mut p = params::charmm19_eef1();
                p.cutoff_override = nonbonded_cutoff.or(p.cutoff_override);
                Ok(run(pdb, &p, nbl_threshold))
            }
            "amber" | "amber96" => {
                let mut p = params::amber96();
                p.cutoff_override = nonbonded_cutoff.or(p.cutoff_override);
                Ok(run(pdb, &p, nbl_threshold))
            }
            "amber96_obc" | "amber96+obc" | "amber96_obc2" => {
                let mut p = params::amber96_obc();
                p.cutoff_override = nonbonded_cutoff.or(p.cutoff_override);
                Ok(run(pdb, &p, nbl_threshold))
            }
            "amber96_obc_cutoff" | "amber96+obc_cutoff" => {
                // CutoffNonPeriodic GB (truncated + reaction-field shift; matches
                // OpenMM GBSAOBCForce with a cutoff). GB cutoff follows the
                // nonbonded cutoff override.
                let mut p = params::amber96_obc_cutoff();
                p.cutoff_override = nonbonded_cutoff.or(p.cutoff_override);
                Ok(run(pdb, &p, nbl_threshold))
            }
            _ => Err(format!(
                "Unknown force field '{ff}'. Use 'charmm19_eef1', 'amber96', \
                 'amber96_obc' (aliases: 'amber96+obc', 'amber96_obc2'), or \
                 'amber96_obc_cutoff'."
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
                "force-field computation failed on this input (usually an \
                 unparameterized residue or atom): {detail}"
            ))
        }
    }
}
