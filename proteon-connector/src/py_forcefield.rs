//! PyO3 bindings for AMBER force field and energy minimization.

use std::panic::{catch_unwind, AssertUnwindSafe};

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::batch::{make_batch_result, BatchOutcome, PyBatchResult};
use crate::forcefield::{api, energy, md, minimize, params, params::ForceField, topology};
use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

/// Dump the raw topology (atom-index tuples for every bond, angle,
/// torsion, improper torsion) as Python lists. Used by the AMBER96 oracle
/// to diff proteon's torsion list against OpenMM's PeriodicTorsionForce
/// contents — atom-index-only so the two sets can be compared directly.
///
/// Accepts the same force-field aliases as `compute_energy`, including
/// `amber96_obc` (with `amber96+obc` / `amber96_obc2` aliases) so OBC
/// topology / charge diagnostics stay on-parity with the energy path.
#[pyfunction]
#[pyo3(signature = (pdb, ff="amber96"))]
pub(crate) fn dump_topology(py: Python<'_>, pdb: &PyPDB, ff: &str) -> PyResult<PyObject> {
    let topo = guard_panic("dump_topology", || -> PyResult<topology::Topology> {
        Ok(match ff {
            "charmm" | "charmm19" | "charmm19_eef1" => {
                topology::build_topology(&pdb.inner, &params::charmm19_eef1())
            }
            "amber" | "amber96" => topology::build_topology(&pdb.inner, &params::amber96()),
            // OBC variants share AMBER96 topology (GB only affects solvation).
            "amber96_obc" | "amber96+obc" | "amber96_obc2" | "amber96_obc_cutoff"
            | "amber96+obc_cutoff" => topology::build_topology(&pdb.inner, &params::amber96_obc()),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown force field '{ff}'. Use 'amber96', 'amber96_obc' \
                     (aliases: 'amber96+obc', 'amber96_obc2'), 'amber96_obc_cutoff', \
                     or 'charmm19_eef1'."
                )))
            }
        })
    })??;
    let dict = pyo3::types::PyDict::new(py);
    let bonds: Vec<(usize, usize)> = topo.bonds.iter().map(|b| (b.i, b.j)).collect();
    let angles: Vec<(usize, usize, usize)> = topo.angles.iter().map(|a| (a.i, a.j, a.k)).collect();
    let torsions: Vec<(usize, usize, usize, usize)> =
        topo.torsions.iter().map(|t| (t.i, t.j, t.k, t.l)).collect();
    let impropers: Vec<(usize, usize, usize, usize)> = topo
        .improper_torsions
        .iter()
        .map(|t| (t.i, t.j, t.k, t.l))
        .collect();
    // Map each topology atom back to its (residue_idx, residue_name,
    // atom_name, amber_type, charge) so diff tools can print both
    // identity and the assigned force-field class.
    let atom_identities: Vec<(usize, String, String)> = topo
        .atoms
        .iter()
        .map(|a| (a.residue_idx, a.residue_name.clone(), a.atom_name.clone()))
        .collect();
    let atom_types: Vec<String> = topo.atoms.iter().map(|a| a.amber_type.clone()).collect();
    let atom_charges: Vec<f64> = topo.atoms.iter().map(|a| a.charge).collect();
    dict.set_item("bonds", bonds)?;
    dict.set_item("angles", angles)?;
    dict.set_item("torsions", torsions)?;
    dict.set_item("impropers", impropers)?;
    dict.set_item("atom_identities", atom_identities)?;
    dict.set_item("atom_types", atom_types)?;
    dict.set_item("atom_charges", atom_charges)?;
    Ok(dict.into_any().unbind())
}

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Run a force-field computation, converting a Rust panic into a clean
/// Python `ValueError`.
///
/// Force-field internals (`topology::build_topology`, the energy and OBC
/// kernels, …) assert hard invariants — e.g. that every atom's AMBER class
/// has a parameter entry. A user-supplied structure with an unusual residue
/// or atom can violate one of those, which panics. Without this boundary the
/// panic surfaces as `pyo3_runtime.PanicException`, which subclasses
/// `BaseException` and so slips past a caller's `except Exception`. Catching
/// it here turns it into an ordinary, catchable exception that still carries
/// the original panic message (the OBC missing-parameter panic, for one,
/// already names the offending atom index).
fn guard_panic<T>(what: &str, f: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "internal panic with no message".to_string());
        pyo3::exceptions::PyValueError::new_err(format!(
            "{what}: force-field computation failed on this input. This \
             usually means the structure contains a residue or atom the \
             force field has no parameters for. Details: {detail}"
        ))
    })
}

/// Compute force field energy of a structure.
///
/// Returns dict with energy components:
///   bond_stretch, angle_bend, torsion, improper_torsion, vdw,
///   electrostatic, solvation, total
///   (all in kcal/mol internally; Python wrapper can convert)
///
/// Args:
///     pdb: Structure to evaluate.
///     ff: Force field name ("amber96" or "charmm19_eef1").
///     nbl_threshold: Optional override for the neighbor-list atom-count
///         threshold. If None, uses the library default (2000 atoms).
///         Set to 0 to force the NBL path for any structure, or to a very
///         large value (e.g. 10_000_000) to force the O(N²) exact path.
///         Primarily intended for regression testing — exposing the two
///         paths so cross-path parity can be verified from Python.
#[pyfunction]
#[pyo3(signature = (pdb, ff="amber96", nbl_threshold=None, nonbonded_cutoff=None))]
pub(crate) fn compute_energy(
    py: Python<'_>,
    pdb: &PyPDB,
    ff: &str,
    nbl_threshold: Option<usize>,
    nonbonded_cutoff: Option<f64>,
) -> PyResult<PyObject> {
    // The ff-string → params → topology → energy pipeline (incl. the
    // amber96_obc OBC2 routing and the catch_unwind panic boundary) lives in
    // forcefield::api so the `proteon energy` CLI computes through the exact
    // same code. Here we only release the GIL around it and shape the dict.
    let report = py
        .allow_threads(|| api::energy_from_pdb(&pdb.inner, ff, nbl_threshold, nonbonded_cutoff))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let r = &report.energy;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bond_stretch", r.bond_stretch)?;
    dict.set_item("angle_bend", r.angle_bend)?;
    dict.set_item("torsion", r.torsion)?;
    dict.set_item("improper_torsion", r.improper_torsion)?;
    dict.set_item("vdw", r.vdw)?;
    dict.set_item("electrostatic", r.electrostatic)?;
    dict.set_item("solvation", r.solvation)?;
    dict.set_item("total", r.total)?;
    dict.set_item("n_unassigned_atoms", report.n_unassigned_atoms)?;
    // Topology counts — diagnostic data for cross-tool oracle comparison.
    // Proteon silently drops hydrogens whose names aren't in the FF
    // residue template (see should_include_atom in topology.rs); without
    // these counts it's impossible to tell from the outside whether a
    // given PDB's H atoms made it into the bonded/nonbonded sums or not.
    dict.set_item("n_topo_atoms", report.n_topo_atoms)?;
    dict.set_item("n_bonds", report.n_bonds)?;
    dict.set_item("n_angles", report.n_angles)?;
    dict.set_item("n_torsions", report.n_torsions)?;
    dict.set_item("n_impropers", report.n_impropers)?;
    dict.set_item("n_excluded_pairs", report.n_excluded_pairs)?;
    dict.set_item("n_14_pairs", report.n_14_pairs)?;
    Ok(dict.into_any().unbind())
}

/// Minimize hydrogen positions using AMBER force field.
///
/// Freezes all heavy atoms and optimizes only hydrogen positions
/// using steepest descent with adaptive step size.
///
/// Args:
///     pdb: Structure to minimize.
///     max_steps: Maximum optimization steps (default 500).
///     gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
///
/// Returns dict with:
///     coords: Nx3 optimized coordinates
///     initial_energy: energy before minimization
///     final_energy: energy after minimization
///     energy_components: dict of bond/angle/torsion/vdw/es
///     steps: number of steps taken
///     converged: whether optimization converged
fn run_minimize(
    coords: &[[f64; 3]],
    topo: &topology::Topology,
    amber: &impl ForceField,
    max_steps: usize,
    gradient_tolerance: f64,
    constrained: &[bool],
    method: &str,
) -> minimize::MinimizeResult {
    match method {
        "sd" | "steepest_descent" => minimize::steepest_descent(
            coords,
            topo,
            amber,
            max_steps,
            gradient_tolerance,
            constrained,
        ),
        "cg" | "conjugate_gradient" => minimize::conjugate_gradient(
            coords,
            topo,
            amber,
            max_steps,
            gradient_tolerance,
            constrained,
        ),
        "lbfgs" | "l-bfgs" => minimize::lbfgs(
            coords,
            topo,
            amber,
            max_steps,
            gradient_tolerance,
            constrained,
        ),
        _ => {
            // Default to SD for backward compat, but this should ideally error.
            // The Python layer validates method names before calling.
            minimize::steepest_descent(
                coords,
                topo,
                amber,
                max_steps,
                gradient_tolerance,
                constrained,
            )
        }
    }
}

#[pyfunction]
#[pyo3(signature = (pdb, max_steps=500, gradient_tolerance=0.1, method="sd"))]
pub(crate) fn minimize_hydrogens(
    py: Python<'_>,
    pdb: &PyPDB,
    max_steps: usize,
    gradient_tolerance: f64,
    method: &str,
) -> PyResult<PyObject> {
    let method = method.to_string();

    let result = guard_panic("minimize_hydrogens", || {
        let amber = params::amber96();
        let topo = topology::build_topology(&pdb.inner, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();
        py.allow_threads(|| {
            run_minimize(
                &coords,
                &topo,
                &amber,
                max_steps,
                gradient_tolerance,
                &constrained,
                &method,
            )
        })
    })?;

    let n = result.coords.len();
    let flat: Vec<f64> = result
        .coords
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();

    let dict = pyo3::types::PyDict::new(py);
    let coords_arr = PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape");
    dict.set_item("coords", coords_arr)?;
    dict.set_item("initial_energy", result.initial_energy)?;
    dict.set_item("final_energy", result.energy.total)?;
    dict.set_item("steps", result.steps)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("accepted_steps", result.accepted_steps)?;
    dict.set_item("status", result.status.as_str())?;

    let components = pyo3::types::PyDict::new(py);
    components.set_item("bond_stretch", result.energy.bond_stretch)?;
    components.set_item("angle_bend", result.energy.angle_bend)?;
    components.set_item("torsion", result.energy.torsion)?;
    components.set_item("improper_torsion", result.energy.improper_torsion)?;
    components.set_item("vdw", result.energy.vdw)?;
    components.set_item("electrostatic", result.energy.electrostatic)?;
    components.set_item("solvation", result.energy.solvation)?;
    dict.set_item("energy_components", components)?;

    Ok(dict.into_any().unbind())
}

/// Full structure energy minimization.
///
/// Args:
///     pdb: Structure to minimize.
///     max_steps: Maximum optimization steps (default 1000).
///     gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
///     method: Optimizer ("sd"/"cg"/"lbfgs", default "lbfgs"). LBFGS has a real
///         line search and is robust on clashing inputs; the old "sd" default could
///         silently no-op on a high-energy structure (see MINIMIZE_RELIABILITY_PLAN).
///     ff: Force field — "amber96", "amber96_obc", or "charmm19_eef1"
///         (default "amber96"). CHARMM19+EEF1 uses united-atom polar-H
///         placement; pass a structure prepared with
///         `place_peptide_hydrogens` rather than `add_hydrogens`.
///
/// Returns dict with same format as minimize_hydrogens.
#[pyfunction]
#[pyo3(signature = (pdb, max_steps=1000, gradient_tolerance=0.1, method="lbfgs", ff="amber96"))]
pub(crate) fn minimize_structure(
    py: Python<'_>,
    pdb: &PyPDB,
    max_steps: usize,
    gradient_tolerance: f64,
    method: &str,
    ff: &str,
) -> PyResult<PyObject> {
    let method = method.to_string();
    let result = guard_panic(
        "minimize_structure",
        || -> PyResult<minimize::MinimizeResult> {
            Ok(match ff {
                "charmm" | "charmm19" | "charmm19_eef1" => {
                    let charmm = params::charmm19_eef1();
                    let topo = topology::build_topology(&pdb.inner, &charmm);
                    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                    let constrained = vec![false; coords.len()];
                    py.allow_threads(|| {
                        run_minimize(
                            &coords,
                            &topo,
                            &charmm,
                            max_steps,
                            gradient_tolerance,
                            &constrained,
                            &method,
                        )
                    })
                }
                "amber96_obc" | "amber96+obc" | "amber96_obc2" => {
                    let amber = params::amber96_obc();
                    let topo = topology::build_topology(&pdb.inner, &amber);
                    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                    let constrained = vec![false; coords.len()];
                    py.allow_threads(|| {
                        run_minimize(
                            &coords,
                            &topo,
                            &amber,
                            max_steps,
                            gradient_tolerance,
                            &constrained,
                            &method,
                        )
                    })
                }
                "amber96_obc_cutoff" | "amber96+obc_cutoff" => {
                    // CutoffNonPeriodic GB runs on the CPU path (the GPU OBC
                    // kernels are NoCutoff-only and GpuStructState::new refuses
                    // the cutoff method, so it degrades to CPU automatically).
                    let amber = params::amber96_obc_cutoff();
                    let topo = topology::build_topology(&pdb.inner, &amber);
                    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                    let constrained = vec![false; coords.len()];
                    py.allow_threads(|| {
                        run_minimize(
                            &coords,
                            &topo,
                            &amber,
                            max_steps,
                            gradient_tolerance,
                            &constrained,
                            &method,
                        )
                    })
                }
                "amber" | "amber96" => {
                    let amber = params::amber96();
                    let topo = topology::build_topology(&pdb.inner, &amber);
                    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                    let constrained = vec![false; coords.len()];
                    py.allow_threads(|| {
                        run_minimize(
                            &coords,
                            &topo,
                            &amber,
                            max_steps,
                            gradient_tolerance,
                            &constrained,
                            &method,
                        )
                    })
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown force field '{ff}'. Use 'amber96', 'amber96_obc' \
                     (aliases: 'amber96+obc', 'amber96_obc2'), 'amber96_obc_cutoff', \
                     or 'charmm19_eef1'."
                    )));
                }
            })
        },
    )??;

    let n = result.coords.len();
    let flat: Vec<f64> = result
        .coords
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();

    let dict = pyo3::types::PyDict::new(py);
    let coords_arr = PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape");
    dict.set_item("coords", coords_arr)?;
    dict.set_item("initial_energy", result.initial_energy)?;
    dict.set_item("final_energy", result.energy.total)?;
    dict.set_item("steps", result.steps)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("accepted_steps", result.accepted_steps)?;
    dict.set_item("status", result.status.as_str())?;

    let components = pyo3::types::PyDict::new(py);
    components.set_item("bond_stretch", result.energy.bond_stretch)?;
    components.set_item("angle_bend", result.energy.angle_bend)?;
    components.set_item("torsion", result.energy.torsion)?;
    components.set_item("improper_torsion", result.energy.improper_torsion)?;
    components.set_item("vdw", result.energy.vdw)?;
    components.set_item("electrostatic", result.energy.electrostatic)?;
    components.set_item("solvation", result.energy.solvation)?;
    dict.set_item("energy_components", components)?;

    Ok(dict.into_any().unbind())
}

// ===========================================================================
// Batch parallel minimization
// ===========================================================================

/// Internal: run H minimization on a single PDB, return result struct.
fn minimize_h_single(
    pdb: &pdbtbx::PDB,
    max_steps: usize,
    gradient_tolerance: f64,
    method: &str,
) -> minimize::MinimizeResult {
    let amber = params::amber96();
    let topo = topology::build_topology(pdb, &amber);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();
    run_minimize(
        &coords,
        &topo,
        &amber,
        max_steps,
        gradient_tolerance,
        &constrained,
        method,
    )
}

/// Per-structure result captured by the parallel `batch_compute_energy` loop.
/// Keeps owned scalars so the parallel body never touches Python objects.
struct EnergyOutcome {
    energy: energy::EnergyResult,
    n_unassigned_atoms: usize,
    n_topo_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_torsions: usize,
    n_impropers: usize,
    n_excluded_pairs: usize,
    n_14_pairs: usize,
}

#[derive(Clone, Copy)]
enum FfKind {
    Charmm19Eef1,
    Amber96,
    Amber96Obc,
    Amber96ObcCutoff,
}

fn parse_ff(ff: &str) -> PyResult<FfKind> {
    match ff {
        "charmm" | "charmm19" | "charmm19_eef1" => Ok(FfKind::Charmm19Eef1),
        "amber" | "amber96" => Ok(FfKind::Amber96),
        "amber96_obc" | "amber96+obc" | "amber96_obc2" => Ok(FfKind::Amber96Obc),
        "amber96_obc_cutoff" | "amber96+obc_cutoff" => Ok(FfKind::Amber96ObcCutoff),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown force field '{ff}'. Use 'amber96', 'amber96_obc' \
             (aliases: 'amber96+obc', 'amber96_obc2'), 'amber96_obc_cutoff', \
             or 'charmm19_eef1'."
        ))),
    }
}

fn compute_energy_one(
    pdb: &pdbtbx::PDB,
    ff_kind: FfKind,
    nbl_threshold: Option<usize>,
    nonbonded_cutoff: Option<f64>,
) -> EnergyOutcome {
    macro_rules! run {
        ($params_expr:expr) => {{
            let mut params = $params_expr;
            if let Some(c) = nonbonded_cutoff {
                params.cutoff_override = Some(c);
            }
            let topo = topology::build_topology(pdb, &params);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            let result = match nbl_threshold {
                Some(t) => energy::compute_energy_auto(&coords, &topo, &params, t),
                None => energy::compute_energy(&coords, &topo, &params),
            };
            (topo, result)
        }};
    }
    let (topo, result) = match ff_kind {
        FfKind::Charmm19Eef1 => run!(params::charmm19_eef1()),
        FfKind::Amber96 => run!(params::amber96()),
        FfKind::Amber96Obc => run!(params::amber96_obc()),
        FfKind::Amber96ObcCutoff => run!(params::amber96_obc_cutoff()),
    };
    EnergyOutcome {
        energy: result,
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

fn energy_outcome_to_dict<'py>(
    py: Python<'py>,
    outcome: &EnergyOutcome,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bond_stretch", outcome.energy.bond_stretch)?;
    dict.set_item("angle_bend", outcome.energy.angle_bend)?;
    dict.set_item("torsion", outcome.energy.torsion)?;
    dict.set_item("improper_torsion", outcome.energy.improper_torsion)?;
    dict.set_item("vdw", outcome.energy.vdw)?;
    dict.set_item("electrostatic", outcome.energy.electrostatic)?;
    dict.set_item("solvation", outcome.energy.solvation)?;
    dict.set_item("total", outcome.energy.total)?;
    dict.set_item("n_unassigned_atoms", outcome.n_unassigned_atoms)?;
    dict.set_item("n_topo_atoms", outcome.n_topo_atoms)?;
    dict.set_item("n_bonds", outcome.n_bonds)?;
    dict.set_item("n_angles", outcome.n_angles)?;
    dict.set_item("n_torsions", outcome.n_torsions)?;
    dict.set_item("n_impropers", outcome.n_impropers)?;
    dict.set_item("n_excluded_pairs", outcome.n_excluded_pairs)?;
    dict.set_item("n_14_pairs", outcome.n_14_pairs)?;
    Ok(dict)
}

/// Compute force-field energy for many structures in parallel (Rust + rayon).
///
/// Returns list of dicts with the same shape as :func:`compute_energy`.
/// Same FF aliases and cutoff arguments. Structures are processed in chunks
/// to bound peak memory from PDB cloning.
#[pyfunction]
#[pyo3(signature = (structures, ff="amber96", nbl_threshold=None, nonbonded_cutoff=None, n_threads=None))]
pub(crate) fn batch_compute_energy<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    ff: &str,
    nbl_threshold: Option<usize>,
    nonbonded_cutoff: Option<f64>,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyObject>> {
    let ff_kind = parse_ff(ff)?;
    let n = resolve_threads(n_threads);

    // Pass raw &pdbtbx::PDB pointers across py.allow_threads instead of
    // cloning. Safety: the input PyList borrow outlives this call, so
    // the PyPDB instances (and the pdbtbx::PDB they own) stay valid for
    // the rayon section. Read-only access in the rayon body.
    let pdb_addrs: Vec<usize> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            let ptr: *const pdbtbx::PDB = &pdb.inner;
            Ok(ptr as usize)
        })
        .collect::<PyResult<_>>()?;

    let outcomes: Vec<EnergyOutcome> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pdb_addrs
                .par_iter()
                .map(|&addr| {
                    // Safety: see batch_atom_sasa in py_sasa.rs.
                    let pdb: &pdbtbx::PDB = unsafe { &*(addr as *const pdbtbx::PDB) };
                    compute_energy_one(pdb, ff_kind, nbl_threshold, nonbonded_cutoff)
                })
                .collect()
        })
    });

    outcomes
        .iter()
        .map(|o| energy_outcome_to_dict(py, o).map(|d| d.into_any().unbind()))
        .collect()
}

/// Batch minimize hydrogen positions for many structures in parallel.
///
/// Returns list of dicts (same format as minimize_hydrogens).
#[pyfunction]
#[pyo3(signature = (structures, max_steps=500, gradient_tolerance=0.1, n_threads=None, method="sd"))]
pub(crate) fn batch_minimize_hydrogens<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    max_steps: usize,
    gradient_tolerance: f64,
    n_threads: Option<i32>,
    method: &str,
) -> PyResult<Vec<PyObject>> {
    let n = resolve_threads(n_threads);
    let method = method.to_string();
    let total = structures.len();
    let chunk_size = 500;
    let mut all_results: Vec<minimize::MinimizeResult> = Vec::with_capacity(total);

    // Process in chunks to avoid cloning all structures at once
    for start in (0..total).step_by(chunk_size) {
        let end = (start + chunk_size).min(total);

        let chunk_pdbs: Vec<pdbtbx::PDB> = (start..end)
            .map(|i| {
                let item = structures.get_item(i)?;
                let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
                Ok(pdb.inner.clone())
            })
            .collect::<PyResult<_>>()?;

        let results: Vec<minimize::MinimizeResult> = py.allow_threads(|| {
            let pool = build_pool(n);
            pool.install(|| {
                chunk_pdbs
                    .par_iter()
                    .map(|pdb| minimize_h_single(pdb, max_steps, gradient_tolerance, &method))
                    .collect()
            })
        });

        all_results.extend(results);
    }

    let results = all_results;

    // Convert to Python dicts
    Ok(results
        .into_iter()
        .map(|result| {
            let n = result.coords.len();
            let flat: Vec<f64> = result
                .coords
                .iter()
                .flat_map(|c| c.iter().copied())
                .collect();

            let dict = pyo3::types::PyDict::new(py);
            let coords_arr = PyArray1::from_vec(py, flat)
                .reshape([n, 3])
                .expect("reshape");
            dict.set_item("coords", coords_arr).unwrap();
            dict.set_item("initial_energy", result.initial_energy)
                .unwrap();
            dict.set_item("final_energy", result.energy.total).unwrap();
            dict.set_item("steps", result.steps).unwrap();
            dict.set_item("converged", result.converged).unwrap();
            dict.set_item("accepted_steps", result.accepted_steps)
                .unwrap();
            dict.set_item("status", result.status.as_str()).unwrap();

            let components = pyo3::types::PyDict::new(py);
            components
                .set_item("bond_stretch", result.energy.bond_stretch)
                .unwrap();
            components
                .set_item("angle_bend", result.energy.angle_bend)
                .unwrap();
            components
                .set_item("torsion", result.energy.torsion)
                .unwrap();
            components
                .set_item("improper_torsion", result.energy.improper_torsion)
                .unwrap();
            components.set_item("vdw", result.energy.vdw).unwrap();
            components
                .set_item("electrostatic", result.energy.electrostatic)
                .unwrap();
            dict.set_item("energy_components", components).unwrap();

            dict.into_any().unbind()
        })
        .collect())
}

/// Load files and minimize hydrogens in one parallel call (zero GIL).
///
/// Returns a `BatchResult` with one item per path (input order); each
/// successful item value is a dict (same shape as `minimize_hydrogens`). A
/// file that fails to load is recorded as a failed item carrying the parse
/// error — failures are no longer silently dropped. Pass `strict=True` to
/// raise on the first failure instead.
#[pyfunction]
#[pyo3(signature = (paths, max_steps=500, gradient_tolerance=0.1, n_threads=None, method="sd", strict=false))]
pub(crate) fn load_and_minimize_hydrogens(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    max_steps: usize,
    gradient_tolerance: f64,
    n_threads: Option<i32>,
    method: &str,
    strict: bool,
) -> PyResult<PyBatchResult> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);
    let method = method.to_string();

    // Load + minimize entirely in Rust. A load failure becomes a per-path
    // Err(String) instead of being filtered out of the result set.
    let results: Vec<Result<minimize::MinimizeResult, String>> = py.allow_threads(|| {
        let mut parsing = pdbtbx::ParsingLevel::all();
        parsing.set_cryst1(false);
        parsing.set_master(false);
        let mut opts = pdbtbx::ReadOptions::new();
        opts.set_level(pdbtbx::StrictnessLevel::Loose)
            .set_parsing_level(&parsing);

        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .map(|path| {
                    opts.read(path)
                        .map(|(pdb, _)| {
                            minimize_h_single(&pdb, max_steps, gradient_tolerance, &method)
                        })
                        .map_err(|errs| {
                            errs.iter()
                                .map(|e| e.to_string())
                                .collect::<Vec<_>>()
                                .join("; ")
                        })
                })
                .collect()
        })
    });

    let outcomes: Vec<BatchOutcome> = results
        .into_iter()
        .map(|r| match r {
            Ok(result) => {
                let nn = result.coords.len();
                let flat: Vec<f64> = result
                    .coords
                    .iter()
                    .flat_map(|c| c.iter().copied())
                    .collect();

                let dict = pyo3::types::PyDict::new(py);
                let coords_arr = PyArray1::from_vec(py, flat)
                    .reshape([nn, 3])
                    .map_err(|e| e.to_string())?;
                dict.set_item("coords", coords_arr)
                    .map_err(|e| e.to_string())?;
                dict.set_item("initial_energy", result.initial_energy)
                    .map_err(|e| e.to_string())?;
                dict.set_item("final_energy", result.energy.total)
                    .map_err(|e| e.to_string())?;
                dict.set_item("steps", result.steps)
                    .map_err(|e| e.to_string())?;
                dict.set_item("converged", result.converged)
                    .map_err(|e| e.to_string())?;
                dict.set_item("accepted_steps", result.accepted_steps)
                    .map_err(|e| e.to_string())?;
                dict.set_item("status", result.status.as_str())
                    .map_err(|e| e.to_string())?;
                Ok(dict.into_any().unbind())
            }
            Err(e) => Err(e),
        })
        .collect();
    make_batch_result(py, outcomes, strict)
}

// ---------------------------------------------------------------------------
// Molecular dynamics
// ---------------------------------------------------------------------------

/// Run molecular dynamics simulation using Velocity Verlet integration.
///
/// Args:
///     pdb: Structure to simulate.
///     n_steps: Number of MD steps (default 1000).
///     dt: Time step in picoseconds (default 0.001 = 1 fs).
///     temperature: Initial/target temperature in Kelvin (default 300).
///     thermostat_tau: Berendsen coupling time in ps. 0 = NVE (default 0.2 = NVT).
///     snapshot_freq: Record trajectory frame every N steps (default 10).
///
/// Returns dict with:
///     coords: final coordinates (N, 3).
///     velocities: final velocities (N, 3).
///     trajectory: list of dicts with step, time_ps, kinetic_energy,
///                 potential_energy, total_energy, temperature.
///     trajectory_coords: list of (N, 3) coordinate arrays at each snapshot.
///     energy: final energy components dict.
#[pyfunction]
#[pyo3(signature = (pdb, n_steps=1000, dt=0.001, temperature=300.0, thermostat_tau=0.2, snapshot_freq=10, shake=false))]
pub(crate) fn run_md(
    py: Python<'_>,
    pdb: &PyPDB,
    n_steps: usize,
    dt: f64,
    temperature: f64,
    thermostat_tau: f64,
    snapshot_freq: usize,
    shake: bool,
) -> PyResult<PyObject> {
    let result = guard_panic("run_md", || {
        let amber = params::amber96();
        let topo = topology::build_topology(&pdb.inner, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
        let snap_freq = snapshot_freq.max(1);

        // Build H-bond constraints if SHAKE enabled
        let constraints = if shake {
            md::build_h_constraints(&topo, &amber)
        } else {
            Vec::new()
        };

        // Run MD (release GIL)
        py.allow_threads(|| {
            md::velocity_verlet_constrained(
                &coords,
                &topo,
                &amber,
                n_steps,
                dt,
                temperature,
                thermostat_tau,
                snap_freq,
                &constraints,
            )
        })
    })?;
    let n = result.coords.len();

    // Build result dict
    let dict = pyo3::types::PyDict::new(py);

    // Final coords
    let flat: Vec<f64> = result
        .coords
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let coords_arr = PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape");
    dict.set_item("coords", coords_arr)?;

    // Final velocities
    let flat_v: Vec<f64> = result
        .velocities
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let vel_arr = PyArray1::from_vec(py, flat_v)
        .reshape([n, 3])
        .expect("reshape");
    dict.set_item("velocities", vel_arr)?;

    // Trajectory frames
    let frames = pyo3::types::PyList::empty(py);
    for frame in &result.frames {
        let fd = pyo3::types::PyDict::new(py);
        fd.set_item("step", frame.step)?;
        fd.set_item("time_ps", frame.time_ps)?;
        fd.set_item("kinetic_energy", frame.kinetic_energy)?;
        fd.set_item("potential_energy", frame.potential_energy)?;
        fd.set_item("total_energy", frame.total_energy)?;
        fd.set_item("temperature", frame.temperature)?;
        frames.append(fd)?;
    }
    dict.set_item("trajectory", frames)?;

    // Final energy components
    let components = pyo3::types::PyDict::new(py);
    components.set_item("bond_stretch", result.energy.bond_stretch)?;
    components.set_item("angle_bend", result.energy.angle_bend)?;
    components.set_item("torsion", result.energy.torsion)?;
    components.set_item("improper_torsion", result.energy.improper_torsion)?;
    components.set_item("vdw", result.energy.vdw)?;
    components.set_item("electrostatic", result.energy.electrostatic)?;
    components.set_item("solvation", result.energy.solvation)?;
    dict.set_item("energy", components)?;

    dict.set_item("n_steps", n_steps)?;
    dict.set_item("dt", dt)?;
    dict.set_item("temperature_target", temperature)?;
    dict.set_item("thermostat_tau", thermostat_tau)?;

    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// GPU status API
// ---------------------------------------------------------------------------

/// Check if CUDA GPU acceleration is available.
///
/// Returns True if the binary was compiled with the `cuda` feature AND a
/// GPU was detected at runtime. This is the same check the minimizer and
/// SASA functions use internally to decide whether to dispatch to GPU.
#[pyfunction]
pub(crate) fn gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::forcefield::gpu::GpuContext::try_global().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get GPU device info as a dict, or None if no GPU available.
///
/// Returns dict with keys: name, compute_capability, total_memory_mb,
/// cuda_compiled (bool — whether the binary has the cuda feature).
#[pyfunction]
pub(crate) fn gpu_info(py: Python<'_>) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);

    #[cfg(feature = "cuda")]
    {
        dict.set_item("cuda_compiled", true)?;
        if let Some(ctx) = crate::forcefield::gpu::GpuContext::try_global() {
            dict.set_item("available", true)?;
            dict.set_item("name", ctx.device_name())?;
            let (major, minor) = ctx.compute_capability();
            dict.set_item("compute_capability", format!("{}.{}", major, minor))?;
            dict.set_item("total_memory_mb", ctx.total_memory_mb())?;
        } else {
            dict.set_item("available", false)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        dict.set_item("cuda_compiled", false)?;
        dict.set_item("available", false)?;
    }

    Ok(dict.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_forcefield(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_energy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_energy, m)?)?;
    m.add_function(wrap_pyfunction!(dump_topology, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_structure, m)?)?;
    m.add_function(wrap_pyfunction!(batch_minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_minimize_hydrogens, m)?)?;
    crate::batch::register(m)?;
    m.add_function(wrap_pyfunction!(run_md, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    Ok(())
}
