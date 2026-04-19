//! PyO3 bindings for AMBER force field and energy minimization.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::forcefield::{energy, md, minimize, params, params::ForceField, topology};
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
    let topo = match ff {
        "charmm" | "charmm19" | "charmm19_eef1" => {
            topology::build_topology(&pdb.inner, &params::charmm19_eef1())
        }
        "amber" | "amber96" => topology::build_topology(&pdb.inner, &params::amber96()),
        "amber96_obc" | "amber96+obc" | "amber96_obc2" => {
            topology::build_topology(&pdb.inner, &params::amber96_obc())
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown force field '{ff}'. Use 'amber96', 'amber96_obc' \
                 (aliases: 'amber96+obc', 'amber96_obc2'), or 'charmm19_eef1'."
            )))
        }
    };
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
    let (topo, result) = match ff {
        "charmm" | "charmm19" | "charmm19_eef1" => {
            let charmm = params::charmm19_eef1();
            let topo = topology::build_topology(&pdb.inner, &charmm);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            // Cutoff override not supported for CHARMM19 yet.
            if nonbonded_cutoff.is_some() {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "nonbonded_cutoff override is only implemented for ff='amber96'",
                ));
            }
            let result = py.allow_threads(|| match nbl_threshold {
                Some(t) => energy::compute_energy_auto(&coords, &topo, &charmm, t),
                None => energy::compute_energy(&coords, &topo, &charmm),
            });
            (topo, result)
        }
        "amber" | "amber96" => {
            let mut amber = params::amber96();
            if let Some(c) = nonbonded_cutoff {
                amber.cutoff_override = Some(c);
            }
            let topo = topology::build_topology(&pdb.inner, &amber);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            let result = py.allow_threads(|| match nbl_threshold {
                Some(t) => energy::compute_energy_auto(&coords, &topo, &amber, t),
                None => energy::compute_energy(&coords, &topo, &amber),
            });
            (topo, result)
        }
        "amber96_obc" | "amber96+obc" | "amber96_obc2" => {
            // AMBER96 with OBC2 implicit solvent (α=1.0, β=0.8, γ=4.85).
            // This matches OpenMM's ForceField("amber96.xml",
            // "amber96_obc.xml") + AMBER's gbsa=OBC2 setting: the OpenMM
            // kernel hardcodes `ObcParameters::ObcTypeII` regardless of
            // whether the XML is named amber96_obc.xml or charmm36_obc2.xml,
            // because the XML only serializes per-atom (radius, scale)
            // and leaves α/β/γ to the code. If/when proteon grows
            // genuine OBC1 support it will ship as a distinct ff string
            // ("amber96_obc1") routed through a separate params loader,
            // NOT as an alias here.
            let mut amber = params::amber96_obc();
            if let Some(c) = nonbonded_cutoff {
                amber.cutoff_override = Some(c);
            }
            let topo = topology::build_topology(&pdb.inner, &amber);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            let result = py.allow_threads(|| match nbl_threshold {
                Some(t) => energy::compute_energy_auto(&coords, &topo, &amber, t),
                None => energy::compute_energy(&coords, &topo, &amber),
            });
            (topo, result)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown force field '{ff}'. Use 'amber96', 'amber96_obc' \
                     (alias: 'amber96+obc', 'amber96_obc2'), or 'charmm19_eef1'."
            )));
        }
    };

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bond_stretch", result.bond_stretch)?;
    dict.set_item("angle_bend", result.angle_bend)?;
    dict.set_item("torsion", result.torsion)?;
    dict.set_item("improper_torsion", result.improper_torsion)?;
    dict.set_item("vdw", result.vdw)?;
    dict.set_item("electrostatic", result.electrostatic)?;
    dict.set_item("solvation", result.solvation)?;
    dict.set_item("total", result.total)?;
    dict.set_item("n_unassigned_atoms", topo.unassigned_atoms.len())?;
    // Topology counts — diagnostic data for cross-tool oracle comparison.
    // Proteon silently drops hydrogens whose names aren't in the FF
    // residue template (see should_include_atom in topology.rs); without
    // these counts it's impossible to tell from the outside whether a
    // given PDB's H atoms made it into the bonded/nonbonded sums or not.
    dict.set_item("n_topo_atoms", topo.atoms.len())?;
    dict.set_item("n_bonds", topo.bonds.len())?;
    dict.set_item("n_angles", topo.angles.len())?;
    dict.set_item("n_torsions", topo.torsions.len())?;
    dict.set_item("n_impropers", topo.improper_torsions.len())?;
    dict.set_item("n_excluded_pairs", topo.excluded_pairs.len())?;
    dict.set_item("n_14_pairs", topo.pairs_14.len())?;
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
    let amber = params::amber96();
    let topo = topology::build_topology(&pdb.inner, &amber);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();
    let method = method.to_string();

    let result = py.allow_threads(|| {
        run_minimize(
            &coords,
            &topo,
            &amber,
            max_steps,
            gradient_tolerance,
            &constrained,
            &method,
        )
    });

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

/// Full structure energy minimization using AMBER force field.
///
/// Args:
///     pdb: Structure to minimize.
///     max_steps: Maximum optimization steps (default 1000).
///     gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
///
/// Returns dict with same format as minimize_hydrogens.
#[pyfunction]
#[pyo3(signature = (pdb, max_steps=1000, gradient_tolerance=0.1, method="sd"))]
pub(crate) fn minimize_structure(
    py: Python<'_>,
    pdb: &PyPDB,
    max_steps: usize,
    gradient_tolerance: f64,
    method: &str,
) -> PyResult<PyObject> {
    let amber = params::amber96();
    let topo = topology::build_topology(&pdb.inner, &amber);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let constrained = vec![false; coords.len()];
    let method = method.to_string();

    let result = py.allow_threads(|| {
        run_minimize(
            &coords,
            &topo,
            &amber,
            max_steps,
            gradient_tolerance,
            &constrained,
            &method,
        )
    });

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
#[pyfunction]
#[pyo3(signature = (paths, max_steps=500, gradient_tolerance=0.1, n_threads=None, method="sd"))]
pub(crate) fn load_and_minimize_hydrogens<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    max_steps: usize,
    gradient_tolerance: f64,
    n_threads: Option<i32>,
    method: &str,
) -> PyResult<Vec<(usize, PyObject)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);
    let method = method.to_string();

    // Load + minimize entirely in Rust
    let results: Vec<(usize, minimize::MinimizeResult)> = py.allow_threads(|| {
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
                .enumerate()
                .filter_map(|(i, path)| {
                    opts.read(path).ok().map(|(pdb, _)| {
                        let result =
                            minimize_h_single(&pdb, max_steps, gradient_tolerance, &method);
                        (i, result)
                    })
                })
                .collect()
        })
    });

    // Convert to Python
    Ok(results
        .into_iter()
        .map(|(idx, result)| {
            let nn = result.coords.len();
            let flat: Vec<f64> = result
                .coords
                .iter()
                .flat_map(|c| c.iter().copied())
                .collect();

            let dict = pyo3::types::PyDict::new(py);
            let coords_arr = PyArray1::from_vec(py, flat)
                .reshape([nn, 3])
                .expect("reshape");
            dict.set_item("coords", coords_arr).unwrap();
            dict.set_item("initial_energy", result.initial_energy)
                .unwrap();
            dict.set_item("final_energy", result.energy.total).unwrap();
            dict.set_item("steps", result.steps).unwrap();
            dict.set_item("converged", result.converged).unwrap();
            (idx, dict.into_any().unbind())
        })
        .collect())
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
    let amber = params::amber96();
    let topo = topology::build_topology(&pdb.inner, &amber);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let n = coords.len();

    let snap_freq = snapshot_freq.max(1);
    let topo_clone = topo.clone();
    let amber_clone = amber.clone();

    // Build H-bond constraints if SHAKE enabled
    let constraints = if shake {
        md::build_h_constraints(&topo, &amber)
    } else {
        Vec::new()
    };

    // Run MD (release GIL)
    let result = py.allow_threads(move || {
        md::velocity_verlet_constrained(
            &coords,
            &topo_clone,
            &amber_clone,
            n_steps,
            dt,
            temperature,
            thermostat_tau,
            snap_freq,
            &constraints,
        )
    });

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
    m.add_function(wrap_pyfunction!(dump_topology, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_structure, m)?)?;
    m.add_function(wrap_pyfunction!(batch_minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(run_md, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    Ok(())
}
