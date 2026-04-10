//! PyO3 bindings for AMBER force field and energy minimization.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::forcefield::{energy, md, minimize, params, params::ForceField, topology};
use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Compute AMBER force field energy of a structure.
///
/// Returns dict with energy components:
///   bond_stretch, angle_bend, torsion, vdw, electrostatic, total
///   (all in kcal/mol)
#[pyfunction]
#[pyo3(signature = (pdb, ff="amber96"))]
pub fn compute_energy(py: Python<'_>, pdb: &PyPDB, ff: &str) -> PyResult<PyObject> {
    let (topo, result) = match ff {
        "charmm" | "charmm19" | "charmm19_eef1" => {
            let charmm = params::charmm19_eef1();
            let topo = topology::build_topology(&pdb.inner, &charmm);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            let result = py.allow_threads(|| energy::compute_energy(&coords, &topo, &charmm));
            (topo, result)
        }
        "amber" | "amber96" => {
            let amber = params::amber96();
            let topo = topology::build_topology(&pdb.inner, &amber);
            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
            let result = py.allow_threads(|| energy::compute_energy(&coords, &topo, &amber));
            (topo, result)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown force field '{}'. Use 'amber96' or 'charmm19_eef1'.", ff)
            ));
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
        "sd" | "steepest_descent" => {
            minimize::steepest_descent(coords, topo, amber, max_steps, gradient_tolerance, constrained)
        }
        "cg" | "conjugate_gradient" => {
            minimize::conjugate_gradient(coords, topo, amber, max_steps, gradient_tolerance, constrained)
        }
        "lbfgs" | "l-bfgs" => {
            minimize::lbfgs(coords, topo, amber, max_steps, gradient_tolerance, constrained)
        }
        _ => {
            // Default to SD for backward compat, but this should ideally error.
            // The Python layer validates method names before calling.
            minimize::steepest_descent(coords, topo, amber, max_steps, gradient_tolerance, constrained)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (pdb, max_steps=500, gradient_tolerance=0.1, method="sd"))]
pub fn minimize_hydrogens<'py>(
    py: Python<'py>,
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
        run_minimize(&coords, &topo, &amber, max_steps, gradient_tolerance, &constrained, &method)
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
pub fn minimize_structure<'py>(
    py: Python<'py>,
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
        run_minimize(&coords, &topo, &amber, max_steps, gradient_tolerance, &constrained, &method)
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
    run_minimize(&coords, &topo, &amber, max_steps, gradient_tolerance, &constrained, method)
}

/// Batch minimize hydrogen positions for many structures in parallel.
///
/// Returns list of dicts (same format as minimize_hydrogens).
#[pyfunction]
#[pyo3(signature = (structures, max_steps=500, gradient_tolerance=0.1, n_threads=None, method="sd"))]
pub fn batch_minimize_hydrogens<'py>(
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
            dict.set_item("initial_energy", result.initial_energy).unwrap();
            dict.set_item("final_energy", result.energy.total).unwrap();
            dict.set_item("steps", result.steps).unwrap();
            dict.set_item("converged", result.converged).unwrap();

            let components = pyo3::types::PyDict::new(py);
            components.set_item("bond_stretch", result.energy.bond_stretch).unwrap();
            components.set_item("angle_bend", result.energy.angle_bend).unwrap();
            components.set_item("torsion", result.energy.torsion).unwrap();
            components.set_item("improper_torsion", result.energy.improper_torsion).unwrap();
            components.set_item("vdw", result.energy.vdw).unwrap();
            components.set_item("electrostatic", result.energy.electrostatic).unwrap();
            dict.set_item("energy_components", components).unwrap();

            dict.into_any().unbind()
        })
        .collect())
}

/// Load files and minimize hydrogens in one parallel call (zero GIL).
#[pyfunction]
#[pyo3(signature = (paths, max_steps=500, gradient_tolerance=0.1, n_threads=None, method="sd"))]
pub fn load_and_minimize_hydrogens<'py>(
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
                        let result = minimize_h_single(&pdb, max_steps, gradient_tolerance, &method);
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
            dict.set_item("initial_energy", result.initial_energy).unwrap();
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
pub fn run_md(
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
            &coords, &topo_clone, &amber_clone, n_steps, dt, temperature,
            thermostat_tau, snap_freq, &constraints,
        )
    });

    // Build result dict
    let dict = pyo3::types::PyDict::new(py);

    // Final coords
    let flat: Vec<f64> = result.coords.iter().flat_map(|c| c.iter().copied()).collect();
    let coords_arr = PyArray1::from_vec(py, flat).reshape([n, 3]).expect("reshape");
    dict.set_item("coords", coords_arr)?;

    // Final velocities
    let flat_v: Vec<f64> = result.velocities.iter().flat_map(|v| v.iter().copied()).collect();
    let vel_arr = PyArray1::from_vec(py, flat_v).reshape([n, 3]).expect("reshape");
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
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub fn py_forcefield(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_energy, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_structure, m)?)?;
    m.add_function(wrap_pyfunction!(batch_minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_minimize_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(run_md, m)?)?;
    Ok(())
}
