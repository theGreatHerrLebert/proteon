//! PyO3 bindings for SASA (Solvent Accessible Surface Area).
//!
//! Single-structure and batch-parallel variants, plus load+analyze combos.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::IntoPyObject;
use rayon::prelude::*;

use crate::batch::{make_batch_result, BatchOutcome, PyBatchResult};
use crate::parallel::{auto_threads, build_pool};
use crate::py_pdb::PyPDB;
use crate::sasa;

/// Estimated peak per-task memory for SASA.
///
/// The theoretical minimum is just the CellList (~80 MB capped) plus
/// per-task temporaries. But empirically on monster3 (120 cores, 250 GB),
/// 120 threads OOM while 32 threads work fine. The effective per-thread
/// footprint from glibc arena fragmentation and transient allocations is
/// much larger than the logical working set.
///
/// Budget 3 GB/thread so auto_threads clamps 250 GB / (3 GB / 0.75 headroom)
/// ≈ 62 threads on monster3 — safely in the known-good zone.
const SASA_PER_TASK_BYTES: usize = 3 * 1024 * 1024 * 1024;

fn parse_radii(radii: &str) -> PyResult<sasa::RadiiSet> {
    match radii.to_lowercase().as_str() {
        "protor" | "naccess" | "freesasa" => Ok(sasa::RadiiSet::ProtOr),
        "bondi" => Ok(sasa::RadiiSet::Bondi),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown radii set '{}'. Use 'bondi' or 'protor'.",
            radii
        ))),
    }
}

fn validate_n_points(n_points: usize) -> PyResult<()> {
    if n_points == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_points must be > 0",
        ));
    }
    Ok(())
}

fn extract_radii(pdb: &pdbtbx::PDB, radii_set: sasa::RadiiSet) -> (Vec<[f64; 3]>, Vec<f64>) {
    let mut coords = Vec::new();
    let mut radii = Vec::new();
    // Use first model only (consistent with atom_count(), DSSP, hbonds, etc.)
    // pdb.chains() iterates ALL models, which inflates NMR ensembles by n_models×
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return (coords, radii),
    };
    for chain in first_model.chains() {
        for residue in chain.residues() {
            let res_name = residue.name().unwrap_or("");
            for atom in crate::altloc::residue_atoms_primary(residue) {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);
                let elem = atom.element().map(|e| e.symbol()).unwrap_or("");
                let r = match radii_set {
                    sasa::RadiiSet::Bondi => sasa::vdw_radius(elem).unwrap_or(sasa::DEFAULT_RADIUS),
                    sasa::RadiiSet::ProtOr => sasa::protor_radius(atom.name(), res_name, elem),
                };
                radii.push(r);
            }
        }
    }
    (coords, radii)
}

// ===========================================================================
// Single-structure SASA
// ===========================================================================

/// Compute per-atom SASA for a structure.
///
/// Args:
///     pdb: Structure to analyze.
///     probe: Probe radius in Angstroms (default 1.4 for water).
///     n_points: Test points per sphere (default 960).
///
/// Returns:
///     1D numpy array of per-atom SASA in Angstroms².
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub(crate) fn atom_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let result = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    Ok(result.into_pyarray(py))
}

/// Compute per-residue SASA for a structure.
///
/// Returns:
///     1D numpy array of per-residue SASA in Angstroms².
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub(crate) fn residue_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let atom_areas = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    let res_areas = sasa::residue_sasa(&pdb.inner, &atom_areas);
    Ok(res_areas.into_pyarray(py))
}

/// Compute relative solvent accessibility (RSA) per residue.
///
/// RSA = residue_sasa / max_sasa_for_residue_type.
/// Values > 0.25 are typically considered "exposed".
///
/// Returns:
///     1D numpy array of RSA values (0.0 to ~1.0+). NaN for unknown residue types.
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub(crate) fn relative_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let atom_areas = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    let res_areas = sasa::residue_sasa(&pdb.inner, &atom_areas);

    let mut rsa = Vec::with_capacity(res_areas.len());
    for (i, chain) in pdb.inner.chains().enumerate() {
        let _ = i;
        for residue in chain.residues() {
            let name = residue.name().unwrap_or("?");
            let max = sasa::max_sasa(name);
            let idx = rsa.len();
            if idx < res_areas.len() {
                match max {
                    Some(m) if m > 0.0 => rsa.push(res_areas[idx] / m),
                    _ => rsa.push(f64::NAN),
                }
            }
        }
    }

    Ok(rsa.into_pyarray(py))
}

/// Total SASA of a structure.
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub(crate) fn total_sasa(
    py: Python<'_>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> PyResult<f64> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let areas = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    Ok(areas.iter().sum())
}

// ===========================================================================
// Batch SASA (rayon over structures)
// ===========================================================================

/// Compute per-atom SASA for many structures in parallel.
///
/// Returns list of 1D numpy arrays.
#[pyfunction]
#[pyo3(signature = (structures, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub(crate) fn batch_atom_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    // Pass raw &pdbtbx::PDB pointers across py.allow_threads instead of
    // cloning each PDB. Safety: the input PyList borrow outlives this
    // function call, so the underlying PyPDB instances (and the
    // pdbtbx::PDB they own) stay valid for the whole rayon section.
    // Only read access happens in the rayon body, no aliased mutation.
    let pdb_addrs: Vec<usize> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            let ptr: *const pdbtbx::PDB = &pdb.inner;
            Ok(ptr as usize)
        })
        .collect::<PyResult<_>>()?;

    let n = auto_threads(n_threads, SASA_PER_TASK_BYTES);

    let results: Vec<Vec<f64>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pdb_addrs
                .par_iter()
                .map(|&addr| {
                    // Safety: addr came from a live &pdbtbx::PDB held by a
                    // PyPDB inside the input structures list, which is
                    // borrowed for 'py and outlives this closure.
                    let pdb: &pdbtbx::PDB = unsafe { &*(addr as *const pdbtbx::PDB) };
                    let (coords, radii) = extract_radii(pdb, rs);
                    // Use serial shrake_rupley: we already get parallelism
                    // across structures; nested shrake_rupley_parallel inside
                    // a busy pool just adds scheduling overhead without
                    // additional speedup.
                    sasa::shrake_rupley(&coords, &radii, probe, n_points)
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|areas| areas.into_pyarray(py))
        .collect())
}

/// Compute total SASA for many structures in parallel.
///
/// Returns 1D numpy array of total SASA values.
#[pyfunction]
#[pyo3(signature = (structures, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub(crate) fn batch_total_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let data: Vec<(Vec<[f64; 3]>, Vec<f64>)> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_radii(&pdb.inner, rs))
        })
        .collect::<PyResult<_>>()?;

    let n = auto_threads(n_threads, SASA_PER_TASK_BYTES);

    let results: Vec<f64> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            data.par_iter()
                .map(|(coords, radii)| {
                    let areas = sasa::shrake_rupley(coords, radii, probe, n_points);
                    areas.iter().sum()
                })
                .collect()
        })
    });

    Ok(results.into_pyarray(py))
}

/// Compute per-residue SASA for many structures in parallel.
///
/// Returns list of 1D numpy arrays, one per structure, each of length
/// equal to that structure's residue count.
#[pyfunction]
#[pyo3(signature = (structures, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub(crate) fn batch_residue_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    // See batch_atom_sasa for the safety rationale.
    let pdb_addrs: Vec<usize> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            let ptr: *const pdbtbx::PDB = &pdb.inner;
            Ok(ptr as usize)
        })
        .collect::<PyResult<_>>()?;

    let n = auto_threads(n_threads, SASA_PER_TASK_BYTES);

    let results: Vec<Vec<f64>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pdb_addrs
                .par_iter()
                .map(|&addr| {
                    // Safety: see batch_atom_sasa.
                    let pdb: &pdbtbx::PDB = unsafe { &*(addr as *const pdbtbx::PDB) };
                    let (coords, radii) = extract_radii(pdb, rs);
                    let atom_areas = sasa::shrake_rupley(&coords, &radii, probe, n_points);
                    sasa::residue_sasa(pdb, &atom_areas)
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|areas| areas.into_pyarray(py))
        .collect())
}

/// Compute relative SASA per residue (RSA) for many structures in parallel.
///
/// Returns list of 1D numpy arrays. NaN for non-standard residue types.
#[pyfunction]
#[pyo3(signature = (structures, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub(crate) fn batch_relative_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    // See batch_atom_sasa for the safety rationale.
    let pdb_addrs: Vec<usize> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            let ptr: *const pdbtbx::PDB = &pdb.inner;
            Ok(ptr as usize)
        })
        .collect::<PyResult<_>>()?;

    let n = auto_threads(n_threads, SASA_PER_TASK_BYTES);

    let results: Vec<Vec<f64>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pdb_addrs
                .par_iter()
                .map(|&addr| {
                    // Safety: see batch_atom_sasa.
                    let pdb: &pdbtbx::PDB = unsafe { &*(addr as *const pdbtbx::PDB) };
                    let (coords, radii) = extract_radii(pdb, rs);
                    let atom_areas = sasa::shrake_rupley(&coords, &radii, probe, n_points);
                    let res_areas = sasa::residue_sasa(pdb, &atom_areas);
                    // Mirror py_sasa::relative_sasa exactly so parity holds.
                    let mut rsa = Vec::with_capacity(res_areas.len());
                    for chain in pdb.chains() {
                        for residue in chain.residues() {
                            let name = residue.name().unwrap_or("?");
                            let max = sasa::max_sasa(name);
                            let idx = rsa.len();
                            if idx < res_areas.len() {
                                match max {
                                    Some(m) if m > 0.0 => rsa.push(res_areas[idx] / m),
                                    _ => rsa.push(f64::NAN),
                                }
                            }
                        }
                    }
                    rsa
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|areas| areas.into_pyarray(py))
        .collect())
}

// ===========================================================================
// Load + SASA (zero GIL pipeline)
// ===========================================================================

fn permissive_load(path: &str) -> Result<pdbtbx::PDB, String> {
    let mut parsing = pdbtbx::ParsingLevel::all();
    parsing.set_cryst1(false);
    parsing.set_master(false);

    let mut opts = pdbtbx::ReadOptions::new();
    opts.set_level(pdbtbx::StrictnessLevel::Loose)
        .set_parsing_level(&parsing);

    opts.read(path).map(|(pdb, _)| pdb).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })
}

/// Load files and compute total SASA in one parallel call.
///
/// Returns a `BatchResult` with one item per path (input order); each
/// successful item value is the structure's total SASA (float). A file that
/// fails to load is recorded as a failed item carrying the parse error —
/// failures are no longer silently dropped from the output. Pass
/// `strict=True` to raise on the first failure instead.
#[pyfunction]
#[pyo3(signature = (paths, probe=1.4, n_points=960, n_threads=None, radii="bondi", strict=false))]
pub(crate) fn load_and_sasa(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
    strict: bool,
) -> PyResult<PyBatchResult> {
    validate_n_points(n_points)?;
    let rs = parse_radii(radii)?;
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = auto_threads(n_threads, SASA_PER_TASK_BYTES);

    let results: Vec<Result<f64, String>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .map(|path| {
                    permissive_load(path).map(|pdb| {
                        let areas = sasa::sasa_from_pdb(&pdb, probe, n_points, rs);
                        areas.iter().sum()
                    })
                })
                .collect()
        })
    });

    let outcomes: Vec<BatchOutcome> = results
        .into_iter()
        .map(|r| match r {
            Ok(total) => total
                .into_pyobject(py)
                .map(|b| b.into_any().unbind())
                .map_err(|e| e.to_string()),
            Err(e) => Err(e),
        })
        .collect();
    make_batch_result(py, outcomes, strict)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_sasa(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(atom_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(relative_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_atom_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_relative_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_sasa, m)?)?;
    crate::batch::register(m)?;
    Ok(())
}
