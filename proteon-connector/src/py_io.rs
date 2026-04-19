//! PyO3 I/O functions for loading and saving PDB/mmCIF files.
//!
//! Includes batch_load for parallel I/O with rayon (GIL released).

use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::py_pdb::PyPDB;

/// Create permissive ReadOptions that skip problematic records.
///
/// Many PDB files from the PDB archive have non-standard space groups
/// or MASTER checksum mismatches. We skip CRYST1 and MASTER parsing
/// to maximize compatibility — these records aren't needed for
/// structural analysis (coordinates, alignment, etc.).
fn permissive_options() -> pdbtbx::ReadOptions {
    let mut parsing = pdbtbx::ParsingLevel::all();
    parsing.set_cryst1(false);
    parsing.set_master(false);

    let mut opts = pdbtbx::ReadOptions::new();
    opts.set_level(pdbtbx::StrictnessLevel::Loose)
        .set_parsing_level(&parsing);
    opts
}

/// Load a structure from a PDB or mmCIF file (auto-detected by extension).
///
/// Args:
///     path: Path to the file (.pdb, .cif, .mmcif, .pdb.gz, .cif.gz).
///
/// Returns:
///     PyPDB: The parsed structure.
#[pyfunction]
pub(crate) fn load(path: &str) -> PyResult<PyPDB> {
    let (pdb, _errors) = permissive_options().read(path).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {msg}"))
    })?;
    Ok(PyPDB::from_inner(pdb))
}

/// Load a structure, forcing PDB format.
#[pyfunction]
pub(crate) fn load_pdb(path: &str) -> PyResult<PyPDB> {
    let mut opts = permissive_options();
    opts.set_format(pdbtbx::Format::Pdb);

    let (pdb, _errors) = opts.read(path).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {msg}"))
    })?;
    Ok(PyPDB::from_inner(pdb))
}

/// Load a structure, forcing mmCIF format.
#[pyfunction]
pub(crate) fn load_mmcif(path: &str) -> PyResult<PyPDB> {
    let mut opts = permissive_options();
    opts.set_format(pdbtbx::Format::Mmcif);

    let (pdb, _errors) = opts.read(path).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {msg}"))
    })?;
    Ok(PyPDB::from_inner(pdb))
}

/// Save a structure to a PDB or mmCIF file (format auto-detected by extension).
///
/// Args:
///     pdb: The structure to save.
///     path: Output file path (.pdb or .cif/.mmcif).
#[pyfunction]
pub(crate) fn save(pdb: &PyPDB, path: &str) -> PyResult<()> {
    pdbtbx::save(&pdb.inner, path, pdbtbx::StrictnessLevel::Loose).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to save {path}: {msg}"))
    })
}

/// Save a structure, forcing PDB format.
#[pyfunction]
pub(crate) fn save_pdb(pdb: &PyPDB, path: &str) -> PyResult<()> {
    pdbtbx::save_pdb(&pdb.inner, path, pdbtbx::StrictnessLevel::Loose).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to save {path}: {msg}"))
    })
}

/// Save a structure, forcing mmCIF format.
#[pyfunction]
pub(crate) fn save_mmcif(pdb: &PyPDB, path: &str) -> PyResult<()> {
    pdbtbx::save_mmcif(&pdb.inner, path, pdbtbx::StrictnessLevel::Loose).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        pyo3::exceptions::PyIOError::new_err(format!("Failed to save {path}: {msg}"))
    })
}

// ---------------------------------------------------------------------------
// Batch loading (rayon parallel, GIL released)
// ---------------------------------------------------------------------------

use crate::parallel::resolve_threads;

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Load a single PDB/mmCIF file permissively. Returns Ok(PDB) or Err(message).
fn load_one(path: &str) -> Result<pdbtbx::PDB, String> {
    permissive_options()
        .read(path)
        .map(|(pdb, _errors)| pdb)
        .map_err(|errs| {
            errs.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        })
}

/// Load many structures in parallel using rayon. GIL is released.
///
/// Args:
///     paths: List of file paths (str).
///     n_threads: Number of threads. None/-1 = all cores.
///
/// Returns:
///     List of PyPDB objects (same order as paths).
///
/// Raises:
///     IOError if any file fails to load.
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn batch_load(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyPDB>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<Result<pdbtbx::PDB, String>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| path_strs.par_iter().map(|p| load_one(p)).collect())
    });

    results
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            r.map(PyPDB::from_inner).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to read {}: {e}",
                    path_strs[i]
                ))
            })
        })
        .collect()
}

/// Load many structures in parallel, skipping failures.
///
/// Args:
///     paths: List of file paths.
///     n_threads: Number of threads. None/-1 = all cores.
///
/// Returns:
///     List of (index, PyPDB) tuples for files that loaded successfully.
///     The index is the position in the original paths list.
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn batch_load_tolerant(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, PyPDB)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<Result<pdbtbx::PDB, String>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| path_strs.par_iter().map(|p| load_one(p)).collect())
    });

    Ok(results
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| r.ok().map(|pdb| (i, PyPDB::from_inner(pdb))))
        .collect())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_io(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(load_mmcif, m)?)?;
    m.add_function(wrap_pyfunction!(batch_load, m)?)?;
    m.add_function(wrap_pyfunction!(batch_load_tolerant, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(save_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(save_mmcif, m)?)?;
    Ok(())
}
