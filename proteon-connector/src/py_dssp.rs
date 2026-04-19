//! PyO3 bindings for DSSP secondary structure assignment.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::dssp;
use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Assign DSSP secondary structure to a protein structure.
///
/// Returns a string of one-letter codes per amino acid residue:
///   H = alpha helix, G = 3-10 helix, I = pi helix,
///   E = extended strand, B = isolated bridge,
///   T = turn, S = bend, C = coil.
///
/// Uses the Kabsch-Sander hydrogen bond energy criterion
/// (Kabsch & Sander, 1983).
#[pyfunction]
#[pyo3(name = "dssp")]
pub(crate) fn compute_dssp(py: Python<'_>, pdb: &PyPDB) -> String {
    let residues = dssp::extract_dssp_residues(&pdb.inner);
    py.allow_threads(|| dssp::assign_dssp(&residues))
}

/// Assign DSSP as a numpy array of u8 character codes.
///
/// Same as dssp() but returns a numpy array for vectorized operations.
#[pyfunction]
pub(crate) fn dssp_array<'py>(py: Python<'py>, pdb: &PyPDB) -> Bound<'py, PyArray1<u8>> {
    let residues = dssp::extract_dssp_residues(&pdb.inner);
    let ss = py.allow_threads(|| dssp::assign_dssp(&residues));
    ss.bytes().collect::<Vec<u8>>().into_pyarray(py)
}

/// Batch DSSP for many structures in parallel.
///
/// Returns list of SS strings.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub(crate) fn batch_dssp(
    py: Python<'_>,
    structures: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<String>> {
    // Extract residue data on main thread
    let all_residues: Vec<Vec<dssp::DsspResidue>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(dssp::extract_dssp_residues(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    // Parallelize the SS assignment
    let results: Vec<String> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_residues
                .par_iter()
                .map(|residues| dssp::assign_dssp(residues))
                .collect()
        })
    });

    Ok(results)
}

/// Load files and compute DSSP in one parallel call (zero GIL).
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn load_and_dssp(
    py: Python<'_>,
    paths: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, String)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, String)> = py.allow_threads(|| {
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
                        let residues = dssp::extract_dssp_residues(&pdb);
                        let ss = dssp::assign_dssp(&residues);
                        (i, ss)
                    })
                })
                .collect()
        })
    });

    Ok(results)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_dssp(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_dssp, m)?)?;
    m.add_function(wrap_pyfunction!(dssp_array, m)?)?;
    m.add_function(wrap_pyfunction!(batch_dssp, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_dssp, m)?)?;
    Ok(())
}
