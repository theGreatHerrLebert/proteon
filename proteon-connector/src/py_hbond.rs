//! PyO3 bindings for hydrogen bond detection.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::hbond;
use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Detect backbone hydrogen bonds using the Kabsch-Sander energy criterion.
///
/// Returns Nx4 numpy array where each row is:
///   [acceptor_residue_idx, donor_residue_idx, energy_kcal, distance_ON_angstrom]
///
/// Args:
///     pdb: Structure to analyze.
///     energy_cutoff: Energy threshold in kcal/mol (default -0.5).
///
/// Returns:
///     Nx4 float64 array of H-bond data.
#[pyfunction]
#[pyo3(signature = (pdb, energy_cutoff=-0.5))]
pub(crate) fn backbone_hbonds<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    energy_cutoff: f64,
) -> Bound<'py, PyArray2<f64>> {
    let bonds = py.allow_threads(|| hbond::backbone_hbonds(&pdb.inner, energy_cutoff));

    let n = bonds.len();
    let flat: Vec<f64> = bonds
        .iter()
        .flat_map(|b| [b.acceptor as f64, b.donor as f64, b.energy, b.dist_on])
        .collect();

    PyArray1::from_vec(py, flat)
        .reshape([n, 4])
        .expect("reshape to Nx4")
}

/// Detect geometric hydrogen bonds between polar atoms.
///
/// Uses donor-acceptor distance criterion (default 3.5 Å).
/// Donors: N, O atoms. Acceptors: O, S atoms.
///
/// Returns Nx3 numpy array: [donor_atom_idx, acceptor_atom_idx, distance]
#[pyfunction]
#[pyo3(signature = (pdb, dist_cutoff=3.5))]
pub(crate) fn geometric_hbonds<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    dist_cutoff: f64,
) -> Bound<'py, PyArray2<f64>> {
    let bonds = py.allow_threads(|| hbond::geometric_hbonds(&pdb.inner, dist_cutoff));

    let n = bonds.len();
    let flat: Vec<f64> = bonds
        .iter()
        .flat_map(|b| [b.donor_atom as f64, b.acceptor_atom as f64, b.distance])
        .collect();

    PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape to Nx3")
}

/// Count backbone H-bonds per residue (how many H-bonds each residue participates in).
///
/// Returns 1D array of length n_residues.
#[pyfunction]
#[pyo3(signature = (pdb, energy_cutoff=-0.5))]
pub(crate) fn hbond_count_per_residue<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    energy_cutoff: f64,
) -> Bound<'py, PyArray1<u32>> {
    let residues = crate::dssp::extract_dssp_residues(&pdb.inner);
    let bonds = py.allow_threads(|| hbond::backbone_hbonds(&pdb.inner, energy_cutoff));

    let n = residues.len();
    let mut counts = vec![0u32; n];
    for b in &bonds {
        if b.acceptor < n {
            counts[b.acceptor] += 1;
        }
        if b.donor < n {
            counts[b.donor] += 1;
        }
    }

    counts.into_pyarray(py)
}

/// Batch backbone H-bonds for many structures in parallel.
///
/// Returns list of Nx4 arrays.
#[pyfunction]
#[pyo3(signature = (structures, energy_cutoff=-0.5, n_threads=None))]
pub(crate) fn batch_backbone_hbonds<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    energy_cutoff: f64,
    n_threads: Option<i32>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    // Extract only DSSP residue data on main thread (no PDB clone)
    let all_residues: Vec<Vec<crate::dssp::DsspResidue>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(crate::dssp::extract_dssp_residues(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<Vec<hbond::BackboneHBond>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_residues
                .par_iter()
                .map(|residues| hbond::backbone_hbonds_from_residues(residues, energy_cutoff))
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|bonds| {
            let nb = bonds.len();
            let flat: Vec<f64> = bonds
                .iter()
                .flat_map(|b| [b.acceptor as f64, b.donor as f64, b.energy, b.dist_on])
                .collect();
            PyArray1::from_vec(py, flat)
                .reshape([nb, 4])
                .expect("reshape")
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_hbond(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backbone_hbonds, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_hbonds, m)?)?;
    m.add_function(wrap_pyfunction!(hbond_count_per_residue, m)?)?;
    m.add_function(wrap_pyfunction!(batch_backbone_hbonds, m)?)?;
    Ok(())
}
