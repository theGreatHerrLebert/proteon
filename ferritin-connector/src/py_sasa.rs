//! PyO3 bindings for SASA (Solvent Accessible Surface Area).
//!
//! Single-structure and batch-parallel variants, plus load+analyze combos.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::py_pdb::PyPDB;
use crate::sasa;

fn parse_radii(radii: &str) -> sasa::RadiiSet {
    match radii.to_lowercase().as_str() {
        "protor" | "naccess" | "freesasa" => sasa::RadiiSet::ProtOr,
        _ => sasa::RadiiSet::Bondi,
    }
}

fn extract_radii(pdb: &pdbtbx::PDB, radii_set: sasa::RadiiSet) -> (Vec<[f64; 3]>, Vec<f64>) {
    let mut coords = Vec::new();
    let mut radii = Vec::new();
    for chain in pdb.chains() {
        for residue in chain.residues() {
            let res_name = residue.name().unwrap_or("");
            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);
                let elem = atom.element().map(|e| e.symbol()).unwrap_or("");
                let r = match radii_set {
                    sasa::RadiiSet::Bondi => {
                        sasa::vdw_radius(elem).unwrap_or(sasa::DEFAULT_RADIUS)
                    }
                    sasa::RadiiSet::ProtOr => {
                        sasa::protor_radius(atom.name(), res_name, elem)
                    }
                };
                radii.push(r);
            }
        }
    }
    (coords, radii)
}

fn resolve_threads(n: Option<i32>) -> usize {
    match n {
        None | Some(-1) => 0,
        Some(n) => n.max(1) as usize,
    }
}

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
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
pub fn atom_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> Bound<'py, PyArray1<f64>> {
    let rs = parse_radii(radii);
    let result = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    result.into_pyarray(py)
}

/// Compute per-residue SASA for a structure.
///
/// Returns:
///     1D numpy array of per-residue SASA in Angstroms².
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub fn residue_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> Bound<'py, PyArray1<f64>> {
    let rs = parse_radii(radii);
    let atom_areas =
        py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    let res_areas = sasa::residue_sasa(&pdb.inner, &atom_areas);
    res_areas.into_pyarray(py)
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
pub fn relative_sasa<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> Bound<'py, PyArray1<f64>> {
    let rs = parse_radii(radii);
    let atom_areas =
        py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
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

    rsa.into_pyarray(py)
}

/// Total SASA of a structure.
#[pyfunction]
#[pyo3(signature = (pdb, probe=1.4, n_points=960, radii="bondi"))]
pub fn total_sasa(
    py: Python<'_>,
    pdb: &PyPDB,
    probe: f64,
    n_points: usize,
    radii: &str,
) -> f64 {
    let rs = parse_radii(radii);
    let areas = py.allow_threads(|| sasa::sasa_from_pdb(&pdb.inner, probe, n_points, rs));
    areas.iter().sum()
}

// ===========================================================================
// Batch SASA (rayon over structures)
// ===========================================================================

/// Compute per-atom SASA for many structures in parallel.
///
/// Returns list of 1D numpy arrays.
#[pyfunction]
#[pyo3(signature = (structures, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub fn batch_atom_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let rs = parse_radii(radii);
    let data: Vec<(Vec<[f64; 3]>, Vec<f64>)> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_radii(&pdb.inner, rs))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    // Parallelize the SASA computation
    let results: Vec<Vec<f64>> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            data.par_iter()
                .map(|(coords, radii)| sasa::shrake_rupley(coords, radii, probe, n_points))
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
pub fn batch_total_sasa<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let rs = parse_radii(radii);
    let data: Vec<(Vec<[f64; 3]>, Vec<f64>)> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_radii(&pdb.inner, rs))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

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

    opts.read(path)
        .map(|(pdb, _)| pdb)
        .map_err(|errs| {
            errs.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        })
}

/// Load files and compute total SASA in one parallel call.
///
/// Returns list of (index, total_sasa) for files that loaded successfully.
#[pyfunction]
#[pyo3(signature = (paths, probe=1.4, n_points=960, n_threads=None, radii="bondi"))]
pub fn load_and_sasa<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    probe: f64,
    n_points: usize,
    n_threads: Option<i32>,
    radii: &str,
) -> PyResult<Vec<(usize, f64)>> {
    let rs = parse_radii(radii);
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, f64)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let areas = sasa::sasa_from_pdb(&pdb, probe, n_points, rs);
                        (i, areas.iter().sum())
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
pub fn py_sasa(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(atom_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(relative_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_atom_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(batch_total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_sasa, m)?)?;
    Ok(())
}
