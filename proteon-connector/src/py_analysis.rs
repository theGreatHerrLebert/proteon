//! Rust-native structural analysis functions with rayon parallelism.
//!
//! All batch functions release the GIL and use rayon for true parallelism.
//! Single-structure functions are also implemented in Rust for speed.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::altloc::{pdb_atom_count_primary, pdb_atoms_primary, residue_atoms_primary};
use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Extract CA atom coordinates from a pdbtbx PDB (first model, primary
/// conformer per residue — avoids altloc duplication; see crate::altloc).
fn extract_ca(pdb: &pdbtbx::PDB) -> Vec<[f64; 3]> {
    pdb_atoms_primary(pdb)
        .filter(|a| a.name().trim() == "CA")
        .map(|a| {
            let (x, y, z) = a.pos();
            [x, y, z]
        })
        .collect()
}

/// Extract backbone (N, CA, C) coordinates grouped by residue from a pdbtbx PDB.
/// Returns (triples, breaks) where breaks[i] marks residue indices at which the
/// backbone is discontinuous (chain boundaries and CA-CA gaps > 4.5 Å, e.g. from
/// insertion-code interleaving or missing residues).
fn extract_backbone(pdb: &pdbtbx::PDB) -> (Vec<([f64; 3], [f64; 3], [f64; 3])>, Vec<usize>) {
    let mut result: Vec<([f64; 3], [f64; 3], [f64; 3])> = Vec::new();
    let mut breaks = Vec::new();

    // Use first model only (consistent with atom_count(), SASA, DSSP, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return (result, breaks),
    };
    for chain in first_model.chains() {
        let start = result.len();
        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if !is_aa {
                continue;
            }

            let mut n_pos: Option<[f64; 3]> = None;
            let mut ca_pos: Option<[f64; 3]> = None;
            let mut c_pos: Option<[f64; 3]> = None;

            for atom in residue_atoms_primary(residue) {
                let name = atom.name().trim();
                let (x, y, z) = atom.pos();
                match name {
                    "N" => n_pos = Some([x, y, z]),
                    "CA" => ca_pos = Some([x, y, z]),
                    "C" => c_pos = Some([x, y, z]),
                    _ => {}
                }
            }

            if let (Some(n), Some(ca), Some(c)) = (n_pos, ca_pos, c_pos) {
                // Detect backbone discontinuity via CA-CA distance
                if let Some(prev) = result.last() {
                    let prev_ca: &[f64; 3] = &prev.1;
                    let dx = ca[0] - prev_ca[0];
                    let dy = ca[1] - prev_ca[1];
                    let dz = ca[2] - prev_ca[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist > 4.5 {
                        breaks.push(result.len());
                    }
                }
                result.push((n, ca, c));
            }
        }
        if result.len() > start {
            breaks.push(start);
        }
    }

    (result, breaks)
}

/// Compute dihedral angle in degrees from four points.
fn dihedral(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
    let b1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b3 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];

    let n1 = cross(&b1, &b2);
    let n2 = cross(&b2, &b3);

    let n1_len = norm(&n1);
    let n2_len = norm(&n2);

    if n1_len < 1e-10 || n2_len < 1e-10 {
        return 0.0;
    }

    let n1_hat = [n1[0] / n1_len, n1[1] / n1_len, n1[2] / n1_len];
    let n2_hat = [n2[0] / n2_len, n2[1] / n2_len, n2[2] / n2_len];

    let b2_len = norm(&b2);
    let b2_hat = if b2_len > 1e-10 {
        [b2[0] / b2_len, b2[1] / b2_len, b2[2] / b2_len]
    } else {
        [0.0, 0.0, 1.0]
    };

    let m1 = cross(&n1_hat, &b2_hat);

    let x = dot(&n1_hat, &n2_hat);
    let y = dot(&m1, &n2_hat);

    (-y).atan2(x).to_degrees()
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Compute backbone dihedrals from backbone triples + chain break indices.
/// Returns (phi, psi, omega) arrays. NaN for undefined (chain termini).
fn compute_dihedrals(
    bb: &[([f64; 3], [f64; 3], [f64; 3])],
    chain_starts: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = bb.len();
    let mut phi = vec![f64::NAN; n];
    let mut psi = vec![f64::NAN; n];
    let mut omega = vec![f64::NAN; n];

    for i in 1..n {
        // Skip across chain boundaries
        if chain_starts.contains(&i) {
            continue;
        }
        // phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        phi[i] = dihedral(&bb[i - 1].2, &bb[i].0, &bb[i].1, &bb[i].2);
        // omega[i] = dihedral(CA[i-1], C[i-1], N[i], CA[i])
        omega[i] = dihedral(&bb[i - 1].1, &bb[i - 1].2, &bb[i].0, &bb[i].1);
    }
    for i in 0..n.saturating_sub(1) {
        if chain_starts.contains(&(i + 1)) {
            continue;
        }
        // psi[i] = dihedral(N[i], CA[i], C[i], N[i+1])
        psi[i] = dihedral(&bb[i].0, &bb[i].1, &bb[i].2, &bb[i + 1].0);
    }

    (phi, psi, omega)
}

/// Compute pairwise distance matrix from coordinates.
fn distance_matrix_rust(coords: &[[f64; 3]]) -> Vec<f64> {
    let n = coords.len();
    let mut dm = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            dm[i * n + j] = d;
            dm[j * n + i] = d;
        }
    }
    dm
}

// ===========================================================================
// Single-structure functions (exposed to Python)
// ===========================================================================

/// Extract CA coordinates from a structure as Mx3 numpy array.
#[pyfunction]
pub(crate) fn extract_ca_coords<'py>(py: Python<'py>, pdb: &PyPDB) -> Bound<'py, PyArray2<f64>> {
    let ca = extract_ca(&pdb.inner);
    let n = ca.len();
    let flat: Vec<f64> = ca.into_iter().flat_map(|c| c.into_iter()).collect();
    PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape to Mx3")
}

/// Compute CA-CA distance matrix as NxN numpy array.
#[pyfunction]
pub(crate) fn ca_distance_matrix<'py>(py: Python<'py>, pdb: &PyPDB) -> Bound<'py, PyArray2<f64>> {
    let ca = extract_ca(&pdb.inner);
    let n = ca.len();
    let dm = py.allow_threads(|| distance_matrix_rust(&ca));
    PyArray1::from_vec(py, dm)
        .reshape([n, n])
        .expect("reshape to NxN")
}

/// Compute CA-CA contact map as NxN boolean numpy array.
#[pyfunction]
pub(crate) fn ca_contact_map<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    cutoff: f64,
) -> Bound<'py, PyArray2<bool>> {
    let ca = extract_ca(&pdb.inner);
    let n = ca.len();
    let dm = py.allow_threads(|| distance_matrix_rust(&ca));
    let contacts: Vec<bool> = dm.iter().map(|&d| d <= cutoff).collect();
    PyArray1::from_vec(py, contacts)
        .reshape([n, n])
        .expect("reshape to NxN")
}

/// Compute backbone phi/psi/omega angles.
/// Returns tuple of (phi, psi, omega) as 1D numpy arrays. NaN for undefined.
#[pyfunction]
pub(crate) fn backbone_dihedrals<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let (bb, chain_starts) = extract_backbone(&pdb.inner);
    let (phi, psi, omega) = py.allow_threads(|| compute_dihedrals(&bb, &chain_starts));
    (
        phi.into_pyarray(py),
        psi.into_pyarray(py),
        omega.into_pyarray(py),
    )
}

/// Compute centroid of all atom coordinates (first model, primary conformer).
#[pyfunction]
pub(crate) fn centroid<'py>(py: Python<'py>, pdb: &PyPDB) -> Bound<'py, PyArray1<f64>> {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    let mut n = 0usize;
    for atom in pdb_atoms_primary(&pdb.inner) {
        let (x, y, z) = atom.pos();
        sx += x;
        sy += y;
        sz += z;
        n += 1;
    }
    let nf = n as f64;
    vec![sx / nf, sy / nf, sz / nf].into_pyarray(py)
}

/// Compute radius of gyration (first model, primary conformer).
#[pyfunction]
pub(crate) fn radius_of_gyration(_py: Python<'_>, pdb: &PyPDB) -> f64 {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    let mut n = 0usize;
    for atom in pdb_atoms_primary(&pdb.inner) {
        let (x, y, z) = atom.pos();
        sx += x;
        sy += y;
        sz += z;
        n += 1;
    }
    let nf = n as f64;
    let cx = sx / nf;
    let cy = sy / nf;
    let cz = sz / nf;

    let mut sum_sq = 0.0f64;
    for atom in pdb_atoms_primary(&pdb.inner) {
        let (x, y, z) = atom.pos();
        let dx = x - cx;
        let dy = y - cy;
        let dz = z - cz;
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / nf).sqrt()
}

// ===========================================================================
// Batch-parallel functions (rayon + GIL release)
// ===========================================================================

// Pre-extract lightweight data from PDB list on the main thread, then
// parallelize the computation with rayon. This avoids cloning entire PDB
// structures across threads.

/// Batch extract CA coordinates from many structures in parallel.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
#[allow(unused_variables)]
pub(crate) fn batch_extract_ca<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    // Extract CA coords on main thread (needs PDB access)
    let all_ca: Vec<Vec<[f64; 3]>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_ca(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    // Convert to numpy (no rayon needed — extraction was the work)
    Ok(all_ca
        .into_iter()
        .map(|ca| {
            let nres = ca.len();
            let flat: Vec<f64> = ca.into_iter().flat_map(|c| c.into_iter()).collect();
            PyArray1::from_vec(py, flat)
                .reshape([nres, 3])
                .expect("reshape")
        })
        .collect())
}

/// Batch compute CA-CA distance matrices in parallel.
///
/// Extraction happens on main thread; O(n^2) distance computation is parallelized.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub(crate) fn batch_distance_matrices<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    // Extract CA coords on main thread
    let all_ca: Vec<Vec<[f64; 3]>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_ca(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    // Parallelize the O(n^2) distance computation
    let results: Vec<(usize, Vec<f64>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_ca
                .par_iter()
                .map(|ca| {
                    let nres = ca.len();
                    let dm = distance_matrix_rust(ca);
                    (nres, dm)
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(nres, dm)| {
            PyArray1::from_vec(py, dm)
                .reshape([nres, nres])
                .expect("reshape")
        })
        .collect())
}

/// Batch compute CA-CA contact maps in parallel.
#[pyfunction]
#[pyo3(signature = (structures, cutoff, n_threads=None))]
pub(crate) fn batch_contact_maps<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    cutoff: f64,
    n_threads: Option<i32>,
) -> PyResult<Vec<Bound<'py, PyArray2<bool>>>> {
    let all_ca: Vec<Vec<[f64; 3]>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_ca(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, Vec<bool>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_ca
                .par_iter()
                .map(|ca| {
                    let nres = ca.len();
                    let dm = distance_matrix_rust(ca);
                    let contacts: Vec<bool> = dm.iter().map(|&d| d <= cutoff).collect();
                    (nres, contacts)
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(nres, contacts)| {
            PyArray1::from_vec(py, contacts)
                .reshape([nres, nres])
                .expect("reshape")
        })
        .collect())
}

/// Batch compute backbone dihedrals in parallel.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub(crate) fn batch_dihedrals<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<
    Vec<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )>,
> {
    // Extract backbone on main thread
    let all_bb: Vec<(Vec<([f64; 3], [f64; 3], [f64; 3])>, Vec<usize>)> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_backbone(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    // Parallelize dihedral computation
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_bb
                .par_iter()
                .map(|(bb, cs)| compute_dihedrals(bb, cs))
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(phi, psi, omega)| {
            (
                phi.into_pyarray(py),
                psi.into_pyarray(py),
                omega.into_pyarray(py),
            )
        })
        .collect())
}

/// Batch compute radius of gyration in parallel.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub(crate) fn batch_radius_of_gyration<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Extract all coords on main thread
    let all_coords: Vec<Vec<[f64; 3]>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            let coords: Vec<[f64; 3]> = pdb_atoms_primary(&pdb.inner)
                .map(|a| {
                    let (x, y, z) = a.pos();
                    [x, y, z]
                })
                .collect();
            Ok(coords)
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<f64> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_coords
                .par_iter()
                .map(|coords| {
                    let nf = coords.len() as f64;
                    let (sx, sy, sz) = coords.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), c| {
                        (sx + c[0], sy + c[1], sz + c[2])
                    });
                    let cx = sx / nf;
                    let cy = sy / nf;
                    let cz = sz / nf;
                    let sum_sq: f64 = coords
                        .iter()
                        .map(|c| {
                            let dx = c[0] - cx;
                            let dy = c[1] - cy;
                            let dz = c[2] - cz;
                            dx * dx + dy * dy + dz * dz
                        })
                        .sum();
                    (sum_sq / nf).sqrt()
                })
                .collect()
        })
    });

    Ok(results.into_pyarray(py))
}

// ===========================================================================
// Load + Analyze in one shot (full pipeline in Rust, zero GIL)
// ===========================================================================

/// Permissive load options (same as py_io).
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

/// Load files + extract CA coordinates in one parallel call.
/// Entire pipeline (I/O → parse → extract) runs in Rust with rayon.
///
/// Returns list of (index, Mx3 array) for files that loaded successfully.
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn load_and_extract_ca<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, Bound<'py, PyArray2<f64>>)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, Vec<[f64; 3]>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let ca = extract_ca(&pdb);
                        (i, ca)
                    })
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(i, ca)| {
            let nres = ca.len();
            let flat: Vec<f64> = ca.into_iter().flat_map(|c| c.into_iter()).collect();
            let arr = PyArray1::from_vec(py, flat)
                .reshape([nres, 3])
                .expect("reshape");
            (i, arr)
        })
        .collect())
}

/// Load files + compute CA distance matrices in one parallel call.
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn load_and_distance_matrices<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, Bound<'py, PyArray2<f64>>)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, usize, Vec<f64>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let ca = extract_ca(&pdb);
                        let nres = ca.len();
                        let dm = distance_matrix_rust(&ca);
                        (i, nres, dm)
                    })
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(i, nres, dm)| {
            let arr = PyArray1::from_vec(py, dm)
                .reshape([nres, nres])
                .expect("reshape");
            (i, arr)
        })
        .collect())
}

/// Load files + compute contact maps in one parallel call.
#[pyfunction]
#[pyo3(signature = (paths, cutoff, n_threads=None))]
pub(crate) fn load_and_contact_maps<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    cutoff: f64,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, Bound<'py, PyArray2<bool>>)>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, usize, Vec<bool>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let ca = extract_ca(&pdb);
                        let nres = ca.len();
                        let dm = distance_matrix_rust(&ca);
                        let contacts: Vec<bool> = dm.iter().map(|&d| d <= cutoff).collect();
                        (i, nres, contacts)
                    })
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(i, nres, contacts)| {
            let arr = PyArray1::from_vec(py, contacts)
                .reshape([nres, nres])
                .expect("reshape");
            (i, arr)
        })
        .collect())
}

/// Load files + compute backbone dihedrals in one parallel call.
#[pyfunction]
#[pyo3(signature = (paths, n_threads=None))]
pub(crate) fn load_and_dihedrals<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<
    Vec<(
        usize,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )>,
> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let (bb, chain_starts) = extract_backbone(&pdb);
                        let (phi, psi, omega) = compute_dihedrals(&bb, &chain_starts);
                        (i, phi, psi, omega)
                    })
                })
                .collect()
        })
    });

    Ok(results
        .into_iter()
        .map(|(i, phi, psi, omega)| {
            (
                i,
                phi.into_pyarray(py),
                psi.into_pyarray(py),
                omega.into_pyarray(py),
            )
        })
        .collect())
}

/// Load files + compute everything: CA coords, distance matrix, contact map,
/// dihedrals, Rg — all in one parallel call. Zero GIL.
///
/// Returns list of dicts with all results for each successfully loaded structure.
#[pyfunction]
#[pyo3(signature = (paths, cutoff=8.0, n_threads=None))]
pub(crate) fn load_and_analyze<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyList>,
    cutoff: f64,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyObject>> {
    let path_strs: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    // Everything happens in Rust: load → parse → extract → compute
    let results: Vec<AnalysisBundle> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            path_strs
                .par_iter()
                .enumerate()
                .filter_map(|(i, path)| {
                    permissive_load(path).ok().map(|pdb| {
                        let ca = extract_ca(&pdb);
                        let nres = ca.len();
                        let dm = distance_matrix_rust(&ca);
                        let contacts: Vec<bool> = dm.iter().map(|&d| d <= cutoff).collect();

                        let (bb, chain_starts) = extract_backbone(&pdb);
                        let (phi, psi, omega) = compute_dihedrals(&bb, &chain_starts);

                        let n_atoms = pdb_atom_count_primary(&pdb);
                        let n_chains = pdb.chain_count();
                        let n_residues = pdb.residue_count();

                        // Rg — primary conformer only (see crate::altloc).
                        let all_coords: Vec<[f64; 3]> = pdb_atoms_primary(&pdb)
                            .map(|a| {
                                let (x, y, z) = a.pos();
                                [x, y, z]
                            })
                            .collect();
                        let nf = all_coords.len() as f64;
                        let (sx, sy, sz) =
                            all_coords.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), c| {
                                (sx + c[0], sy + c[1], sz + c[2])
                            });
                        let cx = sx / nf;
                        let cy = sy / nf;
                        let cz = sz / nf;
                        let rg = {
                            let sum_sq: f64 = all_coords
                                .iter()
                                .map(|c| {
                                    let dx = c[0] - cx;
                                    let dy = c[1] - cy;
                                    let dz = c[2] - cz;
                                    dx * dx + dy * dy + dz * dz
                                })
                                .sum();
                            (sum_sq / nf).sqrt()
                        };

                        AnalysisBundle {
                            index: i,
                            path: path.clone(),
                            n_atoms,
                            n_chains,
                            n_residues,
                            nres,
                            ca,
                            dm,
                            contacts,
                            phi,
                            psi,
                            omega,
                            rg,
                        }
                    })
                })
                .collect()
        })
    });

    // Convert to Python dicts
    let pydicts: Vec<PyObject> = results
        .into_iter()
        .map(|b| {
            let dict = pyo3::types::PyDict::new(py);

            dict.set_item("index", b.index).unwrap();
            dict.set_item("path", &b.path).unwrap();
            dict.set_item("n_atoms", b.n_atoms).unwrap();
            dict.set_item("n_chains", b.n_chains).unwrap();
            dict.set_item("n_residues", b.n_residues).unwrap();
            dict.set_item("n_ca", b.nres).unwrap();
            dict.set_item("rg", b.rg).unwrap();

            let ca_flat: Vec<f64> = b.ca.into_iter().flat_map(|c| c.into_iter()).collect();
            let ca_arr = PyArray1::from_vec(py, ca_flat)
                .reshape([b.nres, 3])
                .expect("reshape");
            dict.set_item("ca_coords", ca_arr).unwrap();

            let dm_arr = PyArray1::from_vec(py, b.dm)
                .reshape([b.nres, b.nres])
                .expect("reshape");
            dict.set_item("distance_matrix", dm_arr).unwrap();

            let cm_arr = PyArray1::from_vec(py, b.contacts)
                .reshape([b.nres, b.nres])
                .expect("reshape");
            dict.set_item("contact_map", cm_arr).unwrap();

            dict.set_item("phi", b.phi.into_pyarray(py)).unwrap();
            dict.set_item("psi", b.psi.into_pyarray(py)).unwrap();
            dict.set_item("omega", b.omega.into_pyarray(py)).unwrap();

            dict.into_any().unbind()
        })
        .collect();

    Ok(pydicts)
}

/// Bundle of all analysis results for one structure.
struct AnalysisBundle {
    index: usize,
    path: String,
    n_atoms: usize,
    n_chains: usize,
    n_residues: usize,
    nres: usize,
    ca: Vec<[f64; 3]>,
    dm: Vec<f64>,
    contacts: Vec<bool>,
    phi: Vec<f64>,
    psi: Vec<f64>,
    omega: Vec<f64>,
    rg: f64,
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_analysis(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Single-structure
    m.add_function(wrap_pyfunction!(extract_ca_coords, m)?)?;
    m.add_function(wrap_pyfunction!(ca_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ca_contact_map, m)?)?;
    m.add_function(wrap_pyfunction!(backbone_dihedrals, m)?)?;
    m.add_function(wrap_pyfunction!(centroid, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration, m)?)?;
    // Batch from pre-loaded structures
    m.add_function(wrap_pyfunction!(batch_extract_ca, m)?)?;
    m.add_function(wrap_pyfunction!(batch_distance_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(batch_contact_maps, m)?)?;
    m.add_function(wrap_pyfunction!(batch_dihedrals, m)?)?;
    m.add_function(wrap_pyfunction!(batch_radius_of_gyration, m)?)?;
    // Load + analyze (full pipeline, zero GIL)
    m.add_function(wrap_pyfunction!(load_and_extract_ca, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_distance_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_contact_maps, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_dihedrals, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_analyze, m)?)?;
    Ok(())
}
