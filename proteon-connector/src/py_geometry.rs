//! PyO3 building blocks: Kabsch superposition, RMSD, secondary structure,
//! TM-score computation.
//!
//! These are general-purpose structural geometry functions useful beyond
//! alignment — for MD analysis, docking, fitting, etc.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use proteon_align::core::kabsch::{kabsch, KabschMode};
use proteon_align::core::secondary_structure::make_sec;
use proteon_align::core::tmscore::standard_tmscore;
use proteon_align::core::types::{Coord3D, MolType, Transform};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an Nx3 numpy array to Vec<Coord3D>.
fn numpy_to_coords(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Coord3D>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected Nx3 array",
        ));
    }
    let n = shape[0];
    let slice = arr.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array not contiguous: {e}"))
    })?;
    let mut coords = Vec::with_capacity(n);
    for i in 0..n {
        coords.push([slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]]);
    }
    Ok(coords)
}

/// Convert Vec<Coord3D> to a flat Vec<f64> for numpy (Nx3).
fn coords_to_flat(coords: &[Coord3D]) -> Vec<f64> {
    coords.iter().flat_map(|c| c.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// Kabsch superposition
// ---------------------------------------------------------------------------

/// Compute optimal rotation and translation to superpose point set x onto y
/// using the Kabsch algorithm.
///
/// Args:
///     x: Nx3 numpy array of coordinates (mobile).
///     y: Nx3 numpy array of coordinates (reference).
///
/// Returns:
///     Tuple of (rmsd, rotation_matrix_3x3, translation_3).
///     RMSD is the root-mean-square deviation after optimal superposition.
#[pyfunction]
pub(crate) fn kabsch_superpose<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xc = numpy_to_coords(&x)?;
    let yc = numpy_to_coords(&y)?;
    if xc.len() != yc.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must have the same number of points",
        ));
    }
    let n = xc.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Empty coordinate arrays",
        ));
    }

    let result = kabsch(&xc, &yc, KabschMode::Both)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Kabsch algorithm failed"))?;

    let rmsd = (result.rms / n as f64).sqrt();

    let rot_flat: Vec<f64> = result
        .transform
        .u
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let rotation = PyArray1::from_vec(py, rot_flat)
        .reshape([3, 3])
        .expect("reshape to 3x3");
    let translation = result.transform.t.to_vec().into_pyarray(py);

    Ok((rmsd, rotation, translation))
}

/// Compute RMSD between two equal-length coordinate sets (without superposition).
///
/// Args:
///     x: Nx3 numpy array.
///     y: Nx3 numpy array.
///
/// Returns:
///     RMSD as float.
#[pyfunction]
pub(crate) fn rmsd_no_super<'py>(
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let xc = numpy_to_coords(&x)?;
    let yc = numpy_to_coords(&y)?;
    if xc.len() != yc.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must have the same number of points",
        ));
    }
    let n = xc.len();
    if n == 0 {
        return Ok(0.0);
    }
    let sum_sq: f64 = xc
        .iter()
        .zip(yc.iter())
        .map(|(a, b)| {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();
    Ok((sum_sq / n as f64).sqrt())
}

/// Compute RMSD after optimal Kabsch superposition.
///
/// Args:
///     x: Nx3 numpy array (mobile).
///     y: Nx3 numpy array (reference).
///
/// Returns:
///     RMSD after optimal superposition.
#[pyfunction]
pub(crate) fn rmsd<'py>(
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let xc = numpy_to_coords(&x)?;
    let yc = numpy_to_coords(&y)?;
    if xc.len() != yc.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must have the same number of points",
        ));
    }
    let n = xc.len();
    if n == 0 {
        return Ok(0.0);
    }
    let result = kabsch(&xc, &yc, KabschMode::RmsOnly)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Kabsch algorithm failed"))?;
    Ok((result.rms / n as f64).sqrt())
}

// ---------------------------------------------------------------------------
// Apply transform
// ---------------------------------------------------------------------------

/// Apply a rotation + translation to coordinates: y = R @ x + t.
///
/// Args:
///     coords: Nx3 numpy array.
///     rotation: 3x3 rotation matrix.
///     translation: length-3 translation vector.
///
/// Returns:
///     Nx3 numpy array of transformed coordinates.
#[pyfunction]
pub(crate) fn apply_transform<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rotation: PyReadonlyArray2<'py, f64>,
    translation: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xc = numpy_to_coords(&coords)?;

    let rot_shape = rotation.shape();
    if rot_shape != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Rotation must be 3x3",
        ));
    }
    let t_shape = translation.shape();
    if t_shape != [3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Translation must have length 3",
        ));
    }

    let r = rotation.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Rotation not contiguous: {e}"))
    })?;
    let t = translation.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Translation not contiguous: {e}"))
    })?;

    let transform = Transform {
        u: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
        t: [t[0], t[1], t[2]],
    };

    let n = xc.len();
    let mut result = vec![[0.0f64; 3]; n];
    transform.apply_batch(&xc, &mut result);

    let flat = coords_to_flat(&result);
    Ok(PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .expect("reshape to Nx3"))
}

// ---------------------------------------------------------------------------
// Secondary structure assignment
// ---------------------------------------------------------------------------

/// Assign secondary structure from CA coordinates using distance geometry.
///
/// Returns H (helix), E (sheet), T (turn), C (coil) for each residue.
/// This is a fast approximation — not DSSP, but useful for alignment
/// initialization and quick structural characterization.
///
/// Args:
///     coords: Nx3 numpy array of CA coordinates.
///
/// Returns:
///     String of length N with characters H/E/T/C.
#[pyfunction]
pub(crate) fn assign_secondary_structure(coords: PyReadonlyArray2<'_, f64>) -> PyResult<String> {
    let xc = numpy_to_coords(&coords)?;
    let sec = make_sec(&xc);
    Ok(sec.into_iter().collect())
}

// ---------------------------------------------------------------------------
// TM-score
// ---------------------------------------------------------------------------

/// Compute TM-score for a pre-existing alignment.
///
/// Given two coordinate arrays and an alignment map (invmap[j] = i means
/// y[j] aligns to x[i], -1 means gap), compute the TM-score normalized
/// by the length of y.
///
/// Args:
///     x: Nx3 numpy array (structure 1 CA coordinates).
///     y: Mx3 numpy array (structure 2 CA coordinates, normalization reference).
///     invmap: Length-M integer array. invmap[j] = i means y[j] aligns to x[i].
///
/// Returns:
///     Tuple of (tm_score, n_aligned, rmsd, rotation_3x3, translation_3).
#[pyfunction]
pub(crate) fn tm_score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    invmap: PyReadonlyArray1<'py, i32>,
) -> PyResult<(
    f64,
    usize,
    f64,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let xc = numpy_to_coords(&x)?;
    let yc = numpy_to_coords(&y)?;
    let map_slice = invmap.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invmap not contiguous: {e}"))
    })?;

    if map_slice.len() != yc.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invmap length must match y length",
        ));
    }

    let score_d8 = 1.5; // standard cutoff
    let (tmscore, n_ali, rmsd_val, transform) =
        standard_tmscore(&xc, &yc, map_slice, score_d8, MolType::Protein);

    let rot_flat: Vec<f64> = transform
        .u
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let rotation = PyArray1::from_vec(py, rot_flat)
        .reshape([3, 3])
        .expect("reshape to 3x3");
    let translation = transform.t.to_vec().into_pyarray(py);

    Ok((tmscore, n_ali, rmsd_val, rotation, translation))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_geometry(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kabsch_superpose, m)?)?;
    m.add_function(wrap_pyfunction!(rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(rmsd_no_super, m)?)?;
    m.add_function(wrap_pyfunction!(apply_transform, m)?)?;
    m.add_function(wrap_pyfunction!(assign_secondary_structure, m)?)?;
    m.add_function(wrap_pyfunction!(tm_score, m)?)?;
    Ok(())
}
