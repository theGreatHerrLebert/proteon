//! PyO3 bindings for alignment results and functions.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use proteon_align::core::types::AlignResult;

/// Python-visible alignment result.
#[pyclass]
#[derive(Clone)]
pub struct PyAlignResult {
    pub inner: AlignResult,
}

#[pymethods]
impl PyAlignResult {
    /// TM-score normalized by length of chain 1.
    #[getter]
    fn tm_score_chain1(&self) -> f64 {
        self.inner.tm_score_chain1
    }

    /// TM-score normalized by length of chain 2.
    #[getter]
    fn tm_score_chain2(&self) -> f64 {
        self.inner.tm_score_chain2
    }

    /// RMSD of aligned residues.
    #[getter]
    fn rmsd(&self) -> f64 {
        self.inner.rmsd
    }

    /// Number of aligned residue pairs.
    #[getter]
    fn n_aligned(&self) -> usize {
        self.inner.n_aligned
    }

    /// Sequence identity (fraction).
    #[getter]
    fn seq_identity(&self) -> f64 {
        self.inner.seq_identity
    }

    /// Aligned sequence of structure 1 (with gaps).
    #[getter]
    fn aligned_seq_x(&self) -> String {
        self.inner.aligned_seq_x.clone()
    }

    /// Aligned sequence of structure 2 (with gaps).
    #[getter]
    fn aligned_seq_y(&self) -> String {
        self.inner.aligned_seq_y.clone()
    }

    /// Alignment markers (`:` = close, `.` = aligned, ` ` = gap).
    #[getter]
    fn alignment_markers(&self) -> String {
        self.inner.alignment_markers.clone()
    }

    /// Rotation matrix as 3x3 numpy array.
    #[getter]
    fn rotation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let u = self.inner.transform.u;
        let flat: Vec<f64> = u.iter().flat_map(|row| row.iter().copied()).collect();
        numpy::PyArray1::from_vec(py, flat).reshape([3, 3]).unwrap()
    }

    /// Translation vector as [x, y, z].
    #[getter]
    fn translation(&self) -> [f64; 3] {
        self.inner.transform.t
    }

    fn __repr__(&self) -> String {
        format!(
            "AlignResult(TM1={:.4}, TM2={:.4}, RMSD={:.2}, Lali={})",
            self.inner.tm_score_chain1,
            self.inner.tm_score_chain2,
            self.inner.rmsd,
            self.inner.n_aligned,
        )
    }
}

impl PyAlignResult {
    pub fn from_inner(result: AlignResult) -> Self {
        PyAlignResult { inner: result }
    }
}

#[pymodule]
pub(crate) fn py_align(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlignResult>()?;
    Ok(())
}
