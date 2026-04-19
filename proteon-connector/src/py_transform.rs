//! PyO3 bindings for transformation (rotation + translation).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use proteon_align::core::types::Transform;

/// Python-visible rotation/translation transform.
#[pyclass]
#[derive(Clone)]
pub struct PyTransform {
    pub inner: Transform,
}

#[pymethods]
impl PyTransform {
    /// 3x3 rotation matrix as numpy array.
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self
            .inner
            .u
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        PyArray1::from_vec(py, flat).reshape([3, 3]).unwrap()
    }

    /// Translation vector as numpy array of shape (3,).
    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.t.to_vec().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Transform(t=[{:.3}, {:.3}, {:.3}])",
            self.inner.t[0], self.inner.t[1], self.inner.t[2],
        )
    }
}

impl PyTransform {
    pub fn from_inner(t: Transform) -> Self {
        PyTransform { inner: t }
    }
}

#[pymodule]
pub(crate) fn py_transform(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransform>()?;
    Ok(())
}
