//! PyO3 bindings for structure data.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use proteon_align::core::types::StructureData;

/// Python-visible structure data loaded from PDB/mmCIF.
#[pyclass]
#[derive(Clone)]
pub struct PyStructureData {
    pub inner: StructureData,
}

#[pymethods]
impl PyStructureData {
    /// CA/C3' coordinates as Nx3 numpy array.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.coords.len();
        let flat: Vec<f64> = self
            .inner
            .coords
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        numpy::PyArray1::from_vec(py, flat).reshape([n, 3]).unwrap()
    }

    /// One-letter sequence string.
    #[getter]
    fn sequence(&self) -> String {
        self.inner.sequence.iter().collect()
    }

    /// Secondary structure string (H/E/T/C).
    #[getter]
    fn sec_structure(&self) -> String {
        self.inner.sec_structure.iter().collect()
    }

    /// Chain identifier.
    #[getter]
    fn chain_id(&self) -> String {
        self.inner.chain_id.clone()
    }

    /// Number of residues.
    fn __len__(&self) -> usize {
        self.inner.coords.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "StructureData(chain='{}', n_residues={}, mol_type={:?})",
            self.inner.chain_id,
            self.inner.coords.len(),
            self.inner.mol_type,
        )
    }
}

impl PyStructureData {
    pub fn from_inner(data: StructureData) -> Self {
        PyStructureData { inner: data }
    }
}

#[pymodule]
pub(crate) fn py_structure(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStructureData>()?;
    Ok(())
}
