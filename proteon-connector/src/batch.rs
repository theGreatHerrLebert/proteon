//! Uniform batch-result contract for connector batch APIs.
//!
//! A batch call processes N inputs and returns a [`PyBatchResult`] holding
//! exactly N [`PyBatchItem`]s in input order. Each item is either a success
//! (carrying the per-input value) or a failure (carrying the error message
//! and the input index).
//!
//! This replaces the two older, inconsistent batch styles:
//!
//! * **fail-fast** — one bad input aborted the whole batch with an exception,
//!   discarding every result that did succeed;
//! * **silent-skip** — failed inputs were dropped from the output with no
//!   diagnostic, so a short result list was the only (ambiguous) signal that
//!   anything went wrong.
//!
//! Batch functions are tolerant by default. Pass `strict=True` to raise on
//! the first failure instead — use that when the caller needs output
//! cardinality to match input cardinality or not run at all.

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;

/// Outcome of one input in a batch: a Python value, or an error message.
pub(crate) type BatchOutcome = Result<Py<PyAny>, String>;

/// The outcome of processing a single input in a batch call.
#[pyclass(name = "BatchItem", module = "proteon_connector", frozen)]
pub struct PyBatchItem {
    /// Position of this input in the batch (0-based, input order).
    #[pyo3(get)]
    pub index: usize,
    /// True if the input was processed successfully.
    #[pyo3(get)]
    pub ok: bool,
    value: Option<Py<PyAny>>,
    /// Error message if `ok` is false, else None.
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyBatchItem {
    /// The per-input result if `ok`, else None.
    #[getter]
    fn value(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.value.as_ref().map(|v| v.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        if self.ok {
            format!("BatchItem(index={}, ok=True)", self.index)
        } else {
            format!(
                "BatchItem(index={}, ok=False, error={:?})",
                self.index,
                self.error.as_deref().unwrap_or(""),
            )
        }
    }
}

/// Result of a batch call: exactly one [`PyBatchItem`] per input, input order.
#[pyclass(name = "BatchResult", module = "proteon_connector", frozen)]
pub struct PyBatchResult {
    items: Vec<Py<PyBatchItem>>,
}

#[pymethods]
impl PyBatchResult {
    /// Number of inputs processed (equals the length of the batch input).
    #[getter]
    fn n_attempted(&self) -> usize {
        self.items.len()
    }

    /// Number of inputs that succeeded.
    #[getter]
    fn n_ok(&self) -> usize {
        self.items.iter().filter(|it| it.get().ok).count()
    }

    /// Number of inputs that failed.
    #[getter]
    fn n_failed(&self) -> usize {
        self.items.iter().filter(|it| !it.get().ok).count()
    }

    /// `(index, error)` for every failed input, in input order.
    #[getter]
    fn failures(&self) -> Vec<(usize, String)> {
        self.items
            .iter()
            .filter_map(|it| {
                let it = it.get();
                it.error.as_ref().map(|e| (it.index, e.clone()))
            })
            .collect()
    }

    /// The successful per-input values, in input order (failures omitted).
    #[getter]
    fn values(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.items
            .iter()
            .filter_map(|it| it.get().value.as_ref().map(|v| v.clone_ref(py)))
            .collect()
    }

    /// All items, in input order.
    #[getter]
    fn items(&self, py: Python<'_>) -> Vec<Py<PyBatchItem>> {
        self.items.iter().map(|it| it.clone_ref(py)).collect()
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyBatchItem>> {
        let len = self.items.len() as isize;
        let idx = if index < 0 { index + len } else { index };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("batch item index out of range"));
        }
        Ok(self.items[idx as usize].clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchResult(n_attempted={}, n_ok={}, n_failed={})",
            self.n_attempted(),
            self.n_ok(),
            self.n_failed(),
        )
    }
}

/// Assemble a [`PyBatchResult`] from per-input outcomes (input order).
///
/// With `strict`, the first failure is raised as a `RuntimeError` instead of
/// being recorded as an item.
pub(crate) fn make_batch_result(
    py: Python<'_>,
    outcomes: Vec<BatchOutcome>,
    strict: bool,
) -> PyResult<PyBatchResult> {
    if strict {
        for (i, o) in outcomes.iter().enumerate() {
            if let Err(e) = o {
                return Err(PyRuntimeError::new_err(format!(
                    "batch item {i} failed (strict mode): {e}"
                )));
            }
        }
    }
    let items = outcomes
        .into_iter()
        .enumerate()
        .map(|(index, o)| {
            let item = match o {
                Ok(v) => PyBatchItem {
                    index,
                    ok: true,
                    value: Some(v),
                    error: None,
                },
                Err(e) => PyBatchItem {
                    index,
                    ok: false,
                    value: None,
                    error: Some(e),
                },
            };
            Py::new(py, item)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyBatchResult { items })
}

/// Register the batch-contract classes on a connector submodule.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchItem>()?;
    m.add_class::<PyBatchResult>()?;
    Ok(())
}
