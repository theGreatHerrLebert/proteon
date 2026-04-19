//! Arrow export/import for Python via IPC format.
//!
//! Serializes RecordBatches to Arrow IPC bytes, which Python reads
//! via `pyarrow.ipc.read_record_batch()`. No version coupling between
//! Rust arrow crate and Python pyarrow.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use proteon_arrow::convert::{atom_batch_to_pdbs, pdb_to_atom_batch, pdb_to_structure_batch};

use crate::py_pdb::PyPDB;

/// Serialize an Arrow RecordBatch to IPC bytes.
fn batch_to_ipc_bytes(batch: &arrow::array::RecordBatch) -> PyResult<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        writer
            .write(batch)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        writer
            .finish()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(buf)
}

/// Deserialize Arrow IPC bytes to a RecordBatch.
fn ipc_bytes_to_batch(data: &[u8]) -> PyResult<arrow::array::RecordBatch> {
    let cursor = std::io::Cursor::new(data);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let batches: Result<Vec<_>, _> = reader.collect();
    let batches = batches.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    if batches.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No record batches in IPC data",
        ));
    }

    // Concatenate if multiple batches
    if batches.len() == 1 {
        Ok(batches.into_iter().next().unwrap())
    } else {
        arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Convert a loaded structure to Arrow IPC bytes (per-atom schema).
///
/// Returns bytes that can be read in Python with:
///   `pyarrow.ipc.open_file(bytes).read_all()`
///
/// Or with `polars.read_ipc(bytes)`.
#[pyfunction]
#[pyo3(signature = (pdb, structure_id=None))]
fn to_arrow_ipc<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    structure_id: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let sid = structure_id.unwrap_or("unknown");
    let batch = py
        .allow_threads(|| pdb_to_atom_batch(&pdb.inner, sid))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let ipc_bytes = batch_to_ipc_bytes(&batch)?;
    Ok(PyBytes::new(py, &ipc_bytes))
}

/// Convert a loaded structure to Arrow IPC bytes (per-structure summary).
#[pyfunction]
#[pyo3(signature = (pdb, structure_id=None))]
fn to_structure_arrow_ipc<'py>(
    py: Python<'py>,
    pdb: &PyPDB,
    structure_id: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let sid = structure_id.unwrap_or("unknown");
    let batch = py
        .allow_threads(|| pdb_to_structure_batch(&pdb.inner, sid))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let ipc_bytes = batch_to_ipc_bytes(&batch)?;
    Ok(PyBytes::new(py, &ipc_bytes))
}

/// Convert Arrow IPC bytes back into PDB structures.
///
/// Returns a list of (structure_id, PyPDB) tuples.
#[pyfunction]
fn from_arrow_ipc(py: Python<'_>, data: &[u8]) -> PyResult<Vec<(String, PyPDB)>> {
    let batch = ipc_bytes_to_batch(data)?;
    let pdbs = py
        .allow_threads(|| atom_batch_to_pdbs(&batch))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(pdbs
        .into_iter()
        .map(|(sid, pdb)| (sid, PyPDB::from_inner(pdb)))
        .collect())
}

/// Write a structure directly to a Parquet file.
#[pyfunction]
#[pyo3(signature = (pdb, path, structure_id=None))]
fn to_parquet(py: Python<'_>, pdb: &PyPDB, path: &str, structure_id: Option<&str>) -> PyResult<()> {
    let sid = structure_id.unwrap_or("unknown");
    let batch = py
        .allow_threads(|| pdb_to_atom_batch(&pdb.inner, sid))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let schema = proteon_arrow::atom::atom_schema();
    proteon_arrow::writer::write_parquet(std::path::Path::new(path), &schema, &[batch])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Read structures from a Parquet file (per-atom schema).
///
/// Returns a list of (structure_id, PyPDB) tuples.
#[pyfunction]
fn from_parquet(py: Python<'_>, path: &str) -> PyResult<Vec<(String, PyPDB)>> {
    let batch = py
        .allow_threads(|| proteon_arrow::reader::read_parquet(std::path::Path::new(path)))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pdbs = py
        .allow_threads(|| atom_batch_to_pdbs(&batch))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(pdbs
        .into_iter()
        .map(|(sid, pdb)| (sid, PyPDB::from_inner(pdb)))
        .collect())
}

#[pymodule]
pub(crate) fn py_arrow(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_arrow_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(to_structure_arrow_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(from_arrow_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(from_parquet, m)?)?;
    Ok(())
}
