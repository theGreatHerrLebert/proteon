use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod dssp;
mod hbond;
mod py_align;
mod py_align_funcs;
mod py_analysis;
mod py_dssp;
mod py_geometry;
mod py_hbond;
mod py_io;
mod py_pdb;
mod py_sasa;
mod py_structure;
mod py_transform;
mod sasa;

/// ferritin_connector — PyO3 bindings for the ferritin structural bioinformatics toolkit.
#[pymodule]
fn ferritin_connector(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(py_align::py_align))?;
    m.add_wrapped(wrap_pymodule!(py_align_funcs::py_align_funcs))?;
    m.add_wrapped(wrap_pymodule!(py_analysis::py_analysis))?;
    m.add_wrapped(wrap_pymodule!(py_dssp::py_dssp))?;
    m.add_wrapped(wrap_pymodule!(py_geometry::py_geometry))?;
    m.add_wrapped(wrap_pymodule!(py_hbond::py_hbond))?;
    m.add_wrapped(wrap_pymodule!(py_io::py_io))?;
    m.add_wrapped(wrap_pymodule!(py_pdb::py_pdb))?;
    m.add_wrapped(wrap_pymodule!(py_sasa::py_sasa))?;
    m.add_wrapped(wrap_pymodule!(py_structure::py_structure))?;
    m.add_wrapped(wrap_pymodule!(py_transform::py_transform))?;
    Ok(())
}
