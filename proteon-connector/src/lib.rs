use pyo3::prelude::*;
use pyo3::wrap_pymodule;

// Pure-Rust compute now lives in proteon-core. Re-export the modules the
// `py_*` shims reach via `crate::<module>` so those shims need no edits, and
// so the connector still presents the same Rust surface.
pub use proteon_core::{
    add_hydrogens, altloc, bond_order, dssp, forcefield, fragment_templates, hbond, prepare,
    reconstruct, sasa,
};

mod batch;
mod parallel;
mod py_add_hydrogens;
mod py_align;
mod py_align_funcs;
mod py_analysis;
mod py_arrow;
mod py_dssp;
mod py_electrostatics;
mod py_forcefield;
mod py_geometry;
mod py_hbond;
mod py_io;
mod py_msa;
mod py_pdb;
mod py_sasa;
mod py_search;
mod py_structure;
mod py_supervision;
mod py_surface;
mod py_transform;

/// proteon_connector — PyO3 bindings for the proteon structural bioinformatics toolkit.
#[pymodule]
fn proteon_connector(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(py_add_hydrogens::py_add_hydrogens))?;
    m.add_wrapped(wrap_pymodule!(py_align::py_align))?;
    m.add_wrapped(wrap_pymodule!(py_align_funcs::py_align_funcs))?;
    m.add_wrapped(wrap_pymodule!(py_arrow::py_arrow))?;
    m.add_wrapped(wrap_pymodule!(py_analysis::py_analysis))?;
    m.add_wrapped(wrap_pymodule!(py_dssp::py_dssp))?;
    m.add_wrapped(wrap_pymodule!(py_electrostatics::py_electrostatics))?;
    m.add_wrapped(wrap_pymodule!(py_forcefield::py_forcefield))?;
    m.add_wrapped(wrap_pymodule!(py_geometry::py_geometry))?;
    m.add_wrapped(wrap_pymodule!(py_hbond::py_hbond))?;
    m.add_wrapped(wrap_pymodule!(py_io::py_io))?;
    m.add_wrapped(wrap_pymodule!(py_msa::py_msa))?;
    m.add_wrapped(wrap_pymodule!(py_pdb::py_pdb))?;
    m.add_wrapped(wrap_pymodule!(py_sasa::py_sasa))?;
    m.add_wrapped(wrap_pymodule!(py_search::py_search))?;
    m.add_wrapped(wrap_pymodule!(py_supervision::py_supervision))?;
    m.add_wrapped(wrap_pymodule!(py_structure::py_structure))?;
    m.add_wrapped(wrap_pymodule!(py_surface::py_surface))?;
    m.add_wrapped(wrap_pymodule!(py_transform::py_transform))?;
    Ok(())
}
