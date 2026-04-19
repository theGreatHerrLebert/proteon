//! PyO3 wrappers for pdbtbx's PDB hierarchy.
//!
//! Wraps PDB → Model → Chain → Residue → Atom with Pythonic access,
//! numpy array extraction, and hierarchy navigation.
//!
//! Altloc handling: every atom-level iteration in this module goes
//! through [`crate::altloc`] primary-conformer helpers. See the
//! module-level doc in `altloc.rs` for the full story — in short,
//! pdbtbx's `Residue::atoms()` / `PDB::atoms()` return duplicated
//! backbone entries for altloc residues, which we cannot expose to
//! Python consumers or the force-field topology builder.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;

use crate::altloc::{
    chain_atom_count_primary, model_atom_count_primary, pdb_atom_count_primary, pdb_atoms_primary,
    pdb_total_atom_count_primary, residue_atom_count_primary, residue_atoms_primary,
};

// ---------------------------------------------------------------------------
// PyAtom
// ---------------------------------------------------------------------------

/// A single atom in a structure.
#[pyclass]
#[derive(Clone)]
pub struct PyAtom {
    pub inner: pdbtbx::Atom,
    /// Residue name from the conformer this atom belongs to.
    pub res_name: String,
    /// Chain ID this atom belongs to.
    pub chain_id: String,
    /// Residue serial number.
    pub res_serial: isize,
    /// Residue insertion code.
    pub res_ins_code: Option<String>,
}

#[pymethods]
impl PyAtom {
    /// Atom name (e.g. "CA", "N", "CB").
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Atom serial number.
    #[getter]
    fn serial_number(&self) -> usize {
        self.inner.serial_number()
    }

    /// X coordinate in Angstroms.
    #[getter]
    fn x(&self) -> f64 {
        self.inner.x()
    }

    /// Y coordinate in Angstroms.
    #[getter]
    fn y(&self) -> f64 {
        self.inner.y()
    }

    /// Z coordinate in Angstroms.
    #[getter]
    fn z(&self) -> f64 {
        self.inner.z()
    }

    /// Position as (x, y, z) tuple.
    #[getter]
    fn pos(&self) -> (f64, f64, f64) {
        self.inner.pos()
    }

    /// Element symbol (e.g. "C", "N", "O") or None.
    #[getter]
    fn element(&self) -> Option<&str> {
        self.inner.element().map(|e| e.symbol())
    }

    /// B-factor (temperature factor).
    #[getter]
    fn b_factor(&self) -> f64 {
        self.inner.b_factor()
    }

    /// Occupancy.
    #[getter]
    fn occupancy(&self) -> f64 {
        self.inner.occupancy()
    }

    /// Formal charge.
    #[getter]
    fn charge(&self) -> isize {
        self.inner.charge()
    }

    /// Whether this is a HETATM record.
    #[getter]
    fn hetero(&self) -> bool {
        self.inner.hetero()
    }

    /// Whether this is a backbone atom (N, CA, C, O).
    #[getter]
    fn is_backbone(&self) -> bool {
        self.inner.is_backbone()
    }

    /// Residue name this atom belongs to (e.g. "ALA", "GLY").
    #[getter]
    fn residue_name(&self) -> &str {
        &self.res_name
    }

    /// Chain ID this atom belongs to.
    #[getter]
    fn chain_id(&self) -> &str {
        &self.chain_id
    }

    /// Residue serial number this atom belongs to.
    #[getter]
    fn residue_serial_number(&self) -> isize {
        self.res_serial
    }

    fn __repr__(&self) -> String {
        format!(
            "Atom(name='{}', serial={}, element={}, pos=({:.3}, {:.3}, {:.3}))",
            self.inner.name(),
            self.inner.serial_number(),
            self.inner.element().map_or("?", |e| e.symbol()),
            self.inner.x(),
            self.inner.y(),
            self.inner.z(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyResidue
// ---------------------------------------------------------------------------

/// A residue (amino acid, nucleotide, or ligand).
///
/// Flattens pdbtbx's Residue+Conformer: exposes atoms from the primary
/// conformer by default, with access to alternates via `conformer_names`.
#[pyclass]
#[derive(Clone)]
pub struct PyResidue {
    pub inner: pdbtbx::Residue,
    pub chain_id: String,
}

#[pymethods]
impl PyResidue {
    /// Residue name (e.g. "ALA", "GLY", "HOH").
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    /// Residue serial number.
    #[getter]
    fn serial_number(&self) -> isize {
        self.inner.serial_number()
    }

    /// Insertion code, if any.
    #[getter]
    fn insertion_code(&self) -> Option<&str> {
        self.inner.insertion_code()
    }

    /// Chain ID this residue belongs to.
    #[getter]
    fn chain_id(&self) -> &str {
        &self.chain_id
    }

    /// Whether this is a standard amino acid.
    #[getter]
    fn is_amino_acid(&self) -> bool {
        self.inner
            .conformers()
            .next()
            .is_some_and(|c| c.is_amino_acid())
    }

    /// Number of atoms in the primary conformer.
    fn __len__(&self) -> usize {
        residue_atom_count_primary(&self.inner)
    }

    /// Names of alternate conformers present.
    #[getter]
    fn conformer_names(&self) -> Vec<String> {
        self.inner
            .conformers()
            .map(|c| {
                let name = c.name().to_string();
                match c.alternative_location() {
                    Some(alt) => format!("{name}:{alt}"),
                    None => name,
                }
            })
            .collect()
    }

    /// Atoms in this residue's primary conformer.
    ///
    /// See the module-level doc on altloc handling: pdbtbx's
    /// `Residue::atoms()` iterates every conformer and would return
    /// duplicated backbone atoms for altloc residues. We select one
    /// primary conformer per residue so the Python view is consistent.
    #[getter]
    fn atoms(&self) -> Vec<PyAtom> {
        let res_name = self.inner.name().unwrap_or("").to_string();
        let serial = self.inner.serial_number();
        let ins = self.inner.insertion_code().map(|s| s.to_string());

        residue_atoms_primary(&self.inner)
            .map(|a| PyAtom {
                inner: a.clone(),
                res_name: res_name.clone(),
                chain_id: self.chain_id.clone(),
                res_serial: serial,
                res_ins_code: ins.clone(),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Residue(name='{}', serial={}, atoms={})",
            self.inner.name().unwrap_or("?"),
            self.inner.serial_number(),
            residue_atom_count_primary(&self.inner),
        )
    }
}

// ---------------------------------------------------------------------------
// PyChain
// ---------------------------------------------------------------------------

/// A chain in a structural model.
#[pyclass]
#[derive(Clone)]
pub struct PyChain {
    pub inner: pdbtbx::Chain,
}

#[pymethods]
impl PyChain {
    /// Chain identifier (e.g. "A", "B").
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    /// Number of residues.
    #[getter]
    fn residue_count(&self) -> usize {
        self.inner.residue_count()
    }

    /// Number of atoms in this chain (primary conformer per residue).
    #[getter]
    fn atom_count(&self) -> usize {
        chain_atom_count_primary(&self.inner)
    }

    /// All residues in this chain.
    #[getter]
    fn residues(&self) -> Vec<PyResidue> {
        let chain_id = self.inner.id().to_string();
        self.inner
            .residues()
            .map(|r| PyResidue {
                inner: r.clone(),
                chain_id: chain_id.clone(),
            })
            .collect()
    }

    /// All atoms in this chain (primary conformer per residue, flattened).
    #[getter]
    fn atoms(&self) -> Vec<PyAtom> {
        let chain_id = self.inner.id().to_string();
        self.inner
            .residues()
            .flat_map(|r| {
                let res_name = r.name().unwrap_or("").to_string();
                let serial = r.serial_number();
                let ins = r.insertion_code().map(|s| s.to_string());
                let cid = chain_id.clone();
                residue_atoms_primary(r).map(move |a| PyAtom {
                    inner: a.clone(),
                    res_name: res_name.clone(),
                    chain_id: cid.clone(),
                    res_serial: serial,
                    res_ins_code: ins.clone(),
                })
            })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.residue_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Chain(id='{}', residues={}, atoms={})",
            self.inner.id(),
            self.inner.residue_count(),
            chain_atom_count_primary(&self.inner),
        )
    }
}

// ---------------------------------------------------------------------------
// PyModel
// ---------------------------------------------------------------------------

/// A structural model (e.g. NMR ensemble member).
#[pyclass]
#[derive(Clone)]
pub struct PyModel {
    pub inner: pdbtbx::Model,
}

#[pymethods]
impl PyModel {
    /// Model serial number (1-based).
    #[getter]
    fn serial_number(&self) -> usize {
        self.inner.serial_number()
    }

    /// Number of chains.
    #[getter]
    fn chain_count(&self) -> usize {
        self.inner.chain_count()
    }

    /// Number of residues.
    #[getter]
    fn residue_count(&self) -> usize {
        self.inner.residue_count()
    }

    /// Number of atoms in this model (primary conformer per residue).
    #[getter]
    fn atom_count(&self) -> usize {
        model_atom_count_primary(&self.inner)
    }

    /// All chains in this model.
    #[getter]
    fn chains(&self) -> Vec<PyChain> {
        self.inner
            .chains()
            .map(|c| PyChain { inner: c.clone() })
            .collect()
    }

    /// All residues (flattened across chains).
    #[getter]
    fn residues(&self) -> Vec<PyResidue> {
        self.inner
            .chains()
            .flat_map(|c| {
                let chain_id = c.id().to_string();
                c.residues().map(move |r| PyResidue {
                    inner: r.clone(),
                    chain_id: chain_id.clone(),
                })
            })
            .collect()
    }

    /// All atoms in this model (primary conformer per residue, flattened
    /// across chains and residues).
    #[getter]
    fn atoms(&self) -> Vec<PyAtom> {
        atoms_from_model(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(serial={}, chains={}, atoms={})",
            self.inner.serial_number(),
            self.inner.chain_count(),
            model_atom_count_primary(&self.inner),
        )
    }
}

// ---------------------------------------------------------------------------
// PyPDB — top-level structure
// ---------------------------------------------------------------------------

/// A parsed PDB or mmCIF structure.
///
/// Provides hierarchy navigation (models/chains/residues/atoms) and
/// bulk numpy array extraction for coordinates, B-factors, etc.
#[pyclass]
pub struct PyPDB {
    pub inner: pdbtbx::PDB,
}

#[pymethods]
impl PyPDB {
    /// PDB identifier (e.g. "1CRN"), if present.
    #[getter]
    fn identifier(&self) -> Option<&str> {
        self.inner.identifier.as_deref()
    }

    // -- counts (first model) -----------------------------------------------

    /// Number of models.
    #[getter]
    fn model_count(&self) -> usize {
        self.inner.model_count()
    }

    /// Number of chains (first model).
    #[getter]
    fn chain_count(&self) -> usize {
        self.inner.chain_count()
    }

    /// Number of residues (first model).
    #[getter]
    fn residue_count(&self) -> usize {
        self.inner.residue_count()
    }

    /// Number of atoms (first model, primary conformer per residue).
    #[getter]
    fn atom_count(&self) -> usize {
        pdb_atom_count_primary(&self.inner)
    }

    /// Total number of atoms across all models (primary conformer per
    /// residue).
    #[getter]
    fn total_atom_count(&self) -> usize {
        pdb_total_atom_count_primary(&self.inner)
    }

    // -- hierarchy navigation -----------------------------------------------

    /// All models in the structure.
    #[getter]
    fn models(&self) -> Vec<PyModel> {
        self.inner
            .models()
            .map(|m| PyModel { inner: m.clone() })
            .collect()
    }

    /// All chains (first model).
    #[getter]
    fn chains(&self) -> Vec<PyChain> {
        self.inner
            .chains()
            .map(|c| PyChain { inner: c.clone() })
            .collect()
    }

    /// All residues (first model, flattened).
    #[getter]
    fn residues(&self) -> Vec<PyResidue> {
        self.inner
            .chains()
            .flat_map(|c| {
                let chain_id = c.id().to_string();
                c.residues().map(move |r| PyResidue {
                    inner: r.clone(),
                    chain_id: chain_id.clone(),
                })
            })
            .collect()
    }

    /// All atoms (first model, flattened).
    #[getter]
    fn atoms(&self) -> Vec<PyAtom> {
        match self.inner.models().next() {
            Some(model) => atoms_from_model(model),
            None => Vec::new(),
        }
    }

    // -- bulk numpy arrays (first model, primary conformer) ----------------

    /// All atom coordinates as Nx3 numpy array (float64).
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        let flat: Vec<f64> = pdb_atoms_primary(&self.inner)
            .flat_map(|a| {
                let (x, y, z) = a.pos();
                [x, y, z]
            })
            .collect();
        let n = flat.len() / 3;
        numpy::PyArray1::from_vec(py, flat)
            .reshape([n, 3])
            .expect("reshape to Nx3")
    }

    /// All B-factors as numpy array (float64).
    #[getter]
    fn b_factors<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let vals: Vec<f64> = pdb_atoms_primary(&self.inner)
            .map(|a| a.b_factor())
            .collect();
        vals.into_pyarray(py)
    }

    /// All occupancies as numpy array (float64).
    #[getter]
    fn occupancies<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let vals: Vec<f64> = pdb_atoms_primary(&self.inner)
            .map(|a| a.occupancy())
            .collect();
        vals.into_pyarray(py)
    }

    // -- bulk metadata lists (first model, primary conformer) --------------

    /// All atom names as a list of strings.
    #[getter]
    fn atom_names(&self) -> Vec<String> {
        pdb_atoms_primary(&self.inner)
            .map(|a| a.name().to_string())
            .collect()
    }

    /// All element symbols as a list of strings.
    #[getter]
    fn elements(&self) -> Vec<String> {
        pdb_atoms_primary(&self.inner)
            .map(|a| {
                a.element()
                    .map_or_else(|| "?".to_string(), |e| e.symbol().to_string())
            })
            .collect()
    }

    /// Residue name for each atom (e.g. ["ALA", "ALA", "ALA", "GLY", ...]).
    #[getter]
    fn residue_names(&self) -> Vec<String> {
        self.inner
            .chains()
            .flat_map(|c| {
                c.residues().flat_map(|r| {
                    let name = r.name().unwrap_or("?").to_string();
                    let count = residue_atom_count_primary(r);
                    std::iter::repeat(name).take(count)
                })
            })
            .collect()
    }

    /// Chain ID for each atom (e.g. ["A", "A", ..., "B", "B", ...]).
    #[getter]
    fn chain_ids(&self) -> Vec<String> {
        self.inner
            .chains()
            .flat_map(|c| {
                let id = c.id().to_string();
                let count = chain_atom_count_primary(c);
                std::iter::repeat(id).take(count)
            })
            .collect()
    }

    /// Residue serial number for each atom.
    #[getter]
    fn residue_serial_numbers<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        let vals: Vec<i64> = self
            .inner
            .chains()
            .flat_map(|c| {
                c.residues().flat_map(|r| {
                    let serial = r.serial_number() as i64;
                    let count = residue_atom_count_primary(r);
                    std::iter::repeat(serial).take(count)
                })
            })
            .collect();
        vals.into_pyarray(py)
    }

    // -- dunder methods -----------------------------------------------------

    fn __len__(&self) -> usize {
        pdb_atom_count_primary(&self.inner)
    }

    fn __repr__(&self) -> String {
        let id = self.inner.identifier.as_deref().unwrap_or("unknown");
        format!(
            "PDB(id='{}', models={}, chains={}, residues={}, atoms={})",
            id,
            self.inner.model_count(),
            self.inner.chain_count(),
            self.inner.residue_count(),
            pdb_atom_count_primary(&self.inner),
        )
    }
}

impl PyPDB {
    pub fn from_inner(pdb: pdbtbx::PDB) -> Self {
        PyPDB { inner: pdb }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract all atoms from a model with full hierarchy context
/// (primary conformer per residue — see module-level doc on altloc).
fn atoms_from_model(model: &pdbtbx::Model) -> Vec<PyAtom> {
    model
        .chains()
        .flat_map(|c| {
            let chain_id = c.id().to_string();
            c.residues().flat_map(move |r| {
                let res_name = r.name().unwrap_or("").to_string();
                let serial = r.serial_number();
                let ins = r.insertion_code().map(|s| s.to_string());
                let chain_id = chain_id.clone();
                residue_atoms_primary(r).map(move |a| PyAtom {
                    inner: a.clone(),
                    res_name: res_name.clone(),
                    chain_id: chain_id.clone(),
                    res_serial: serial,
                    res_ins_code: ins.clone(),
                })
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_pdb(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPDB>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyChain>()?;
    m.add_class::<PyResidue>()?;
    m.add_class::<PyAtom>()?;
    Ok(())
}
