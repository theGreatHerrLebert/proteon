//! PyO3 bindings for peptide hydrogen placement.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::add_hydrogens;
use crate::py_pdb::PyPDB;

fn resolve_threads(n: Option<i32>) -> usize {
    match n {
        None | Some(-1) => 0,
        Some(n) => n.max(1) as usize,
    }
}

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Place peptide backbone hydrogen atoms on a protein structure.
///
/// Adds amide H atoms to backbone nitrogen of non-N-terminal, non-proline
/// amino acid residues. Uses the DSSP bisector method at 1.02 Å from N.
///
/// Modifies the structure in place and returns (n_added, n_skipped).
///
/// Args:
///     pdb: Structure to modify (modified in place).
///
/// Returns:
///     Tuple of (atoms_added, atoms_skipped).
#[pyfunction]
pub fn place_peptide_hydrogens(py: Python<'_>, pdb: &mut PyPDB) -> (usize, usize) {
    let result = py.allow_threads(|| add_hydrogens::place_peptide_hydrogens(&mut pdb.inner));
    (result.added, result.skipped)
}

/// Place peptide hydrogens and return their positions as Nx3 array.
///
/// Same as place_peptide_hydrogens but also returns the coordinates of
/// all placed H atoms for inspection.
///
/// Args:
///     pdb: Structure to modify (modified in place).
///
/// Returns:
///     Tuple of ((n_added, n_skipped), Nx3 float64 array of H positions).
#[pyfunction]
pub fn place_peptide_hydrogens_with_coords<'py>(
    py: Python<'py>,
    pdb: &mut PyPDB,
) -> ((usize, usize), Bound<'py, PyArray2<f64>>) {
    let result = py.allow_threads(|| add_hydrogens::place_peptide_hydrogens(&mut pdb.inner));

    // Collect the placed H positions by scanning the structure
    let mut h_coords: Vec<f64> = Vec::new();
    for chain in pdb.inner.chains() {
        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .map_or(false, |c| c.is_amino_acid());
            if !is_aa {
                continue;
            }
            for atom in residue.atoms() {
                if atom.name().trim() == "H" {
                    let (x, y, z) = atom.pos();
                    h_coords.extend_from_slice(&[x, y, z]);
                }
            }
        }
    }

    let n = h_coords.len() / 3;
    let arr = numpy::PyArray1::from_vec(py, h_coords)
        .reshape([n, 3])
        .expect("reshape to Nx3");

    ((result.added, result.skipped), arr)
}

/// Batch place peptide hydrogens on multiple structures in parallel.
///
/// Returns list of (n_added, n_skipped) tuples.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub fn batch_place_peptide_hydrogens(
    py: Python<'_>,
    structures: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, usize)>> {
    let mut all_pdbs: Vec<pdbtbx::PDB> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(pdb.inner.clone())
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results: Vec<(usize, usize)> = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_pdbs
                .par_iter_mut()
                .map(|pdb| {
                    let r = add_hydrogens::place_peptide_hydrogens(pdb);
                    (r.added, r.skipped)
                })
                .collect()
        })
    });

    // Write back modified structures
    for (item, modified) in structures.iter().zip(all_pdbs) {
        let mut pdb = item.extract::<PyRefMut<'_, PyPDB>>()?;
        pdb.inner = modified;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub fn py_add_hydrogens(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(place_peptide_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(place_peptide_hydrogens_with_coords, m)?)?;
    m.add_function(wrap_pyfunction!(batch_place_peptide_hydrogens, m)?)?;
    Ok(())
}
