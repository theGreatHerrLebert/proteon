//! PyO3 bindings for peptide hydrogen placement.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::add_hydrogens;
use crate::py_pdb::PyPDB;

/// Apply minimized coordinates back to a PDB structure.
///
/// Iterates chains → residues → atoms in the same order as topology building,
/// and sets each atom's position from the flat coordinate array.
/// Panics if the coordinate array doesn't match the atom count.
fn apply_coords_to_pdb(pdb: &mut pdbtbx::PDB, coords: &[[f64; 3]]) {
    let mut idx = 0;
    // Use first model only (consistent with build_topology, etc.)
    let first_model = match pdb.models_mut().next() {
        Some(m) => m,
        None => return,
    };
    for chain in first_model.chains_mut() {
        for residue in chain.residues_mut() {
            for atom in residue.atoms_mut() {
                assert!(
                    idx < coords.len(),
                    "apply_coords_to_pdb: coord array too short ({} coords, atom index {})",
                    coords.len(), idx,
                );
                atom.set_pos((coords[idx][0], coords[idx][1], coords[idx][2]))
                    .expect("apply_coords_to_pdb: invalid coordinates (NaN/Inf)");
                idx += 1;
            }
        }
    }
    assert_eq!(
        idx, coords.len(),
        "apply_coords_to_pdb: coord array length ({}) != atom count ({})",
        coords.len(), idx,
    );
}

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

/// Place sidechain hydrogen atoms on all standard amino acid residues.
///
/// Template-based placement for the 20 standard amino acids.
/// Modifies the structure in place and returns (n_added, n_skipped).
#[pyfunction]
pub fn place_sidechain_hydrogens(py: Python<'_>, pdb: &mut PyPDB) -> (usize, usize) {
    let result = py.allow_threads(|| add_hydrogens::place_sidechain_hydrogens(&mut pdb.inner));
    (result.added, result.skipped)
}

/// Place all hydrogens: backbone amide H + sidechain H.
///
/// Equivalent to calling place_peptide_hydrogens then place_sidechain_hydrogens.
/// Returns (n_added, n_skipped).
#[pyfunction]
pub fn place_all_hydrogens(py: Python<'_>, pdb: &mut PyPDB) -> (usize, usize) {
    let result = py.allow_threads(|| add_hydrogens::place_all_hydrogens(&mut pdb.inner));
    (result.added, result.skipped)
}

/// Place hydrogens on all atoms including non-standard residues and ligands.
///
/// Runs Phase 1 (backbone) + Phase 2 (sidechain templates) + Phase 3
/// (general BALL algorithm for ligands/non-standard residues).
///
/// Args:
///     pdb: Structure to modify.
///     include_water: If True, also place 2 H on each water molecule (default False).
///
/// Returns (n_added, n_skipped).
#[pyfunction]
#[pyo3(signature = (pdb, include_water=false))]
pub fn place_general_hydrogens(py: Python<'_>, pdb: &mut PyPDB, include_water: bool) -> (usize, usize) {
    let result = py.allow_threads(|| add_hydrogens::place_general_hydrogens(&mut pdb.inner, include_water));
    (result.added, result.skipped)
}

/// Reconstruct missing atoms from fragment templates.
///
/// Adds missing heavy atoms and hydrogens to standard amino acid residues
/// by comparing against template structures from the BALL fragment database.
/// Returns the number of atoms added.
#[pyfunction]
pub fn reconstruct_fragments(py: Python<'_>, pdb: &mut PyPDB) -> usize {
    let result = py.allow_threads(|| crate::reconstruct::reconstruct_fragments(&mut pdb.inner));
    result.added
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
    let n = resolve_threads(n_threads);
    let total = structures.len();
    let chunk_size = 500;
    let mut all_results = Vec::with_capacity(total);

    // Process in chunks to avoid cloning all structures at once
    for start in (0..total).step_by(chunk_size) {
        let end = (start + chunk_size).min(total);

        let mut chunk_pdbs: Vec<pdbtbx::PDB> = (start..end)
            .map(|i| {
                let item = structures.get_item(i)?;
                let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
                Ok(pdb.inner.clone())
            })
            .collect::<PyResult<_>>()?;

        let results: Vec<(usize, usize)> = py.allow_threads(|| {
            let pool = build_pool(n);
            pool.install(|| {
                chunk_pdbs
                    .par_iter_mut()
                    .map(|pdb| {
                        let r = add_hydrogens::place_peptide_hydrogens(pdb);
                        (r.added, r.skipped)
                    })
                    .collect()
            })
        });

        // Write back modified structures for this chunk
        for (i, modified) in (start..end).zip(chunk_pdbs) {
            let item = structures.get_item(i)?;
            let mut pdb = item.extract::<PyRefMut<'_, PyPDB>>()?;
            pdb.inner = modified;
        }

        all_results.extend(results);
    }

    Ok(all_results)
}

/// Batch prepare structures in parallel (reconstruct + place H + minimize H).
///
/// Runs the full preparation pipeline on each structure using rayon parallelism.
/// Returns list of dicts with preparation statistics.
#[pyfunction]
#[pyo3(signature = (structures, reconstruct=true, hydrogens="all", include_water=false, minimize=true, minimize_method="lbfgs", minimize_steps=500, gradient_tolerance=0.1, n_threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn batch_prepare(
    py: Python<'_>,
    structures: &Bound<'_, PyList>,
    reconstruct: bool,
    hydrogens: &str,
    include_water: bool,
    minimize: bool,
    minimize_method: &str,
    minimize_steps: usize,
    gradient_tolerance: f64,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyObject>> {
    let n = resolve_threads(n_threads);
    let h_mode = hydrogens.to_string();
    let method = minimize_method.to_string();
    let amber = crate::forcefield::params::amber96();
    let total = structures.len();
    let chunk_size = 200; // prepare is heavier per-structure, smaller chunks
    let mut all_results = Vec::with_capacity(total);

    // Process in chunks to avoid cloning all structures at once
    for start in (0..total).step_by(chunk_size) {
        let end = (start + chunk_size).min(total);

        let mut chunk_pdbs: Vec<pdbtbx::PDB> = (start..end)
            .map(|i| {
                let item = structures.get_item(i)?;
                let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
                Ok(pdb.inner.clone())
            })
            .collect::<PyResult<_>>()?;

        let h_mode = h_mode.clone();
        let method = method.clone();
        let results: Vec<(usize, usize, usize, f64, f64, usize, bool, usize)> = py.allow_threads(|| {
            let pool = build_pool(n);
            pool.install(|| {
                chunk_pdbs
                    .par_iter_mut()
                    .map(|pdb| {
                        // Reconstruct
                        let reconstructed = if reconstruct {
                            crate::reconstruct::reconstruct_fragments(pdb).added
                        } else {
                            0
                        };

                        // Place hydrogens
                        let (h_added, h_skipped) = match h_mode.as_str() {
                            "backbone" => {
                                let r = add_hydrogens::place_peptide_hydrogens(pdb);
                                (r.added, r.skipped)
                            }
                            "general" => {
                                let r = add_hydrogens::place_general_hydrogens(pdb, include_water);
                                (r.added, r.skipped)
                            }
                            "none" => (0, 0),
                            "all" => {
                                let r = add_hydrogens::place_all_hydrogens(pdb);
                                (r.added, r.skipped)
                            }
                            _ => (0, 0),
                        };

                        // Minimize H positions and apply coords back to PDB
                        let has_any_h = pdb.atoms().any(|a| {
                            a.element().map_or(false, |e| e.symbol() == "H" || e.symbol() == "D")
                        });
                        let (init_e, final_e, steps, converged) = if minimize && (h_added > 0 || has_any_h) {
                            let topo = crate::forcefield::topology::build_topology(pdb, &amber);
                            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                            let constrained: Vec<bool> = topo.atoms.iter().map(|a| !a.is_hydrogen).collect();
                            let result = match method.as_str() {
                                "cg" => crate::forcefield::minimize::conjugate_gradient(
                                    &coords, &topo, &amber, minimize_steps, gradient_tolerance, &constrained),
                                "lbfgs" => crate::forcefield::minimize::lbfgs(
                                    &coords, &topo, &amber, minimize_steps, gradient_tolerance, &constrained),
                                _ => crate::forcefield::minimize::steepest_descent(
                                    &coords, &topo, &amber, minimize_steps, gradient_tolerance, &constrained),
                            };
                            apply_coords_to_pdb(pdb, &result.coords);
                            (result.initial_energy, result.energy.total, result.steps, result.converged)
                        } else {
                            (0.0, 0.0, 0, false)
                        };

                        let topo = crate::forcefield::topology::build_topology(pdb, &amber);
                        let n_unassigned = topo.unassigned_atoms.len();

                        (reconstructed, h_added, h_skipped, init_e, final_e, steps, converged, n_unassigned)
                    })
                    .collect()
            })
        });

        // Write back modified structures for this chunk
        for (i, modified) in (start..end).zip(chunk_pdbs) {
            let item = structures.get_item(i)?;
            let mut pdb = item.extract::<PyRefMut<'_, PyPDB>>()?;
            pdb.inner = modified;
        }

        all_results.extend(results);
    }

    // Convert to Python dicts
    Ok(all_results
        .into_iter()
        .map(|(recon, h_add, h_skip, init_e, final_e, steps, conv, unassigned)| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("atoms_reconstructed", recon).unwrap();
            dict.set_item("hydrogens_added", h_add).unwrap();
            dict.set_item("hydrogens_skipped", h_skip).unwrap();
            dict.set_item("initial_energy", init_e).unwrap();
            dict.set_item("final_energy", final_e).unwrap();
            dict.set_item("minimizer_steps", steps).unwrap();
            dict.set_item("converged", conv).unwrap();
            dict.set_item("n_unassigned_atoms", unassigned).unwrap();
            dict.into_any().unbind()
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub fn py_add_hydrogens(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(place_peptide_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(place_peptide_hydrogens_with_coords, m)?)?;
    m.add_function(wrap_pyfunction!(place_sidechain_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(place_all_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(place_general_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_fragments, m)?)?;
    m.add_function(wrap_pyfunction!(batch_place_peptide_hydrogens, m)?)?;
    m.add_function(wrap_pyfunction!(batch_prepare, m)?)?;
    Ok(())
}
