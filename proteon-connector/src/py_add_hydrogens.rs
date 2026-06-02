//! PyO3 bindings for peptide hydrogen placement.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::add_hydrogens;
use crate::py_pdb::PyPDB;

use crate::parallel::resolve_threads;

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
pub(crate) fn place_peptide_hydrogens(py: Python<'_>, pdb: &mut PyPDB) -> (usize, usize) {
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
pub(crate) fn place_peptide_hydrogens_with_coords<'py>(
    py: Python<'py>,
    pdb: &mut PyPDB,
) -> ((usize, usize), Bound<'py, PyArray2<f64>>) {
    let result = py.allow_threads(|| add_hydrogens::place_peptide_hydrogens(&mut pdb.inner));

    // Collect the placed H positions by scanning the structure (first model,
    // primary conformer only — pdb.inner.chains() iterates ALL models).
    let mut h_coords: Vec<f64> = Vec::new();
    if let Some(first_model) = pdb.inner.models().next() {
        for chain in first_model.chains() {
            for residue in chain.residues() {
                let is_aa = residue
                    .conformers()
                    .next()
                    .is_some_and(|c| c.is_amino_acid());
                if !is_aa {
                    continue;
                }
                for atom in crate::altloc::residue_atoms_primary(residue) {
                    if atom.name().trim() == "H" {
                        let (x, y, z) = atom.pos();
                        h_coords.extend_from_slice(&[x, y, z]);
                    }
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
///
/// If `polar_only=True`, only hydrogens bonded to N/O/S are placed
/// (guanidinium, amide, hydroxyl, thiol, imidazole, indole N-H, and
/// NH3+). Non-polar C-H atoms are skipped. Use this when the downstream
/// force field is a polar-H united-atom model like CHARMM19.
#[pyfunction]
#[pyo3(signature = (pdb, polar_only=false))]
pub(crate) fn place_sidechain_hydrogens(
    py: Python<'_>,
    pdb: &mut PyPDB,
    polar_only: bool,
) -> (usize, usize) {
    let result =
        py.allow_threads(|| add_hydrogens::place_sidechain_hydrogens(&mut pdb.inner, polar_only));
    (result.added, result.skipped)
}

/// Place all hydrogens: backbone amide H + sidechain H.
///
/// Equivalent to calling place_peptide_hydrogens then place_sidechain_hydrogens.
/// Returns (n_added, n_skipped).
///
/// If `polar_only=True`, only hydrogens bonded to N/O/S are placed on
/// the sidechain (backbone amide is always placed). Use for CHARMM19.
#[pyfunction]
#[pyo3(signature = (pdb, polar_only=false))]
pub(crate) fn place_all_hydrogens(
    py: Python<'_>,
    pdb: &mut PyPDB,
    polar_only: bool,
) -> (usize, usize) {
    let result =
        py.allow_threads(|| add_hydrogens::place_all_hydrogens(&mut pdb.inner, polar_only));
    (result.added, result.skipped)
}

/// Place hydrogens on all atoms including non-standard residues and ligands.
///
/// Runs Phase 1 (backbone) + Phase 2 (sidechain templates) + Phase 3
/// (general placer for ligands/non-standard residues).
///
/// Args:
///     pdb: Structure to modify.
///     include_water: If True, also place 2 H on each water molecule (default False).
///
/// Returns (n_added, n_skipped).
#[pyfunction]
#[pyo3(signature = (pdb, include_water=false))]
pub(crate) fn place_general_hydrogens(
    py: Python<'_>,
    pdb: &mut PyPDB,
    include_water: bool,
) -> (usize, usize) {
    let result =
        py.allow_threads(|| add_hydrogens::place_general_hydrogens(&mut pdb.inner, include_water));
    (result.added, result.skipped)
}

/// Reconstruct missing atoms from fragment templates.
///
/// Adds missing heavy atoms and hydrogens to standard amino acid residues
/// by comparing against template structures from the BALL fragment database.
/// Returns the number of atoms added.
#[pyfunction]
pub(crate) fn reconstruct_fragments(py: Python<'_>, pdb: &mut PyPDB) -> usize {
    let result = py.allow_threads(|| crate::reconstruct::reconstruct_fragments(&mut pdb.inner));
    result.added
}

/// Batch place peptide hydrogens on multiple structures in parallel.
///
/// Returns list of (n_added, n_skipped) tuples.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub(crate) fn batch_place_peptide_hydrogens(
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
/// The per-structure result type and pipeline body live in [`crate::prepare`]
/// so the CLI (`proteon prepare`/`protonate`/`minimize`) drives the same code.
use crate::prepare::{self, PrepareOptions, PrepareReport};

/// Returns list of dicts with preparation statistics.
///
/// The `ff` parameter picks the force field used by the topology builder
/// and the minimizer. "charmm19_eef1" (the default) gives physically
/// meaningful energies on isolated proteins without explicit water —
/// the EEF1 solvation term dampens the unscreened electrostatic blow-up
/// that makes raw AMBER96 numbers on bare structures useless. "amber96"
/// is provided for like-for-like comparison against other AMBER96
/// implementations (OpenMM, BALL) in the SOTA validation harness.
#[pyfunction]
#[pyo3(signature = (structures, reconstruct=true, hydrogens="all", include_water=false, minimize=true, minimize_method="lbfgs", minimize_steps=500, gradient_tolerance=0.1, n_threads=None, strip_hydrogens=true, ff="charmm19_eef1", constrain_heavy=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_prepare(
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
    strip_hydrogens: bool,
    ff: &str,
    // Whether to freeze heavy atoms during minimization:
    //   * None (default): FF-aware — True for AMBER96 (explicit H absorbs
    //     clashes via hydrogen motion, heavy atoms already roughly at
    //     AMBER's minimum on a crystal structure), False for CHARMM19+EEF1
    //     (united-atom inflated C radii need heavy-atom relaxation).
    //   * Some(true): always freeze heavy atoms, move only H.
    //   * Some(false): always move all atoms.
    // The FF-aware default means existing AMBER96 callers keep exactly the
    // same behavior while CHARMM19 users automatically get full relaxation.
    constrain_heavy: Option<bool>,
) -> PyResult<Vec<PyObject>> {
    let all_results = match ff {
        "amber" | "amber96" => {
            let params = crate::forcefield::params::amber96();
            // Default for AMBER96: freeze heavy atoms (H-only min).
            let constrain_heavy = constrain_heavy.unwrap_or(true);
            batch_prepare_inner(
                py,
                structures,
                reconstruct,
                hydrogens,
                include_water,
                minimize,
                minimize_method,
                minimize_steps,
                gradient_tolerance,
                n_threads,
                strip_hydrogens,
                &params,
                constrain_heavy,
            )?
        }
        "charmm" | "charmm19" | "charmm19_eef1" => {
            let params = crate::forcefield::params::charmm19_eef1();
            // Default for CHARMM19+EEF1: move everything. Heavy atoms must
            // relax against the united-atom inflated carbon radii.
            let constrain_heavy = constrain_heavy.unwrap_or(false);
            batch_prepare_inner(
                py,
                structures,
                reconstruct,
                hydrogens,
                include_water,
                minimize,
                minimize_method,
                minimize_steps,
                gradient_tolerance,
                n_threads,
                strip_hydrogens,
                &params,
                constrain_heavy,
            )?
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown ff '{}'. Use 'amber96' or 'charmm19_eef1'.",
                ff
            )));
        }
    };

    // Convert to Python dicts
    Ok(all_results
        .into_iter()
        .map(|r| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("atoms_reconstructed", r.reconstructed)
                .unwrap();
            dict.set_item("hydrogens_added", r.h_added).unwrap();
            dict.set_item("hydrogens_skipped", r.h_skipped).unwrap();
            dict.set_item("initial_energy", r.init_e).unwrap();
            dict.set_item("final_energy", r.final_e).unwrap();
            dict.set_item("minimizer_steps", r.steps).unwrap();
            dict.set_item("converged", r.converged).unwrap();
            dict.set_item("n_unassigned_atoms", r.n_unassigned).unwrap();
            dict.set_item("skipped_no_protein", r.skipped_no_protein)
                .unwrap();
            // Component breakdown at the post-minimization geometry (all
            // zero if minimize=False or skipped_no_protein).
            let components = pyo3::types::PyDict::new(py);
            components.set_item("bond_stretch", r.bond_stretch).unwrap();
            components.set_item("angle_bend", r.angle_bend).unwrap();
            components.set_item("torsion", r.torsion).unwrap();
            components
                .set_item("improper_torsion", r.improper_torsion)
                .unwrap();
            components.set_item("vdw", r.vdw).unwrap();
            components
                .set_item("electrostatic", r.electrostatic)
                .unwrap();
            components.set_item("solvation", r.solvation).unwrap();
            dict.set_item("components", components).unwrap();
            dict.into_any().unbind()
        })
        .collect())
}

/// Generic inner loop for `batch_prepare`, monomorphized over the force
/// field type so we keep static dispatch inside the hot path (no perf hit
/// from adding the `ff` parameter).
#[allow(clippy::too_many_arguments)]
fn batch_prepare_inner<F>(
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
    strip_hydrogens: bool,
    ff: &F,
    constrain_heavy: bool,
) -> PyResult<Vec<PrepareReport>>
where
    F: crate::forcefield::params::ForceField + Sync,
{
    let n = resolve_threads(n_threads);
    let h_mode = hydrogens.to_string();
    let method = minimize_method.to_string();
    let total = structures.len();
    let chunk_size = 200; // prepare is heavier per-structure, smaller chunks
    let mut all_results: Vec<PrepareReport> = Vec::with_capacity(total);

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

        let opts = PrepareOptions {
            reconstruct,
            hydrogens: h_mode.clone(),
            include_water,
            minimize,
            minimize_method: method.clone(),
            minimize_steps,
            gradient_tolerance,
            strip_hydrogens,
            // The caller already resolved the FF-aware default for this ff.
            constrain_heavy: Some(constrain_heavy),
        };
        let results: Vec<PrepareReport> = py.allow_threads(|| {
            let pool = build_pool(n);
            pool.install(|| {
                chunk_pdbs
                    .par_iter_mut()
                    .map(|pdb| prepare::prepare_structure(pdb, &opts, ff))
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

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub(crate) fn py_add_hydrogens(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
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
