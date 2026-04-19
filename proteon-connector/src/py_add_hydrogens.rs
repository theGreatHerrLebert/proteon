//! PyO3 bindings for peptide hydrogen placement.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::add_hydrogens;
use crate::py_pdb::PyPDB;

/// Apply minimized coordinates back to a PDB structure.
///
/// Iterates chains → residues → **primary conformer** → atoms in the same
/// order as topology building, and sets each atom's position from the flat
/// coordinate array. The primary-conformer selection mirrors
/// [`crate::altloc::primary_conformer`]: blank altLoc first, then "A", then
/// the first available conformer.
///
/// Atoms are included or skipped via `topology::should_include_atom`, the
/// same predicate `build_topology` calls. That function filters out:
///   * water residues (HOH/WAT/...) under vacuum protein force fields,
///   * non-polar hydrogens under polar-H force fields (CHARMM19).
///
/// If this function's skip logic drifts from build_topology's, the
/// coord array length and the atom iteration will desync and the asserts
/// below will fire.
///
/// Takes the force field as a parameter so the predicate can consult
/// the parameter table — same as build_topology does. Note that the
/// residue-variant lookup (terminal "-N"/"-C" tails) is not available
/// here since we don't have a Topology at this point, so we pass the
/// base residue name as `lookup_name`. `get_atom_type` internally falls
/// back from variant to base name, so hydrogens that exist under both
/// variant and base names (all of them, in practice) still resolve.
///
/// Panics if the coordinate array doesn't match the atom count.
fn apply_coords_to_pdb<F: crate::forcefield::params::ForceField + ?Sized>(
    pdb: &mut pdbtbx::PDB,
    coords: &[[f64; 3]],
    params: &F,
) {
    let mut idx = 0;
    // Use first model only (consistent with build_topology, etc.)
    let first_model = match pdb.models_mut().next() {
        Some(m) => m,
        None => return,
    };
    for chain in first_model.chains_mut() {
        for residue in chain.residues_mut() {
            let res_name = residue.name().unwrap_or("UNK").to_string();

            // Determine the primary-conformer alt_loc in an immutable scan,
            // then iterate mutably and update only atoms in that conformer.
            let primary_alt: Option<Option<String>> = {
                let blank = residue
                    .conformers()
                    .find(|c| c.alternative_location().is_none());
                let a = residue
                    .conformers()
                    .find(|c| c.alternative_location() == Some("A"));
                blank
                    .or(a)
                    .or_else(|| residue.conformers().next())
                    .map(|c| c.alternative_location().map(str::to_string))
            };
            let Some(target_alt) = primary_alt else {
                continue;
            };

            for conformer in residue.conformers_mut() {
                let matches = match (conformer.alternative_location(), target_alt.as_deref()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };
                if !matches {
                    continue;
                }
                for atom in conformer.atoms_mut() {
                    let atom_name = atom.name().trim().to_string();
                    let element = atom
                        .element()
                        .map(|e| e.symbol().to_string())
                        .unwrap_or_else(|| "C".to_string());
                    // Shared predicate — identical to build_topology.
                    // Using `res_name` for both residue_name and lookup_name
                    // because we lack Topology's residue_variants map here.
                    if !crate::forcefield::topology::should_include_atom(
                        &res_name, &atom_name, &element, params, &res_name,
                    ) {
                        continue;
                    }
                    assert!(
                        idx < coords.len(),
                        "apply_coords_to_pdb: coord array too short ({} coords, atom index {})",
                        coords.len(),
                        idx,
                    );
                    atom.set_pos((coords[idx][0], coords[idx][1], coords[idx][2]))
                        .expect("apply_coords_to_pdb: invalid coordinates (NaN/Inf)");
                    idx += 1;
                }
                break;
            }
        }
    }
    assert_eq!(
        idx,
        coords.len(),
        "apply_coords_to_pdb: coord array length ({}) != atom count ({})",
        coords.len(),
        idx,
    );
}

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
/// (general BALL algorithm for ligands/non-standard residues).
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
/// Per-structure result from batch_prepare. One of these is produced per
/// input structure and converted to a Python dict at the end.
struct PrepareResult {
    reconstructed: usize,
    h_added: usize,
    h_skipped: usize,
    init_e: f64,
    final_e: f64,
    bond_stretch: f64,
    angle_bend: f64,
    torsion: f64,
    improper_torsion: f64,
    vdw: f64,
    electrostatic: f64,
    solvation: f64,
    steps: usize,
    converged: bool,
    n_unassigned: usize,
    skipped_no_protein: bool,
}

impl PrepareResult {
    fn empty() -> Self {
        Self {
            reconstructed: 0,
            h_added: 0,
            h_skipped: 0,
            init_e: 0.0,
            final_e: 0.0,
            bond_stretch: 0.0,
            angle_bend: 0.0,
            torsion: 0.0,
            improper_torsion: 0.0,
            vdw: 0.0,
            electrostatic: 0.0,
            solvation: 0.0,
            steps: 0,
            converged: false,
            n_unassigned: 0,
            skipped_no_protein: false,
        }
    }
}

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
) -> PyResult<Vec<PrepareResult>>
where
    F: crate::forcefield::params::ForceField + Sync,
{
    let n = resolve_threads(n_threads);
    let h_mode = hydrogens.to_string();
    let method = minimize_method.to_string();
    let total = structures.len();
    let chunk_size = 200; // prepare is heavier per-structure, smaller chunks
    let mut all_results: Vec<PrepareResult> = Vec::with_capacity(total);

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
        let results: Vec<PrepareResult> = py.allow_threads(|| {
            let pool = build_pool(n);
            pool.install(|| {
                chunk_pdbs
                    .par_iter_mut()
                    .map(|pdb| {
                        // Optionally strip pre-existing hydrogens. Used to
                        // rescue structures whose externally-resolved H sit
                        // off the MM minimum and prevent LBFGS convergence.
                        if strip_hydrogens {
                            add_hydrogens::strip_hydrogens(pdb);
                        }

                        // Reconstruct
                        let reconstructed = if reconstruct {
                            crate::reconstruct::reconstruct_fragments(pdb).added
                        } else {
                            0
                        };

                        // Place hydrogens
                        // Layer A: under a polar-H united-atom force field
                        // (CHARMM19+EEF1), only place hydrogens bonded to
                        // N/O/S. Non-polar C-H atoms are absorbed into
                        // united carbon types (CH1E/CH2E/CH3E) and don't
                        // exist as separate atoms. Placing them anyway
                        // produces 40% "unassigned" atoms at build_topology
                        // time, which Layer B skips, but it's cleaner to
                        // not place them in the first place so the output
                        // PDB matches GROMACS's pdb2gmx behavior exactly.
                        let polar_only = ff.has_eef1();
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
                                let r = add_hydrogens::place_all_hydrogens(pdb, polar_only);
                                (r.added, r.skipped)
                            }
                            _ => (0, 0),
                        };

                        // Build topology once. n_unassigned is invariant under
                        // coordinate changes (it depends only on residue/atom
                        // names), so we can compute it before minimization and
                        // reuse the same topology for the minimizer.
                        let topo = crate::forcefield::topology::build_topology(pdb, ff);
                        let n_unassigned = topo.unassigned_atoms.len();

                        // Heuristic: if more than half the NON-WATER atoms have
                        // no force field type (e.g. nucleic acids, ligand-only
                        // entries, exotic non-standard residues) then the
                        // protein prep pipeline is the wrong tool. Skip
                        // minimization and mark the structure so downstream
                        // consumers can distinguish "convergence failed" from
                        // "skipped, not a protein".
                        //
                        // Waters (HOH, WAT, ...) are EXCLUDED from both the
                        // numerator and denominator: they're expected to be
                        // "unassigned" under a protein-only force field, but
                        // their presence doesn't mean the pipeline should give
                        // up. Before this exclusion, 1bpi (58-residue protein
                        // with 167 crystal waters + 2 PO4 ions) got classified
                        // as not-a-protein because after H placement the water
                        // atoms outnumbered the protein atoms by a hair
                        // (544/1065 ≈ 51% unassigned, above the 50% line).
                        // Post-fix, only the non-water atoms count: for 1bpi
                        // that's ~44/565 ≈ 8% unassigned, well below threshold.
                        //
                        // Threshold 50% still picks DNA/RNA cleanly (193d: 394
                        // of ~404 non-water atoms unassigned) without false-
                        // flagging proteins with bound ligands or HETATM tails.
                        let non_water_total = topo
                            .atoms
                            .iter()
                            .filter(|a| !crate::add_hydrogens::is_water_residue(&a.residue_name))
                            .count();
                        let non_water_unassigned = topo
                            .unassigned_atoms
                            .iter()
                            .filter(|s| {
                                // Unassigned entries are formatted "ResName:AtomName".
                                let res = s.split(':').next().unwrap_or("");
                                !crate::add_hydrogens::is_water_residue(res)
                            })
                            .count();
                        let skipped_no_protein =
                            non_water_total > 0 && non_water_unassigned * 2 > non_water_total;

                        let mut out = PrepareResult::empty();
                        out.reconstructed = reconstructed;
                        out.h_added = h_added;
                        out.h_skipped = h_skipped;
                        out.n_unassigned = n_unassigned;
                        out.skipped_no_protein = skipped_no_protein;

                        // Minimize H positions and apply coords back to PDB
                        let has_any_h = crate::altloc::pdb_atoms_primary(pdb).any(|a| {
                            a.element()
                                .is_some_and(|e| e.symbol() == "H" || e.symbol() == "D")
                        });
                        if !skipped_no_protein && minimize && (h_added > 0 || has_any_h) {
                            let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
                            // Freeze heavy atoms (move only H) when `constrain_heavy`
                            // is true. AMBER96 default: True. CHARMM19+EEF1 default:
                            // False — heavy atoms need to relax against united-atom
                            // inflated carbon radii for a negative final energy.
                            let constrained: Vec<bool> = if constrain_heavy {
                                topo.atoms.iter().map(|a| !a.is_hydrogen).collect()
                            } else {
                                vec![false; topo.atoms.len()]
                            };
                            let result = match method.as_str() {
                                "cg" => crate::forcefield::minimize::conjugate_gradient(
                                    &coords,
                                    &topo,
                                    ff,
                                    minimize_steps,
                                    gradient_tolerance,
                                    &constrained,
                                ),
                                "lbfgs" => crate::forcefield::minimize::lbfgs(
                                    &coords,
                                    &topo,
                                    ff,
                                    minimize_steps,
                                    gradient_tolerance,
                                    &constrained,
                                ),
                                _ => crate::forcefield::minimize::steepest_descent(
                                    &coords,
                                    &topo,
                                    ff,
                                    minimize_steps,
                                    gradient_tolerance,
                                    &constrained,
                                ),
                            };
                            apply_coords_to_pdb(pdb, &result.coords, ff);
                            out.init_e = result.initial_energy;
                            out.final_e = result.energy.total;
                            out.bond_stretch = result.energy.bond_stretch;
                            out.angle_bend = result.energy.angle_bend;
                            out.torsion = result.energy.torsion;
                            out.improper_torsion = result.energy.improper_torsion;
                            out.vdw = result.energy.vdw;
                            out.electrostatic = result.energy.electrostatic;
                            out.solvation = result.energy.solvation;
                            out.steps = result.steps;
                            out.converged = result.converged;
                        }
                        out
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
