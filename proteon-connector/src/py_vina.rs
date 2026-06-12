//! PyO3 bindings for the `proteon-vina` AutoDock-Vina scorer.
//!
//! Exposes two entry points:
//!
//! * `score_only(receptor_pdbqt, ligand_pdbqt)` — returns the
//!   8-component energy vector that upstream `vina --score_only`
//!   prints. Matches upstream to ≤ 1 mkcal/mol on our parity
//!   fixtures (incl. macrocycles and metalloproteins).
//! * `local_only(receptor_pdbqt, ligand_pdbqt, max_steps=None,
//!   v_curl=1000.0)` — runs BFGS on the ligand conformation to
//!   its nearest local minimum and returns the refined pose
//!   (coords + fragment IDs), the 8 components at the refined
//!   pose, and BFGS diagnostics.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use crate::parallel::resolve_threads;
use proteon_vina::local_only::{local_only as rust_local_only, LocalOnlyOptions};
use proteon_vina::molecule::Molecule;
use proteon_vina::pdbqt::parse_pdbqt;
use proteon_vina::precalculate::Precalculate;
use proteon_vina::score::{score_only as rust_score_only, ScoreComponents};

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// The eight energy components upstream Vina reports after scoring.
/// Field names and semantics match upstream's `show_score` output.
#[pyclass(name = "VinaScoreComponents")]
#[derive(Clone)]
pub struct PyScoreComponents {
    pub(crate) inner: ScoreComponents,
}

#[pymethods]
impl PyScoreComponents {
    /// Conf-independent-adjusted total (what upstream prints as
    /// "Estimated Free Energy of Binding"). kcal/mol.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }
    /// Ligand–receptor pair energy (upstream "Ligand - Receptor").
    #[getter]
    fn lig_grids(&self) -> f64 {
        self.inner.lig_grids
    }
    /// Ligand–flex side chain pair energy. Always 0 in v0 (no flex).
    #[getter]
    fn inter_pairs(&self) -> f64 {
        self.inner.inter_pairs
    }
    /// Flex side chain–receptor pair energy. Always 0 in v0.
    #[getter]
    fn flex_grids(&self) -> f64 {
        self.inner.flex_grids
    }
    /// Flex–flex pair energy. Always 0 in v0.
    #[getter]
    fn intra_pairs(&self) -> f64 {
        self.inner.intra_pairs
    }
    /// Intra-ligand pair energy across rotatable bonds (upstream
    /// "Final Total Internal Energy / Ligand").
    #[getter]
    fn lig_intra(&self) -> f64 {
        self.inner.lig_intra
    }
    /// Conformational-independent penalty (upstream "Torsional Free
    /// Energy").
    #[getter]
    fn conf_independent(&self) -> f64 {
        self.inner.conf_independent
    }
    /// Intramolecular reference energy (upstream "Unbound System's
    /// Energy"). For SF_VINA this equals `lig_intra`.
    #[getter]
    fn intramolecular(&self) -> f64 {
        self.inner.intramolecular
    }

    /// Convert to a plain dict for convenient interop.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("total", self.inner.total)?;
        d.set_item("lig_grids", self.inner.lig_grids)?;
        d.set_item("inter_pairs", self.inner.inter_pairs)?;
        d.set_item("flex_grids", self.inner.flex_grids)?;
        d.set_item("intra_pairs", self.inner.intra_pairs)?;
        d.set_item("lig_intra", self.inner.lig_intra)?;
        d.set_item("conf_independent", self.inner.conf_independent)?;
        d.set_item("intramolecular", self.inner.intramolecular)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "VinaScoreComponents(total={:.4}, lig_grids={:.4}, lig_intra={:.4}, \
             conf_independent={:.4}, intramolecular={:.4})",
            self.inner.total,
            self.inner.lig_grids,
            self.inner.lig_intra,
            self.inner.conf_independent,
            self.inner.intramolecular,
        )
    }
}

/// BFGS convergence statistics surfaced from `local_only`.
#[pyclass(name = "BfgsOutcome")]
#[derive(Clone)]
pub struct PyBfgsOutcome {
    pub(crate) inner: proteon_vina::bfgs::BfgsOutcome,
}

#[pymethods]
impl PyBfgsOutcome {
    /// Energy at the starting pose (before minimisation).
    #[getter]
    fn initial_energy(&self) -> f64 {
        self.inner.initial_energy
    }
    /// Energy at the minimised pose. Equals `initial_energy` if the
    /// safeguard reverted because line search never improved.
    #[getter]
    fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }
    /// Total energy+gradient evaluations (incl. line-search misses).
    #[getter]
    fn n_evals(&self) -> usize {
        self.inner.n_evals
    }
    /// Outer BFGS iterations actually performed.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }
    /// True if convergence hit (gradient norm < 1e-5).
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "BfgsOutcome(n_steps={}, n_evals={}, converged={}, \
             initial_energy={:.4}, final_energy={:.4})",
            self.inner.n_steps,
            self.inner.n_evals,
            self.inner.converged,
            self.inner.initial_energy,
            self.inner.final_energy,
        )
    }
}

/// Full output of `local_only`: refined pose + score + BFGS stats.
#[pyclass(name = "VinaLocalOnlyOutcome")]
pub struct PyLocalOnlyOutcome {
    components: PyScoreComponents,
    bfgs: PyBfgsOutcome,
    coords: Vec<[f64; 3]>,
    serials: Vec<u32>,
}

#[pymethods]
impl PyLocalOnlyOutcome {
    /// 8-component energy vector at the minimised pose. Same layout
    /// as `score_only`'s return value.
    #[getter]
    fn components(&self) -> PyScoreComponents {
        self.components.clone()
    }
    /// BFGS trajectory statistics.
    #[getter]
    fn bfgs(&self) -> PyBfgsOutcome {
        self.bfgs.clone()
    }
    /// Minimised ligand coordinates as an (N, 3) float64 numpy array.
    /// Indices match `original_serials`.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self
            .coords
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect();
        let n = self.coords.len();
        PyArray1::from_vec(py, flat).reshape([n, 3]).unwrap()
    }
    /// Original PDBQT serial numbers of the retained ligand atoms
    /// (parallel to `coords`).
    #[getter]
    fn original_serials<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.serials.clone().into_pyarray(py)
    }
    /// Shortcut: refined total energy (== `components.total`).
    #[getter]
    fn total(&self) -> f64 {
        self.components.inner.total
    }

    fn __repr__(&self) -> String {
        format!(
            "VinaLocalOnlyOutcome(total={:.4}, n_steps={}, converged={})",
            self.components.inner.total, self.bfgs.inner.n_steps, self.bfgs.inner.converged,
        )
    }
}

/// Score a ligand pose against a receptor with AutoDock-Vina's
/// scoring function. Inputs are PDBQT text (not file paths).
///
/// Matches `vina --score_only` to ≤ 1 mkcal/mol across every
/// reported component on proteon-vina's parity fixtures.
#[pyfunction]
#[pyo3(name = "score_only")]
pub(crate) fn py_score_only(
    py: Python<'_>,
    receptor_pdbqt: &str,
    ligand_pdbqt: &str,
) -> PyResult<PyScoreComponents> {
    let rec_text = receptor_pdbqt.to_owned();
    let lig_text = ligand_pdbqt.to_owned();
    let components: ScoreComponents = py.allow_threads(move || -> PyResult<ScoreComponents> {
        let receptor = Molecule::from_pdbqt_str(&rec_text).map_err(to_py_err)?;
        let ligand = Molecule::from_pdbqt_str(&lig_text).map_err(to_py_err)?;
        let file = parse_pdbqt(&lig_text).map_err(to_py_err)?;
        let precalc = Precalculate::vina();
        Ok(rust_score_only(
            &receptor,
            &ligand,
            &file.rotatable_bonds,
            &precalc,
            1000.0,
        ))
    })?;
    Ok(PyScoreComponents { inner: components })
}

/// Run BFGS to the ligand's nearest local minimum and return the
/// refined pose plus score. Inputs are PDBQT text (not file paths).
///
/// Matches `vina --local_only` to ≤ 80 mkcal/mol on drug-like
/// fixtures; up to ~1.5 kcal/mol on high-DoF macrocycles where
/// BFGS lands in a neighbouring local basin (see
/// proteon-vina docs).
#[pyfunction]
#[pyo3(name = "local_only", signature = (receptor_pdbqt, ligand_pdbqt, *, max_steps=None, v_curl=1000.0))]
pub(crate) fn py_local_only(
    py: Python<'_>,
    receptor_pdbqt: &str,
    ligand_pdbqt: &str,
    max_steps: Option<usize>,
    v_curl: f64,
) -> PyResult<PyLocalOnlyOutcome> {
    let rec_text = receptor_pdbqt.to_owned();
    let lig_text = ligand_pdbqt.to_owned();
    let outcome = py.allow_threads(move || -> PyResult<PyLocalOnlyOutcome> {
        let receptor = Molecule::from_pdbqt_str(&rec_text).map_err(to_py_err)?;
        let ligand = Molecule::from_pdbqt_str(&lig_text).map_err(to_py_err)?;
        let file = parse_pdbqt(&lig_text).map_err(to_py_err)?;
        let precalc = Precalculate::vina();
        let opts = LocalOnlyOptions { max_steps, v_curl };
        let r = rust_local_only(&receptor, &ligand, &file, &precalc, opts);

        // Re-apply the refined conformation to get world coords.
        use proteon_vina::torsion::TorsionTree;
        let tree = TorsionTree::from_molecule(&ligand, &file);
        let coords = tree.apply(&r.conf);
        Ok(PyLocalOnlyOutcome {
            components: PyScoreComponents { inner: r.components },
            bfgs: PyBfgsOutcome { inner: r.bfgs },
            coords,
            serials: ligand.original_serials.clone(),
        })
    })?;
    Ok(outcome)
}

fn to_py_err(e: proteon_vina::pdbqt::PdbqtError) -> PyErr {
    PyValueError::new_err(format!("PDBQT parse error: {e}"))
}

/// Batch `score_only`: score every ligand in `ligands_pdbqt` against
/// a single `receptor_pdbqt`. The receptor is parsed once and the
/// 2 MB `Precalculate` table is built once; ligand parsing + scoring
/// runs in parallel on a rayon pool.
///
/// `n_threads`: `None` (or 0) uses every core; positive values cap
/// the pool size. Matches proteon's batch API convention.
///
/// Order of the returned list matches the input order (rayon's
/// `par_iter().collect()` is order-preserving).
#[pyfunction]
#[pyo3(name = "batch_score_only", signature = (receptor_pdbqt, ligands_pdbqt, *, n_threads=None))]
pub(crate) fn py_batch_score_only(
    py: Python<'_>,
    receptor_pdbqt: &str,
    ligands_pdbqt: Vec<String>,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyScoreComponents>> {
    let rec_text = receptor_pdbqt.to_owned();
    let n = resolve_threads(n_threads);
    let results: Vec<ScoreComponents> = py.allow_threads(move || -> PyResult<Vec<ScoreComponents>> {
        let receptor = Molecule::from_pdbqt_str(&rec_text).map_err(to_py_err)?;
        let precalc = Precalculate::vina();
        let pool = build_pool(n);
        pool.install(|| {
            ligands_pdbqt
                .par_iter()
                .map(|lig_text| {
                    let ligand = Molecule::from_pdbqt_str(lig_text).map_err(to_py_err)?;
                    let file = parse_pdbqt(lig_text).map_err(to_py_err)?;
                    Ok(rust_score_only(
                        &receptor,
                        &ligand,
                        &file.rotatable_bonds,
                        &precalc,
                        1000.0,
                    ))
                })
                .collect()
        })
    })?;
    Ok(results.into_iter().map(|c| PyScoreComponents { inner: c }).collect())
}

/// Batch `local_only`: refine every ligand in `ligands_pdbqt`
/// against a single `receptor_pdbqt`. Same parallelism model as
/// `batch_score_only`. Each element of the returned list has the
/// refined coords, the 8-component score at the refined pose, and
/// BFGS stats for that specific ligand.
#[pyfunction]
#[pyo3(name = "batch_local_only", signature = (receptor_pdbqt, ligands_pdbqt, *, n_threads=None, max_steps=None, v_curl=1000.0))]
pub(crate) fn py_batch_local_only(
    py: Python<'_>,
    receptor_pdbqt: &str,
    ligands_pdbqt: Vec<String>,
    n_threads: Option<i32>,
    max_steps: Option<usize>,
    v_curl: f64,
) -> PyResult<Vec<PyLocalOnlyOutcome>> {
    let rec_text = receptor_pdbqt.to_owned();
    let n = resolve_threads(n_threads);
    let opts = LocalOnlyOptions { max_steps, v_curl };
    let outcomes = py.allow_threads(move || -> PyResult<Vec<PyLocalOnlyOutcome>> {
        let receptor = Molecule::from_pdbqt_str(&rec_text).map_err(to_py_err)?;
        let precalc = Precalculate::vina();
        let pool = build_pool(n);
        pool.install(|| {
            ligands_pdbqt
                .par_iter()
                .map(|lig_text| -> PyResult<PyLocalOnlyOutcome> {
                    let ligand = Molecule::from_pdbqt_str(lig_text).map_err(to_py_err)?;
                    let file = parse_pdbqt(lig_text).map_err(to_py_err)?;
                    let r = rust_local_only(&receptor, &ligand, &file, &precalc, opts);
                    use proteon_vina::torsion::TorsionTree;
                    let tree = TorsionTree::from_molecule(&ligand, &file);
                    let coords = tree.apply(&r.conf);
                    Ok(PyLocalOnlyOutcome {
                        components: PyScoreComponents { inner: r.components },
                        bfgs: PyBfgsOutcome { inner: r.bfgs },
                        coords,
                        serials: ligand.original_serials.clone(),
                    })
                })
                .collect()
        })
    })?;
    Ok(outcomes)
}

/// Python module entry point: proteon_connector.py_vina
#[pymodule]
pub(crate) fn py_vina(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScoreComponents>()?;
    m.add_class::<PyBfgsOutcome>()?;
    m.add_class::<PyLocalOnlyOutcome>()?;
    m.add_function(wrap_pyfunction!(py_score_only, m)?)?;
    m.add_function(wrap_pyfunction!(py_local_only, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_score_only, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_local_only, m)?)?;
    Ok(())
}
