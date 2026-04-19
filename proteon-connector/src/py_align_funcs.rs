//! PyO3 alignment functions: TM-align, SOI-align, FlexAlign, MM-align.
//!
//! Each has single-pair, one-to-many, and many-to-many variants.
//! Uses rayon for parallel batch alignment with configurable thread count.
//! GIL is released during computation via `py.allow_threads()`.

use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use proteon_align::core::align::tmalign::tmalign;
use proteon_align::core::types::{AlignOptions, AlignResult, StructureData, Transform};
use proteon_align::ext::flexalign::{flexalign_main, FlexOptions};
use proteon_align::ext::mmalign::{mmalign_complex, ChainData, MMAlignResult};
use proteon_align::ext::soialign::{soialign_main, SoiOptions, SoiResult};
use proteon_io::pdb_io::{extract_chains_for_alignment, extract_for_alignment};

use crate::py_align::PyAlignResult;
use crate::py_pdb::PyPDB;

// ===========================================================================
// Helpers
// ===========================================================================

/// Extract StructureData from a PyPDB, converting errors to PyErr.
fn extract(pdb: &PyPDB, chain: Option<&str>) -> PyResult<StructureData> {
    extract_for_alignment(&pdb.inner, chain)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Convert `Vec<char>` to `Vec<u8>` (needed by ext functions).
fn chars_to_u8(v: &[char]) -> Vec<u8> {
    v.iter().map(|c| *c as u8).collect()
}

use crate::parallel::resolve_threads;

/// Build a rayon thread pool with the given thread count.
fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

/// Extract list of PyPDB from a Python list.
fn extract_list(list: &Bound<'_, PyList>, chain: Option<&str>) -> PyResult<Vec<StructureData>> {
    list.iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            extract(&pdb, chain)
        })
        .collect()
}

// ===========================================================================
// TM-align
// ===========================================================================

fn run_tmalign(s1: &StructureData, s2: &StructureData, fast: bool) -> Result<AlignResult, String> {
    let opts = AlignOptions {
        fast_opt: fast,
        ..AlignOptions::default()
    };
    tmalign(
        &s1.coords,
        &s2.coords,
        &s1.sequence,
        &s2.sequence,
        &s1.sec_structure,
        &s2.sec_structure,
        &opts,
    )
    .map_err(|e| e.to_string())
}

/// Align two structures using TM-align.
#[pyfunction]
#[pyo3(signature = (pdb1, pdb2, chain1=None, chain2=None, fast=false))]
pub(crate) fn tm_align_pair(
    py: Python<'_>,
    pdb1: &PyPDB,
    pdb2: &PyPDB,
    chain1: Option<&str>,
    chain2: Option<&str>,
    fast: bool,
) -> PyResult<PyAlignResult> {
    let s1 = extract(pdb1, chain1)?;
    let s2 = extract(pdb2, chain2)?;
    let result = py.allow_threads(|| run_tmalign(&s1, &s2, fast));
    match result {
        Ok(r) => Ok(PyAlignResult::from_inner(r)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// Align one query against many targets in parallel (TM-align).
#[pyfunction]
#[pyo3(signature = (query, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn tm_align_one_to_many(
    py: Python<'_>,
    query: &PyPDB,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<PyAlignResult>> {
    let query_sd = extract(query, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            target_sds
                .par_iter()
                .map(|target| run_tmalign(&query_sd, target, fast))
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|r| match r {
            Ok(result) => Ok(PyAlignResult::from_inner(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        })
        .collect()
}

/// Align all pairs between two lists (Cartesian product, TM-align).
#[pyfunction]
#[pyo3(signature = (queries, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn tm_align_many_to_many(
    py: Python<'_>,
    queries: &Bound<'_, PyList>,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<(usize, usize, PyAlignResult)>> {
    let query_sds = extract_list(queries, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let pairs: Vec<(usize, usize)> = (0..query_sds.len())
        .flat_map(|qi| (0..target_sds.len()).map(move |ti| (qi, ti)))
        .collect();

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pairs
                .par_iter()
                .map(|&(qi, ti)| {
                    let r = run_tmalign(&query_sds[qi], &target_sds[ti], fast);
                    (qi, ti, r)
                })
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|(qi, ti, r)| match r {
            Ok(result) => Ok((qi, ti, PyAlignResult::from_inner(result))),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        })
        .collect()
}

// ===========================================================================
// SOI-align result wrapper
// ===========================================================================

/// Result from SOI-align (sequence-order independent alignment).
#[pyclass]
pub struct PySoiAlignResult {
    inner: SoiResult,
}

#[pymethods]
impl PySoiAlignResult {
    /// TM-score normalized by length of structure 1.
    #[getter]
    fn tm_score_chain1(&self) -> f64 {
        self.inner.tm1
    }

    /// TM-score normalized by length of structure 2.
    #[getter]
    fn tm_score_chain2(&self) -> f64 {
        self.inner.tm2
    }

    /// RMSD of aligned residues.
    #[getter]
    fn rmsd(&self) -> f64 {
        self.inner.rmsd
    }

    /// Number of aligned residue pairs.
    #[getter]
    fn n_aligned(&self) -> usize {
        self.inner.n_ali8
    }

    /// Sequence identity (count of identical aligned residues).
    #[getter]
    fn seq_identity(&self) -> f64 {
        self.inner.liden
    }

    /// Aligned sequence of structure 1 (with gaps).
    #[getter]
    fn aligned_seq_x(&self) -> &str {
        &self.inner.seq_x_aligned
    }

    /// Aligned sequence of structure 2 (with gaps).
    #[getter]
    fn aligned_seq_y(&self) -> &str {
        &self.inner.seq_y_aligned
    }

    /// Rotation matrix as 3x3 numpy array.
    #[getter]
    fn rotation_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        let u = self.inner.transform.u;
        let flat: Vec<f64> = u.iter().flat_map(|row| row.iter().copied()).collect();
        numpy::PyArray1::from_vec(py, flat)
            .reshape([3, 3])
            .expect("reshape to 3x3")
    }

    /// Translation vector.
    #[getter]
    fn translation(&self) -> [f64; 3] {
        self.inner.transform.t
    }

    fn __repr__(&self) -> String {
        format!(
            "SoiAlignResult(TM1={:.4}, TM2={:.4}, RMSD={:.2}, Lali={})",
            self.inner.tm1, self.inner.tm2, self.inner.rmsd, self.inner.n_ali8,
        )
    }
}

// ===========================================================================
// FlexAlign result wrapper
// ===========================================================================

/// Result from FlexAlign (flexible hinge-based alignment).
#[pyclass]
pub struct PyFlexAlignResult {
    tm1: f64,
    tm2: f64,
    rmsd: f64,
    n_aligned: usize,
    seq_identity: f64,
    aligned_seq_x: String,
    aligned_seq_y: String,
    hinge_count: usize,
    transforms_flat: Vec<f64>,
}

#[pymethods]
impl PyFlexAlignResult {
    /// TM-score normalized by length of structure 1.
    #[getter]
    fn tm_score_chain1(&self) -> f64 {
        self.tm1
    }

    /// TM-score normalized by length of structure 2.
    #[getter]
    fn tm_score_chain2(&self) -> f64 {
        self.tm2
    }

    /// RMSD of aligned residues.
    #[getter]
    fn rmsd(&self) -> f64 {
        self.rmsd
    }

    /// Number of aligned residue pairs.
    #[getter]
    fn n_aligned(&self) -> usize {
        self.n_aligned
    }

    /// Sequence identity (count of identical residues).
    #[getter]
    fn seq_identity(&self) -> f64 {
        self.seq_identity
    }

    /// Aligned sequence of structure 1 (with gaps).
    #[getter]
    fn aligned_seq_x(&self) -> &str {
        &self.aligned_seq_x
    }

    /// Aligned sequence of structure 2 (with gaps).
    #[getter]
    fn aligned_seq_y(&self) -> &str {
        &self.aligned_seq_y
    }

    /// Number of hinge points detected.
    #[getter]
    fn hinge_count(&self) -> usize {
        self.hinge_count
    }

    /// Per-hinge-segment rotation matrices as Kx3x3 numpy array (K = hinge_count + 1).
    #[getter]
    fn rotation_matrices<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray3<f64>> {
        let k = self.hinge_count + 1;
        // transforms_flat has k * 12 entries (9 rotation + 3 translation per segment)
        let mut rotations = Vec::with_capacity(k * 9);
        for i in 0..k {
            let base = i * 12;
            rotations.extend_from_slice(&self.transforms_flat[base..base + 9]);
        }
        numpy::PyArray1::from_vec(py, rotations)
            .reshape([k, 3, 3])
            .expect("reshape to Kx3x3")
    }

    /// Per-hinge-segment translations as Kx3 numpy array.
    #[getter]
    fn translations<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        let k = self.hinge_count + 1;
        let mut trans = Vec::with_capacity(k * 3);
        for i in 0..k {
            let base = i * 12 + 9;
            trans.extend_from_slice(&self.transforms_flat[base..base + 3]);
        }
        numpy::PyArray1::from_vec(py, trans)
            .reshape([k, 3])
            .expect("reshape to Kx3")
    }

    fn __repr__(&self) -> String {
        format!(
            "FlexAlignResult(TM1={:.4}, TM2={:.4}, RMSD={:.2}, Lali={}, hinges={})",
            self.tm1, self.tm2, self.rmsd, self.n_aligned, self.hinge_count,
        )
    }
}

/// Pack a FlexResult into a PyFlexAlignResult (extracts fields to avoid Send issues).
fn pack_flex_result(r: proteon_align::ext::flexalign::FlexResult) -> PyFlexAlignResult {
    let se = &r.se_result;
    // Flatten transforms: [u11,u12,...,u33,t1,t2,t3] per segment
    let mut transforms_flat = Vec::with_capacity(r.transforms.len() * 12);
    for t in &r.transforms {
        for row in &t.u {
            transforms_flat.extend_from_slice(row);
        }
        transforms_flat.extend_from_slice(&t.t);
    }
    PyFlexAlignResult {
        tm1: se.tm1,
        tm2: se.tm2,
        rmsd: se.rmsd,
        n_aligned: se.n_ali8,
        seq_identity: se.liden,
        aligned_seq_x: se.seq_x_aligned.clone(),
        aligned_seq_y: se.seq_y_aligned.clone(),
        hinge_count: r.hinge_count,
        transforms_flat,
    }
}

// ===========================================================================
// SOI-align functions
// ===========================================================================

fn run_soialign(s1: &StructureData, s2: &StructureData, fast: bool) -> SoiResult {
    let seqx = chars_to_u8(&s1.sequence);
    let seqy = chars_to_u8(&s2.sequence);
    let secx = chars_to_u8(&s1.sec_structure);
    let secy = chars_to_u8(&s2.sec_structure);
    let identity = Transform {
        t: [0.0; 3],
        u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let opts = SoiOptions {
        mol_type: s1.mol_type,
        fast_opt: fast,
        ..SoiOptions::default()
    };
    soialign_main(
        &s1.coords, &s2.coords, &seqx, &seqy, &secx, &secy, &identity, &opts,
    )
}

/// Align two structures using SOI-align (sequence-order independent).
#[pyfunction]
#[pyo3(signature = (pdb1, pdb2, chain1=None, chain2=None, fast=false))]
pub(crate) fn soi_align_pair(
    py: Python<'_>,
    pdb1: &PyPDB,
    pdb2: &PyPDB,
    chain1: Option<&str>,
    chain2: Option<&str>,
    fast: bool,
) -> PyResult<PySoiAlignResult> {
    let s1 = extract(pdb1, chain1)?;
    let s2 = extract(pdb2, chain2)?;
    let result = py.allow_threads(|| run_soialign(&s1, &s2, fast));
    Ok(PySoiAlignResult { inner: result })
}

/// SOI-align one query against many targets in parallel.
#[pyfunction]
#[pyo3(signature = (query, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn soi_align_one_to_many(
    py: Python<'_>,
    query: &PyPDB,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<PySoiAlignResult>> {
    let query_sd = extract(query, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            target_sds
                .par_iter()
                .map(|target| run_soialign(&query_sd, target, fast))
                .collect::<Vec<_>>()
        })
    });

    Ok(results
        .into_iter()
        .map(|r| PySoiAlignResult { inner: r })
        .collect())
}

/// SOI-align all pairs between two lists (Cartesian product).
#[pyfunction]
#[pyo3(signature = (queries, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn soi_align_many_to_many(
    py: Python<'_>,
    queries: &Bound<'_, PyList>,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<(usize, usize, PySoiAlignResult)>> {
    let query_sds = extract_list(queries, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let pairs: Vec<(usize, usize)> = (0..query_sds.len())
        .flat_map(|qi| (0..target_sds.len()).map(move |ti| (qi, ti)))
        .collect();

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pairs
                .par_iter()
                .map(|&(qi, ti)| {
                    let r = run_soialign(&query_sds[qi], &target_sds[ti], fast);
                    (qi, ti, r)
                })
                .collect::<Vec<_>>()
        })
    });

    Ok(results
        .into_iter()
        .map(|(qi, ti, r)| (qi, ti, PySoiAlignResult { inner: r }))
        .collect())
}

// ===========================================================================
// FlexAlign functions
// ===========================================================================

fn run_flexalign(
    s1: &StructureData,
    s2: &StructureData,
    fast: bool,
) -> Result<PyFlexAlignResult, String> {
    let seqx = chars_to_u8(&s1.sequence);
    let seqy = chars_to_u8(&s2.sequence);
    let secx = chars_to_u8(&s1.sec_structure);
    let secy = chars_to_u8(&s2.sec_structure);
    let opts = FlexOptions {
        mol_type: s1.mol_type,
        fast_opt: fast,
        ..FlexOptions::default()
    };
    let result = flexalign_main(
        &s1.coords,
        &s2.coords,
        &seqx,
        &seqy,
        &secx,
        &secy,
        &opts,
        &[],
    )
    .map_err(|e| e.to_string())?;
    Ok(pack_flex_result(result))
}

/// Align two structures using FlexAlign (flexible, hinge-based).
#[pyfunction]
#[pyo3(signature = (pdb1, pdb2, chain1=None, chain2=None, fast=false))]
pub(crate) fn flex_align_pair(
    py: Python<'_>,
    pdb1: &PyPDB,
    pdb2: &PyPDB,
    chain1: Option<&str>,
    chain2: Option<&str>,
    fast: bool,
) -> PyResult<PyFlexAlignResult> {
    let s1 = extract(pdb1, chain1)?;
    let s2 = extract(pdb2, chain2)?;
    let result = py.allow_threads(|| run_flexalign(&s1, &s2, fast));
    result.map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// FlexAlign one query against many targets in parallel.
#[pyfunction]
#[pyo3(signature = (query, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn flex_align_one_to_many(
    py: Python<'_>,
    query: &PyPDB,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<PyFlexAlignResult>> {
    let query_sd = extract(query, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            target_sds
                .par_iter()
                .map(|target| run_flexalign(&query_sd, target, fast))
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|r| r.map_err(pyo3::exceptions::PyRuntimeError::new_err))
        .collect()
}

/// FlexAlign all pairs between two lists (Cartesian product).
#[pyfunction]
#[pyo3(signature = (queries, targets, n_threads=None, chain=None, fast=false))]
pub(crate) fn flex_align_many_to_many(
    py: Python<'_>,
    queries: &Bound<'_, PyList>,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
    chain: Option<&str>,
    fast: bool,
) -> PyResult<Vec<(usize, usize, PyFlexAlignResult)>> {
    let query_sds = extract_list(queries, chain)?;
    let target_sds = extract_list(targets, chain)?;
    let n = resolve_threads(n_threads);

    let pairs: Vec<(usize, usize)> = (0..query_sds.len())
        .flat_map(|qi| (0..target_sds.len()).map(move |ti| (qi, ti)))
        .collect();

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pairs
                .par_iter()
                .map(|&(qi, ti)| {
                    let r = run_flexalign(&query_sds[qi], &target_sds[ti], fast);
                    (qi, ti, r)
                })
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|(qi, ti, r)| match r {
            Ok(result) => Ok((qi, ti, result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        })
        .collect()
}

// ===========================================================================
// MM-align (multi-chain complex alignment)
// ===========================================================================

/// Per-chain alignment result within an MM-align result.
#[pyclass]
#[derive(Clone)]
pub struct PyChainPairResult {
    /// Chain index in complex 1.
    #[pyo3(get)]
    pub query_chain: usize,
    /// Chain index in complex 2.
    #[pyo3(get)]
    pub target_chain: usize,
    /// TM-score normalized by target chain length.
    #[pyo3(get)]
    pub tm_score: f64,
    /// RMSD of aligned residues.
    #[pyo3(get)]
    pub rmsd: f64,
    /// Number of aligned residues.
    #[pyo3(get)]
    pub n_aligned: usize,
    /// Aligned sequence of chain in complex 1 (with gaps).
    #[pyo3(get)]
    pub aligned_seq_x: String,
    /// Aligned sequence of chain in complex 2 (with gaps).
    #[pyo3(get)]
    pub aligned_seq_y: String,
}

/// Result of MM-align (multi-chain complex alignment).
#[pyclass]
pub struct PyMMAlignResult {
    /// Total complex TM-score.
    #[pyo3(get)]
    pub total_score: f64,
    /// Per-chain-pair results.
    #[pyo3(get)]
    pub chain_pairs: Vec<PyChainPairResult>,
    /// Chain assignment: list of (query_chain_idx, target_chain_idx).
    #[pyo3(get)]
    pub chain_assignments: Vec<(usize, usize)>,
}

#[pymethods]
impl PyMMAlignResult {
    fn __repr__(&self) -> String {
        format!(
            "MMAlignResult(total_score={:.4}, chain_pairs={})",
            self.total_score,
            self.chain_pairs.len(),
        )
    }
}

fn pack_mmalign_result(r: MMAlignResult) -> PyMMAlignResult {
    let chain_pairs: Vec<PyChainPairResult> = r
        .chain_assignments
        .iter()
        .zip(r.per_chain_results.iter())
        .map(|(&(qi, ti), se)| PyChainPairResult {
            query_chain: qi,
            target_chain: ti,
            tm_score: se.tm1,
            rmsd: se.rmsd,
            n_aligned: se.n_ali8,
            aligned_seq_x: se.seq_x_aligned.clone(),
            aligned_seq_y: se.seq_y_aligned.clone(),
        })
        .collect();

    PyMMAlignResult {
        total_score: r.total_score,
        chain_assignments: r.chain_assignments,
        chain_pairs,
    }
}

/// Extract ChainData from a PyPDB.
fn extract_chains(pdb: &PyPDB) -> PyResult<Vec<ChainData>> {
    extract_chains_for_alignment(&pdb.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Align two multi-chain complexes using MM-align.
///
/// Automatically determines chain-to-chain correspondence and computes
/// complex-level TM-score.
#[pyfunction]
#[pyo3(signature = (pdb1, pdb2))]
pub(crate) fn mm_align_pair(
    py: Python<'_>,
    pdb1: &PyPDB,
    pdb2: &PyPDB,
) -> PyResult<PyMMAlignResult> {
    let chains1 = extract_chains(pdb1)?;
    let chains2 = extract_chains(pdb2)?;

    let result = py
        .allow_threads(|| mmalign_complex(&chains1, &chains2))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(pack_mmalign_result(result))
}

/// MM-align one query complex against many target complexes in parallel.
#[pyfunction]
#[pyo3(signature = (query, targets, n_threads=None))]
pub(crate) fn mm_align_one_to_many(
    py: Python<'_>,
    query: &PyPDB,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyMMAlignResult>> {
    let chains1 = extract_chains(query)?;
    let target_chains: Vec<Vec<ChainData>> = targets
        .iter()
        .map(|t| {
            let pdb = t.extract::<PyRef<'_, PyPDB>>()?;
            extract_chains(&pdb)
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            target_chains
                .par_iter()
                .map(|tc| {
                    mmalign_complex(&chains1, tc)
                        .map(pack_mmalign_result)
                        .map_err(|e| e.to_string())
                })
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|r| r.map_err(pyo3::exceptions::PyRuntimeError::new_err))
        .collect()
}

/// MM-align all pairs between two lists of complexes (Cartesian product).
#[pyfunction]
#[pyo3(signature = (queries, targets, n_threads=None))]
pub(crate) fn mm_align_many_to_many(
    py: Python<'_>,
    queries: &Bound<'_, PyList>,
    targets: &Bound<'_, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<(usize, usize, PyMMAlignResult)>> {
    let query_chains: Vec<Vec<ChainData>> = queries
        .iter()
        .map(|q| {
            let pdb = q.extract::<PyRef<'_, PyPDB>>()?;
            extract_chains(&pdb)
        })
        .collect::<PyResult<_>>()?;

    let target_chains: Vec<Vec<ChainData>> = targets
        .iter()
        .map(|t| {
            let pdb = t.extract::<PyRef<'_, PyPDB>>()?;
            extract_chains(&pdb)
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);

    let pairs: Vec<(usize, usize)> = (0..query_chains.len())
        .flat_map(|qi| (0..target_chains.len()).map(move |ti| (qi, ti)))
        .collect();

    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            pairs
                .par_iter()
                .map(|&(qi, ti)| {
                    let r = mmalign_complex(&query_chains[qi], &target_chains[ti])
                        .map(pack_mmalign_result)
                        .map_err(|e| e.to_string());
                    (qi, ti, r)
                })
                .collect::<Vec<_>>()
        })
    });

    results
        .into_iter()
        .map(|(qi, ti, r)| match r {
            Ok(result) => Ok((qi, ti, result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        })
        .collect()
}

// ===========================================================================
// Module registration
// ===========================================================================

#[pymodule]
pub(crate) fn py_align_funcs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TM-align
    m.add_function(wrap_pyfunction!(tm_align_pair, m)?)?;
    m.add_function(wrap_pyfunction!(tm_align_one_to_many, m)?)?;
    m.add_function(wrap_pyfunction!(tm_align_many_to_many, m)?)?;
    // SOI-align
    m.add_function(wrap_pyfunction!(soi_align_pair, m)?)?;
    m.add_function(wrap_pyfunction!(soi_align_one_to_many, m)?)?;
    m.add_function(wrap_pyfunction!(soi_align_many_to_many, m)?)?;
    // FlexAlign
    m.add_function(wrap_pyfunction!(flex_align_pair, m)?)?;
    m.add_function(wrap_pyfunction!(flex_align_one_to_many, m)?)?;
    m.add_function(wrap_pyfunction!(flex_align_many_to_many, m)?)?;
    // MM-align
    m.add_function(wrap_pyfunction!(mm_align_pair, m)?)?;
    m.add_function(wrap_pyfunction!(mm_align_one_to_many, m)?)?;
    m.add_function(wrap_pyfunction!(mm_align_many_to_many, m)?)?;
    // Result types
    m.add_class::<PySoiAlignResult>()?;
    m.add_class::<PyFlexAlignResult>()?;
    m.add_class::<PyMMAlignResult>()?;
    m.add_class::<PyChainPairResult>()?;
    Ok(())
}
