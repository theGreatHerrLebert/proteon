//! PyO3 bindings for ferritin-search → MSA tensor assembly.
//!
//! Exposes a [`PySearchEngine`] class to Python with two operations:
//!
//! - `.search(query)` → list of hit dicts (target_id, score, coords, CIGAR).
//! - `.search_and_build_msa(query, max_seqs, gap_idx)` → numpy-array dict
//!   matching the field shapes of
//!   `packages/ferritin/src/ferritin/sequence_example.py::SequenceExample`,
//!   so the Python caller can splat it into `build_sequence_example(...)`
//!   without copying field by field.

use ferritin_search::alphabet::Alphabet;
use ferritin_search::matrix::SubstitutionMatrix;
use ferritin_search::msa::{MsaAssembly, MsaOptions};
use ferritin_search::search::{SearchEngine as CoreSearchEngine, SearchOptions};
use ferritin_search::sequence::Sequence;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Wraps [`ferritin_search::search::SearchEngine`] so Python can drive it.
///
/// Construction takes the targets and indexing options, building the
/// k-mer index up-front. `.search()` and `.search_and_build_msa()` are
/// per-query.
#[pyclass(name = "SearchEngine", module = "ferritin_connector.py_msa")]
pub struct PySearchEngine {
    inner: CoreSearchEngine,
    alphabet: Alphabet,
}

#[pymethods]
impl PySearchEngine {
    /// Build a search engine over a target corpus.
    ///
    /// `targets` is a list of `(seq_id, sequence_str)` pairs. Sequences
    /// are ASCII protein letters (FASTA-style). Defaults match
    /// upstream MMseqs2 protein-search settings.
    #[new]
    #[pyo3(signature = (
        targets,
        k = 6,
        reduce_to = Some(13),
        bit_factor = 2.0,
        gap_open = -11,
        gap_extend = -1,
        min_score = 0,
        max_prefilter_hits = Some(1000),
        max_results = None,
    ))]
    fn new(
        targets: Vec<(u32, String)>,
        k: usize,
        reduce_to: Option<usize>,
        bit_factor: f32,
        gap_open: i32,
        gap_extend: i32,
        min_score: i32,
        max_prefilter_hits: Option<usize>,
        max_results: Option<usize>,
    ) -> PyResult<Self> {
        let alphabet = Alphabet::protein();
        let matrix = SubstitutionMatrix::blosum62();
        let opts = SearchOptions {
            k,
            reduce_to,
            bit_factor,
            diagonal_score_threshold: 0,
            max_prefilter_hits,
            gap_open,
            gap_extend,
            min_score,
            max_results,
        };
        let target_seqs: Vec<(u32, Sequence)> = targets
            .into_iter()
            .map(|(id, s)| (id, Sequence::from_ascii(alphabet.clone(), s.as_bytes())))
            .collect();
        let inner = CoreSearchEngine::build(target_seqs, &matrix, alphabet.clone(), opts)
            .map_err(|e| PyValueError::new_err(format!("engine build failed: {e}")))?;
        Ok(Self { inner, alphabet })
    }

    /// Number of targets indexed.
    fn target_count(&self) -> usize {
        self.inner.target_count()
    }

    /// Run a single query, return a list of hit dicts sorted by gapped
    /// alignment score descending.
    fn search<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyList>> {
        let q = Sequence::from_ascii(self.alphabet.clone(), query.as_bytes());
        let hits = self.inner.search(&q);
        let list = PyList::empty(py);
        for h in hits {
            let d = PyDict::new(py);
            d.set_item("target_id", h.target_id)?;
            d.set_item("prefilter_score", h.prefilter_score)?;
            d.set_item("best_diagonal", h.best_diagonal)?;
            d.set_item("ungapped_score", h.ungapped_score)?;
            d.set_item("score", h.alignment.score)?;
            d.set_item("query_start", h.alignment.query_start)?;
            d.set_item("query_end", h.alignment.query_end)?;
            d.set_item("target_start", h.alignment.target_start)?;
            d.set_item("target_end", h.alignment.target_end)?;
            d.set_item("cigar", h.alignment.cigar_string())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Run a query and assemble an AF2-style MSA tensor bundle.
    ///
    /// Returns a dict with keys matching `SequenceExample`'s field names:
    ///
    /// - `aatype` — `(L,)` `uint8` query residues encoded.
    /// - `seq_mask` — `(L,)` `float32`, all 1.0.
    /// - `msa` — `(N, L)` `uint8`, row 0 query, rows 1+ homologs in
    ///   query coords. `gap_idx` fills uncovered positions.
    /// - `deletion_matrix` — `(N, L)` `uint8`, AF2 deletion convention.
    /// - `msa_mask` — `(N, L)` `float32`, 1.0 inside alignment range.
    /// - `n_seqs`, `query_len`, `gap_idx` — scalar metadata.
    #[pyo3(signature = (query, max_seqs = 256, gap_idx = 21))]
    fn search_and_build_msa<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        max_seqs: usize,
        gap_idx: u8,
    ) -> PyResult<Bound<'py, PyDict>> {
        let q = Sequence::from_ascii(self.alphabet.clone(), query.as_bytes());
        let opts = MsaOptions { max_seqs, gap_idx };
        let msa = self.inner.search_and_build_msa(&q, opts);
        msa_to_pydict(py, msa)
    }
}

fn msa_to_pydict<'py>(py: Python<'py>, msa: MsaAssembly) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let n = msa.n_seqs;
    let l = msa.query_len;

    dict.set_item("query_len", l)?;
    dict.set_item("n_seqs", n)?;
    dict.set_item("gap_idx", msa.gap_idx)?;
    dict.set_item("aatype", msa.aatype.into_pyarray(py))?;
    dict.set_item("seq_mask", msa.seq_mask.into_pyarray(py))?;

    let msa_flat: Vec<u8> = msa.msa.into_iter().flatten().collect();
    dict.set_item(
        "msa",
        PyArray1::from_vec(py, msa_flat)
            .reshape([n, l])
            .map_err(|e| PyValueError::new_err(format!("reshape msa: {e}")))?,
    )?;

    let del_flat: Vec<u8> = msa.deletion_matrix.into_iter().flatten().collect();
    dict.set_item(
        "deletion_matrix",
        PyArray1::from_vec(py, del_flat)
            .reshape([n, l])
            .map_err(|e| PyValueError::new_err(format!("reshape deletion_matrix: {e}")))?,
    )?;

    let mask_flat: Vec<f32> = msa.msa_mask.into_iter().flatten().collect();
    dict.set_item(
        "msa_mask",
        PyArray1::from_vec(py, mask_flat)
            .reshape([n, l])
            .map_err(|e| PyValueError::new_err(format!("reshape msa_mask: {e}")))?,
    )?;

    Ok(dict)
}

#[pymodule]
pub fn py_msa(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySearchEngine>()?;
    Ok(())
}
