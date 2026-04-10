//! PyO3 bindings for structural alphabet encoding.

use ferritin_align::search::alphabet::{encode_structure, BackboneAtoms};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::parallel::resolve_threads;
use crate::py_pdb::PyPDB;

fn build_pool(n_threads: usize) -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if n_threads > 0 {
        builder = builder.num_threads(n_threads);
    }
    builder.build().expect("failed to build rayon thread pool")
}

const AA_ALPHABET: &[u8] = b"ARNDCQEGHILKMFPSTWYVX";
const SA_ALPHABET: &[u8] = b"ACDEFGHIKLMNPQRSTVWYX";

const BLOSUM62: [[i16; 21]; 21] = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const MAT3DI: [[i16; 21]; 21] = [
    [6, -3, 1, 2, 3, -2, -2, -7, -3, -3, -10, -5, -1, 1, -4, -7, -5, -6, 0, -2, 0],
    [-3, 6, -2, -8, -5, -4, -4, -12, -13, 1, -14, 0, 0, 1, -1, 0, -8, 1, -7, -9, 0],
    [1, -2, 4, -3, 0, 1, 1, -3, -5, -4, -5, -2, 1, -1, -1, -4, -2, -3, -2, -2, 0],
    [2, -8, -3, 9, -2, -7, -4, -12, -10, -7, -17, -8, -6, -3, -8, -10, -10, -13, -6, -3, 0],
    [3, -5, 0, -2, 7, -3, -3, -5, 1, -3, -9, -5, -2, 2, -5, -8, -3, -7, 4, -4, 0],
    [-2, -4, 1, -7, -3, 6, 3, 0, -7, -7, -1, -2, -2, -4, 3, -3, 4, -6, -4, -2, 0],
    [-2, -4, 1, -4, -3, 3, 6, -4, -7, -6, -6, 0, -1, -3, 1, -3, -1, -5, -5, 3, 0],
    [-7, -12, -3, -12, -5, 0, -4, 8, -5, -11, 7, -7, -6, -6, -3, -9, 6, -12, -5, -8, 0],
    [-3, -13, -5, -10, 1, -7, -7, -5, 9, -11, -8, -12, -6, -5, -9, -14, -5, -15, 5, -8, 0],
    [-3, 1, -4, -7, -3, -7, -6, -11, -11, 6, -16, -3, -2, 2, -4, -4, -9, 0, -8, -9, 0],
    [-10, -14, -5, -17, -9, -1, -6, 7, -8, -16, 10, -9, -9, -10, -5, -10, 3, -16, -6, -9, 0],
    [-5, 0, -2, -8, -5, -2, 0, -7, -12, -3, -9, 7, 0, -2, 2, 3, -4, 0, -8, -5, 0],
    [-1, 0, 1, -6, -2, -2, -1, -6, -6, -2, -9, 0, 4, 0, 0, -2, -4, 0, -4, -5, 0],
    [1, 1, -1, -3, 2, -4, -3, -6, -5, 2, -10, -2, 0, 5, -2, -4, -5, -1, -2, -5, 0],
    [-4, -1, -1, -8, -5, 3, 1, -3, -9, -4, -5, 2, 0, -2, 6, 2, 0, -1, -6, -3, 0],
    [-7, 0, -4, -10, -8, -3, -3, -9, -14, -4, -10, 3, -2, -4, 2, 6, -6, 0, -11, -9, 0],
    [-5, -8, -2, -10, -3, 4, -1, 6, -5, -9, 3, -4, -4, -5, 0, -6, 8, -9, -5, -5, 0],
    [-6, 1, -3, -13, -7, -6, -5, -12, -15, 0, -16, 0, 0, -1, -1, 0, -9, 3, -10, -11, 0],
    [0, -7, -2, -6, 4, -4, -5, -5, 5, -8, -6, -8, -4, -2, -6, -11, -5, -10, 8, -6, 0],
    [-2, -9, -2, -3, -4, -2, 3, -8, -8, -9, -9, -5, -5, -5, -3, -9, -5, -11, -6, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

fn alphabet_index(alphabet: &[u8], ch: u8) -> usize {
    alphabet
        .iter()
        .position(|&aa| aa == ch.to_ascii_uppercase())
        .unwrap_or(alphabet.len() - 1)
}

fn ungapped_diagonal_score(query_sa: &str, query_aa: &str, target_sa: &str, target_aa: &str) -> f64 {
    let q_sa = query_sa.as_bytes();
    let q_aa = query_aa.as_bytes();
    let t_sa = target_sa.as_bytes();
    let t_aa = target_aa.as_bytes();
    let q_len = q_sa.len().min(q_aa.len());
    let t_len = t_sa.len().min(t_aa.len());
    if q_len == 0 || t_len == 0 {
        return 0.0;
    }

    let mut best = 0_i32;
    for diagonal in -((t_len as isize) - 1)..(q_len as isize) {
        let q_start = diagonal.max(0) as usize;
        let t_start = (-diagonal).max(0) as usize;
        let length = (q_len - q_start).min(t_len - t_start);
        let mut score = 0_i32;
        for offset in 0..length {
            let qi = q_start + offset;
            let ti = t_start + offset;
            let q3 = alphabet_index(SA_ALPHABET, q_sa[qi]);
            let t3 = alphabet_index(SA_ALPHABET, t_sa[ti]);
            let qa = alphabet_index(AA_ALPHABET, q_aa[qi]);
            let ta = alphabet_index(AA_ALPHABET, t_aa[ti]);
            score = (score + MAT3DI[q3][t3] as i32 + BLOSUM62[qa][ta] as i32).max(0);
            best = best.max(score);
        }
    }
    best as f64 / q_len.max(t_len) as f64
}

#[derive(Debug, Clone)]
struct ResidueBackboneRecord {
    chain_id: String,
    residue_name: String,
    residue_number: i32,
    insertion_code: Option<String>,
    n: [f64; 3],
    ca: [f64; 3],
    c: [f64; 3],
    cb: [f64; 3],
}

fn extract_backbone_records(pdb: &pdbtbx::PDB) -> Vec<ResidueBackboneRecord> {
    let mut records = Vec::new();

    for chain in pdb.chains() {
        let chain_id = chain.id().to_string();
        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if !is_aa {
                continue;
            }

            let mut n = [f64::NAN; 3];
            let mut ca = [f64::NAN; 3];
            let mut c = [f64::NAN; 3];
            let mut cb = [f64::NAN; 3];

            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                let pos = [x, y, z];
                match atom.name().trim() {
                    "N" => n = pos,
                    "CA" => ca = pos,
                    "C" => c = pos,
                    "CB" => cb = pos,
                    _ => {}
                }
            }

            records.push(ResidueBackboneRecord {
                chain_id: chain_id.clone(),
                residue_name: residue.name().unwrap_or("UNK").to_string(),
                residue_number: residue.serial_number() as i32,
                insertion_code: residue.insertion_code().map(|s| s.to_string()),
                n,
                ca,
                c,
                cb,
            });
        }
    }

    records
}

fn encode_records(records: &[ResidueBackboneRecord]) -> ferritin_align::search::alphabet::AlphabetResult {
    let ca: Vec<[f64; 3]> = records.iter().map(|r| r.ca).collect();
    let n: Vec<[f64; 3]> = records.iter().map(|r| r.n).collect();
    let c: Vec<[f64; 3]> = records.iter().map(|r| r.c).collect();
    let cb: Vec<[f64; 3]> = records.iter().map(|r| r.cb).collect();

    let atoms = BackboneAtoms {
        ca: &ca,
        n: &n,
        c: &c,
        cb: &cb,
    };
    encode_structure(&atoms)
}

fn build_result_dict<'py>(
    py: Python<'py>,
    records: &[ResidueBackboneRecord],
    result: ferritin_align::search::alphabet::AlphabetResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("alphabet", result.to_string())?;
    dict.set_item("states", result.states.into_pyarray(py))?;
    dict.set_item("valid_mask", result.valid.into_pyarray(py))?;
    dict.set_item("partners", result.partners.into_pyarray(py))?;

    let n_rows = result.features.len();
    let flat_features: Vec<f64> = result
        .features
        .into_iter()
        .flat_map(|row| row.into_iter())
        .collect();
    let features = PyArray1::from_vec(py, flat_features)
        .reshape([n_rows, 10])
        .expect("reshape alphabet features to Nx10");
    dict.set_item("features", features)?;

    let chain_ids: Vec<String> = records.iter().map(|r| r.chain_id.clone()).collect();
    let residue_names: Vec<String> = records.iter().map(|r| r.residue_name.clone()).collect();
    let residue_numbers: Vec<i32> = records.iter().map(|r| r.residue_number).collect();
    let insertion_codes = PyList::new(
        py,
        records
            .iter()
            .map(|r| r.insertion_code.as_deref())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("chain_ids", chain_ids)?;
    dict.set_item("residue_names", residue_names)?;
    dict.set_item("residue_numbers", residue_numbers.into_pyarray(py))?;
    dict.set_item("insertion_codes", insertion_codes)?;

    Ok(dict)
}

/// Encode a structure into a learned 20-state structural alphabet.
///
/// Returns a dict containing:
///   states: N-length uint8 array of state indices
///   alphabet: N-length alphabet string
///   valid_mask: N-length bool array
///   partners: N-length int32 array of nearest-neighbor residue indices
///   features: Nx10 float64 array of geometric features
///   chain_ids, residue_names, residue_numbers, insertion_codes metadata
#[pyfunction]
pub fn encode_alphabet(py: Python<'_>, pdb: &PyPDB) -> PyResult<PyObject> {
    let records = extract_backbone_records(&pdb.inner);
    let result = py.allow_threads(|| encode_records(&records));
    let dict = build_result_dict(py, &records, result)?;
    Ok(dict.into_any().unbind())
}

/// Encode many structures into the structural alphabet in parallel.
///
/// Returns a list of dicts with the same schema as `encode_alphabet`.
#[pyfunction]
#[pyo3(signature = (structures, n_threads=None))]
pub fn batch_encode_alphabet<'py>(
    py: Python<'py>,
    structures: &Bound<'py, PyList>,
    n_threads: Option<i32>,
) -> PyResult<Vec<PyObject>> {
    let all_records: Vec<Vec<ResidueBackboneRecord>> = structures
        .iter()
        .map(|item| {
            let pdb = item.extract::<PyRef<'_, PyPDB>>()?;
            Ok(extract_backbone_records(&pdb.inner))
        })
        .collect::<PyResult<_>>()?;

    let n = resolve_threads(n_threads);
    let results = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            all_records
                .par_iter()
                .map(|records| encode_records(records))
                .collect::<Vec<_>>()
        })
    });

    all_records
        .iter()
        .zip(results)
        .map(|(records, result)| {
            let dict = build_result_dict(py, records, result)?;
            Ok(dict.into_any().unbind())
        })
        .collect()
}

/// Score candidate target strings with Foldseek-style ungapped diagonal scoring.
#[pyfunction]
#[pyo3(signature = (query_alphabet, query_aa, target_alphabets, target_aas, n_threads=None))]
pub fn diagonal_rescore_batch(
    py: Python<'_>,
    query_alphabet: String,
    query_aa: String,
    target_alphabets: Vec<String>,
    target_aas: Vec<String>,
    n_threads: Option<i32>,
) -> PyResult<Vec<f64>> {
    if target_alphabets.len() != target_aas.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "target_alphabets and target_aas must have the same length, got {} and {}",
            target_alphabets.len(),
            target_aas.len()
        )));
    }

    let n = resolve_threads(n_threads);
    let scores = py.allow_threads(|| {
        let pool = build_pool(n);
        pool.install(|| {
            target_alphabets
                .par_iter()
                .zip(target_aas.par_iter())
                .map(|(target_alphabet, target_aa)| {
                    ungapped_diagonal_score(
                        &query_alphabet,
                        &query_aa,
                        target_alphabet,
                        target_aa,
                    )
                })
                .collect::<Vec<_>>()
        })
    });
    Ok(scores)
}

#[pymodule]
pub fn py_search(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_alphabet, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode_alphabet, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_rescore_batch, m)?)?;
    Ok(())
}
