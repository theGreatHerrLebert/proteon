//! End-to-end search orchestrator.
//!
//! Ties the four prefilter sub-phases plus gapped alignment together
//! into a single API:
//!
//! 1. Reduce target sequences to a smaller alphabet (2.3c) and build a
//!    k-mer index over them (2.2).
//! 2. Per query: reduce, run diagonal-voting prefilter (2.3a, optionally
//!    with similar-k-mer expansion 2.3b), ungapped extension along the
//!    best diagonal (2.3d), Smith-Waterman gapped alignment (2.4) on
//!    the full alphabet.
//! 3. Sort hits by gapped alignment score; apply per-query result limit.
//!
//! [`SearchEngine`] is built once over a target corpus and reused across
//! many queries — that's the production usage pattern (one DB, many
//! queries) and matches upstream's `createindex` + `search` split.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::alphabet::Alphabet;
use crate::db::{DBReader, DbError};
use crate::gapped::{smith_waterman, GappedAlignment};
use crate::kmer::{KmerEncoder, KmerIndex, KmerIndexError};
use crate::kmer_generator::widen_to_i32;
use crate::matrix::SubstitutionMatrix;
use crate::prefilter::{diagonal_prefilter, PrefilterHit, PrefilterOptions};
use crate::reduced_alphabet::ReducedAlphabet;
use crate::sequence::Sequence;
use crate::ungapped::ungapped_alignment;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("index build failed: {0}")]
    IndexBuild(#[from] KmerIndexError),
    #[error("reduced-alphabet construction failed: invalid reduce_to or unknown index")]
    BadReduction,
}

/// Error type for [`SearchEngine::build_from_mmseqs_db`].
///
/// Distinct from [`SearchError`] because opening the DB and building the
/// engine are independently fallible and callers may want to handle them
/// separately (e.g. "missing DB" vs "bad k value").
#[derive(Debug, Error)]
pub enum BuildFromDbError {
    #[error("mmseqs DB open failed: {0}")]
    DbOpen(#[from] DbError),
    #[error("search engine build failed: {0}")]
    Build(#[from] SearchError),
}

/// Tuning knobs for [`SearchEngine`].
///
/// Defaults mirror common MMseqs2 protein-search settings: k=6, BLOSUM62
/// at bit_factor=2, gap_open=-11, gap_extend=-1, alphabet reduction to
/// 13 letters. They're intentionally conservative — tighten / loosen for
/// your workload.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub k: usize,
    /// `Some(n)` reduces the alphabet to `n` letters before building
    /// the k-mer index (collapses 21^k offsets to n^k); `None` indexes
    /// in the full alphabet.
    pub reduce_to: Option<usize>,
    /// Score scaling for [`SubstitutionMatrix::to_integer_matrix`].
    pub bit_factor: f32,
    /// Drop prefilter candidates whose diagonal k-mer count falls below this.
    pub diagonal_score_threshold: u32,
    /// Cap the number of prefilter candidates passed to ungapped + gapped.
    pub max_prefilter_hits: Option<usize>,
    /// Affine gap penalties for Smith-Waterman.
    pub gap_open: i32,
    pub gap_extend: i32,
    /// Drop final hits whose gapped alignment score falls below this.
    pub min_score: i32,
    /// Cap the number of hits returned per query.
    pub max_results: Option<usize>,
    /// Use the GPU-batched dispatch path when the `cuda` feature is
    /// compiled in and a device is detected at runtime. `false` forces
    /// the CPU path even on GPU hosts — useful for debugging or
    /// reproducing exact upstream ordering. Ignored when the feature
    /// is not compiled.
    pub use_gpu: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 6,
            reduce_to: Some(13),
            bit_factor: 2.0,
            diagonal_score_threshold: 0,
            max_prefilter_hits: Some(1000),
            gap_open: -11,
            gap_extend: -1,
            min_score: 0,
            max_results: None,
            use_gpu: true,
        }
    }
}

/// One search hit: a target sequence and the full chain of intermediate
/// scores for it.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub target_id: u32,
    pub prefilter_score: u32,
    pub best_diagonal: i32,
    pub ungapped_score: i32,
    pub alignment: GappedAlignment,
}

/// Pre-built search engine over a fixed target corpus.
///
/// Construction does the expensive work once: alphabet reduction,
/// k-mer index build, integer score matrix conversion, hashmap from
/// `seq_id → target sequence bytes` for fast lookup during query.
pub struct SearchEngine {
    /// Target sequences in full-alphabet encoding, indexed by seq_id.
    targets_full: HashMap<u32, Vec<u8>>,
    /// K-mer index over reduced-alphabet (or full if no reduction) targets.
    index: KmerIndex,
    /// Flat alphabet_size² i32 score matrix for ungapped + gapped.
    matrix_int: Vec<i32>,
    full_alphabet_size: usize,
    /// Reducer used for query encoding before prefilter; `None` means the
    /// k-mer index is in the full alphabet.
    reducer: Option<ReducedAlphabet>,
    /// Skip index passed to k-mer iteration (X in reduced or full space).
    skip_idx: u8,
    opts: SearchOptions,
}

impl SearchEngine {
    /// Build the engine over a target corpus.
    ///
    /// `targets` is consumed; the engine retains only the encoded byte
    /// vectors. `alphabet` must match the encoding used when constructing
    /// the `Sequence` values (typically `Alphabet::protein()`).
    pub fn build(
        targets: impl IntoIterator<Item = (u32, Sequence)>,
        matrix: &SubstitutionMatrix,
        alphabet: Alphabet,
        opts: SearchOptions,
    ) -> Result<Self, SearchError> {
        let full_alphabet_size = alphabet.size();
        let x_full = alphabet.encode(b'X');

        // Materialize targets into the lookup map and a flat list for indexing.
        let mut targets_full: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut order: Vec<(u32, Vec<u8>)> = Vec::new();
        for (id, seq) in targets {
            targets_full.insert(id, seq.data.clone());
            order.push((id, seq.data));
        }

        // Optional alphabet reduction. When reducing, we keep X as a
        // singleton class (so the skip semantic survives — the
        // unknown_reduced_idx is what we pass to k-mer iter as skip_idx).
        let reducer = match opts.reduce_to {
            Some(r) => Some(
                ReducedAlphabet::from_matrix(matrix, r, Some(x_full))
                    .ok_or(SearchError::BadReduction)?,
            ),
            None => None,
        };

        let (kmer_alphabet_size, skip_idx, indexed_targets): (usize, u8, Vec<(u32, Vec<u8>)>) =
            match &reducer {
                Some(r) => {
                    let skip = r.unknown_reduced_idx.ok_or(SearchError::BadReduction)?;
                    let reduced = order
                        .iter()
                        .map(|(id, s)| (*id, r.reduce_sequence(s)))
                        .collect();
                    (r.reduced_size, skip, reduced)
                }
                None => (full_alphabet_size, x_full, order),
            };

        let encoder = KmerEncoder::new(kmer_alphabet_size as u32, opts.k);
        let pairs: Vec<(u32, &[u8])> = indexed_targets
            .iter()
            .map(|(id, s)| (*id, s.as_slice()))
            .collect();
        let index = KmerIndex::build(encoder, pairs, skip_idx)?;

        let matrix_int = widen_to_i32(&matrix.to_integer_matrix(opts.bit_factor, 0.0));

        Ok(Self {
            targets_full,
            index,
            matrix_int,
            full_alphabet_size,
            reducer,
            skip_idx,
            opts,
        })
    }

    /// Build the engine from an on-disk MMseqs2-compatible DB.
    ///
    /// Opens the DB at `prefix` (anything [`crate::db::DBReader`] accepts
    /// — e.g. output of `mmseqs createdb` or ferritin-search's own
    /// [`crate::db::DBWriter`]) and streams its sequences into
    /// [`SearchEngine::build`]. Sequence ids are taken straight from the
    /// DB's index keys so lookups round-trip against the DB's
    /// `.lookup` file if present.
    ///
    /// Payloads are ASCII and may carry a trailing `\n` before the
    /// `\0` terminator. [`crate::db::DBReader::get_payload`] strips the
    /// `\0`; [`Sequence::from_ascii`] strips whitespace including the
    /// `\n`, so each entry is passed through exactly once without any
    /// intermediate Python `List[(id, str)]` materialization.
    pub fn build_from_mmseqs_db(
        prefix: impl AsRef<Path>,
        matrix: &SubstitutionMatrix,
        alphabet: Alphabet,
        opts: SearchOptions,
    ) -> Result<Self, BuildFromDbError> {
        let reader = DBReader::open(prefix)?;
        // Encode all payloads up-front, then drop the reader — releases
        // the DB's bulk `data: Vec<u8>` buffer before `build` allocates
        // its own encoded targets map. Peak memory is raw-DB + encoded
        // sequences briefly; once `reader` drops it's just the encoded
        // copies that `build` retains. No Python `List[(id, str)]`
        // intermediate either way.
        let targets: Vec<(u32, Sequence)> = reader
            .index
            .iter()
            .map(|entry| {
                let payload = reader.get_payload(entry);
                (entry.key, Sequence::from_ascii(alphabet.clone(), payload))
            })
            .collect();
        drop(reader);
        Ok(Self::build(targets, matrix, alphabet, opts)?)
    }

    /// Run a single query against the indexed corpus.
    ///
    /// Returns hits sorted by gapped alignment score descending.
    ///
    /// Dispatches to [`SearchEngine::search_gpu`] when the `cuda` feature
    /// is compiled in, `opts.use_gpu` is `true`, and a device is
    /// detected at runtime. Silent CPU fallback otherwise.
    pub fn search(&self, query: &Sequence) -> Vec<SearchHit> {
        let query_for_prefilter = match &self.reducer {
            Some(r) => r.reduce_sequence(&query.data),
            None => query.data.clone(),
        };

        let prefilter_hits = diagonal_prefilter(
            &self.index,
            &query_for_prefilter,
            self.skip_idx,
            &PrefilterOptions {
                score_threshold: self.opts.diagonal_score_threshold,
                max_hits: self.opts.max_prefilter_hits,
                exclude_self: None,
            },
        );

        #[cfg(feature = "cuda")]
        {
            if self.opts.use_gpu && crate::gpu::is_available() {
                return self.search_gpu(&query.data, &prefilter_hits);
            }
        }

        self.search_cpu(&query.data, &prefilter_hits)
    }

    /// CPU reference path. Also the fallback when GPU dispatch errors
    /// mid-flight. Parity target for the GPU path — same inputs must
    /// produce the same `SearchHit` ordering.
    fn search_cpu(&self, query: &[u8], prefilter_hits: &[PrefilterHit]) -> Vec<SearchHit> {
        let mut results: Vec<SearchHit> = Vec::new();
        for ph in prefilter_hits {
            let target = match self.targets_full.get(&ph.seq_id) {
                Some(t) => t,
                None => continue,
            };

            let ungapped = match ungapped_alignment(
                query,
                target,
                ph.best_diagonal,
                &self.matrix_int,
                self.full_alphabet_size,
            ) {
                Some(u) => u,
                None => continue,
            };

            let gapped = match smith_waterman(
                query,
                target,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Some(a) if a.score >= self.opts.min_score => a,
                _ => continue,
            };

            results.push(SearchHit {
                target_id: ph.seq_id,
                prefilter_score: ph.diagonal_score,
                best_diagonal: ph.best_diagonal,
                ungapped_score: ungapped.score,
                alignment: gapped,
            });
        }

        results.sort_by(|a, b| b.alignment.score.cmp(&a.alignment.score));
        if let Some(limit) = self.opts.max_results {
            results.truncate(limit);
        }
        results
    }

    /// GPU-batched path: dispatch ungapped + Smith-Waterman scoring
    /// across all prefilter candidates in two batched GPU calls, then
    /// run CPU `smith_waterman` only on the surviving top-K to recover
    /// CIGAR traceback (the GPU score-batch kernel deliberately skips
    /// traceback — see [`crate::gpu::sw`]).
    ///
    /// On any GPU infrastructure error, falls back to the full CPU
    /// path for this query so correctness is never at risk.
    #[cfg(feature = "cuda")]
    fn search_gpu(&self, query: &[u8], prefilter_hits: &[PrefilterHit]) -> Vec<SearchHit> {
        use crate::gpu::{diagonal, sw};

        struct Candidate<'a> {
            target: &'a [u8],
            seq_id: u32,
            prefilter_score: u32,
            best_diagonal: i32,
        }

        let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(prefilter_hits.len());
        for ph in prefilter_hits {
            if let Some(target) = self.targets_full.get(&ph.seq_id) {
                candidates.push(Candidate {
                    target: target.as_slice(),
                    seq_id: ph.seq_id,
                    prefilter_score: ph.diagonal_score,
                    best_diagonal: ph.best_diagonal,
                });
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }

        let pairs: Vec<(&[u8], i32)> = candidates
            .iter()
            .map(|c| (c.target, c.best_diagonal))
            .collect();

        // Batched ungapped extension. Err → full CPU fallback.
        let ungapped = match diagonal::ungapped_alignment_batch_gpu(
            query,
            &pairs,
            &self.matrix_int,
            self.full_alphabet_size,
        ) {
            Ok(u) => u,
            Err(e) => {
                eprintln!(
                    "[ferritin-search] GPU ungapped batch failed; CPU fallback: {e:#}"
                );
                return self.search_cpu(query, prefilter_hits);
            }
        };

        // Keep only ungapped survivors (CPU path's "skip on None" semantic).
        let mut surviving: Vec<(usize, i32)> = Vec::with_capacity(candidates.len());
        for (i, u) in ungapped.iter().enumerate() {
            if let Some(hit) = u {
                surviving.push((i, hit.score));
            }
        }
        if surviving.is_empty() {
            return Vec::new();
        }

        let surviving_targets: Vec<&[u8]> = surviving.iter().map(|(i, _)| candidates[*i].target).collect();

        // Batched SW score+endpoint. Dispatch by query length:
        //   query_len ≤ 256              → warp singletile (4.5a), fastest
        //   256 < query_len ≤ 2048       → warp multitile (4.5b)
        //   query_len > 2048             → thread-per-pair kernel (sw)
        //
        // All three are batched GPU kernels; any infrastructure error
        // falls the whole query back to the CPU path for correctness.
        use crate::gpu::pssm_sw_warp::{self, MAX_QUERY_LEN as SINGLETILE_MAX_Q};
        use crate::gpu::pssm_sw_warp_multitile::{
            self, MAX_QUERY_LEN as MULTITILE_MAX_Q,
        };
        use crate::pssm::Pssm;
        let sw_scores = if query.len() <= SINGLETILE_MAX_Q {
            let pssm = Pssm::build(query, &self.matrix_int, self.full_alphabet_size);
            match pssm_sw_warp::pssm_sw_warp_batch_gpu(
                &pssm,
                &surviving_targets,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[ferritin-search] GPU warp singletile SW failed; CPU fallback: {e:#}"
                    );
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        } else if query.len() <= MULTITILE_MAX_Q {
            let pssm = Pssm::build(query, &self.matrix_int, self.full_alphabet_size);
            match pssm_sw_warp_multitile::pssm_sw_warp_multitile_batch_gpu(
                &pssm,
                &surviving_targets,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[ferritin-search] GPU warp multitile SW failed; CPU fallback: {e:#}"
                    );
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        } else {
            match sw::smith_waterman_score_batch_gpu(
                query,
                &surviving_targets,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "[ferritin-search] GPU SW batch failed; CPU fallback: {e:#}"
                    );
                    return self.search_cpu(query, prefilter_hits);
                }
            }
        };

        // Threshold + rank on GPU scores so the expensive CPU traceback
        // only runs on the final top-K. This is the whole point of the
        // GPU path: one cheap batched score sweep, then targeted CPU
        // work — mirrors how upstream MMseqs2 hands off to CPU for output.
        let mut ranked: Vec<(usize, i32, i32)> = Vec::with_capacity(surviving.len()); // (idx_in_surviving, sw_score, ungapped_score)
        for (pos, (_, ungapped_score)) in surviving.iter().enumerate() {
            let sw_score = sw_scores[pos].score;
            if sw_score >= self.opts.min_score {
                ranked.push((pos, sw_score, *ungapped_score));
            }
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(limit) = self.opts.max_results {
            ranked.truncate(limit);
        }

        // CPU traceback only on the retained hits.
        let mut results: Vec<SearchHit> = Vec::with_capacity(ranked.len());
        for (pos, _, ungapped_score) in ranked {
            let (cand_idx, _) = surviving[pos];
            let c = &candidates[cand_idx];
            let gapped = match smith_waterman(
                query,
                c.target,
                &self.matrix_int,
                self.full_alphabet_size,
                self.opts.gap_open,
                self.opts.gap_extend,
            ) {
                Some(a) if a.score >= self.opts.min_score => a,
                _ => continue,
            };
            results.push(SearchHit {
                target_id: c.seq_id,
                prefilter_score: c.prefilter_score,
                best_diagonal: c.best_diagonal,
                ungapped_score,
                alignment: gapped,
            });
        }
        results
    }

    /// Number of targets indexed.
    pub fn target_count(&self) -> usize {
        self.targets_full.len()
    }

    /// Look up an indexed target's full-alphabet bytes by `seq_id`.
    /// Used by [`crate::msa::assemble_msa`] to project hits into the
    /// query coordinate frame.
    pub fn target_bytes(&self, seq_id: u32) -> Option<&[u8]> {
        self.targets_full.get(&seq_id).map(Vec::as_slice)
    }

    /// Convenience: run [`SearchEngine::search`] for the given query and
    /// pipe the hits straight into [`crate::msa::assemble_msa`], returning
    /// an AF2-style MSA tensor bundle. Equivalent to:
    ///
    /// ```ignore
    /// let hits = engine.search(&query);
    /// assemble_msa(&query, &hits, |id| engine.target_bytes(id), opts)
    /// ```
    pub fn search_and_build_msa(
        &self,
        query: &Sequence,
        msa_opts: crate::msa::MsaOptions,
    ) -> crate::msa::MsaAssembly {
        let hits = self.search(query);
        crate::msa::assemble_msa(query, &hits, |id| self.target_bytes(id), msa_opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_and_matrix() -> (Alphabet, SubstitutionMatrix) {
        (Alphabet::protein(), SubstitutionMatrix::blosum62())
    }

    #[test]
    fn self_search_finds_self_as_top_hit() {
        // Three distinct proteins; querying any one of them must rank
        // that one as the top hit.
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let opts = SearchOptions { k: 3, reduce_to: Some(13), ..Default::default() };
        let engine = SearchEngine::build(targets.clone(), &m, alpha.clone(), opts).unwrap();
        for (qid, seq) in &targets {
            let hits = engine.search(seq);
            assert!(!hits.is_empty(), "query {qid} returned no hits");
            assert_eq!(
                hits[0].target_id, *qid,
                "query {qid}'s top hit was {} (expected self)",
                hits[0].target_id,
            );
        }
    }

    #[test]
    fn search_returns_hits_sorted_by_alignment_score_desc() {
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MKLVRQPSTNLKACDFGHIY".as_slice()),
            (2u32, b"WWWWWWWWWWWWWWWWWWWW".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let opts = SearchOptions { k: 3, reduce_to: Some(13), ..Default::default() };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();

        let query = Sequence::from_ascii(alpha, b"MKLVRQPSTNLKACDFGHIY");
        let hits = engine.search(&query);
        for w in hits.windows(2) {
            assert!(
                w[0].alignment.score >= w[1].alignment.score,
                "hits not sorted by gapped score desc",
            );
        }
    }

    #[test]
    fn search_respects_min_score_threshold() {
        let (alpha, m) = alpha_and_matrix();
        let seqs = [(1u32, b"MKLVRQPSTNL".as_slice())];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();

        // Self-search with reachable score → finds it.
        let opts1 = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 0,
            ..Default::default()
        };
        let engine = SearchEngine::build(
            targets.clone(),
            &m,
            alpha.clone(),
            opts1,
        )
        .unwrap();
        let q = Sequence::from_ascii(alpha.clone(), b"MKLVRQPSTNL");
        assert!(!engine.search(&q).is_empty());

        // Same search with absurdly high threshold → no hits.
        let opts2 = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 100_000,
            ..Default::default()
        };
        let engine2 =
            SearchEngine::build(targets, &m, alpha.clone(), opts2).unwrap();
        assert!(engine2.search(&q).is_empty());
    }

    #[test]
    fn search_respects_max_results_cap() {
        // Five identical targets with distinct ids → all should self-match
        // any query; max_results=2 keeps just the top 2.
        let (alpha, m) = alpha_and_matrix();
        let seq = b"MKLVRQPSTNLAACDF".as_slice();
        let targets: Vec<(u32, Sequence)> = (1..=5)
            .map(|id| (id as u32, Sequence::from_ascii(alpha.clone(), seq)))
            .collect();

        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            max_results: Some(2),
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, seq);
        let hits = engine.search(&q);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn search_works_without_reduction() {
        // reduce_to: None → k-mer index lives in the full 21-letter
        // alphabet. Smaller k keeps table_size sane.
        let (alpha, m) = alpha_and_matrix();
        let targets: Vec<(u32, Sequence)> =
            vec![(1u32, Sequence::from_ascii(alpha.clone(), b"MKLVRQPSTNLAACDF"))];
        let opts = SearchOptions {
            k: 3,
            reduce_to: None,
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, b"MKLVRQPSTNLAACDF");
        let hits = engine.search(&q);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].target_id, 1);
    }

    /// End-to-end parity: GPU dispatch path returns the same hits in
    /// the same order as the CPU path. Skips when no GPU is present.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_search_matches_cpu_search() {
        if !crate::gpu::is_available() {
            eprintln!("SKIP gpu_search_matches_cpu_search: no GPU available");
            return;
        }
        let (alpha, m) = alpha_and_matrix();
        let seqs = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
            (4u32, b"MKLVRQPSTNLKACDFGHIYMKLVRQPSTNLKACDFGHIY".as_slice()),
            (5u32, b"WWWWWWWWWWWWWWWWWWWW".as_slice()),
        ];
        let targets: Vec<(u32, Sequence)> = seqs
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();

        let opts_cpu = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            use_gpu: false,
            ..Default::default()
        };
        let opts_gpu = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            use_gpu: true,
            ..Default::default()
        };
        let engine_cpu =
            SearchEngine::build(targets.clone(), &m, alpha.clone(), opts_cpu).unwrap();
        let engine_gpu =
            SearchEngine::build(targets.clone(), &m, alpha.clone(), opts_gpu).unwrap();

        for (qid, seq) in &targets {
            let cpu = engine_cpu.search(seq);
            let gpu = engine_gpu.search(seq);
            assert_eq!(
                cpu.len(),
                gpu.len(),
                "query {qid}: CPU returned {} hits, GPU returned {}",
                cpu.len(),
                gpu.len(),
            );
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert_eq!(
                    c.target_id, g.target_id,
                    "query {qid} pos {i}: target_id CPU={} GPU={}",
                    c.target_id, g.target_id,
                );
                assert_eq!(
                    c.alignment.score, g.alignment.score,
                    "query {qid} pos {i}: score CPU={} GPU={}",
                    c.alignment.score, g.alignment.score,
                );
                assert_eq!(c.ungapped_score, g.ungapped_score);
                assert_eq!(c.alignment.query_end, g.alignment.query_end);
                assert_eq!(c.alignment.target_end, g.alignment.target_end);
            }
        }
    }

    #[test]
    fn build_from_mmseqs_db_round_trips_against_in_memory_build() {
        use crate::db::{DBWriter, Dbtype};
        use tempfile::tempdir;

        // Canonical DB payload format: bytes + trailing `\n\0` that
        // `DBWriter::write_entry` appends. `build_from_mmseqs_db` must
        // strip the `\0` (via `get_payload`) and let `Sequence::from_ascii`
        // drop the `\n`, producing the same encoded sequences as an
        // in-memory `SearchEngine::build` over identical inputs.
        let dir = tempdir().unwrap();
        let prefix = dir.path().join("targets");
        let mut w = DBWriter::create(&prefix, Dbtype::AMINO_ACIDS).unwrap();
        let entries = [
            (1u32, b"MNALVVKFGGTSVANAERFLRVADILESNARQGQ".as_slice()),
            (2u32, b"WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP".as_slice()),
            (3u32, b"MEAFRKQLPCFRSGAQQVKEHFKQVAEKHHGFLEEFCAR".as_slice()),
        ];
        for (key, payload) in &entries {
            w.write_entry(*key, payload).unwrap();
        }
        w.finish().unwrap();

        let (alpha, m) = alpha_and_matrix();
        let opts = SearchOptions { k: 3, reduce_to: Some(13), ..Default::default() };

        let from_db = SearchEngine::build_from_mmseqs_db(
            &prefix,
            &m,
            alpha.clone(),
            opts.clone(),
        )
        .expect("build_from_mmseqs_db");

        let in_mem_targets: Vec<(u32, Sequence)> = entries
            .iter()
            .map(|(id, s)| (*id, Sequence::from_ascii(alpha.clone(), s)))
            .collect();
        let in_mem = SearchEngine::build(in_mem_targets.clone(), &m, alpha.clone(), opts)
            .expect("SearchEngine::build");

        // Per-query parity: same top-hit target_id + gapped score for every input.
        for (qid, qseq) in &entries {
            let query = Sequence::from_ascii(alpha.clone(), qseq);
            let db_hits = from_db.search(&query);
            let mem_hits = in_mem.search(&query);
            assert_eq!(
                db_hits.len(),
                mem_hits.len(),
                "hit count mismatch for query {qid}",
            );
            if !db_hits.is_empty() {
                assert_eq!(db_hits[0].target_id, mem_hits[0].target_id);
                assert_eq!(db_hits[0].alignment.score, mem_hits[0].alignment.score);
            }
        }

        // Self-hit sanity: every DB-built query finds itself at rank 0.
        for (qid, qseq) in &entries {
            let q = Sequence::from_ascii(alpha.clone(), qseq);
            let hits = from_db.search(&q);
            assert!(!hits.is_empty(), "DB-built engine: query {qid} had no hits");
            assert_eq!(hits[0].target_id, *qid);
        }
    }

    #[test]
    fn build_from_mmseqs_db_surfaces_missing_db_as_db_open_error() {
        let (alpha, m) = alpha_and_matrix();
        let result = SearchEngine::build_from_mmseqs_db(
            "/nonexistent/path/to/mmseqs-db",
            &m,
            alpha,
            SearchOptions::default(),
        );
        match result {
            Err(BuildFromDbError::DbOpen(_)) => {}
            Err(e) => panic!("expected DbOpen error, got {e}"),
            Ok(_) => panic!("expected error for missing DB, got Ok"),
        }
    }

    #[test]
    fn unrelated_query_returns_no_hits_at_reasonable_threshold() {
        let (alpha, m) = alpha_and_matrix();
        let targets = vec![(1u32, Sequence::from_ascii(alpha.clone(), b"AAAAAAAAAAAAAAAAAAAA"))];
        let opts = SearchOptions {
            k: 3,
            reduce_to: Some(13),
            min_score: 50, // generous; A-A under BLOSUM62 won't approach this
            ..Default::default()
        };
        let engine = SearchEngine::build(targets, &m, alpha.clone(), opts).unwrap();
        let q = Sequence::from_ascii(alpha, b"WWWWWWWWWWWWWWWWWWWW");
        assert!(engine.search(&q).is_empty());
    }
}
