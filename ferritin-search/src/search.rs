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

use thiserror::Error;

use crate::alphabet::Alphabet;
use crate::gapped::{smith_waterman, GappedAlignment};
use crate::kmer::{KmerEncoder, KmerIndex, KmerIndexError};
use crate::kmer_generator::widen_to_i32;
use crate::matrix::SubstitutionMatrix;
use crate::prefilter::{diagonal_prefilter, PrefilterOptions};
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

    /// Run a single query against the indexed corpus.
    ///
    /// Returns hits sorted by gapped alignment score descending.
    pub fn search(&self, query: &Sequence) -> Vec<SearchHit> {
        // Prefilter operates on the reduced (or full) alphabet matching
        // the index; ungapped + gapped use the full-alphabet matrix.
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

        let mut results: Vec<SearchHit> = Vec::new();
        for ph in &prefilter_hits {
            let target = match self.targets_full.get(&ph.seq_id) {
                Some(t) => t,
                None => continue,
            };

            let ungapped = match ungapped_alignment(
                &query.data,
                target,
                ph.best_diagonal,
                &self.matrix_int,
                self.full_alphabet_size,
            ) {
                Some(u) => u,
                None => continue,
            };

            let gapped = match smith_waterman(
                &query.data,
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

    /// Number of targets indexed.
    pub fn target_count(&self) -> usize {
        self.targets_full.len()
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
