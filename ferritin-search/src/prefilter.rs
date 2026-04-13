//! K-mer diagonal-voting prefilter.
//!
//! Phase 2.3a: given a query sequence and a [`KmerIndex`] over a target
//! corpus, count k-mer matches per `(target_seq_id, diagonal)` and emit
//! each target's best-scoring diagonal.
//!
//! The **diagonal** of a k-mer match is `target_pos - query_pos`. Matches
//! along a shared diagonal are consistent with an ungapped alignment
//! between the two sequences, so a high diagonal count is a strong signal
//! that query and target share a homologous region. This is the core
//! prefilter signal in upstream MMseqs2 (see
//! `src/prefiltering/QueryMatcher.cpp`).
//!
//! Scope limitations (addressed in subsequent phases):
//!   - **2.3b** adds similar-k-mer expansion (`KmerGenerator`) so a query
//!     k-mer can match target k-mers that score above a substitution-matrix
//!     threshold rather than only exact matches.
//!   - **2.3c** adds the reduced alphabet so `table_size` is tractable at
//!     production k.
//!   - **2.3d** extends the best diagonal into an ungapped HSP score.
//!
//! The algorithm used here is naïve relative to upstream's cache-friendly
//! sort-based counting — we accumulate into a `HashMap<(seq_id, diagonal)>`
//! per query. Correct and easy to review; optimize once a benchmark pins
//! the actual bottleneck.

use std::collections::HashMap;

use crate::kmer::KmerIndex;

/// One prefilter result: the target sequence, its best-scoring diagonal,
/// and the k-mer count on that diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefilterHit {
    pub seq_id: u32,
    /// Number of k-mer matches on [`PrefilterHit::best_diagonal`]. Upstream
    /// calls this `prefScore` (still an unscored k-mer count here; real
    /// ungapped score comes in 2.3d).
    pub diagonal_score: u32,
    /// The best diagonal, defined as `target_pos - query_pos`. May be
    /// negative (target match position precedes query position). Upstream
    /// stores this as `u16` with wraparound; we keep `i32` so the sign is
    /// explicit and ordering is unambiguous for downstream consumers.
    pub best_diagonal: i32,
}

/// Prefilter tuning knobs. Defaults keep every seq with at least one
/// diagonal hit and return them all.
#[derive(Debug, Clone)]
pub struct PrefilterOptions {
    /// Drop hits whose `diagonal_score` is below this threshold. `0` keeps
    /// everything.
    pub score_threshold: u32,
    /// Keep at most this many hits (by score desc, then seq_id asc). `None`
    /// returns all hits.
    pub max_hits: Option<usize>,
    /// If set, this `seq_id` is excluded from the results — the typical
    /// "don't return the query against itself" case.
    pub exclude_self: Option<u32>,
}

impl Default for PrefilterOptions {
    fn default() -> Self {
        Self { score_threshold: 0, max_hits: None, exclude_self: None }
    }
}

/// Run the diagonal-voting prefilter for a single query.
///
/// `query` is an already-encoded sequence (alphabet indices, same
/// alphabet used to build `index`). `skip_idx` is the alphabet index for
/// X / unknown, used to skip k-mer windows containing it.
pub fn diagonal_prefilter(
    index: &KmerIndex,
    query: &[u8],
    skip_idx: u8,
    opts: &PrefilterOptions,
) -> Vec<PrefilterHit> {
    // Count (seq_id, diagonal) co-occurrences across query k-mer matches.
    let mut counts: HashMap<(u32, i32), u32> = HashMap::new();
    for (q_pos, hash) in index.encoder.iter_kmers(query, skip_idx) {
        for hit in index.lookup_hash(hash) {
            let diagonal = hit.pos as i32 - q_pos as i32;
            *counts.entry((hit.seq_id, diagonal)).or_insert(0) += 1;
        }
    }

    // Reduce to the best diagonal per target sequence. Tie-break: smaller
    // diagonal wins — chosen for determinism; upstream also has a total
    // order via its sort keys, so we match the spirit not the exact key.
    let mut best: HashMap<u32, PrefilterHit> = HashMap::new();
    for ((seq_id, diagonal), score) in counts {
        let entry = best.entry(seq_id).or_insert(PrefilterHit {
            seq_id,
            diagonal_score: 0,
            best_diagonal: diagonal,
        });
        if score > entry.diagonal_score
            || (score == entry.diagonal_score && diagonal < entry.best_diagonal)
        {
            entry.diagonal_score = score;
            entry.best_diagonal = diagonal;
        }
    }

    // Apply exclude_self + score_threshold filters, then sort.
    let mut hits: Vec<PrefilterHit> = best
        .into_values()
        .filter(|h| {
            h.diagonal_score >= opts.score_threshold
                && opts.exclude_self != Some(h.seq_id)
        })
        .collect();

    // Upstream's compareHitsByScoreAndId: score desc, seq_id asc.
    hits.sort_by(|a, b| {
        b.diagonal_score
            .cmp(&a.diagonal_score)
            .then_with(|| a.seq_id.cmp(&b.seq_id))
    });

    if let Some(limit) = opts.max_hits {
        hits.truncate(limit);
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::kmer::{KmerEncoder, KmerIndex};
    use crate::sequence::Sequence;

    /// Build a KmerIndex over a small synthetic corpus for tests.
    fn build_small_index() -> KmerIndex {
        // Alphabet size 4, k=2. 4-nt "DNA" encoded 0=A, 1=C, 2=G, 3=T.
        let enc = KmerEncoder::new(4, 2);
        // target 10: ACGT  → (AC@0, CG@1, GT@2)
        // target 20: CGTA  → (CG@0, GT@1, TA@2)
        // target 30: AAAA  → (AA@0, AA@1, AA@2)
        let a = vec![0u8, 1, 2, 3];
        let b = vec![1u8, 2, 3, 0];
        let c = vec![0u8, 0, 0, 0];
        KmerIndex::build(
            enc,
            [(10u32, a.as_slice()), (20u32, b.as_slice()), (30u32, c.as_slice())],
            99,
        )
        .unwrap()
    }

    #[test]
    fn diagonal_count_matches_manual_calculation() {
        // Query "ACGT" against the small corpus.
        // Query k-mers: AC@0, CG@1, GT@2.
        // Target 10 (ACGT): AC@0 (diag=0), CG@1 (diag=0), GT@2 (diag=0) → 3 hits on diag 0
        // Target 20 (CGTA): CG@0 (diag=-1), GT@1 (diag=-1) → 2 hits on diag -1
        // Target 30 (AAAA): no matches on these k-mers
        let idx = build_small_index();
        let query = vec![0u8, 1, 2, 3];
        let hits = diagonal_prefilter(&idx, &query, 99, &PrefilterOptions::default());
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0],
            PrefilterHit { seq_id: 10, diagonal_score: 3, best_diagonal: 0 }
        );
        assert_eq!(
            hits[1],
            PrefilterHit { seq_id: 20, diagonal_score: 2, best_diagonal: -1 }
        );
    }

    #[test]
    fn best_diagonal_picks_higher_count_not_lower_index() {
        // Craft a case where a target has matches on two diagonals but one
        // has more hits, to prove the best-diagonal reduction picks by count.
        // Target/query: ACGTAC (seq shared; AC occurs at positions 0 and 4).
        // Target k-mer → positions: AC→{0,4}, CG→{1}, GT→{2}, TA→{3}.
        // Per-query-kmer diagonals (target_pos - query_pos):
        //   q AC@0 → target AC@0,4 → diags 0, 4
        //   q CG@1 → target CG@1   → diag 0
        //   q GT@2 → target GT@2   → diag 0
        //   q TA@3 → target TA@3   → diag 0
        //   q AC@4 → target AC@0,4 → diags -4, 0
        // Per-diagonal counts: diag 0 = 5, diag 4 = 1, diag -4 = 1.
        let enc = KmerEncoder::new(4, 2);
        let seq = vec![0u8, 1, 2, 3, 0, 1];
        let idx = KmerIndex::build(enc, [(1u32, seq.as_slice())], 99).unwrap();
        let hits = diagonal_prefilter(&idx, &seq, 99, &PrefilterOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].seq_id, 1);
        assert_eq!(hits[0].best_diagonal, 0);
        assert_eq!(hits[0].diagonal_score, 5);
    }

    #[test]
    fn score_threshold_filters_low_scoring_hits() {
        let idx = build_small_index();
        let query = vec![0u8, 1, 2, 3];
        let opts = PrefilterOptions { score_threshold: 3, ..Default::default() };
        let hits = diagonal_prefilter(&idx, &query, 99, &opts);
        // Only target 10 has diagonal_score >= 3; target 20 (score=2) drops.
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].seq_id, 10);
    }

    #[test]
    fn max_hits_keeps_highest_scoring() {
        let idx = build_small_index();
        let query = vec![0u8, 1, 2, 3];
        let opts = PrefilterOptions { max_hits: Some(1), ..Default::default() };
        let hits = diagonal_prefilter(&idx, &query, 99, &opts);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].seq_id, 10); // higher score wins
    }

    #[test]
    fn exclude_self_drops_query_against_itself() {
        let idx = build_small_index();
        let query = vec![0u8, 1, 2, 3];
        let opts = PrefilterOptions { exclude_self: Some(10), ..Default::default() };
        let hits = diagonal_prefilter(&idx, &query, 99, &opts);
        // seq 10 excluded; seq 20 should remain.
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].seq_id, 20);
    }

    #[test]
    fn hits_sorted_by_score_desc_then_seq_id_asc() {
        // Two targets identical to query → both score max, should sort by seq_id.
        let enc = KmerEncoder::new(4, 2);
        let s = vec![0u8, 1, 2, 3];
        let idx = KmerIndex::build(
            enc,
            [(77u32, s.as_slice()), (33u32, s.as_slice()), (55u32, s.as_slice())],
            99,
        )
        .unwrap();
        let hits = diagonal_prefilter(&idx, &s, 99, &PrefilterOptions::default());
        assert_eq!(hits.len(), 3);
        assert!(hits.iter().all(|h| h.diagonal_score == 3 && h.best_diagonal == 0));
        // Same score → seq_id ascending: 33, 55, 77.
        assert_eq!(hits[0].seq_id, 33);
        assert_eq!(hits[1].seq_id, 55);
        assert_eq!(hits[2].seq_id, 77);
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let idx = build_small_index();
        let hits = diagonal_prefilter(&idx, &[], 99, &PrefilterOptions::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn query_shorter_than_k_returns_no_hits() {
        let idx = build_small_index();
        let hits = diagonal_prefilter(&idx, &[0u8], 99, &PrefilterOptions::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn query_with_only_x_windows_returns_no_hits() {
        let idx = build_small_index();
        let query = vec![99u8, 99, 99, 99]; // every window contains skip_idx
        let hits = diagonal_prefilter(&idx, &query, 99, &PrefilterOptions::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn end_to_end_via_protein_alphabet() {
        let alpha = Alphabet::protein();
        let x = alpha.encode(b'X');
        let enc = KmerEncoder::new(alpha.size() as u32, 3);

        // Two targets: one is a shifted prefix of the query, one is unrelated.
        let q = Sequence::from_ascii(alpha.clone(), b"MKLVRQ");
        let t1 = Sequence::from_ascii(alpha.clone(), b"MKLVRQ"); // identical
        let t2 = Sequence::from_ascii(alpha.clone(), b"WWWWWW"); // no k-mer overlap

        let idx = KmerIndex::build(
            enc,
            [(1u32, t1.data.as_slice()), (2u32, t2.data.as_slice())],
            x,
        )
        .unwrap();
        let hits = diagonal_prefilter(&idx, &q.data, x, &PrefilterOptions::default());
        assert_eq!(hits.len(), 1, "only the identical target should hit");
        assert_eq!(hits[0].seq_id, 1);
        assert_eq!(hits[0].best_diagonal, 0);
        // Query len 6 at k=3 → 4 k-mers, all match on diagonal 0.
        assert_eq!(hits[0].diagonal_score, 4);
    }
}
