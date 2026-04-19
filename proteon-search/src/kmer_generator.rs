//! Similar-k-mer generation for sensitive prefilter.
//!
//! Given a query k-mer and a substitution score matrix, generate every
//! k-mer that scores at or above a threshold when aligned position-for-
//! position against the query. This is the sensitivity layer on top of
//! exact k-mer matching: without it, the prefilter only finds targets
//! that share the exact same k-mers as the query; with it, biologically
//! similar k-mers also count.
//!
//! Algorithm: branch-and-bound DFS. For each prefix position we try every
//! alphabet substitution, accumulating score; we prune as soon as the
//! optimistic best-case completion of the prefix can no longer reach
//! `threshold`. Upstream (`src/prefiltering/KmerGenerator.cpp`) uses a
//! more sophisticated divide-and-merge pattern (precompute top-scoring
//! similars for chunks, then merge) — faster for production k/alphabet
//! combinations but much more code. We take the DFS path here; the
//! divide-and-merge speedup is a follow-up once benchmarks prove it
//! matters on real workloads.

use crate::kmer::KmerEncoder;

/// Flattened `alphabet_size × alphabet_size` score matrix, row-major.
/// `scores[query_aa * alphabet_size + candidate_aa]` is the contribution of
/// substituting `candidate_aa` at a position where the query has `query_aa`.
/// Typically produced from [`crate::matrix::SubstitutionMatrix::to_integer_matrix`]
/// widened to `i32` (k-mer scores sum `kmer_size` entries so `i8` would
/// overflow for long k).
pub type ScoreMatrix<'a> = &'a [i32];

/// Generate all k-mers whose position-wise substitution score against
/// `query_kmer` is `>= threshold`.
///
/// Returns `(encoded_kmer_hash, score)` pairs, where the hash is produced
/// by `encoder` so it can feed directly into [`crate::kmer::KmerIndex::lookup_hash`].
/// The query k-mer itself is included in the output iff its self-score
/// passes the threshold (which is the typical case).
///
/// # Panics
///
/// All three panics are caller-contract violations, not runtime conditions:
///
/// - `query_kmer.len() != encoder.kmer_size()`
/// - `scores.len() != alphabet_size^2`
/// - any byte in `query_kmer` is `>= alphabet_size` — caller must pass
///   already-alphabet-encoded bytes (typically produced via
///   [`crate::alphabet::Alphabet::encode`] or
///   [`crate::sequence::Sequence::from_ascii`]). Raw ASCII won't work.
pub fn generate_similar_kmers(
    encoder: &KmerEncoder,
    query_kmer: &[u8],
    scores: ScoreMatrix<'_>,
    threshold: i32,
) -> Vec<(u64, i32)> {
    let k = encoder.kmer_size();
    let a = encoder.alphabet_size() as usize;
    assert_eq!(
        query_kmer.len(),
        k,
        "query_kmer length must equal kmer_size"
    );
    assert_eq!(
        scores.len(),
        a * a,
        "scores length must equal alphabet_size^2"
    );
    // Upfront validation avoids the less-helpful slice-OOB panic deep in
    // the DFS. Cost is O(k) against the caller's usual k <= 16.
    for (i, &b) in query_kmer.iter().enumerate() {
        assert!(
            (b as usize) < a,
            "query_kmer[{i}] = {b} is out of range for alphabet size {a}; \
             callers must pass alphabet-encoded bytes, not raw ASCII",
        );
    }

    // Precompute per-position maximum remaining score: max_remaining[i] is
    // the upper bound on additional score achievable starting from
    // position i (inclusive). Used for pruning.
    let mut max_remaining = vec![0i32; k + 1];
    for i in (0..k).rev() {
        let q = query_kmer[i] as usize;
        let row = &scores[q * a..q * a + a];
        let best: i32 = *row.iter().max().expect("alphabet_size >= 1");
        max_remaining[i] = best + max_remaining[i + 1];
    }

    // Early-out: even the best-case full k-mer can't reach threshold.
    if max_remaining[0] < threshold {
        return Vec::new();
    }

    let mut out: Vec<(u64, i32)> = Vec::new();
    let mut prefix = Vec::with_capacity(k);
    dfs(
        0,
        k,
        a,
        query_kmer,
        scores,
        threshold,
        &max_remaining,
        &mut prefix,
        0,
        encoder,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    pos: usize,
    k: usize,
    a: usize,
    query_kmer: &[u8],
    scores: ScoreMatrix<'_>,
    threshold: i32,
    max_remaining: &[i32],
    prefix: &mut Vec<u8>,
    running_score: i32,
    encoder: &KmerEncoder,
    out: &mut Vec<(u64, i32)>,
) {
    if pos == k {
        // Leaf: the upper-bound pruning above only guarantees that *some*
        // completion could reach `threshold`, not that the completion we
        // actually took did. Score must be re-checked at the leaf.
        if running_score >= threshold {
            out.push((encoder.encode(prefix), running_score));
        }
        return;
    }
    // Prune: even the best-case completion of this prefix can't reach threshold.
    if running_score + max_remaining[pos] < threshold {
        return;
    }
    let q = query_kmer[pos] as usize;
    let row = &scores[q * a..q * a + a];
    for candidate in 0..a {
        let s = row[candidate];
        prefix.push(candidate as u8);
        dfs(
            pos + 1,
            k,
            a,
            query_kmer,
            scores,
            threshold,
            max_remaining,
            prefix,
            running_score + s,
            encoder,
            out,
        );
        prefix.pop();
    }
}

/// Widen a 21×21 or 5×5 `i8` matrix (from
/// [`crate::matrix::SubstitutionMatrix::to_integer_matrix`]) into the
/// `i32` row-major layout [`generate_similar_kmers`] expects.
pub fn widen_to_i32(matrix: &[i8]) -> Vec<i32> {
    matrix.iter().map(|&v| v as i32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::kmer::KmerEncoder;
    use crate::matrix::SubstitutionMatrix;

    /// Identity scoring matrix: match = match_score, mismatch = mismatch_score.
    fn identity_matrix(alphabet_size: usize, match_score: i32, mismatch_score: i32) -> Vec<i32> {
        let mut m = vec![mismatch_score; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            m[i * alphabet_size + i] = match_score;
        }
        m
    }

    #[test]
    fn self_is_always_included_when_threshold_leq_self_score() {
        // Alphabet 4, k=3, identity +2/-1, threshold 0.
        // Self-score = 3 * 2 = 6 >= 0 → identity present.
        let enc = KmerEncoder::new(4, 3);
        let scores = identity_matrix(4, 2, -1);
        let q = vec![0u8, 1, 2];
        let mut results = generate_similar_kmers(&enc, &q, &scores, 0);
        results.sort_by_key(|&(h, _)| h);
        let self_hash = enc.encode(&q);
        assert!(results.iter().any(|&(h, s)| h == self_hash && s == 6));
    }

    #[test]
    fn count_matches_hand_computed_exact_threshold() {
        // Alphabet 2, k=3, identity +1/0, threshold 2.
        // Total k-mers: 2^3 = 8. Each scores = count of matching positions.
        // Against query [0,0,0], scores:
        //   [0,0,0]=3  [0,0,1]=2  [0,1,0]=2  [1,0,0]=2
        //   [0,1,1]=1  [1,0,1]=1  [1,1,0]=1  [1,1,1]=0
        // threshold=2 keeps the first four.
        let enc = KmerEncoder::new(2, 3);
        let scores = identity_matrix(2, 1, 0);
        let q = vec![0u8, 0, 0];
        let results = generate_similar_kmers(&enc, &q, &scores, 2);
        assert_eq!(results.len(), 4);

        // Verify the exact set by decoding hashes back.
        let mut found: Vec<(u64, i32)> = results;
        found.sort_unstable();
        let expected = [
            (enc.encode(&[0, 0, 0]), 3),
            (enc.encode(&[0, 0, 1]), 2),
            (enc.encode(&[0, 1, 0]), 2),
            (enc.encode(&[1, 0, 0]), 2),
        ];
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        assert_eq!(found, expected);
    }

    #[test]
    fn returns_empty_when_threshold_exceeds_max_self_score() {
        // Alphabet 2, k=3, identity +1/0, threshold 4 (exceeds max achievable = 3).
        let enc = KmerEncoder::new(2, 3);
        let scores = identity_matrix(2, 1, 0);
        let q = vec![0u8, 0, 0];
        let results = generate_similar_kmers(&enc, &q, &scores, 4);
        assert!(results.is_empty());
    }

    #[test]
    fn threshold_zero_includes_every_kmer_when_min_score_is_zero() {
        // Identity matrix with +1 match, 0 mismatch, threshold 0 keeps all 2^3 = 8.
        let enc = KmerEncoder::new(2, 3);
        let scores = identity_matrix(2, 1, 0);
        let q = vec![0u8, 0, 0];
        let results = generate_similar_kmers(&enc, &q, &scores, 0);
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn pruning_does_not_miss_k_mers_only_reachable_via_non_greedy_path() {
        // Setup: 3-letter alphabet, k=3.
        // Matrix where query letter 0 scores:
        //   vs 0: +5  vs 1: +3  vs 2: +1
        // query letter 1 scores:
        //   vs 0: +1  vs 1: +5  vs 2: +3
        // query letter 2 scores:
        //   vs 0: +3  vs 1: +1  vs 2: +5
        // Query: [0, 1, 2]. Threshold: 9.
        //
        // A DFS that explores all substitutions correctly must find every
        // triple whose score sums to >= 9. Exhaustive enumeration gives:
        //   [0,1,2]=15, [0,1,0]=11, [0,1,1]=11, [0,2,0]=11, [0,2,2]=13,
        //   [1,1,2]=13, [2,1,2]=11, [0,0,2]=11, [1,1,0]=9, [0,2,1]=9,
        //   [1,2,2]=11, [2,0,2]=9, [2,1,0]=7 (no, sum=7 < 9, excluded), ...
        // Easier: compute against every triple and cross-check.
        let enc = KmerEncoder::new(3, 3);
        #[rustfmt::skip]
        let scores = vec![
            5, 3, 1,
            1, 5, 3,
            3, 1, 5,
        ];
        let q = vec![0u8, 1, 2];
        let threshold = 9;

        // Brute-force expected set:
        let mut expected: Vec<(u64, i32)> = Vec::new();
        for a in 0..3u8 {
            for b in 0..3u8 {
                for c in 0..3u8 {
                    let s =
                        scores[a as usize] + scores[3 + b as usize] + scores[2 * 3 + c as usize];
                    if s >= threshold {
                        expected.push((enc.encode(&[a, b, c]), s));
                    }
                }
            }
        }
        expected.sort_unstable();

        let mut actual = generate_similar_kmers(&enc, &q, &scores, threshold);
        actual.sort_unstable();

        assert_eq!(actual, expected);
    }

    #[test]
    fn works_with_real_blosum62_protein_kmer() {
        // End-to-end: use the real BLOSUM62 matrix scaled to integers.
        let m = SubstitutionMatrix::blosum62();
        let alpha = Alphabet::protein();
        let k = 3;
        let enc = KmerEncoder::new(alpha.size() as u32, k);
        let scores_i8 = m.to_integer_matrix(2.0, 0.0);
        let scores_i32 = widen_to_i32(&scores_i8);

        // Query: "MKL" encoded. Self-score >= 3 * match_score, comfortably above
        // any non-trivial threshold. With threshold=15 we expect at least
        // the identity k-mer.
        let q: Vec<u8> = [b'M', b'K', b'L']
            .iter()
            .map(|&c| alpha.encode(c))
            .collect();
        let self_hash = enc.encode(&q);
        let self_score_i32: i32 = (0..k)
            .map(|i| scores_i32[q[i] as usize * alpha.size() + q[i] as usize])
            .sum();

        let results = generate_similar_kmers(&enc, &q, &scores_i32, 15);
        // Result must contain at least the identity k-mer with its self-score.
        assert!(
            results
                .iter()
                .any(|&(h, s)| h == self_hash && s == self_score_i32),
            "expected identity k-mer in generated set; got {} results, self_score={}",
            results.len(),
            self_score_i32,
        );
        // Every returned k-mer's score must be >= threshold.
        for &(_, s) in &results {
            assert!(s >= 15, "score {s} below threshold 15");
        }
    }

    #[test]
    #[should_panic(expected = "out of range for alphabet size")]
    fn panics_with_clear_message_on_out_of_range_query_byte() {
        // Alphabet size 4, but we pass a byte with value 99 (a common mistake:
        // using the X "skip index" sentinel as an alphabet byte, or passing
        // raw ASCII instead of alphabet-encoded data).
        let enc = KmerEncoder::new(4, 3);
        let scores = identity_matrix(4, 2, -1);
        let bad = vec![0u8, 99, 1];
        let _ = generate_similar_kmers(&enc, &bad, &scores, 0);
    }

    #[test]
    fn symmetric_scoring_produces_symmetric_membership() {
        // If score(a,b) == score(b,a) (true for BLOSUM-family) and threshold
        // is set consistently, swapping query and candidate should give the
        // same membership (not necessarily same hashes since hashes encode
        // which is query — but each direction's results should match when
        // we compare against brute force).
        let enc = KmerEncoder::new(4, 2);
        let scores = identity_matrix(4, 3, -1);
        for q in [[0u8, 1], [1, 2], [3, 0], [2, 3]] {
            let r = generate_similar_kmers(&enc, &q, &scores, 2);
            // Every k-mer where both positions match contributes +3+3=6.
            // One match + one mismatch contributes +3-1=2. Two mismatches: -2.
            // At threshold=2 we keep: identity (6) and all (one-match) cases.
            // Count of one-match k-mers: 2 positions × 3 substitutions = 6,
            // plus identity = 7 total.
            assert_eq!(r.len(), 7, "unexpected count for query {q:?}: {}", r.len());
        }
    }
}
