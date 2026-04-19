//! Ungapped alignment extension along a single diagonal.
//!
//! Reference: Steinegger & Söding, *Nat. Biotechnol.* 35, 1026-1028
//! (2017), §2.2 (ungapped alignment stage of the MMseqs2 pipeline).
//!
//! Given a query and target sequence and a diagonal (`target_pos -
//! query_pos`), find the highest-scoring ungapped local segment along
//! that diagonal under a substitution matrix. This is the final
//! prefilter stage in the MMseqs2-style pipeline: the diagonal-voting
//! prefilter (2.3a/b) produces candidate `(seq_id, best_diagonal)`
//! hits using raw k-mer counts, and this module converts each candidate
//! into a concrete HSP score + coordinates that the downstream gapped
//! aligner can extend.
//!
//! Algorithm: Kadane-style max running sum with reset-on-negative,
//! matching upstream `UngappedAlignment::scalarDiagonalScoring` at
//! `src/prefiltering/UngappedAlignment.cpp:45`. We return the
//! coordinates of the max segment in addition to the score so callers
//! can feed an HSP into banded or SW gapped alignment later.
//!
//! Scope: scalar implementation. Upstream's SIMD path scores 4 (SSE) or
//! 8 (AVX2) diagonals in parallel against a shared query profile; we
//! take the scalar approach first for reviewability and port the SIMD
//! unroll when benchmarks demand it.

/// Result of an ungapped alignment on a single diagonal.
///
/// `query_end` / `target_end` are **exclusive** — slice indexing via
/// `query[query_start..query_end]` gives the aligned residues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UngappedHit {
    pub score: i32,
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
}

impl UngappedHit {
    pub fn length(&self) -> usize {
        self.query_end - self.query_start
    }
}

/// Scan the given diagonal for the highest-scoring ungapped local segment.
///
/// `diagonal = target_pos - query_pos`. `scores` is a flat
/// `alphabet_size × alphabet_size` score matrix in row-major order (same
/// layout [`crate::kmer_generator::generate_similar_kmers`] expects).
///
/// Returns `None` if the diagonal has no positive-scoring segment — the
/// caller should treat that as "this candidate didn't actually align".
///
/// Panics if any byte in `query` or `target` (within the scanned range)
/// is `>= alphabet_size`; same caller contract as the rest of the crate.
pub fn ungapped_alignment(
    query: &[u8],
    target: &[u8],
    diagonal: i32,
    scores: &[i32],
    alphabet_size: usize,
) -> Option<UngappedHit> {
    assert_eq!(
        scores.len(),
        alphabet_size * alphabet_size,
        "scores length must equal alphabet_size^2"
    );

    // Clamp the scan to the overlap region: query[q_start..q_end] aligns
    // with target[q_start+diag..q_end+diag], both slices in range.
    let (q_start, q_end) = overlap_range(query.len(), target.len(), diagonal);
    if q_start >= q_end {
        return None;
    }

    let mut running: i32 = 0;
    let mut running_start: usize = q_start;
    let mut best_score: i32 = 0;
    let mut best_start: usize = q_start;
    let mut best_end: usize = q_start;

    for q in q_start..q_end {
        let t = (q as i64 + diagonal as i64) as usize;
        let qb = query[q] as usize;
        let tb = target[t] as usize;
        debug_assert!(qb < alphabet_size && tb < alphabet_size);
        let cell = scores[qb * alphabet_size + tb];

        running += cell;
        if running < 0 {
            running = 0;
            running_start = q + 1;
        } else if running > best_score {
            best_score = running;
            best_start = running_start;
            best_end = q + 1;
        }
    }

    if best_score > 0 {
        let best_tstart = (best_start as i64 + diagonal as i64) as usize;
        let best_tend = (best_end as i64 + diagonal as i64) as usize;
        Some(UngappedHit {
            score: best_score,
            query_start: best_start,
            query_end: best_end,
            target_start: best_tstart,
            target_end: best_tend,
        })
    } else {
        None
    }
}

/// Compute the inclusive..exclusive range of query positions whose
/// target counterpart at the given diagonal is in-bounds.
///
/// All arithmetic is done in `i64` so that the extreme `diagonal`
/// value `i32::MIN` — whose negation overflows `i32` — is handled as
/// an ordinary "no overlap" case and returns an empty range rather
/// than panicking.
fn overlap_range(query_len: usize, target_len: usize, diagonal: i32) -> (usize, usize) {
    let q_len = query_len as i64;
    let t_len = target_len as i64;
    let diag = diagonal as i64;

    // q_pos valid when 0 <= q_pos < query_len AND 0 <= q_pos + diag < target_len.
    // So q_pos in [max(0, -diag), min(query_len, target_len - diag)).
    let q_start_i = 0i64.max(-diag);
    let q_end_i = q_len.min(t_len - diag);

    if q_start_i >= q_end_i || q_start_i < 0 {
        return (0, 0);
    }
    (q_start_i as usize, q_end_i as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::kmer_generator::widen_to_i32;
    use crate::matrix::SubstitutionMatrix;
    use crate::sequence::Sequence;

    /// Identity matrix: match = +match_score, mismatch = mismatch_score.
    fn identity_matrix(alphabet_size: usize, match_score: i32, mismatch_score: i32) -> Vec<i32> {
        let mut m = vec![mismatch_score; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            m[i * alphabet_size + i] = match_score;
        }
        m
    }

    #[test]
    fn identical_sequences_on_diag_0_score_full_length() {
        // Alphabet 4, +2/-1. Sequences identical → ungapped HSP spans the full seq.
        let scores = identity_matrix(4, 2, -1);
        let q: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        let t = q.clone();
        let hit = ungapped_alignment(&q, &t, 0, &scores, 4).unwrap();
        assert_eq!(hit.score, 2 * 6);
        assert_eq!(hit.query_start, 0);
        assert_eq!(hit.query_end, 6);
        assert_eq!(hit.target_start, 0);
        assert_eq!(hit.target_end, 6);
        assert_eq!(hit.length(), 6);
    }

    #[test]
    fn diverged_tails_are_trimmed() {
        // Center matches, flanking regions mismatch. Kadane should trim the tails.
        // Query:  [9,9, 0,1,2,3,  9,9]   (indices 9 = mismatch-heavy sentinel within alphabet)
        // Target: [8,8, 0,1,2,3,  8,8]   (both tails unrelated at same positions)
        // With +3/-2 at alphabet 10 on diag 0, center +12, tails -8 on each side.
        let scores = identity_matrix(10, 3, -2);
        let q: Vec<u8> = vec![9, 9, 0, 1, 2, 3, 9, 9];
        let t: Vec<u8> = vec![8, 8, 0, 1, 2, 3, 8, 8];
        let hit = ungapped_alignment(&q, &t, 0, &scores, 10).unwrap();
        assert_eq!(hit.score, 12, "expected center-only score");
        assert_eq!(hit.query_start, 2);
        assert_eq!(hit.query_end, 6);
    }

    #[test]
    fn returns_none_when_no_positive_segment() {
        // Every position mismatches → running stays non-positive → no HSP.
        let scores = identity_matrix(4, 2, -3);
        let q: Vec<u8> = vec![0, 1, 2, 3];
        let t: Vec<u8> = vec![1, 2, 3, 0]; // pairwise all mismatch
        assert_eq!(ungapped_alignment(&q, &t, 0, &scores, 4), None);
    }

    #[test]
    fn respects_positive_diagonal_boundaries() {
        // diagonal = +2 means target starts 2 positions after query; the
        // first two query positions (q=0,1) have no target counterpart and
        // are skipped. Scanning happens only for q in [0, target.len() - 2).
        let scores = identity_matrix(4, 2, -1);
        // q=[3,3,0,1,2], t=[3,3,3,3,0,1,2]. Length 5 and 7; diag=+2 →
        // q in [0, 5). Pairs: q[0]/t[2]=3/3 match, q[1]/t[3]=3/3 match,
        // q[2]/t[4]=0/0 match, q[3]/t[5]=1/1 match, q[4]/t[6]=2/2 match.
        let q: Vec<u8> = vec![3, 3, 0, 1, 2];
        let t: Vec<u8> = vec![3, 3, 3, 3, 0, 1, 2];
        let hit = ungapped_alignment(&q, &t, 2, &scores, 4).unwrap();
        assert_eq!(hit.score, 2 * 5, "five matches × +2");
        assert_eq!(hit.query_start, 0);
        assert_eq!(hit.query_end, 5);
        assert_eq!(hit.target_start, 2);
        assert_eq!(hit.target_end, 7);
    }

    #[test]
    fn respects_negative_diagonal_boundaries() {
        // diagonal = -2 means target starts 2 positions before query; first
        // two query positions have no target counterpart; scanning starts
        // at q=2.
        let scores = identity_matrix(4, 2, -1);
        // Fill q[0..2] with mismatch-ish values (within alphabet) that
        // don't participate because they're outside the overlap anyway.
        let q: Vec<u8> = vec![3, 3, 0, 1, 2];
        let t: Vec<u8> = vec![0, 1, 2];
        // Pairs at diag -2: q[2]=0/t[0]=0 match, q[3]=1/t[1]=1 match,
        //                   q[4]=2/t[2]=2 match.
        let hit = ungapped_alignment(&q, &t, -2, &scores, 4).unwrap();
        assert_eq!(hit.score, 6);
        assert_eq!(hit.query_start, 2);
        assert_eq!(hit.query_end, 5);
        assert_eq!(hit.target_start, 0);
        assert_eq!(hit.target_end, 3);
    }

    #[test]
    fn empty_overlap_returns_none() {
        let scores = identity_matrix(4, 2, -1);
        let q: Vec<u8> = vec![0, 1];
        let t: Vec<u8> = vec![0, 1];
        // diag >= target.len() → no overlap
        assert_eq!(ungapped_alignment(&q, &t, 5, &scores, 4), None);
        // diag <= -query.len() → no overlap
        assert_eq!(ungapped_alignment(&q, &t, -5, &scores, 4), None);
    }

    #[test]
    fn extreme_diagonal_i32_min_returns_none_without_panic() {
        // Regression: -i32::MIN overflows i32, so the previous
        // `(-diagonal) as usize` path panicked in debug builds. i64-based
        // overlap_range handles it as "no overlap" like any other out-of-
        // range diagonal. Test both MIN and MAX since they're symmetric
        // public-API edges.
        let scores = identity_matrix(4, 2, -1);
        let q: Vec<u8> = vec![0, 1, 2];
        let t: Vec<u8> = vec![0, 1, 2];
        assert_eq!(ungapped_alignment(&q, &t, i32::MIN, &scores, 4), None);
        assert_eq!(ungapped_alignment(&q, &t, i32::MAX, &scores, 4), None);
    }

    #[test]
    fn empty_sequences_return_none() {
        let scores = identity_matrix(4, 2, -1);
        assert_eq!(ungapped_alignment(&[], &[0u8, 1, 2], 0, &scores, 4), None);
        assert_eq!(ungapped_alignment(&[0u8, 1, 2], &[], 0, &scores, 4), None);
    }

    #[test]
    fn kadane_picks_later_higher_segment_over_earlier_positive() {
        // Make sure the Kadane reset-on-negative actually finds the later
        // stronger segment even if the earlier one was positive.
        // Query vs target, pairwise scores over 10 positions:
        //   +3  -2  -2  -2  +3  +3  +3  +3  -2  +3
        //   running: 3, 1, -1→0, -2→0, 3, 6, 9, 12, 10, 13
        // Expected best: 13 covering the whole run from position 4 onward,
        // but actually the best contiguous segment including the final +3 is
        // positions [4..10), score 3+3+3+3-2+3 = 13.
        // Actually let me compute carefully. Kadane returns max running.
        // After running becomes 0 at pos 3, running from pos 4: +3 (3), +3 (6),
        // +3 (9), +3 (12), -2 (10), +3 (13). Max = 13.
        // Meanwhile the first positive segment peaked at 3 (pos 0).
        let scores = identity_matrix(2, 3, -2);
        // Craft query=target at matching positions, mismatch at others.
        // match at 0, mismatch at 1, mismatch at 2, mismatch at 3, then
        // match at 4..8, mismatch at 8, match at 9.
        let q: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let t: Vec<u8> = vec![0, 1, 1, 1, 0, 0, 0, 0, 1, 0];
        let hit = ungapped_alignment(&q, &t, 0, &scores, 2).unwrap();
        assert_eq!(hit.score, 13);
        assert_eq!(hit.query_start, 4);
        assert_eq!(hit.query_end, 10);
    }

    #[test]
    fn end_to_end_blosum62_protein_hsp() {
        // Two proteins with a shared 8-residue core ("HELLOWWR") flanked by
        // unrelated residues. Ungapped alignment at the correct diagonal
        // should find the core at its expected coordinates.
        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let scores = widen_to_i32(&m.to_integer_matrix(2.0, 0.0));

        let q = Sequence::from_ascii(alpha.clone(), b"AAAAHELLOWWRCCCCC");
        let t = Sequence::from_ascii(alpha.clone(), b"DDDDDDDHELLOWWREEEEE");
        // Query "HELLOWWR" starts at q=4, target starts at t=7 → diagonal=+3.
        let hit = ungapped_alignment(&q.data, &t.data, 3, &scores, alpha.size()).unwrap();
        assert!(hit.score > 0);
        assert_eq!(hit.query_start, 4);
        assert_eq!(hit.query_end, 12, "HSP ends after core residues");
        assert_eq!(hit.target_start, 7);
        assert_eq!(hit.target_end, 15);
    }

    #[test]
    fn hit_length_matches_coordinate_diff() {
        let scores = identity_matrix(4, 2, -1);
        let q: Vec<u8> = vec![0, 1, 2, 3];
        let t = q.clone();
        let hit = ungapped_alignment(&q, &t, 0, &scores, 4).unwrap();
        assert_eq!(hit.length(), hit.query_end - hit.query_start);
        assert_eq!(hit.length(), hit.target_end - hit.target_start);
    }
}
