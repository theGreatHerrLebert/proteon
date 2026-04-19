//! Position-Specific Scoring Matrix (PSSM): query × alphabet → score.
//!
//! Precomputes `pssm[i][a] = scores[query[i] * alphabet_size + a]` so the
//! hot inner loop of every alignment kernel does one indirect lookup
//! instead of two. On CPU this is a constant-factor speedup (and a
//! simpler call site); on GPU it's the central perf primitive — the
//! PSSM fits in shared memory for typical query lengths, eliminating
//! global-memory reads for scores entirely.
//!
//! Algorithmically equivalent to inline `scores[q*A+t]` lookup; built
//! to be a drop-in for CPU consumers as well as the storage layout
//! GPU kernels expect.

/// Per-position scoring table. `pssm[i * alphabet_size + a]` is the
/// score for substituting `a` at query position `i`.
///
/// Stored row-major (one row per query position) so that a column
/// scan during DP reads contiguous bytes — the GPU shared-memory
/// access pattern is the same shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pssm {
    pub query_len: usize,
    pub alphabet_size: usize,
    pub data: Vec<i32>,
}

impl Pssm {
    /// Build a PSSM from an alphabet-encoded query and a flat
    /// `alphabet_size × alphabet_size` score matrix.
    ///
    /// Panics if any byte in `query` is `>= alphabet_size` (same
    /// caller contract as the rest of the crate — bytes must come
    /// from `Alphabet::encode` or equivalent).
    pub fn build(query: &[u8], scores: &[i32], alphabet_size: usize) -> Self {
        assert_eq!(
            scores.len(),
            alphabet_size * alphabet_size,
            "scores length must equal alphabet_size^2"
        );
        let mut data = vec![0i32; query.len() * alphabet_size];
        for (i, &q) in query.iter().enumerate() {
            assert!(
                (q as usize) < alphabet_size,
                "query[{i}] = {q} out of range for alphabet size {alphabet_size}"
            );
            let row_start = (q as usize) * alphabet_size;
            let dst_start = i * alphabet_size;
            data[dst_start..dst_start + alphabet_size]
                .copy_from_slice(&scores[row_start..row_start + alphabet_size]);
        }
        Self {
            query_len: query.len(),
            alphabet_size,
            data,
        }
    }

    /// `pssm[i][a]` — score at query position `i` for alphabet index `a`.
    /// O(1).
    pub fn get(&self, i: usize, a: u8) -> i32 {
        debug_assert!(i < self.query_len);
        debug_assert!((a as usize) < self.alphabet_size);
        self.data[i * self.alphabet_size + a as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix(alphabet_size: usize, m: i32, mm: i32) -> Vec<i32> {
        let mut s = vec![mm; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            s[i * alphabet_size + i] = m;
        }
        s
    }

    #[test]
    fn pssm_lookup_matches_inline_scores_lookup() {
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 3, -2);
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);

        // Every (i, a) pair must produce the same value as the inline
        // scores[query[i]*A+a] lookup.
        for (i, &q) in query.iter().enumerate() {
            for a in 0..alphabet_size as u8 {
                assert_eq!(
                    pssm.get(i, a),
                    scores[q as usize * alphabet_size + a as usize],
                    "PSSM diverges from scores lookup at i={i}, a={a}"
                );
            }
        }
    }

    #[test]
    fn pssm_diagonal_is_match_score() {
        // For an identity matrix and any query, pssm[i][query[i]] = match.
        let alphabet_size = 4;
        let scores = identity_matrix(alphabet_size, 5, -1);
        let query: Vec<u8> = vec![2, 0, 3, 1];
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        for (i, &q) in query.iter().enumerate() {
            assert_eq!(pssm.get(i, q), 5, "diagonal at i={i} should be match score");
        }
    }

    #[test]
    fn pssm_shape_matches_query_length() {
        let alphabet_size = 21;
        let scores = identity_matrix(alphabet_size, 3, -1);
        let query: Vec<u8> = (0..50).map(|i| (i % alphabet_size) as u8).collect();
        let pssm = Pssm::build(&query, &scores, alphabet_size);
        assert_eq!(pssm.query_len, 50);
        assert_eq!(pssm.alphabet_size, 21);
        assert_eq!(pssm.data.len(), 50 * 21);
    }

    #[test]
    #[should_panic(expected = "out of range for alphabet size")]
    fn pssm_panics_on_out_of_range_query_byte() {
        let scores = identity_matrix(4, 3, -2);
        let bad: Vec<u8> = vec![0, 99, 1];
        let _ = Pssm::build(&bad, &scores, 4);
    }
}
