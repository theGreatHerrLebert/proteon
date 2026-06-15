//! Smith-Waterman local gapped alignment with affine gap penalties.
//!
//! References:
//! - Smith & Waterman, "Identification of common molecular subsequences",
//!   *J. Mol. Biol.* 147(1), 195-197 (1981) — the local-alignment DP.
//! - Gotoh, "An improved algorithm for matching biological sequences",
//!   *J. Mol. Biol.* 162(3), 705-708 (1982) — the three-matrix formulation
//!   for affine gap penalties used here.
//! - Steinegger & Söding, *Nat. Biotechnol.* 35, 1026-1028 (2017), §2.3
//!   — gapped alignment stage of the MMseqs2 pipeline proteon-search ports.
//!
//! Standard three-matrix Gotoh 1982 formulation:
//!
//! - `M[i,j]` — best score of an alignment ending with `query[i-1]`
//!   aligned to `target[j-1]` (i.e. the last column is a match or mismatch).
//! - `X[i,j]` — best score ending with a gap in the query (target
//!   letter j-1 is a deletion from the query's perspective).
//! - `Y[i,j]` — best score ending with a gap in the target (query
//!   letter i-1 is an insertion relative to the target).
//!
//! All three matrices are clamped at 0 so the alignment is LOCAL — the
//! highest score anywhere in `M` is the returned alignment's endpoint,
//! and traceback stops when the score reaches 0.
//!
//! Scope: scalar O(|q|·|t|) time; memory is **O(|q|·|t|) for traceback (one packed
//! byte per cell) plus O(|t|) score scratch** — the three score matrices are computed
//! with rolling rows, so only the inherently-O(q·t) traceback grid is materialized.
//! Striped SIMD (Farrar SSW) or a block-aligner dependency are natural *speed* paths
//! once benchmarks show SW is the bottleneck; neither is needed for correctness. A
//! true linear-space traceback (Myers–Miller) would drop the last O(q·t) term but is
//! out of scope. Comfortable for protein pairs up to ~10k residues each.

use std::fmt;

/// One segment of a CIGAR string.
///
/// `Match` covers both identity and substitution (the m8 output
/// convention), matching MMseqs2 / BLAST tabular output. If callers want
/// to split into `=` and `X` they can re-walk the aligned residues using
/// the alignment coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
    /// `query[a]` aligned with `target[b]` — either an exact match or a
    /// substitution.
    Match(usize),
    /// Insertion in the query relative to the target: `count` query
    /// residues consumed, no target residues.
    Insert(usize),
    /// Deletion from the query relative to the target: no query residues,
    /// `count` target residues consumed.
    Delete(usize),
}

impl CigarOp {
    pub fn count(&self) -> usize {
        match *self {
            CigarOp::Match(n) | CigarOp::Insert(n) | CigarOp::Delete(n) => n,
        }
    }

    fn letter(&self) -> char {
        match *self {
            CigarOp::Match(_) => 'M',
            CigarOp::Insert(_) => 'I',
            CigarOp::Delete(_) => 'D',
        }
    }
}

/// A local gapped alignment between a query and a target.
///
/// `query_end` and `target_end` are exclusive — slice indexing via
/// `query[query_start..query_end]` yields the aligned query region,
/// likewise for target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GappedAlignment {
    pub score: i32,
    pub query_start: usize,
    pub query_end: usize,
    pub target_start: usize,
    pub target_end: usize,
    pub cigar: Vec<CigarOp>,
}

impl GappedAlignment {
    /// CIGAR string in standard bioinformatics notation (e.g. `"42M2I8M"`).
    pub fn cigar_string(&self) -> String {
        let mut out = String::new();
        for op in &self.cigar {
            out.push_str(&op.count().to_string());
            out.push(op.letter());
        }
        out
    }

    /// Number of query residues the alignment consumes.
    pub fn query_len(&self) -> usize {
        self.query_end - self.query_start
    }

    /// Number of target residues the alignment consumes.
    pub fn target_len(&self) -> usize {
        self.target_end - self.target_start
    }

    /// Total number of aligned columns (matches + inserts + deletes).
    pub fn alignment_length(&self) -> usize {
        self.cigar.iter().map(|op| op.count()).sum()
    }
}

impl fmt::Display for GappedAlignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "score={} q=[{}..{}) t=[{}..{}) cigar={}",
            self.score,
            self.query_start,
            self.query_end,
            self.target_start,
            self.target_end,
            self.cigar_string(),
        )
    }
}

/// Smith-Waterman local alignment with affine gap penalties.
///
/// `scores` is a flat `alphabet_size × alphabet_size` substitution matrix
/// in row-major i32. `gap_open` is the cost of opening a new gap (applied
/// once when the gap starts); `gap_extend` is the cost of extending an
/// existing gap by one additional position. Both are typically negative.
///
/// Convention matches BLAST / MMseqs2: the cost of a gap of length `k` is
/// `gap_open + gap_extend * (k - 1)` — i.e. gap_open is the "first gap
/// column" cost and gap_extend is each subsequent column.
///
/// Returns `None` if no positive-scoring local alignment exists.
///
/// Panics on caller-contract violations: score matrix shape mismatch, or
/// sequence bytes `>= alphabet_size` (same contract as the rest of the crate).
pub fn smith_waterman(
    query: &[u8],
    target: &[u8],
    scores: &[i32],
    alphabet_size: usize,
    gap_open: i32,
    gap_extend: i32,
) -> Option<GappedAlignment> {
    assert_eq!(
        scores.len(),
        alphabet_size * alphabet_size,
        "scores length must equal alphabet_size^2",
    );

    let q_len = query.len();
    let t_len = target.len();
    if q_len == 0 || t_len == 0 {
        return None;
    }

    // Gotoh affine DP. The three score matrices (`M`, `X`, `Y`) are computed with
    // only TWO rolling rows each (`prev_*` = row i-1, `cur_*` = row i) — every cell
    // reads either the previous row or the already-filled left part of the current
    // row, so the full (q+1)×(t+1) score grids are unnecessary. Only the traceback is
    // inherently O(q·t); it is stored as ONE packed byte per cell (was three). This
    // cuts memory from 15 bytes/cell to ~1 + O(t) scratch with byte-identical results.
    let rows = q_len + 1;
    let cols = t_len + 1;
    let neg_sent: i32 = i32::MIN / 4; // big negative, but leaves headroom for arithmetic

    // Boundary (empty prefix): M = 0, X/Y = neg_sent — matches the full-matrix row 0.
    let mut prev_m = vec![0i32; cols];
    let mut prev_x = vec![neg_sent; cols];
    let mut prev_y = vec![neg_sent; cols];
    let mut cur_m = vec![0i32; cols];
    let mut cur_x = vec![neg_sent; cols];
    let mut cur_y = vec![neg_sent; cols];

    // Packed traceback, one byte per cell:
    //   bits 0-1 = M predecessor: 0 stop, 1 diagonal-from-M, 2 from-X, 3 from-Y;
    //   bit 2    = X predecessor: 0 from M, 1 from X (X is the gap-in-query / current-row
    //              LEFT dependency → Delete, j-=1);
    //   bit 3    = Y predecessor: 0 from M, 1 from Y (Y is the gap-in-target / prev-row
    //              UP dependency → Insert, i-=1).
    let mut tb = vec![0u8; rows * cols];

    let mut best_score: i32 = 0;
    let mut best_i: usize = 0;
    let mut best_j: usize = 0;

    for i in 1..rows {
        let q_idx = query[i - 1] as usize;
        debug_assert!(q_idx < alphabet_size);
        // Column-0 boundary for this row (M=0, X/Y=neg_sent), as in the full matrix.
        cur_m[0] = 0;
        cur_x[0] = neg_sent;
        cur_y[0] = neg_sent;
        for j in 1..cols {
            let t_idx = target[j - 1] as usize;
            debug_assert!(t_idx < alphabet_size);
            let sub = scores[q_idx * alphabet_size + t_idx];

            // X[i,j] = gap in query (consume target letter j-1), current-row LEFT:
            // = max(M[i, j-1] + gap_open, X[i, j-1] + gap_extend)
            let x_from_m = cur_m[j - 1] + gap_open;
            let x_from_x = cur_x[j - 1] + gap_extend;
            let (x_val, x_is_x) = if x_from_m >= x_from_x {
                (x_from_m, false)
            } else {
                (x_from_x, true)
            };
            cur_x[j] = x_val;

            // Y[i,j] = gap in target (consume query letter i-1), prev-row UP:
            // = max(M[i-1, j] + gap_open, Y[i-1, j] + gap_extend)
            let y_from_m = prev_m[j] + gap_open;
            let y_from_y = prev_y[j] + gap_extend;
            let (y_val, y_is_y) = if y_from_m >= y_from_y {
                (y_from_m, false)
            } else {
                (y_from_y, true)
            };
            cur_y[j] = y_val;

            // M[i,j] = diagonal step + substitution, from M/X/Y at (i-1, j-1) (prev row).
            let m_from_m = prev_m[j - 1];
            let m_from_x = prev_x[j - 1];
            let m_from_y = prev_y[j - 1];
            let mut best = m_from_m;
            let mut m_pred = 1u8;
            if m_from_x > best {
                best = m_from_x;
                m_pred = 2;
            }
            if m_from_y > best {
                best = m_from_y;
                m_pred = 3;
            }
            let m_val = best + sub;
            // Local alignment: clamp at 0, and mark "stop" traceback.
            let (m_final, m_code) = if m_val > 0 { (m_val, m_pred) } else { (0, 0u8) };
            cur_m[j] = m_final;

            tb[i * cols + j] = m_code | (u8::from(x_is_x) << 2) | (u8::from(y_is_y) << 3);

            if m_final > best_score {
                best_score = m_final;
                best_i = i;
                best_j = j;
            }
        }
        // Advance: the row just filled becomes `prev`. Swap all three pairs TOGETHER so
        // `prev_*` stay mutually consistent; the new `cur_*` interiors are overwritten
        // for every j>=1 next row (and column 0 is reset above).
        std::mem::swap(&mut prev_m, &mut cur_m);
        std::mem::swap(&mut prev_x, &mut cur_x);
        std::mem::swap(&mut prev_y, &mut cur_y);
    }

    if best_score == 0 {
        return None;
    }

    // Traceback from (best_i, best_j) in M until a stop cell, unpacking one byte per cell.
    let mut cigar_rev: Vec<CigarOp> = Vec::new();
    let mut i = best_i;
    let mut j = best_j;
    let mut current = 'M';
    while i > 0 && j > 0 {
        let packed = tb[i * cols + j];
        match current {
            'M' => {
                let pred = packed & 0b11;
                if pred == 0 {
                    break;
                }
                push_op(&mut cigar_rev, CigarOp::Match(1));
                i -= 1;
                j -= 1;
                current = match pred {
                    1 => 'M',
                    2 => 'X',
                    3 => 'Y',
                    _ => unreachable!("invalid M predecessor {pred}"),
                };
            }
            'X' => {
                // gap in query: consume target letter → Delete.
                push_op(&mut cigar_rev, CigarOp::Delete(1));
                j -= 1;
                current = if packed & 0b100 != 0 { 'X' } else { 'M' };
            }
            'Y' => {
                // gap in target: consume query letter → Insert.
                push_op(&mut cigar_rev, CigarOp::Insert(1));
                i -= 1;
                current = if packed & 0b1000 != 0 { 'Y' } else { 'M' };
            }
            _ => unreachable!(),
        }
    }
    let query_start = i;
    let target_start = j;

    cigar_rev.reverse();
    Some(GappedAlignment {
        score: best_score,
        query_start,
        query_end: best_i,
        target_start,
        target_end: best_j,
        cigar: cigar_rev,
    })
}

/// Push a 1-count op onto the traceback, coalescing with the previous op
/// of the same kind.
fn push_op(cigar_rev: &mut Vec<CigarOp>, op: CigarOp) {
    match (cigar_rev.last_mut(), op) {
        (Some(CigarOp::Match(n)), CigarOp::Match(1)) => *n += 1,
        (Some(CigarOp::Insert(n)), CigarOp::Insert(1)) => *n += 1,
        (Some(CigarOp::Delete(n)), CigarOp::Delete(1)) => *n += 1,
        _ => cigar_rev.push(op),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::kmer_generator::widen_to_i32;
    use crate::matrix::SubstitutionMatrix;
    use crate::sequence::Sequence;

    fn identity_matrix(alphabet_size: usize, match_score: i32, mismatch_score: i32) -> Vec<i32> {
        let mut m = vec![mismatch_score; alphabet_size * alphabet_size];
        for i in 0..alphabet_size {
            m[i * alphabet_size + i] = match_score;
        }
        m
    }

    #[test]
    fn identical_sequences_produce_full_match_cigar() {
        let scores = identity_matrix(4, 3, -2);
        let q: Vec<u8> = vec![0, 1, 2, 3, 0];
        let t = q.clone();
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        assert_eq!(a.score, 5 * 3);
        assert_eq!(a.query_start, 0);
        assert_eq!(a.query_end, 5);
        assert_eq!(a.target_start, 0);
        assert_eq!(a.target_end, 5);
        assert_eq!(a.cigar, vec![CigarOp::Match(5)]);
        assert_eq!(a.cigar_string(), "5M");
    }

    #[test]
    fn completely_unrelated_sequences_return_none() {
        // Disjoint alphabets per side: no possible matching pair.
        let scores = identity_matrix(4, 2, -10);
        let q: Vec<u8> = vec![0, 0, 0];
        let t: Vec<u8> = vec![1, 1, 1];
        assert!(smith_waterman(&q, &t, &scores, 4, -5, -1).is_none());
    }

    #[test]
    fn empty_input_returns_none() {
        let scores = identity_matrix(4, 2, -1);
        assert!(smith_waterman(&[], &[0u8, 1, 2], &scores, 4, -5, -1).is_none());
        assert!(smith_waterman(&[0u8, 1, 2], &[], &scores, 4, -5, -1).is_none());
    }

    #[test]
    fn alignment_with_single_insertion_in_query() {
        // Query has one extra residue in the middle that the target lacks.
        // Expected CIGAR: some matches, 1I, some matches.
        let scores = identity_matrix(4, 3, -2);
        // q:  A C _ G T    (underscore = the inserted query residue)
        //     A C X G T
        // t:  A C   G T
        // So q = [0, 1, 2, 2, 3] (middle 2 is "extra"; using same alphabet byte
        // for the insertion to still score positive on the diagonals).
        //
        // Simpler: build it directly. q=[0,1,0,2,3], t=[0,1,2,3] → best alignment
        // has a 1I at position 2. Matches score: 0/0, 1/1, 2/2, 3/3 = 4 matches.
        // Insertion penalty = gap_open (1 column). With +3/-2 matrix and gap_open=-5,
        // score = 4*3 + (-5) = 7.
        let q: Vec<u8> = vec![0, 1, 0, 2, 3];
        let t: Vec<u8> = vec![0, 1, 2, 3];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        assert_eq!(a.score, 7);
        assert!(a.cigar.iter().any(|op| matches!(op, CigarOp::Insert(_))));
    }

    #[test]
    fn alignment_with_single_deletion_from_query() {
        let scores = identity_matrix(4, 3, -2);
        // q=[0,1,2,3], t=[0,1,0,2,3] → target has an extra 0 → deletion from query.
        let q: Vec<u8> = vec![0, 1, 2, 3];
        let t: Vec<u8> = vec![0, 1, 0, 2, 3];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        assert_eq!(a.score, 7); // 4 matches × 3 - 5 gap open
        assert!(a.cigar.iter().any(|op| matches!(op, CigarOp::Delete(_))));
    }

    #[test]
    fn affine_vs_linear_gap_behavior() {
        // Affine should coalesce a multi-residue insertion into a single
        // gap with one open + extends, not multiple opens.
        //
        // q has a 2-residue insertion in the middle of an otherwise-perfect
        // 10-residue match. Use a high mismatch penalty so the aligner
        // can't escape the gap by accepting mismatches.
        //
        // q (12): 0 1 2 3 0  X X  1 2 3 0 1
        // t (10): 0 1 2 3 0      1 2 3 0 1
        //
        // Best alignment: 5M 2I 5M.
        // Score: 10*3 + (-5) + (-1) = 24 with affine (single 2I gap).
        // Linear cost (two 1I): 10*3 + (-5) + (-5) = 20.
        // Mismatch-only path scores at most 5*3 = 15 (just one half).
        let scores = identity_matrix(4, 3, -10);
        let q: Vec<u8> = vec![0, 1, 2, 3, 0, /* gap */ 2, 2, /* */ 1, 2, 3, 0, 1];
        let t: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        assert_eq!(a.score, 24, "affine 5M 2I 5M = 30 - 5 - 1 = 24");

        // CIGAR should have a single 2I, not two 1I ops.
        let inserts: Vec<&CigarOp> = a
            .cigar
            .iter()
            .filter(|op| matches!(op, CigarOp::Insert(_)))
            .collect();
        assert_eq!(inserts.len(), 1, "affine should coalesce the gap");
        assert_eq!(*inserts[0], CigarOp::Insert(2));
    }

    #[test]
    fn local_alignment_trims_negative_flanks() {
        // Center matches; flanks mismatch heavily. Local SW must not include
        // the flanks (running score would go negative → new local alignment).
        let scores = identity_matrix(4, 3, -10);
        let q: Vec<u8> = vec![1, 1, 0, 1, 2, 3, 1, 1];
        let t: Vec<u8> = vec![2, 2, 0, 1, 2, 3, 2, 2];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        // HSP spans the 4 matching center residues.
        assert_eq!(a.score, 4 * 3);
        assert_eq!(a.query_start, 2);
        assert_eq!(a.query_end, 6);
        assert_eq!(a.target_start, 2);
        assert_eq!(a.target_end, 6);
        assert_eq!(a.cigar, vec![CigarOp::Match(4)]);
    }

    #[test]
    fn cigar_length_invariants() {
        // Match count sums to query_len and target_len when no gaps;
        // more generally, match_count + insert_count = query_len,
        // match_count + delete_count = target_len.
        let scores = identity_matrix(4, 3, -2);
        let q: Vec<u8> = vec![0, 1, 0, 2, 3];
        let t: Vec<u8> = vec![0, 1, 2, 3];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();

        let mut match_total = 0usize;
        let mut insert_total = 0usize;
        let mut delete_total = 0usize;
        for op in &a.cigar {
            match op {
                CigarOp::Match(n) => match_total += n,
                CigarOp::Insert(n) => insert_total += n,
                CigarOp::Delete(n) => delete_total += n,
            }
        }
        assert_eq!(match_total + insert_total, a.query_len());
        assert_eq!(match_total + delete_total, a.target_len());
    }

    #[test]
    fn cigar_string_format_matches_convention() {
        // Standard BLAST/MMseqs CIGAR: "<count><op>..." with ops MID.
        let scores = identity_matrix(4, 3, -2);
        let q: Vec<u8> = vec![0, 1, 0, 2, 3];
        let t: Vec<u8> = vec![0, 1, 2, 3];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        let s = a.cigar_string();
        // Every character must be a digit or one of M/I/D.
        for c in s.chars() {
            assert!(c.is_ascii_digit() || "MID".contains(c), "bad char {c}");
        }
    }

    #[test]
    fn end_to_end_blosum62_with_gap() {
        // Real BLOSUM62 + a sensible gap. Query has a 2-residue insertion
        // in the middle of an otherwise-identical 30-residue stretch.
        // Local SW must:
        //   - find a positive-scoring alignment
        //   - place the gap as an Insert (q has extra residues vs t)
        //   - cover most of t (long matching flanks justify spanning the gap)
        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let scores = widen_to_i32(&m.to_integer_matrix(2.0, 0.0));

        let t = Sequence::from_ascii(alpha.clone(), b"MNALVVKFGGTSVANAERFLRVADILESNQ");
        // Insert "GG" after position 15 ("MNALVVKFGGTSVAN" + "GG" + "AERFLRVADILESNQ").
        let q = Sequence::from_ascii(alpha.clone(), b"MNALVVKFGGTSVANGGAERFLRVADILESNQ");

        let a = smith_waterman(&q.data, &t.data, &scores, alpha.size(), -11, -1).unwrap();
        assert!(a.score > 0);
        // 15 matches + gap + 15 matches → alignment_length ~ 32, target_len = 30.
        assert!(
            a.target_len() >= 25,
            "expected to cover most of target, got target_len={}",
            a.target_len(),
        );
        let cigar = a.cigar_string();
        assert!(
            cigar.contains('I'),
            "expected an Insert in CIGAR (q has extra residues), got {cigar}",
        );
    }

    /// Reference implementation: the ORIGINAL full-matrix (6-array) Gotoh DP, kept
    /// verbatim so the rolling-row + packed-traceback rewrite can be checked for
    /// byte-identical output (not just equal score — an equally-optimal alternate path
    /// would slip past a score-only or brute-force check).
    fn sw_reference(
        query: &[u8],
        target: &[u8],
        scores: &[i32],
        alphabet_size: usize,
        gap_open: i32,
        gap_extend: i32,
    ) -> Option<GappedAlignment> {
        let q_len = query.len();
        let t_len = target.len();
        if q_len == 0 || t_len == 0 {
            return None;
        }
        let rows = q_len + 1;
        let cols = t_len + 1;
        let neg_sent: i32 = i32::MIN / 4;
        let mut m = vec![0i32; rows * cols];
        let mut x = vec![neg_sent; rows * cols];
        let mut y = vec![neg_sent; rows * cols];
        let mut tb_m = vec![0u8; rows * cols];
        let mut tb_x = vec![0u8; rows * cols];
        let mut tb_y = vec![0u8; rows * cols];
        let (mut best_score, mut best_i, mut best_j) = (0i32, 0usize, 0usize);
        for i in 1..rows {
            let q_idx = query[i - 1] as usize;
            for j in 1..cols {
                let t_idx = target[j - 1] as usize;
                let sub = scores[q_idx * alphabet_size + t_idx];
                let x_from_m = m[i * cols + (j - 1)] + gap_open;
                let x_from_x = x[i * cols + (j - 1)] + gap_extend;
                let (xv, xp) = if x_from_m >= x_from_x {
                    (x_from_m, 4u8)
                } else {
                    (x_from_x, 5u8)
                };
                x[i * cols + j] = xv;
                tb_x[i * cols + j] = xp;
                let y_from_m = m[(i - 1) * cols + j] + gap_open;
                let y_from_y = y[(i - 1) * cols + j] + gap_extend;
                let (yv, yp) = if y_from_m >= y_from_y {
                    (y_from_m, 6u8)
                } else {
                    (y_from_y, 7u8)
                };
                y[i * cols + j] = yv;
                tb_y[i * cols + j] = yp;
                let mut best = m[(i - 1) * cols + (j - 1)];
                let mut pred = 1u8;
                if x[(i - 1) * cols + (j - 1)] > best {
                    best = x[(i - 1) * cols + (j - 1)];
                    pred = 2;
                }
                if y[(i - 1) * cols + (j - 1)] > best {
                    best = y[(i - 1) * cols + (j - 1)];
                    pred = 3;
                }
                let m_val = best + sub;
                let (mf, mp) = if m_val > 0 { (m_val, pred) } else { (0, 0) };
                m[i * cols + j] = mf;
                tb_m[i * cols + j] = mp;
                if mf > best_score {
                    best_score = mf;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best_score == 0 {
            return None;
        }
        let mut cigar_rev: Vec<CigarOp> = Vec::new();
        let (mut i, mut j, mut current) = (best_i, best_j, 'M');
        while i > 0 && j > 0 {
            match current {
                'M' => {
                    let pred = tb_m[i * cols + j];
                    if pred == 0 {
                        break;
                    }
                    push_op(&mut cigar_rev, CigarOp::Match(1));
                    i -= 1;
                    j -= 1;
                    current = match pred {
                        1 => 'M',
                        2 => 'X',
                        3 => 'Y',
                        _ => unreachable!(),
                    };
                }
                'X' => {
                    let pred = tb_x[i * cols + j];
                    push_op(&mut cigar_rev, CigarOp::Delete(1));
                    j -= 1;
                    current = match pred {
                        4 => 'M',
                        5 => 'X',
                        _ => unreachable!(),
                    };
                }
                'Y' => {
                    let pred = tb_y[i * cols + j];
                    push_op(&mut cigar_rev, CigarOp::Insert(1));
                    i -= 1;
                    current = match pred {
                        6 => 'M',
                        7 => 'Y',
                        _ => unreachable!(),
                    };
                }
                _ => unreachable!(),
            }
        }
        cigar_rev.reverse();
        Some(GappedAlignment {
            score: best_score,
            query_start: i,
            query_end: best_i,
            target_start: j,
            target_end: best_j,
            cigar: cigar_rev,
        })
    }

    /// Tiny deterministic LCG (no `rand` dependency) for the differential test.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next_u32() % ((hi - lo + 1) as u32)) as i32
        }
    }

    #[test]
    fn differential_matches_reference_random() {
        // The rolling-row + packed-traceback rewrite must be byte-identical to the
        // original full-matrix DP across many random (incl. tie-heavy) inputs.
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        for _ in 0..4000 {
            let a = (rng.next_u32() % 5 + 1) as usize; // alphabet size 1..=5
            let ql = (rng.next_u32() % 9) as usize; // 0..=8 (incl. empty)
            let tl = (rng.next_u32() % 9) as usize;
            let q: Vec<u8> = (0..ql)
                .map(|_| (rng.next_u32() as usize % a) as u8)
                .collect();
            let t: Vec<u8> = (0..tl)
                .map(|_| (rng.next_u32() as usize % a) as u8)
                .collect();
            // Random (possibly asymmetric) substitution matrix, small range to force ties.
            let scores: Vec<i32> = (0..a * a).map(|_| rng.in_range(-3, 3)).collect();
            // Gap penalties incl. 0 (tie-prone) and open==extend.
            let gap_open = rng.in_range(-6, 0);
            let gap_extend = rng.in_range(-3, 0);
            let got = smith_waterman(&q, &t, &scores, a, gap_open, gap_extend);
            let want = sw_reference(&q, &t, &scores, a, gap_open, gap_extend);
            assert_eq!(
                got, want,
                "mismatch q={q:?} t={t:?} a={a} scores={scores:?} go={gap_open} ge={gap_extend}"
            );
        }
    }

    #[test]
    fn differential_matches_reference_exhaustive_small() {
        // Exhaustive over all q,t up to length 4 on a 2-letter alphabet, with a few
        // matrices (identity, asymmetric, all-equal → maximal ties) and gap settings.
        let a = 2;
        let matrices = [
            vec![2, -1, -1, 2], // identity-ish
            vec![1, 0, -2, 1],  // asymmetric
            vec![1, 1, 1, 1],   // all-equal → every M predecessor ties
        ];
        let gaps = [(-1, -1), (0, 0), (-3, -1), (-2, -2)];
        for len_q in 0..=4usize {
            for len_t in 0..=4usize {
                for qbits in 0..(1u32 << len_q) {
                    let q: Vec<u8> = (0..len_q).map(|k| ((qbits >> k) & 1) as u8).collect();
                    for tbits in 0..(1u32 << len_t) {
                        let t: Vec<u8> = (0..len_t).map(|k| ((tbits >> k) & 1) as u8).collect();
                        for scores in &matrices {
                            for &(go, ge) in &gaps {
                                let got = smith_waterman(&q, &t, scores, a, go, ge);
                                let want = sw_reference(&q, &t, scores, a, go, ge);
                                assert_eq!(
                                    got, want,
                                    "q={q:?} t={t:?} m={scores:?} go={go} ge={ge}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "stress: ~4000x4000 pair; confirms the rolling-row layout no longer needs the full grid"]
    fn large_pair_does_not_allocate_full_grid() {
        let scores = identity_matrix(4, 3, -2);
        let n = 4000;
        let mut rng = Lcg(7);
        let q: Vec<u8> = (0..n).map(|_| (rng.next_u32() % 4) as u8).collect();
        let t: Vec<u8> = (0..n).map(|_| (rng.next_u32() % 4) as u8).collect();
        // Would have been ~3*4000*4000*4 + 3*4000*4000 ≈ 240 MB of score+tb grids;
        // now the score grids are gone (rolling rows) and traceback is 1 byte/cell.
        let _ = smith_waterman(&q, &t, &scores, 4, -5, -1);
    }

    #[test]
    fn cigar_match_display_implements() {
        let scores = identity_matrix(4, 3, -2);
        let q: Vec<u8> = vec![0, 1];
        let t: Vec<u8> = vec![0, 1];
        let a = smith_waterman(&q, &t, &scores, 4, -5, -1).unwrap();
        let s = format!("{a}");
        assert!(s.contains("score="));
        assert!(s.contains("cigar=2M"));
    }
}
