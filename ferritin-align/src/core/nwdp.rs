//! Needleman-Wunsch dynamic programming alignment variants.
//!
//! Ported from C++ TMalign (lines 1321-1645). Four variants that differ only
//! in how the diagonal (match) score is computed:
//! 1. Pre-computed score matrix
//! 2. Coordinate distance with rotation
//! 3. Coordinate distance without rotation
//! 4. Secondary structure matching
//!
//! Reference: Needleman & Wunsch, "A general method applicable to the search
//! for similarities in the amino acid sequence of two proteins", *J. Mol.
//! Biol.* 48(3), 443-453 (1970).

use crate::core::types::{dist_squared, Coord3D, DPWorkspace, Transform};

/// Core DP fill + traceback shared by all variants.
///
/// `score_fn(i, j)` returns the diagonal score for aligning position i-1 in seq1
/// with position j-1 in seq2 (1-indexed i,j as in the DP matrix).
///
/// Returns alignment map: `j2i[j] = i` means position j in seq2 aligns to i in seq1,
/// or -1 for gap. Length = len2.
fn nwdp_core(
    ws: &mut DPWorkspace,
    len1: usize,
    len2: usize,
    gap_open: f64,
    score_fn: impl Fn(usize, usize) -> f64,
) -> Vec<i32> {
    ws.ensure_size(len1, len2);

    // Initialization
    for i in 0..=len1 {
        ws.val[i][0] = 0.0;
        ws.path[i][0] = false;
    }
    for j in 0..=len2 {
        ws.val[0][j] = 0.0;
        ws.path[0][j] = false;
    }

    // DP fill
    for i in 1..=len1 {
        for j in 1..=len2 {
            let d = ws.val[i - 1][j - 1] + score_fn(i, j);

            let mut h = ws.val[i - 1][j];
            if ws.path[i - 1][j] {
                h += gap_open;
            }

            let mut v = ws.val[i][j - 1];
            if ws.path[i][j - 1] {
                v += gap_open;
            }

            if d >= h && d >= v {
                ws.path[i][j] = true;
                ws.val[i][j] = d;
            } else {
                ws.path[i][j] = false;
                ws.val[i][j] = if v >= h { v } else { h };
            }
        }
    }

    // Traceback
    let mut j2i = vec![-1_i32; len2];
    let mut i = len1;
    let mut j = len2;
    while i > 0 && j > 0 {
        if ws.path[i][j] {
            j2i[j - 1] = (i - 1) as i32;
            i -= 1;
            j -= 1;
        } else {
            let mut h = ws.val[i - 1][j];
            if ws.path[i - 1][j] {
                h += gap_open;
            }
            let mut v = ws.val[i][j - 1];
            if ws.path[i][j - 1] {
                v += gap_open;
            }
            if v >= h {
                j -= 1;
            } else {
                i -= 1;
            }
        }
    }

    j2i
}

/// NW-DP alignment using a pre-computed score matrix.
///
/// `score` is 1-indexed: `score[i][j]` for i in 1..=len1, j in 1..=len2.
/// Corresponds to C++ `NWDP_TM(double **score, ...)`.
pub fn nwdp_score_matrix(
    ws: &mut DPWorkspace,
    score: &[Vec<f64>],
    len1: usize,
    len2: usize,
    gap_open: f64,
) -> Vec<i32> {
    nwdp_core(ws, len1, len2, gap_open, |i, j| score[i][j])
}

/// NW-DP alignment using coordinate distance with rotation.
///
/// Score = 1/(1 + dist²/d02) where dist is after applying transform.
/// Corresponds to C++ `NWDP_TM(..., double t[3], double u[3][3], double d02, ...)`.
pub fn nwdp_coords(
    ws: &mut DPWorkspace,
    x: &[Coord3D],
    y: &[Coord3D],
    transform: &Transform,
    d02: f64,
    gap_open: f64,
) -> Vec<i32> {
    let len1 = x.len();
    let len2 = y.len();

    // Pre-compute transformed x coordinates
    let xt: Vec<Coord3D> = x.iter().map(|xi| transform.apply(xi)).collect();

    nwdp_core(ws, len1, len2, gap_open, |i, j| {
        let dij = dist_squared(&xt[i - 1], &y[j - 1]);
        1.0 / (1.0 + dij / d02)
    })
}

/// NW-DP alignment using coordinate distance without rotation.
///
/// Score = 1/(1 + dist²/d02) using raw distances.
/// Corresponds to C++ `NWDP_SE(...)`.
pub fn nwdp_coords_no_rotation(
    ws: &mut DPWorkspace,
    x: &[Coord3D],
    y: &[Coord3D],
    d02: f64,
    gap_open: f64,
) -> Vec<i32> {
    let len1 = x.len();
    let len2 = y.len();
    nwdp_core(ws, len1, len2, gap_open, |i, j| {
        let dij = dist_squared(&x[i - 1], &y[j - 1]);
        1.0 / (1.0 + dij / d02)
    })
}

/// NW-DP alignment using secondary structure matching.
///
/// Score = 1.0 if secx[i-1] == secy[j-1], else 0.0.
/// Corresponds to C++ `NWDP_TM(..., const char *secx, const char *secy, ...)`.
pub fn nwdp_secondary_structure(
    ws: &mut DPWorkspace,
    secx: &[char],
    secy: &[char],
    gap_open: f64,
) -> Vec<i32> {
    let len1 = secx.len();
    let len2 = secy.len();
    nwdp_core(ws, len1, len2, gap_open, |i, j| {
        if secx[i - 1] == secy[j - 1] {
            1.0
        } else {
            0.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences_perfect_alignment() {
        let mut ws = DPWorkspace::new(10, 10);
        let sec = vec!['H', 'H', 'E', 'E', 'C'];
        let j2i = nwdp_secondary_structure(&mut ws, &sec, &sec, -1.0);
        // Perfect diagonal alignment expected
        for (j, &val) in j2i.iter().enumerate() {
            assert_eq!(val, j as i32, "position {j} should align to itself");
        }
    }

    #[test]
    fn completely_different_ss() {
        let mut ws = DPWorkspace::new(10, 10);
        let secx = vec!['H', 'H', 'H'];
        let secy = vec!['E', 'E', 'E'];
        let j2i = nwdp_secondary_structure(&mut ws, &secx, &secy, -1.0);
        // With gap_open=-1.0 and score=0 for mismatches, alignment is still forced
        // but all scores are 0 so traceback depends on tie-breaking
        // Just verify it doesn't crash and returns correct length
        assert_eq!(j2i.len(), 3);
    }

    #[test]
    fn coords_identical_points() {
        let mut ws = DPWorkspace::new(10, 10);
        let pts: Vec<Coord3D> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let transform = Transform::default(); // identity
        let d02 = 1.0;
        let j2i = nwdp_coords(&mut ws, &pts, &pts, &transform, d02, -0.6);
        for (j, &val) in j2i.iter().enumerate() {
            assert_eq!(val, j as i32, "position {j} should align to itself");
        }
    }

    #[test]
    fn score_matrix_simple() {
        let mut ws = DPWorkspace::new(5, 5);
        // 3x3 score matrix (1-indexed), high score on diagonal
        let score = vec![
            vec![0.0; 4], // row 0 (unused)
            vec![0.0, 1.0, 0.1, 0.1],
            vec![0.0, 0.1, 1.0, 0.1],
            vec![0.0, 0.1, 0.1, 1.0],
        ];
        let j2i = nwdp_score_matrix(&mut ws, &score, 3, 3, -0.6);
        assert_eq!(j2i, vec![0, 1, 2]);
    }

    #[test]
    fn unequal_lengths() {
        let mut ws = DPWorkspace::new(10, 10);
        let secx = vec!['H', 'H', 'E', 'E', 'C'];
        let secy = vec!['H', 'E', 'C'];
        let j2i = nwdp_secondary_structure(&mut ws, &secx, &secy, -1.0);
        assert_eq!(j2i.len(), 3);
        // Each position in secy should align to a valid position in secx or -1
        for &val in &j2i {
            assert!(val == -1 || (0..5).contains(&val));
        }
    }
}
