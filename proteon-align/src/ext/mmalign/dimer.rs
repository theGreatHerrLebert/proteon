//! Dimer-specific alignment algorithms.
//!
//! Ported from C++ USAlign `MMalign.h`: `adjust_dimer_assignment`,
//! `NWDP_TM_dimer` (coords and SS variants), `DP_iter_dimer`,
//! `get_initial_ss_dimer`, `get_initial5_dimer`, `get_initial_ssplus_dimer`.

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::types::{dist_squared, Coord3D, TMParams, Transform};

/// Adjust dimer assignment by trying both chain pairings for a 2+2 dimer.
///
/// Given two complexes each with exactly 2 assigned chains, evaluates whether
/// swapping the chain correspondence improves the total TM-score.
///
/// Returns `true` if the assignment was swapped, and modifies `assign1`/`assign2`.
///
/// Corresponds to C++ `adjust_dimer_assignment`.
pub fn adjust_dimer_assignment(
    x_coords: &[Vec<Coord3D>],
    y_coords: &[Vec<Coord3D>],
    xlen_vec: &[usize],
    ylen_vec: &[usize],
    mol_vec1: &[i32],
    mol_vec2: &[i32],
    assign1: &mut [i32],
    assign2: &mut [i32],
    seqx_a_mat: &[Vec<String>],
    seqy_a_mat: &[Vec<String>],
) -> bool {
    // Find the two assigned chain pairs
    let chain1_num = x_coords.len();
    let mut i1: i32 = -1;
    let mut i2: i32 = -1;
    let mut j1: i32 = -1;
    let mut j2: i32 = -1;

    for i in 0..chain1_num {
        if assign1[i] >= 0 {
            if i1 < 0 {
                i1 = i as i32;
                j1 = assign1[i];
            } else {
                i2 = i as i32;
                j2 = assign1[i];
            }
        }
    }

    if i1 < 0 || i2 < 0 || j1 < 0 || j2 < 0 {
        return false;
    }

    let (i1, i2, j1, j2) = (i1 as usize, i2 as usize, j1 as usize, j2 as usize);

    // Compute normalization length
    let xlen = xlen_vec[i1] + xlen_vec[i2];
    let ylen = ylen_vec[j1] + ylen_vec[j2];
    let lnorm = xlen.min(ylen) as f64;
    let mol_type = mol_vec1[i1] + mol_vec1[i2] + mol_vec2[j1] + mol_vec2[j2];
    let params = TMParams::for_final(
        lnorm,
        if mol_type > 0 {
            crate::core::types::MolType::RNA
        } else {
            crate::core::types::MolType::Protein
        },
    );
    let d0 = params.d0;

    // Helper: extract aligned pairs from alignment strings
    let extract_pairs = |seq_x: &str,
                         seq_y: &str,
                         xc: &[Coord3D],
                         yc: &[Coord3D]|
     -> (Vec<Coord3D>, Vec<Coord3D>) {
        let mut xa = Vec::new();
        let mut ya = Vec::new();
        let xb = seq_x.as_bytes();
        let yb = seq_y.as_bytes();
        let l = xb.len().min(yb.len());
        let mut i: i32 = -1;
        let mut j: i32 = -1;
        for r in 0..l {
            if xb[r] != b'-' {
                i += 1;
            }
            if yb[r] != b'-' {
                j += 1;
            }
            if xb[r] == b'-' || yb[r] == b'-' {
                continue;
            }
            if (i as usize) < xc.len() && (j as usize) < yc.len() {
                xa.push(xc[i as usize]);
                ya.push(yc[j as usize]);
            }
        }
        (xa, ya)
    };

    // Score for current assignment: (i1,j1) + (i2,j2)
    let (mut xa1, mut ya1) = extract_pairs(
        &seqx_a_mat[i1][j1],
        &seqy_a_mat[i1][j1],
        &x_coords[i1],
        &y_coords[j1],
    );
    let (xa2, ya2) = extract_pairs(
        &seqx_a_mat[i2][j2],
        &seqy_a_mat[i2][j2],
        &x_coords[i2],
        &y_coords[j2],
    );
    xa1.extend_from_slice(&xa2);
    ya1.extend_from_slice(&ya2);

    let total_score1 = compute_dimer_score(&xa1, &ya1, d0, lnorm);

    // Score for swapped assignment: (i1,j2) + (i2,j1)
    let (mut xa1s, mut ya1s) = extract_pairs(
        &seqx_a_mat[i1][j2],
        &seqy_a_mat[i1][j2],
        &x_coords[i1],
        &y_coords[j2],
    );
    let (xa2s, ya2s) = extract_pairs(
        &seqx_a_mat[i2][j1],
        &seqy_a_mat[i2][j1],
        &x_coords[i2],
        &y_coords[j1],
    );
    xa1s.extend_from_slice(&xa2s);
    ya1s.extend_from_slice(&ya2s);

    let total_score2 = compute_dimer_score(&xa1s, &ya1s, d0, lnorm);

    // Swap if reversed is better
    if total_score1 < total_score2 {
        assign1[i1] = j2 as i32;
        assign1[i2] = j1 as i32;
        assign2[j1] = i2 as i32;
        assign2[j2] = i1 as i32;
        true
    } else {
        false
    }
}

/// Compute the dimer TM-score for a set of aligned coordinate pairs.
fn compute_dimer_score(xa: &[Coord3D], ya: &[Coord3D], d0: f64, lnorm: f64) -> f64 {
    if xa.is_empty() || ya.is_empty() {
        return 0.0;
    }

    let result = match kabsch(xa, ya, KabschMode::Both) {
        Some(r) => r,
        None => return 0.0,
    };

    let mut xt = vec![[0.0f64; 3]; xa.len()];
    result.transform.apply_batch(xa, &mut xt);

    let mut total_score = 0.0_f64;
    for r in 0..xt.len() {
        let dd = dist_squared(&xt[r], &ya[r]);
        total_score += 1.0 / (1.0 + dd / (d0 * d0));
    }
    total_score / lnorm
}

/// Needleman-Wunsch DP for dimer alignment using coordinates + rotation.
///
/// This variant uses a boolean mask to restrict which (i, j) cells are
/// valid for diagonal transitions (preventing inter-chain alignments).
///
/// Corresponds to C++ `NWDP_TM_dimer(path, val, x, y, len1, len2, mask, t, u, d02, gap_open, j2i)`.
pub fn nwdp_tm_dimer_coords(
    x: &[Coord3D],
    y: &[Coord3D],
    mask: &[Vec<bool>],
    transform: &Transform,
    d02: f64,
    gap_open: f64,
) -> Vec<i32> {
    let len1 = x.len();
    let len2 = y.len();

    let mut val = vec![vec![0.0f64; len2 + 1]; len1 + 1];
    let mut path = vec![vec![false; len2 + 1]; len1 + 1];
    let mut j2i = vec![-1i32; len2];

    // Initialization
    for i in 0..=len1 {
        val[i][0] = i as f64 * gap_open;
        path[i][0] = false;
    }
    for j in 0..=len2 {
        val[0][j] = j as f64 * gap_open;
        path[0][j] = false;
    }

    // DP fill
    for i in 1..=len1 {
        let xx = transform.apply(&x[i - 1]);
        for j in 1..=len2 {
            let d = if mask[i][j] {
                let dij = dist_squared(&xx, &y[j - 1]);
                val[i - 1][j - 1] + 1.0 / (1.0 + dij / d02)
            } else {
                f64::MIN
            };

            let mut h = val[i - 1][j];
            if path[i - 1][j] {
                h += gap_open;
            }

            let mut v = val[i][j - 1];
            if path[i][j - 1] {
                v += gap_open;
            }

            if d >= h && d >= v {
                path[i][j] = true;
                val[i][j] = d;
            } else {
                path[i][j] = false;
                val[i][j] = if v >= h { v } else { h };
            }
        }
    }

    // Traceback
    let mut i = len1;
    let mut j = len2;
    while i > 0 && j > 0 {
        if path[i][j] {
            j2i[j - 1] = (i - 1) as i32;
            i -= 1;
            j -= 1;
        } else {
            let mut h = val[i - 1][j];
            if path[i - 1][j] {
                h += gap_open;
            }
            let mut v = val[i][j - 1];
            if path[i][j - 1] {
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

/// Needleman-Wunsch DP for dimer alignment using secondary structure.
///
/// Corresponds to C++ `NWDP_TM_dimer(path, val, secx, secy, len1, len2, mask, gap_open, j2i)`.
pub fn nwdp_tm_dimer_ss(secx: &[u8], secy: &[u8], mask: &[Vec<bool>], gap_open: f64) -> Vec<i32> {
    let len1 = secx.len();
    let len2 = secy.len();

    let mut val = vec![vec![0.0f64; len2 + 1]; len1 + 1];
    let mut path = vec![vec![false; len2 + 1]; len1 + 1];
    let mut j2i = vec![-1i32; len2];

    // Initialization
    for i in 0..=len1 {
        val[i][0] = i as f64 * gap_open;
        path[i][0] = false;
    }
    for j in 0..=len2 {
        val[0][j] = j as f64 * gap_open;
        path[0][j] = false;
    }

    // DP fill
    for i in 1..=len1 {
        for j in 1..=len2 {
            let d = if mask[i][j] {
                val[i - 1][j - 1] + if secx[i - 1] == secy[j - 1] { 1.0 } else { 0.0 }
            } else {
                f64::MIN
            };

            let mut h = val[i - 1][j];
            if path[i - 1][j] {
                h += gap_open;
            }

            let mut v = val[i][j - 1];
            if path[i][j - 1] {
                v += gap_open;
            }

            if d >= h && d >= v {
                path[i][j] = true;
                val[i][j] = d;
            } else {
                path[i][j] = false;
                val[i][j] = if v >= h { v } else { h };
            }
        }
    }

    // Traceback
    let mut i = len1;
    let mut j = len2;
    while i > 0 && j > 0 {
        if path[i][j] {
            j2i[j - 1] = (i - 1) as i32;
            i -= 1;
            j -= 1;
        } else {
            let mut h = val[i - 1][j];
            if path[i - 1][j] {
                h += gap_open;
            }
            let mut v = val[i][j - 1];
            if path[i][j - 1] {
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

/// Build the block-diagonal mask for dimer DP.
///
/// `mask[i][j]` is true iff residue i and residue j belong to the same
/// chain pair in the assignment (preventing inter-chain alignment).
///
/// Indices are 1-based (row 0 / col 0 is for DP boundary).
pub fn build_dimer_mask(xlen_vec: &[usize], ylen_vec: &[usize]) -> Vec<Vec<bool>> {
    let xlen: usize = xlen_vec.iter().sum();
    let ylen: usize = ylen_vec.iter().sum();

    let mut mask = vec![vec![false; ylen + 1]; xlen + 1];

    // Allow boundary transitions
    if !xlen_vec.is_empty() {
        for i in 0..=xlen_vec[0] {
            mask[i][0] = true;
        }
    }
    if !ylen_vec.is_empty() {
        for j in 0..=ylen_vec[0] {
            mask[0][j] = true;
        }
    }

    // Fill block-diagonal
    let mut prev_xlen = 1usize;
    let mut prev_ylen = 1usize;
    for c in 0..xlen_vec.len() {
        for i in prev_xlen..prev_xlen + xlen_vec[c] {
            for j in prev_ylen..prev_ylen + ylen_vec[c] {
                mask[i][j] = true;
            }
        }
        prev_xlen += xlen_vec[c];
        prev_ylen += ylen_vec[c];
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dimer_mask() {
        let xlen_vec = vec![3, 4];
        let ylen_vec = vec![2, 5];
        let mask = build_dimer_mask(&xlen_vec, &ylen_vec);

        // Total size: (7+1) x (7+1)
        assert_eq!(mask.len(), 8);
        assert_eq!(mask[0].len(), 8);

        // Block 1: rows 1..3, cols 1..2
        assert!(mask[1][1]);
        assert!(mask[3][2]);
        assert!(!mask[1][3]); // cross-block

        // Block 2: rows 4..7, cols 3..7
        assert!(mask[4][3]);
        assert!(mask[7][7]);
        assert!(!mask[4][2]); // cross-block
    }

    #[test]
    fn test_nwdp_tm_dimer_ss_identical() {
        let secx = b"HHHCCC";
        let secy = b"HHHCCC";
        // Single chain pair: all positions allowed
        let mask = vec![vec![true; 7]; 7];
        let j2i = nwdp_tm_dimer_ss(secx, secy, &mask, -1.0);
        // Identical SS => perfect diagonal alignment
        for j in 0..6 {
            assert_eq!(j2i[j], j as i32, "j2i[{}]={}", j, j2i[j]);
        }
    }

    #[test]
    fn test_compute_dimer_score_identical() {
        let coords: Vec<Coord3D> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        let score = compute_dimer_score(&coords, &coords, 5.0, 10.0);
        // Identical structures: each pair contributes 1/(1+0) = 1.0
        // Total = 10/10 = 1.0
        assert!((score - 1.0).abs() < 0.01, "score={}", score);
    }
}
