//! Enhanced greedy search with swap refinement for SOI alignment.
//!
//! Ported from C++ USAlign `SOIalign.h` `soi_egs` and `sec2sq`.
//! Two-stage algorithm:
//!   1. Greedy assignment of highest-scoring unassigned pairs
//!   2. Iterative swap refinement until no improvement

use crate::ext::soialign::sec_bond::SecBond;

/// Check if pairing residue `i` (in x) to residue `j` (in y) conforms to
/// sequentiality within SSE boundaries.
///
/// Returns `true` if the pairing is legal (no sequentiality violation),
/// `false` if it would create a crossing within an SSE.
///
/// Corresponds to C++ `sec2sq`.
#[inline]
pub fn sec2sq(
    i: i32,
    j: i32,
    secx_bond: &[SecBond],
    secy_bond: &[SecBond],
    fwdmap: &[i32],
    invmap: &[i32],
) -> bool {
    if i < 0 || j < 0 {
        return true;
    }
    let iu = i as usize;
    let ju = j as usize;

    // Check sequentiality in the x-side SSE
    if secx_bond[iu][0] >= 0 {
        let start = secx_bond[iu][0] as usize;
        let end = secx_bond[iu][1] as usize;
        for ii in start..end {
            let jj = fwdmap[ii];
            if jj >= 0 && (i - ii as i32) as i64 * (j - jj) as i64 <= 0 {
                return false;
            }
        }
    }

    // Check sequentiality in the y-side SSE
    if secy_bond[ju][0] >= 0 {
        let start = secy_bond[ju][0] as usize;
        let end = secy_bond[ju][1] as usize;
        for jj in start..end {
            let ii = invmap[jj];
            if ii >= 0 && (i - ii) as i64 * (j - jj as i32) as i64 <= 0 {
                return false;
            }
        }
    }

    true
}

/// Enhanced greedy search with swap refinement.
///
/// Modifies `invmap` in place (j->i mapping, -1 for unassigned).
/// `score` is 1-indexed: `score[i+1][j+1]` for 0-indexed i,j.
///
/// `mm_opt == 6` enables SSE sequentiality constraints via `sec2sq`.
///
/// Two stages:
/// 1. Greedily assign highest-scoring unassigned pairs
/// 2. Iteratively swap pairs to improve total score
///
/// Corresponds to C++ `soi_egs`.
pub fn soi_egs(
    score: &[Vec<f64>],
    xlen: usize,
    ylen: usize,
    invmap: &mut [i32],
    secx_bond: &[SecBond],
    secy_bond: &[SecBond],
    mm_opt: i32,
) {
    let mut fwdmap = vec![-1i32; xlen];
    for j in 0..ylen {
        let i = invmap[j];
        if i >= 0 {
            fwdmap[i as usize] = j as i32;
        }
    }

    // Stage 1: make initial assignment, starting from the highest score pair
    loop {
        let mut max_score = 0.0_f64;
        let mut maxi: i32 = -1;
        let mut maxj: i32 = -1;

        for i in 0..xlen {
            if fwdmap[i] >= 0 {
                continue;
            }
            for j in 0..ylen {
                if invmap[j] >= 0 || score[i + 1][j + 1] <= max_score {
                    continue;
                }
                if mm_opt == 6 && !sec2sq(i as i32, j as i32, secx_bond, secy_bond, &fwdmap, invmap)
                {
                    continue;
                }
                maxi = i as i32;
                maxj = j as i32;
                max_score = score[i + 1][j + 1];
            }
        }
        if maxi < 0 {
            break; // no more assignments possible
        }
        invmap[maxj as usize] = maxi;
        fwdmap[maxi as usize] = maxj;
    }

    // Compute initial total score
    let mut _total_score = 0.0_f64;
    for j in 0..ylen {
        let i = invmap[j];
        if i >= 0 {
            _total_score += score[i as usize + 1][j + 1];
        }
    }

    // Stage 2: swap assignment until total score cannot be improved
    let max_iter = xlen.min(ylen) * 5;
    for _iter in 0..max_iter {
        let mut improved = false;
        for i in 0..xlen {
            let oldj = fwdmap[i];
            for j in 0..ylen {
                let oldi = invmap[j];
                if score[i + 1][j + 1] <= 0.0 || oldi == i as i32 {
                    continue;
                }
                if mm_opt == 6
                    && (!sec2sq(i as i32, j as i32, secx_bond, secy_bond, &fwdmap, invmap)
                        || !sec2sq(oldi, oldj, secx_bond, secy_bond, &fwdmap, invmap))
                {
                    continue;
                }

                let mut delta_score = score[i + 1][j + 1];
                if oldi >= 0 && oldj >= 0 {
                    delta_score += score[oldi as usize + 1][oldj as usize + 1];
                }
                if oldi >= 0 {
                    delta_score -= score[oldi as usize + 1][j + 1];
                }
                if oldj >= 0 {
                    delta_score -= score[i + 1][oldj as usize + 1];
                }

                if delta_score > 0.0 {
                    // Successful swap
                    fwdmap[i] = j as i32;
                    if oldi >= 0 {
                        fwdmap[oldi as usize] = oldj;
                    }
                    invmap[j] = i as i32;
                    if oldj >= 0 {
                        invmap[oldj as usize] = oldi;
                    }
                    _total_score += delta_score;
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            break; // cannot make further swap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 1-indexed score matrix from a 0-indexed 2D slice.
    fn make_score_matrix(xlen: usize, ylen: usize, data: &[f64]) -> Vec<Vec<f64>> {
        let mut score = vec![vec![0.0; ylen + 1]; xlen + 1];
        for i in 0..xlen {
            for j in 0..ylen {
                score[i + 1][j + 1] = data[i * ylen + j];
            }
        }
        score
    }

    #[test]
    fn test_soi_egs_diagonal_best() {
        // 3x3 identity-like score matrix: diagonal = 1.0, off-diagonal = 0.0
        #[rustfmt::skip]
        let data = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let score = make_score_matrix(3, 3, &data);
        let secx_bond = vec![[-1, -1]; 3];
        let secy_bond = vec![[-1, -1]; 3];
        let mut invmap = vec![-1i32; 3];

        soi_egs(&score, 3, 3, &mut invmap, &secx_bond, &secy_bond, 0);

        assert_eq!(invmap[0], 0);
        assert_eq!(invmap[1], 1);
        assert_eq!(invmap[2], 2);
    }

    #[test]
    fn test_soi_egs_swap_improves() {
        // Score matrix where greedy picks (0->0, 1->1) but swap to (0->1, 1->0)
        // is better overall.
        //       y0    y1
        // x0   0.5   0.9
        // x1   0.9   0.5
        #[rustfmt::skip]
        let data = [
            0.5, 0.9,
            0.9, 0.5,
        ];
        let score = make_score_matrix(2, 2, &data);
        let secx_bond = vec![[-1, -1]; 2];
        let secy_bond = vec![[-1, -1]; 2];
        let mut invmap = vec![-1i32; 2];

        soi_egs(&score, 2, 2, &mut invmap, &secx_bond, &secy_bond, 0);

        // Greedy first picks max=0.9 at (0,1) or (1,0), then fills remaining.
        // Either way total is 0.9+0.5=1.4, which is optimal.
        let total: f64 = (0..2)
            .filter_map(|j| {
                let i = invmap[j];
                if i >= 0 {
                    Some(score[i as usize + 1][j + 1])
                } else {
                    None
                }
            })
            .sum();
        assert!(total >= 1.3, "total={}", total);
    }

    #[test]
    fn test_soi_egs_empty() {
        // All zero scores: nothing should be assigned
        let data = [0.0; 4];
        let score = make_score_matrix(2, 2, &data);
        let secx_bond = vec![[-1, -1]; 2];
        let secy_bond = vec![[-1, -1]; 2];
        let mut invmap = vec![-1i32; 2];

        soi_egs(&score, 2, 2, &mut invmap, &secx_bond, &secy_bond, 0);

        assert_eq!(invmap[0], -1);
        assert_eq!(invmap[1], -1);
    }

    #[test]
    fn test_sec2sq_no_sse() {
        let secx_bond = vec![[-1i32, -1]; 5];
        let secy_bond = vec![[-1i32, -1]; 5];
        let fwdmap = vec![-1i32; 5];
        let invmap = vec![-1i32; 5];
        // No SSE constraints: always legal
        assert!(sec2sq(0, 0, &secx_bond, &secy_bond, &fwdmap, &invmap));
        assert!(sec2sq(2, 3, &secx_bond, &secy_bond, &fwdmap, &invmap));
    }

    #[test]
    fn test_sec2sq_negative_indices() {
        let secx_bond: Vec<SecBond> = vec![];
        let secy_bond: Vec<SecBond> = vec![];
        let fwdmap: Vec<i32> = vec![];
        let invmap: Vec<i32> = vec![];
        assert!(sec2sq(-1, 3, &secx_bond, &secy_bond, &fwdmap, &invmap));
        assert!(sec2sq(2, -1, &secx_bond, &secy_bond, &fwdmap, &invmap));
    }
}
