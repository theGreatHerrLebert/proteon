//! Chain-to-chain assignment algorithms for multi-chain alignment.
//!
//! Ported from C++ USAlign `MMalign.h`: `enhanced_greedy_search`,
//! `count_assign_pair`, `homo_refined_greedy_search`,
//! `hetero_refined_greedy_search`, `check_heterooligomer`.

use crate::core::types::{dist_squared, Coord3D, Transform};

use super::complex_score::cal_mm_score;

/// Count how many chains in complex 1 are paired (assigned to a chain in complex 2).
///
/// Corresponds to C++ `count_assign_pair`.
pub fn count_assign_pair(assign1_list: &[i32]) -> usize {
    assign1_list.iter().filter(|&&v| v >= 0).count()
}

/// Greedy chain assignment with swap refinement.
///
/// Given a `tm_mat[i][j]` matrix of chain-pair TM-scores, greedily assigns
/// chains from complex 1 to complex 2 (highest score first), then iteratively
/// attempts swaps to improve total score.
///
/// Corresponds to C++ `enhanced_greedy_search`.
///
/// Returns `(assign1, assign2, total_score)` where:
/// - `assign1[i]` = index of chain in complex 2 assigned to chain i, or -1
/// - `assign2[j]` = index of chain in complex 1 assigned to chain j, or -1
pub fn enhanced_greedy_search(
    tm_mat: &[Vec<f64>],
    chain1_num: usize,
    chain2_num: usize,
) -> (Vec<i32>, Vec<i32>, f64) {
    let mut assign1 = vec![-1i32; chain1_num];
    let mut assign2 = vec![-1i32; chain2_num];
    let mut total_score = 0.0_f64;

    // Greedy phase: in each iteration, pick the highest unassigned pair
    loop {
        let mut tmp_score = -1.0_f64;
        let mut maxi = 0usize;
        let mut maxj = 0usize;

        for i in 0..chain1_num {
            if assign1[i] >= 0 {
                continue;
            }
            for j in 0..chain2_num {
                if assign2[j] >= 0 || tm_mat[i][j] <= 0.0 {
                    continue;
                }
                if tm_mat[i][j] > tmp_score {
                    maxi = i;
                    maxj = j;
                    tmp_score = tm_mat[i][j];
                }
            }
        }

        if tmp_score <= 0.0 {
            break;
        }
        assign1[maxi] = maxj as i32;
        assign2[maxj] = maxi as i32;
        total_score += tmp_score;
    }

    if total_score <= 0.0 {
        return (assign1, assign2, total_score);
    }

    // Swap refinement phase
    let mut assign1_tmp = assign1.clone();
    let mut assign2_tmp = assign2.clone();
    let max_iter = chain1_num.min(chain2_num) * 5;

    for _iter in 0..max_iter {
        let mut delta_score = -1.0_f64;

        'outer: for i in 0..chain1_num {
            let old_j = assign1[i];

            for j in 0..chain2_num {
                if j as i32 == assign1[i] || tm_mat[i][j] <= 0.0 {
                    continue;
                }
                let old_i = assign2[j];

                // Attempt swap
                assign1_tmp[i] = j as i32;
                if old_i >= 0 {
                    assign1_tmp[old_i as usize] = old_j;
                }
                assign2_tmp[j] = i as i32;
                if old_j >= 0 {
                    assign2_tmp[old_j as usize] = old_i;
                }

                delta_score = tm_mat[i][j];
                if old_j >= 0 {
                    delta_score -= tm_mat[i][old_j as usize];
                }
                if old_i >= 0 {
                    delta_score -= tm_mat[old_i as usize][j];
                }
                if old_i >= 0 && old_j >= 0 {
                    delta_score += tm_mat[old_i as usize][old_j as usize];
                }

                if delta_score > 0.0 {
                    // Accept swap
                    assign1[i] = j as i32;
                    if old_i >= 0 {
                        assign1[old_i as usize] = old_j;
                    }
                    assign2[j] = i as i32;
                    if old_j >= 0 {
                        assign2[old_j as usize] = old_i;
                    }
                    total_score += delta_score;
                    break 'outer;
                }
                // Revert
                assign1_tmp[i] = assign1[i];
                if old_i >= 0 {
                    assign1_tmp[old_i as usize] = assign1[old_i as usize];
                }
                assign2_tmp[j] = assign2[j];
                if old_j >= 0 {
                    assign2_tmp[old_j as usize] = assign2[old_j as usize];
                }
            }
            if delta_score > 0.0 {
                break;
            }
        }

        if delta_score <= 0.0 {
            break;
        }
    }

    (assign1, assign2, total_score)
}

/// Check if alignment is heterooligomer vs homooligomer.
///
/// Returns `het_deg` in [0, 1]. Larger = more "hetero", smaller = more "homo".
///
/// Corresponds to C++ `check_heterooligomer`.
pub fn check_heterooligomer(tm_mat: &[Vec<f64>], chain1_num: usize, chain2_num: usize) -> f64 {
    let mut min_tm = -1.0_f64;
    let mut max_tm = -1.0_f64;

    for i in 0..chain1_num {
        for j in 0..chain2_num {
            if min_tm < 0.0 || tm_mat[i][j] < min_tm {
                min_tm = tm_mat[i][j];
            }
            if max_tm < 0.0 || tm_mat[i][j] >= max_tm {
                max_tm = tm_mat[i][j];
            }
        }
    }

    if max_tm <= 0.0 {
        return 0.0;
    }
    (max_tm - min_tm) / max_tm
}

/// Refined greedy search for homooligomers.
///
/// Uses per-chain rotation matrices (from pairwise TM-align) to superpose
/// centroids and re-evaluate chain assignments based on both monomer TM-score
/// and centroid proximity.
///
/// Corresponds to C++ `homo_refined_greedy_search`.
///
/// `ut_mat[c1 * chain2_num + c2]` contains the flattened `[u00..u22, t0, t1, t2]`
/// rotation matrix from aligning chain c1 to chain c2.
///
/// Returns the maximum MMscore and updates `assign1`/`assign2`.
pub fn homo_refined_greedy_search(
    tm_mat: &[Vec<f64>],
    chain1_num: usize,
    chain2_num: usize,
    xcentroids: &[Coord3D],
    ycentroids: &[Coord3D],
    d0mm: f64,
    total_len: usize,
    ut_mat: &[Vec<f64>],
) -> (Vec<i32>, Vec<i32>, f64) {
    let chain_num = chain1_num.min(chain2_num);
    let total_pair = chain1_num * chain2_num;
    let mut assign1_best = vec![-1i32; chain1_num];
    let mut assign2_best = vec![-1i32; chain2_num];
    let mut mm_score_max = 0.0_f64;

    let mut assign1_tmp = vec![-1i32; chain1_num];
    let mut assign2_tmp = vec![-1i32; chain2_num];
    let mut xt = vec![[0.0f64; 3]; chain1_num];

    for c1 in 0..chain1_num {
        for c2 in 0..chain2_num {
            if tm_mat[c1][c2] <= 0.0 {
                continue;
            }
            let ut_idx = c1 * chain2_num + c2;
            if ut_idx >= ut_mat.len() || ut_mat[ut_idx].len() < 12 {
                continue;
            }

            // Extract rotation matrix from ut_mat
            let ut = &ut_mat[ut_idx];
            let transform = Transform {
                u: [
                    [ut[0], ut[1], ut[2]],
                    [ut[3], ut[4], ut[5]],
                    [ut[6], ut[7], ut[8]],
                ],
                t: [ut[9], ut[10], ut[11]],
            };

            // Rotate x centroids
            transform.apply_batch(xcentroids, &mut xt[..chain1_num]);

            // Compute combined TM-score * centroid proximity for each pair
            let mut ut_tmc_mat = vec![0.0f64; total_pair];
            let mut ut_tm_vec: Vec<(f64, usize)> = vec![(0.0, 0); total_pair];

            for i in 0..chain1_num {
                assign1_tmp[i] = -1;
            }
            for j in 0..chain2_num {
                assign2_tmp[j] = -1;
            }

            for i in 0..chain1_num {
                for j in 0..chain2_num {
                    let idx = i * chain2_num + j;
                    ut_tmc_mat[idx] = 0.0;
                    ut_tm_vec[idx] = (-1.0, idx);
                    if tm_mat[i][j] <= 0.0 {
                        continue;
                    }
                    let dd = dist_squared(&xt[i], &ycentroids[j]);
                    ut_tmc_mat[idx] = 1.0 / (1.0 + dd / (d0mm * d0mm));
                    ut_tm_vec[idx].0 = ut_tmc_mat[idx] * tm_mat[i][j];
                }
            }

            // Initial assignment: seed with (c1, c2)
            assign1_tmp[c1] = c2 as i32;
            assign2_tmp[c2] = c1 as i32;
            let mut tm_sum = tm_mat[c1][c2];
            let mut tm_score_sum = ut_tmc_mat[c1 * chain2_num + c2];

            // Sort in ascending order, iterate from highest
            ut_tm_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            for k in (0..total_pair).rev() {
                let idx = ut_tm_vec[k].1;
                let j = idx % chain2_num;
                let i = idx / chain2_num;
                if tm_mat[i][j] <= 0.0 {
                    break;
                }
                if assign1_tmp[i] >= 0 || assign2_tmp[j] >= 0 {
                    continue;
                }
                assign1_tmp[i] = j as i32;
                assign2_tmp[j] = i as i32;
                tm_sum += tm_mat[i][j];
                tm_score_sum += ut_tmc_mat[i * chain2_num + j];
            }

            // Compute MMscore
            let mm_score = (tm_sum / total_len as f64) * (tm_score_sum / chain_num as f64);
            if mm_score > mm_score_max {
                mm_score_max = mm_score;
                assign1_best.copy_from_slice(&assign1_tmp);
                assign2_best.copy_from_slice(&assign2_tmp);
            }
        }
    }

    (assign1_best, assign2_best, mm_score_max)
}

/// Refined greedy search for heterooligomers.
///
/// Uses centroid-based MMscore evaluation to iteratively swap chain assignments.
///
/// Corresponds to C++ `hetero_refined_greedy_search`.
///
/// Returns updated assignments and the final MMscore.
pub fn hetero_refined_greedy_search(
    tm_mat: &[Vec<f64>],
    assign1_init: &[i32],
    assign2_init: &[i32],
    chain1_num: usize,
    chain2_num: usize,
    xcentroids: &[Coord3D],
    ycentroids: &[Coord3D],
    d0mm: f64,
    total_len: usize,
) -> (Vec<i32>, Vec<i32>, f64) {
    let mut assign1 = assign1_init.to_vec();
    let mut assign2 = assign2_init.to_vec();

    // Calculate initial MMscore
    let mut mm_score_old = cal_mm_score(
        tm_mat, &assign1, chain1_num, chain2_num, xcentroids, ycentroids, d0mm, total_len,
    );

    // Iterative swap refinement
    let mut assign1_tmp = assign1.clone();
    let mut assign2_tmp = assign2.clone();

    for _iter in 0..chain1_num * chain2_num {
        let mut delta_score = -1.0_f64;

        for i in 0..chain1_num {
            let old_j = assign1[i];
            for j in 0..chain2_num {
                if j as i32 == assign1[i] || tm_mat[i][j] <= 0.0 {
                    continue;
                }
                let old_i = assign2[j];

                // Attempt swap
                assign1_tmp[i] = j as i32;
                if old_i >= 0 {
                    assign1_tmp[old_i as usize] = old_j;
                }
                assign2_tmp[j] = i as i32;
                if old_j >= 0 {
                    assign2_tmp[old_j as usize] = old_i;
                }

                let mm_score = cal_mm_score(
                    tm_mat,
                    &assign1_tmp,
                    chain1_num,
                    chain2_num,
                    xcentroids,
                    ycentroids,
                    d0mm,
                    total_len,
                );

                if mm_score > mm_score_old {
                    // Accept swap
                    assign1[i] = j as i32;
                    if old_i >= 0 {
                        assign1[old_i as usize] = old_j;
                    }
                    assign2[j] = i as i32;
                    if old_j >= 0 {
                        assign2[old_j as usize] = old_i;
                    }
                    delta_score = mm_score - mm_score_old;
                    mm_score_old = mm_score;
                    break;
                }
                // Revert
                assign1_tmp[i] = assign1[i];
                if old_i >= 0 {
                    assign1_tmp[old_i as usize] = assign1[old_i as usize];
                }
                assign2_tmp[j] = assign2[j];
                if old_j >= 0 {
                    assign2_tmp[old_j as usize] = assign2[old_j as usize];
                }
            }
        }

        if delta_score <= 0.0 {
            break;
        }
    }

    (assign1, assign2, mm_score_old)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_greedy_search_simple() {
        // 2x2 matrix: chain 0->1, chain 1->0 should give the best assignment
        let tm_mat = vec![vec![0.3, 0.9], vec![0.8, 0.2]];
        let (assign1, assign2, score) = enhanced_greedy_search(&tm_mat, 2, 2);
        assert_eq!(assign1[0], 1); // chain 0 maps to chain 1
        assert_eq!(assign1[1], 0); // chain 1 maps to chain 0
        assert_eq!(assign2[0], 1);
        assert_eq!(assign2[1], 0);
        assert!((score - 1.7).abs() < 1e-10);
    }

    #[test]
    fn test_enhanced_greedy_search_uneven() {
        // 3x2 matrix: one chain unassigned
        let tm_mat = vec![vec![0.1, 0.9], vec![0.8, 0.2], vec![0.0, 0.0]];
        let (assign1, _assign2, score) = enhanced_greedy_search(&tm_mat, 3, 2);
        assert_eq!(count_assign_pair(&assign1), 2);
        assert!(score > 0.0);
    }

    #[test]
    fn test_check_heterooligomer() {
        // Homooligomer: all scores similar
        let tm_mat = vec![vec![0.8, 0.75], vec![0.78, 0.82]];
        let het = check_heterooligomer(&tm_mat, 2, 2);
        assert!(het < 0.2, "het_deg={}", het);

        // Heterooligomer: very different scores
        let tm_mat2 = vec![vec![0.9, 0.1], vec![0.1, 0.8]];
        let het2 = check_heterooligomer(&tm_mat2, 2, 2);
        assert!(het2 > 0.5, "het_deg={}", het2);
    }

    #[test]
    fn test_count_assign_pair() {
        assert_eq!(count_assign_pair(&[-1, 0, -1, 2]), 2);
        assert_eq!(count_assign_pair(&[-1, -1, -1]), 0);
        assert_eq!(count_assign_pair(&[0, 1, 2]), 3);
    }
}
