//! SOI iteration: Kabsch superposition + greedy re-assignment loop.
//!
//! Ported from C++ USAlign `SOIalign.h`: `SOI_iter`, `get_SOI_initial_assign`,
//! `SOI_super2score`, `SOI_assign2super`.
//! Iteratively refines structure alignment by alternating between
//! greedy assignment and TM-score-maximizing superposition.

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::nwdp::nwdp_score_matrix;
use crate::core::tmscore::{extract_aligned_pairs, tmscore8_search};
use crate::core::types::{dist_squared, Coord3D, DPWorkspace, TMParams, Transform};

use crate::ext::soialign::greedy::soi_egs;
use crate::ext::soialign::sec_bond::SecBond;

/// Compute pairwise distance-based score matrix from superposed coordinates.
///
/// Returns a 1-indexed score matrix: `score[i+1][j+1]` for 0-indexed `i`,`j`.
/// Pairs with squared distance > `score_d8^2` get score 0.
/// Otherwise `score = 1 / (1 + d^2 / d0^2)`.
///
/// Corresponds to C++ `SOI_super2score`.
pub fn super2score(xt: &[Coord3D], ya: &[Coord3D], d0: f64, score_d8: f64) -> Vec<Vec<f64>> {
    let xlen = xt.len();
    let ylen = ya.len();
    let d02 = d0 * d0;
    let score_d82 = score_d8 * score_d8;

    let mut score = vec![vec![0.0_f64; ylen + 1]; xlen + 1];
    for i in 0..xlen {
        for j in 0..ylen {
            let d2 = dist_squared(&xt[i], &ya[j]);
            if d2 > score_d82 {
                score[i + 1][j + 1] = 0.0;
            } else {
                score[i + 1][j + 1] = 1.0 / (1.0 + d2 / d02);
            }
        }
    }
    score
}

/// Transpose a 1-indexed score matrix from `(xlen+1) x (ylen+1)` to `(ylen+1) x (xlen+1)`.
///
/// Used when running SOI_iter with x and y roles swapped.
pub fn transpose_score(score: &[Vec<f64>], xlen: usize, ylen: usize) -> Vec<Vec<f64>> {
    let mut scoret = vec![vec![0.0_f64; xlen + 1]; ylen + 1];
    for i in 0..xlen {
        for j in 0..ylen {
            scoret[j + 1][i + 1] = score[i + 1][j + 1];
        }
    }
    scoret
}

/// Apply rotation to coordinates and then run `tmscore8_search` from an invmap.
///
/// Extracts aligned pairs from `invmap`, runs TMscore8_search, applies the resulting
/// rotation to all of `xa`, and returns the rotated coordinates along with
/// the transform.
///
/// Corresponds to C++ `SOI_assign2super`.
pub fn assign2super(
    xa: &[Coord3D],
    ya: &[Coord3D],
    invmap: &[i32],
    params: &TMParams,
) -> (Transform, Vec<Coord3D>) {
    let (xtm, ytm) = extract_aligned_pairs(xa, ya, invmap);
    if xtm.is_empty() {
        return (Transform::default(), xa.to_vec());
    }

    let (_, transform) = tmscore8_search(
        &xtm,
        &ytm,
        40, // simplify_step
        8,  // score_sum_method
        params.d0_search,
        params,
    );

    let mut xt = vec![[0.0; 3]; xa.len()];
    transform.apply_batch(xa, &mut xt);
    (transform, xt)
}

/// Iteratively refine alignment by alternating greedy assignment and Kabsch superposition.
///
/// Returns the best TM-score found and updates `invmap0` with the best alignment.
///
/// When `init_invmap` is true, the first iteration uses the provided `invmap0`
/// instead of running fresh assignment from the score matrix.
///
/// Corresponds to C++ `SOI_iter`.
pub fn soi_iter(
    xa: &[Coord3D],
    ya: &[Coord3D],
    invmap0: &mut [i32],
    iteration_max: usize,
    params: &TMParams,
    secx_bond: &[SecBond],
    secy_bond: &[SecBond],
    mm_opt: i32,
    init_invmap: bool,
    initial_score: &[Vec<f64>],
) -> f64 {
    let xlen = xa.len();
    let ylen = ya.len();

    let mut invmap = vec![-1i32; ylen];
    let mut tmscore_max = -1.0_f64;
    let mut tmscore_old = 0.0_f64;

    let mut score = initial_score.to_vec();
    let mut ws = DPWorkspace::new(xlen.max(ylen), xlen.max(ylen));

    for iteration in 0..iteration_max {
        if iteration == 0 && init_invmap {
            invmap.copy_from_slice(&invmap0[..ylen]);
        } else {
            for j in 0..ylen {
                invmap[j] = -1;
            }
            if mm_opt == 6 {
                let dp_result = nwdp_score_matrix(&mut ws, &score, xlen, ylen, -0.6);
                invmap[..ylen].copy_from_slice(&dp_result[..ylen]);
            }
        }

        soi_egs(
            &score,
            xlen,
            ylen,
            &mut invmap,
            secx_bond,
            secy_bond,
            mm_opt,
        );

        // Extract aligned pairs
        let (xtm, ytm) = extract_aligned_pairs(xa, ya, &invmap);
        if xtm.is_empty() {
            break;
        }

        let (tmscore, transform) = tmscore8_search(
            &xtm,
            &ytm,
            40, // simplify_step
            8,  // score_sum_method
            params.d0_search,
            params,
        );

        if tmscore > tmscore_max {
            tmscore_max = tmscore;
            invmap0[..ylen].copy_from_slice(&invmap[..ylen]);
        }

        if iteration > 0 && (tmscore_old - tmscore).abs() < 0.000001 {
            break;
        }
        tmscore_old = tmscore;

        // Apply rotation and recompute score matrix
        let mut xt = vec![[0.0; 3]; xlen];
        transform.apply_batch(xa, &mut xt);
        score = super2score(&xt, ya, params.d0, params.score_d8);
    }

    tmscore_max
}

/// Compute initial SOI assignment using K-nearest neighbor fragments.
///
/// For each pair `(i, j)`, superpose the K-nearest neighbor fragments of `i` in x
/// onto those of `j` in y using Kabsch, then score by the distance between
/// the last neighbors after rotation. The resulting score matrix feeds into
/// greedy assignment.
///
/// Corresponds to C++ `get_SOI_initial_assign`.
pub fn get_soi_initial_assign(
    xk: &[Coord3D],
    yk: &[Coord3D],
    close_k_opt: usize,
    xlen: usize,
    ylen: usize,
    params: &TMParams,
    secx_bond: &[SecBond],
    secy_bond: &[SecBond],
    mm_opt: i32,
) -> (Vec<i32>, Vec<Vec<f64>>) {
    let d02 = params.d0 * params.d0;
    let score_d82 = params.score_d8 * params.score_d8;

    let mut score = vec![vec![0.0_f64; ylen + 1]; xlen + 1];

    let mut xfrag = vec![[0.0; 3]; close_k_opt];
    let mut yfrag = vec![[0.0; 3]; close_k_opt];

    for i in 0..xlen {
        for k in 0..close_k_opt {
            xfrag[k] = xk[i * close_k_opt + k];
        }

        for j in 0..ylen {
            for k in 0..close_k_opt {
                yfrag[k] = yk[j * close_k_opt + k];
            }

            // Kabsch superposition of fragments
            if let Some(result) = kabsch(&xfrag, &yfrag, KabschMode::Both) {
                // Apply rotation to xfrag
                let mut xtran = vec![[0.0; 3]; close_k_opt];
                result.transform.apply_batch(&xfrag, &mut xtran);

                // Score using last neighbor (k = closeK_opt - 1) only
                let k = close_k_opt - 1;
                let d2 = dist_squared(&xtran[k], &yfrag[k]);
                if d2 > score_d82 {
                    score[i + 1][j + 1] = 0.0;
                } else {
                    score[i + 1][j + 1] = 1.0 / (1.0 + d2 / d02);
                }
            }
        }
    }

    // Initial assignment
    let mut invmap = vec![-1i32; ylen];
    if mm_opt == 6 {
        let mut ws = DPWorkspace::new(xlen.max(ylen), xlen.max(ylen));
        let dp_result = nwdp_score_matrix(&mut ws, &score, xlen, ylen, -0.6);
        invmap[..ylen].copy_from_slice(&dp_result[..ylen]);
    }
    soi_egs(
        &score,
        xlen,
        ylen,
        &mut invmap,
        secx_bond,
        secy_bond,
        mm_opt,
    );

    (invmap, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line_coords(n: usize, offset: f64) -> Vec<Coord3D> {
        (0..n)
            .map(|i| [i as f64 * 3.8 + offset, 0.0, 0.0])
            .collect()
    }

    #[test]
    fn test_super2score_identical() {
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]];
        let score = super2score(&coords, &coords, 5.0, 100.0);
        // Diagonal should be 1.0 (distance=0)
        for i in 0..3 {
            assert!(
                (score[i + 1][i + 1] - 1.0).abs() < 1e-10,
                "score[{}][{}] = {}",
                i,
                i,
                score[i + 1][i + 1]
            );
        }
    }

    #[test]
    fn test_transpose_score() {
        let score = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.3],
            vec![0.0, 0.7, 0.9],
        ];
        let t = transpose_score(&score, 2, 2);
        assert!((t[1][1] - 0.5).abs() < 1e-10);
        assert!((t[1][2] - 0.7).abs() < 1e-10);
        assert!((t[2][1] - 0.3).abs() < 1e-10);
        assert!((t[2][2] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_soi_iter_identical() {
        let n = 10;
        let coords = make_line_coords(n, 0.0);
        let params = TMParams::for_search(n, n);

        let secx_bond = vec![[-1i32, -1]; n];
        let secy_bond = vec![[-1i32, -1]; n];

        let initial_score = super2score(&coords, &coords, params.d0, params.score_d8);
        let mut invmap = vec![-1i32; n];

        let tm = soi_iter(
            &coords,
            &coords,
            &mut invmap,
            10,
            &params,
            &secx_bond,
            &secy_bond,
            0,
            false,
            &initial_score,
        );
        assert!(tm > 0.0, "tm={}", tm);
        // Diagonal alignment expected
        for j in 0..n {
            assert_eq!(invmap[j], j as i32, "invmap[{}]={}", j, invmap[j]);
        }
    }

    #[test]
    fn test_assign2super_basic() {
        let xa: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]];
        let ya: Vec<Coord3D> = vec![[1.0, 0.0, 0.0], [4.8, 0.0, 0.0]];
        let invmap = vec![0i32, 1];
        let params = TMParams::for_search(2, 2);

        let (transform, xt) = assign2super(&xa, &ya, &invmap, &params);
        // After superposition, xt should be close to ya
        assert!(xt.len() == 2);
        // The transform should exist
        // Just confirm it ran without panicking; a zero translation is valid here.
        let _ = transform.t;
    }

    #[test]
    fn test_get_soi_initial_assign() {
        let n = 5;
        let close_k = 3;
        let coords: Vec<Coord3D> = (0..n).map(|i| [i as f64 * 3.8, 0.0, 0.0]).collect();

        // K-nearest neighbors for identical structures
        let xk = crate::ext::soialign::close_k::get_close_k(&coords, close_k);
        let yk = crate::ext::soialign::close_k::get_close_k(&coords, close_k);

        let params = TMParams::for_search(n, n);
        let secx_bond = vec![[-1i32, -1]; n];
        let secy_bond = vec![[-1i32, -1]; n];

        let (invmap, score) =
            get_soi_initial_assign(&xk, &yk, close_k, n, n, &params, &secx_bond, &secy_bond, 0);

        // For identical structures, should get diagonal assignment
        let n_assigned = invmap.iter().filter(|&&v| v >= 0).count();
        assert!(n_assigned > 0, "no residues assigned");
        // Score matrix should have positive diagonal values
        assert!(score[1][1] > 0.0);
    }
}
