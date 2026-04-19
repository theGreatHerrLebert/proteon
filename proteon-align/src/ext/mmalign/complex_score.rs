//! Complex-level scoring for multi-chain alignment.
//!
//! Ported from C++ USAlign `MMalign.h`: `calculate_centroids`, `calMMscore`.

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::types::{dist_squared, Coord3D};

/// Calculate the centroid (center of mass) of each chain.
///
/// Also computes `d0MM`, the average nearest-neighbor centroid distance,
/// used as a scaling factor for the MM-score.
///
/// Corresponds to C++ `calculate_centroids`.
///
/// Returns `(centroids, d0mm)`.
pub fn calculate_centroids(chains_coords: &[Vec<Coord3D>]) -> (Vec<Coord3D>, f64) {
    let chain_num = chains_coords.len();
    let mut centroids = vec![[0.0f64; 3]; chain_num];

    for (c, coords) in chains_coords.iter().enumerate() {
        let l = coords.len() as f64;
        if l == 0.0 {
            continue;
        }
        let mut sum = [0.0f64; 3];
        for r in coords {
            sum[0] += r[0];
            sum[1] += r[1];
            sum[2] += r[2];
        }
        centroids[c][0] = sum[0] / l;
        centroids[c][1] = sum[1] / l;
        centroids[c][2] = sum[2] / l;
    }

    // Compute d0MM: average of minimum inter-chain centroid distances
    let mut d0_vec = vec![-1.0f64; chain_num];
    for c in 0..chain_num {
        for c2 in 0..chain_num {
            if c2 == c {
                continue;
            }
            let d = dist_squared(&centroids[c], &centroids[c2]).sqrt();
            if d0_vec[c] <= 0.0 {
                d0_vec[c] = d;
            } else {
                d0_vec[c] = d0_vec[c].min(d);
            }
        }
    }

    let d0mm = if chain_num > 0 {
        d0_vec.iter().sum::<f64>() / chain_num as f64
    } else {
        0.0
    };

    (centroids, d0mm)
}

/// Calculate the MMscore of aligned chains.
///
/// MMscore = `(sum of TMave_mat[i][j] / L) * (centroid_pseudo_tmscore / min(n1, n2))`
///
/// where centroid_pseudo_tmscore uses Kabsch superposition of chain centroids,
/// and `dij^2 / d0MM^2` as the distance penalty.
///
/// Corresponds to C++ `calMMscore`.
pub fn cal_mm_score(
    tm_mat: &[Vec<f64>],
    assign1: &[i32],
    chain1_num: usize,
    chain2_num: usize,
    xcentroids: &[Coord3D],
    ycentroids: &[Coord3D],
    d0mm: f64,
    total_len: usize,
) -> f64 {
    let mut n_ali = 0usize;
    let mut mm_score = 0.0_f64;
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();

    for i in 0..chain1_num {
        let j = assign1[i];
        if j < 0 {
            continue;
        }
        let ju = j as usize;

        r1.push(xcentroids[i]);
        r2.push(ycentroids[ju]);
        n_ali += 1;
        mm_score += tm_mat[i][ju];
    }

    if total_len == 0 {
        return 0.0;
    }
    mm_score /= total_len as f64;

    let chain_num = chain1_num.min(chain2_num);
    let tm_score;

    if n_ali >= 3 {
        // Kabsch superposition of centroids
        if let Some(result) = kabsch(&r1, &r2, KabschMode::Both) {
            let mut xt = vec![[0.0f64; 3]; n_ali];
            result.transform.apply_batch(&r1, &mut xt);
            let mut s = 0.0f64;
            for k in 0..n_ali {
                let dd = dist_squared(&xt[k], &r2[k]);
                s += 1.0 / (1.0 + dd / (d0mm * d0mm));
            }
            tm_score = s;
        } else {
            tm_score = 1.0;
        }
    } else if n_ali == 2 {
        let dd = dist_squared(&r1[0], &r2[0]);
        tm_score = 1.0 / (1.0 + dd / (d0mm * d0mm));
    } else {
        tm_score = 1.0; // only one aligned chain
    }

    let chain_score = tm_score / chain_num as f64;
    mm_score * chain_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_centroids_single_chain() {
        let coords = vec![vec![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]];
        let (centroids, d0mm) = calculate_centroids(&coords);
        assert_eq!(centroids.len(), 1);
        assert!((centroids[0][0] - 2.0).abs() < 1e-10);
        assert!((centroids[0][1] - 3.0).abs() < 1e-10);
        assert!((centroids[0][2] - 4.0).abs() < 1e-10);
        // Only one chain, so d0_vec[0] stays -1, d0mm = -1/1 = -1
        // This edge case is fine; real use is multi-chain
        assert!(d0mm < 0.0);
    }

    #[test]
    fn test_calculate_centroids_two_chains() {
        let coords = vec![
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], // centroid = [1, 0, 0]
            vec![[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]], // centroid = [11, 0, 0]
        ];
        let (centroids, d0mm) = calculate_centroids(&coords);
        assert!((centroids[0][0] - 1.0).abs() < 1e-10);
        assert!((centroids[1][0] - 11.0).abs() < 1e-10);
        assert!((d0mm - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cal_mm_score_single_pair() {
        let tm_mat = vec![vec![0.8]];
        let assign1 = vec![0i32];
        let xcen = vec![[0.0, 0.0, 0.0]];
        let ycen = vec![[0.0, 0.0, 0.0]];
        let score = cal_mm_score(&tm_mat, &assign1, 1, 1, &xcen, &ycen, 5.0, 100);
        // Only one chain: mm_score = 0.8/100 * 1.0/1 = 0.008
        assert!((score - 0.008).abs() < 1e-10, "score={}", score);
    }
}
