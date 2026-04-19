//! SS + previous alignment initialization.
//!
//! Ported from C++ TMalign `get_initial_ssplus` and `score_matrix_rmsd_sec`
//! (lines 2613-2675).

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::nwdp::nwdp_score_matrix;
use crate::core::types::{dist_squared, Coord3D, DPWorkspace};

/// Build a score matrix combining structural distance and SS match.
///
/// For each (i,j) pair:
///   score = 1/(1 + dist²/d02) + 0.5 * (secx[i] == secy[j])
///
/// Uses the rotation derived from the previous alignment `y2x0`.
/// Returns a 1-indexed score matrix: score[i+1][j+1].
fn score_matrix_rmsd_sec(
    x: &[Coord3D],
    y: &[Coord3D],
    secx: &[char],
    secy: &[char],
    y2x0: &[i32],
    d0_min: f64,
    d0: f64,
) -> Vec<Vec<f64>> {
    let xlen = x.len();
    let ylen = y.len();

    let d01 = (d0 + 1.5).max(d0_min);
    let d02 = d01 * d01;

    // Extract aligned pairs from previous alignment
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    for (j, &i) in y2x0.iter().enumerate() {
        if i >= 0 {
            r1.push(x[i as usize]);
            r2.push(y[j]);
        }
    }

    // Compute rotation from previous alignment
    let transform = if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
        result.transform
    } else {
        crate::core::types::Transform::default()
    };

    // Build 1-indexed score matrix
    let mut score = vec![vec![0.0_f64; ylen + 1]; xlen + 1];
    for ii in 0..xlen {
        let xx = transform.apply(&x[ii]);
        for jj in 0..ylen {
            let dij = dist_squared(&xx, &y[jj]);
            let ss_bonus = if secx[ii] == secy[jj] { 0.5 } else { 0.0 };
            score[ii + 1][jj + 1] = 1.0 / (1.0 + dij / d02) + ss_bonus;
        }
    }

    score
}

/// Get initial alignment from SS + previous alignment.
///
/// Builds a distance+SS score matrix from the previous alignment,
/// then runs NW-DP on it.
///
/// Returns alignment map (j2i).
pub fn get_initial_ssplus(
    x: &[Coord3D],
    y: &[Coord3D],
    secx: &[char],
    secy: &[char],
    y2x0: &[i32],
    ws: &mut DPWorkspace,
    d0_min: f64,
    d0: f64,
) -> Vec<i32> {
    let xlen = x.len();
    let ylen = y.len();
    let score = score_matrix_rmsd_sec(x, y, secx, secy, y2x0, d0_min, d0);
    nwdp_score_matrix(ws, &score, xlen, ylen, -1.0)
}
