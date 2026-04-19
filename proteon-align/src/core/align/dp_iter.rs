//! Iterative DP refinement loop.
//!
//! Ported from C++ TMalign `DP_iter` (lines 2979-3044).

use crate::core::nwdp::nwdp_coords;
use crate::core::tmscore::tmscore8_search;
use crate::core::types::{Coord3D, DPWorkspace, TMParams, Transform};

/// Iteratively refines an alignment using NW-DP + TMscore8_search.
///
/// Tries gap penalties from `gap_range` (typically [0..1] or [0..2]),
/// with `[-0.6, 0.0]` as the gap open values.
///
/// Returns (best_tmscore, best_invmap, best_transform).
pub fn dp_iter(
    x: &[Coord3D],
    y: &[Coord3D],
    ws: &mut DPWorkspace,
    transform: &Transform,
    gap_range: std::ops::Range<usize>,
    iteration_max: usize,
    local_d0_search: f64,
    params: &TMParams,
) -> (f64, Vec<i32>, Transform) {
    let gap_open = [-0.6, 0.0];
    let ylen = y.len();
    let d02 = params.d0 * params.d0;
    let score_sum_method = 8u8;
    let simplify_step = 40;

    let mut tmscore_max = -1.0_f64;
    let mut best_invmap = vec![-1_i32; ylen];
    let mut best_transform = transform.clone();
    let mut current_transform = transform.clone();

    for g in gap_range {
        if g >= gap_open.len() {
            break;
        }
        let mut tmscore_old = 0.0_f64;

        for iteration in 0..iteration_max {
            // NW-DP alignment using current rotation
            let invmap = nwdp_coords(ws, x, y, &current_transform, d02, gap_open[g]);

            // Extract aligned pairs
            let mut xtm = Vec::new();
            let mut ytm = Vec::new();
            for (j, &i) in invmap.iter().enumerate() {
                if i >= 0 {
                    xtm.push(x[i as usize]);
                    ytm.push(y[j]);
                }
            }

            if xtm.is_empty() {
                break;
            }

            // Search for best rotation
            let (tmscore, new_transform) = tmscore8_search(
                &xtm,
                &ytm,
                simplify_step,
                score_sum_method,
                local_d0_search,
                params,
            );
            current_transform = new_transform;

            if tmscore > tmscore_max {
                tmscore_max = tmscore;
                best_invmap = invmap;
                best_transform = current_transform.clone();
            }

            if iteration > 0 && (tmscore_old - tmscore).abs() < 0.000001 {
                break;
            }
            tmscore_old = tmscore;
        }
    }

    (tmscore_max, best_invmap, best_transform)
}
