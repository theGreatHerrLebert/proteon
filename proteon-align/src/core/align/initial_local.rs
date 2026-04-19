//! Local superposition-based initialization.
//!
//! Ported from C++ TMalign `get_initial5` (lines 2514-2611).

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::nwdp::nwdp_coords;
use crate::core::tmscore::get_score_fast;
use crate::core::types::{Coord3D, DPWorkspace};

/// Get initial alignment from exhaustive fragment superposition.
///
/// Tries all pairs of fragment positions across both structures,
/// superimposes each pair with Kabsch, then runs NW-DP to find the
/// best global alignment.
///
/// Returns (found, best_invmap). `found` is false if no alignment improved.
pub fn get_initial5(
    x: &[Coord3D],
    y: &[Coord3D],
    ws: &mut DPWorkspace,
    d0: f64,
    d0_search: f64,
    fast_opt: bool,
    d0_min: f64,
) -> (bool, Vec<i32>) {
    let xlen = x.len();
    let ylen = y.len();
    let al = xlen.min(ylen);

    let d01 = (d0 + 1.5).max(d0_min);
    let d02 = d01 * d01;

    // Jump steps for each sequence
    let mut n_jump1 = if xlen > 250 {
        45
    } else if xlen > 200 {
        35
    } else if xlen > 150 {
        25
    } else {
        15
    };
    if n_jump1 > xlen / 3 {
        n_jump1 = xlen / 3;
    }

    let mut n_jump2 = if ylen > 250 {
        45
    } else if ylen > 200 {
        35
    } else if ylen > 150 {
        25
    } else {
        15
    };
    if n_jump2 > ylen / 3 {
        n_jump2 = ylen / 3;
    }

    // Fragment sizes
    let mut n_frag = [20usize, 100];
    if n_frag[0] > al / 3 {
        n_frag[0] = al / 3;
    }
    if n_frag[1] > al / 2 {
        n_frag[1] = al / 2;
    }

    if fast_opt {
        n_jump1 *= 5;
        n_jump2 *= 5;
    }

    // Ensure non-zero jumps
    n_jump1 = n_jump1.max(1);
    n_jump2 = n_jump2.max(1);

    let mut gl_max = 0.0_f64;
    let mut best_y2x = vec![-1_i32; ylen];
    let mut found = false;

    for i_frag in 0..2 {
        let frag_len = n_frag[i_frag];
        if frag_len == 0 {
            continue;
        }
        let m1 = xlen.saturating_sub(frag_len) + 1;
        let m2 = ylen.saturating_sub(frag_len) + 1;

        let mut i = 0;
        while i < m1 {
            let mut j = 0;
            while j < m2 {
                // Extract fragment pairs
                let r1: Vec<Coord3D> = (0..frag_len).map(|k| x[k + i]).collect();
                let r2: Vec<Coord3D> = (0..frag_len).map(|k| y[k + j]).collect();

                // Kabsch on fragment
                if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                    // NW-DP with this rotation
                    let invmap = nwdp_coords(ws, x, y, &result.transform, d02, 0.0);

                    // Score quickly
                    let (gl, _) = get_score_fast(x, y, &invmap, d0, d0_search);
                    if gl > gl_max {
                        gl_max = gl;
                        best_y2x = invmap;
                        found = true;
                    }
                }

                j += n_jump2;
            }
            i += n_jump1;
        }
    }

    (found, best_y2x)
}
