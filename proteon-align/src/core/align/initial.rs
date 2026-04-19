//! Gapless threading initialization.
//!
//! Ported from C++ TMalign `get_initial` (lines 2341-2390).

use crate::core::tmscore::get_score_fast;
use crate::core::types::Coord3D;

/// Find the best gapless threading alignment by trying all offsets.
///
/// Returns (best_score, best_invmap).
pub fn get_initial(
    x: &[Coord3D],
    y: &[Coord3D],
    d0: f64,
    d0_search: f64,
    fast_opt: bool,
) -> (f64, Vec<i32>) {
    let xlen = x.len();
    let ylen = y.len();
    let min_len = xlen.min(ylen);
    if min_len < 3 {
        return (-1.0, vec![-1; ylen]);
    }

    let min_ali = (min_len / 2).max(5);
    let n1 = -(ylen as isize) + min_ali as isize;
    let n2 = xlen as isize - min_ali as isize;

    let step: isize = if fast_opt { 5 } else { 1 };
    let mut tmscore_max = -1.0_f64;
    let mut k_best = n1;

    let mut k = n1;
    while k <= n2 {
        // Build alignment map for offset k
        let y2x: Vec<i32> = (0..ylen)
            .map(|j| {
                let i = j as isize + k;
                if i >= 0 && i < xlen as isize {
                    i as i32
                } else {
                    -1
                }
            })
            .collect();

        let (tmscore, _) = get_score_fast(x, y, &y2x, d0, d0_search);
        if tmscore >= tmscore_max {
            tmscore_max = tmscore;
            k_best = k;
        }
        k += step;
    }

    // Build best alignment map
    let y2x: Vec<i32> = (0..ylen)
        .map(|j| {
            let i = j as isize + k_best;
            if i >= 0 && i < xlen as isize {
                i as i32
            } else {
                -1
            }
        })
        .collect();

    (tmscore_max, y2x)
}
