//! Fragment gapless threading initialization.
//!
//! Ported from C++ TMalign `get_initial_fgt` and `find_max_frag` (lines 2678-2973).

use crate::core::tmscore::get_score_fast;
use crate::core::types::{dist_squared, Coord3D};

/// Find the longest continuous fragment where consecutive CA-CA distances
/// are below `dcu0`. If the fragment is too short, relax the threshold.
///
/// Returns (start, end) inclusive indices.
fn find_max_frag(x: &[Coord3D], dcu0: f64, fast_opt: bool) -> (usize, usize) {
    let len = x.len();
    let fra_min: usize = if fast_opt { 8 } else { 4 };
    let r_min = (len / 3).min(fra_min);

    let mut start_max = 0usize;
    let mut end_max = 0usize;
    let mut inc = 0u32;

    loop {
        let dcu_cut = (1.1_f64.powi(inc as i32) * dcu0).powi(2);
        let mut lfr_max = 0usize;
        let mut j = 1usize; // current fragment length
        let mut start = 0usize;

        for i in 1..len {
            if dist_squared(&x[i - 1], &x[i]) < dcu_cut {
                j += 1;
                if i == len - 1 && j > lfr_max {
                    lfr_max = j;
                    start_max = start;
                    end_max = i;
                }
            } else {
                if j > lfr_max {
                    lfr_max = j;
                    start_max = start;
                    end_max = i - 1;
                }
                j = 1;
                start = i;
            }
        }

        if lfr_max >= r_min || inc > 20 {
            break;
        }
        inc += 1;
    }

    // Handle edge case: single residue
    if len == 1 {
        return (0, 0);
    }

    (start_max, end_max)
}

/// Helper: run gapless threading of fragment indices `ifr` against structure,
/// mapping from fragment space to original indices.
fn thread_fragment_x(
    x: &[Coord3D],
    y: &[Coord3D],
    ifr: &[usize],
    d0: f64,
    d0_search: f64,
    fast_opt: bool,
    fra_min1: usize,
) -> (f64, Vec<i32>) {
    let l1 = ifr.len();
    let ylen = y.len();
    let min_len = l1.min(ylen);
    let min_ali = (min_len * 2 / 5).max(fra_min1); // min_len/2.5
    let n1 = -(ylen as isize) + min_ali as isize;
    let n2 = l1 as isize - min_ali as isize;

    let step: isize = if fast_opt { 3 } else { 1 };
    let mut tmscore_max = -1.0_f64;
    let mut best_y2x = vec![-1_i32; ylen];

    let mut k = n1;
    while k <= n2 {
        let y2x: Vec<i32> = (0..ylen)
            .map(|j| {
                let i = j as isize + k;
                if i >= 0 && (i as usize) < l1 {
                    ifr[i as usize] as i32
                } else {
                    -1
                }
            })
            .collect();

        let (tmscore, _) = get_score_fast(x, y, &y2x, d0, d0_search);
        if tmscore >= tmscore_max {
            tmscore_max = tmscore;
            best_y2x = y2x;
        }
        k += step;
    }

    (tmscore_max, best_y2x)
}

/// Helper: run gapless threading where fragment indices map into y-space.
fn thread_fragment_y(
    x: &[Coord3D],
    y: &[Coord3D],
    ifr: &[usize],
    d0: f64,
    d0_search: f64,
) -> (f64, Vec<i32>) {
    let l2 = ifr.len();
    let xlen = x.len();
    let ylen = y.len();
    let min_len = xlen.min(l2);
    let min_ali = (min_len * 2 / 5).max(3); // fra_min1=3
    let n1 = -(l2 as isize) + min_ali as isize;
    let n2 = xlen as isize - min_ali as isize;

    let mut tmscore_max = -1.0_f64;
    let mut best_y2x = vec![-1_i32; ylen];

    for k in n1..=n2 {
        let mut y2x = vec![-1_i32; ylen];
        for j in 0..l2 {
            let i = j as isize + k;
            if i >= 0 && (i as usize) < xlen {
                y2x[ifr[j]] = i as i32;
            }
        }

        let (tmscore, _) = get_score_fast(x, y, &y2x, d0, d0_search);
        if tmscore >= tmscore_max {
            tmscore_max = tmscore;
            best_y2x = y2x;
        }
    }

    (tmscore_max, best_y2x)
}

/// Trim fragment to middle 80% if it covers the full sequence.
fn trim_fragment(ifr: &[usize], l0: usize) -> Vec<usize> {
    if ifr.len() == l0 {
        let n1 = (l0 as f64 * 0.1) as usize;
        let n2 = (l0 as f64 * 0.89) as usize;
        (n1..=n2).map(|i| ifr[i]).collect()
    } else {
        ifr.to_vec()
    }
}

/// Get initial alignment from fragment gapless threading.
///
/// Finds the longest continuous fragment in each structure,
/// then performs gapless threading of the shorter fragment.
///
/// Returns (best_score, best_invmap).
pub fn get_initial_fgt(
    x: &[Coord3D],
    y: &[Coord3D],
    d0: f64,
    d0_search: f64,
    dcu0: f64,
    fast_opt: bool,
) -> (f64, Vec<i32>) {
    let xlen = x.len();
    let ylen = y.len();
    let fra_min: usize = if fast_opt { 8 } else { 4 };
    let fra_min1 = fra_min - 1;

    let (xstart, xend) = find_max_frag(x, dcu0, fast_opt);
    let (ystart, yend) = find_max_frag(y, dcu0, fast_opt);

    let lx = xend - xstart + 1;
    let ly = yend - ystart + 1;

    if lx < ly || (lx == ly && xlen < ylen) {
        // Use x fragment
        let ifr: Vec<usize> = (xstart..=xend).collect();
        let l0 = xlen.min(ylen);
        let ifr = trim_fragment(&ifr, l0);
        thread_fragment_x(x, y, &ifr, d0, d0_search, fast_opt, fra_min1)
    } else if lx > ly || (lx == ly && xlen > ylen) {
        // Use y fragment
        let ifr: Vec<usize> = (ystart..=yend).collect();
        let l0 = xlen.min(ylen);
        let ifr = trim_fragment(&ifr, l0);
        thread_fragment_y(x, y, &ifr, d0, d0_search)
    } else {
        // Symmetric case: try both and pick best
        let l0 = xlen; // xlen == ylen in this branch

        // Part 1: x fragment
        let ifr_x: Vec<usize> = (xstart..=xend).collect();
        let ifr_x = trim_fragment(&ifr_x, l0);
        let (score1, y2x1) = thread_fragment_x(x, y, &ifr_x, d0, d0_search, fast_opt, fra_min1);

        // Part 2: y fragment
        let ifr_y: Vec<usize> = (ystart..=yend).collect();
        let ifr_y = trim_fragment(&ifr_y, l0);
        let (score2, y2x2) = thread_fragment_y(x, y, &ifr_y, d0, d0_search);

        if score2 > score1 {
            (score2, y2x2)
        } else {
            (score1, y2x1)
        }
    }
}
