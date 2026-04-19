//! Circular permutation alignment.
//!
//! Ported from C++ TMalign `CPalign_main` (lines 4528-4685).

use anyhow::Result;

use crate::core::align::tmalign::tmalign;
use crate::core::types::{AlignOptions, AlignResult, Coord3D};

/// TM-align with circular permutation detection.
///
/// Duplicates chain1, runs fast TM-align on the doubled structure,
/// identifies the best CP point, then runs full TM-align from that point.
pub fn cpalign(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[char],
    seqy: &[char],
    secx: &[char],
    secy: &[char],
    opts: &AlignOptions,
) -> Result<AlignResult> {
    let xlen = xa.len();

    // Duplicate structure: xa_cp = [xa, xa]
    let mut xa_cp: Vec<Coord3D> = Vec::with_capacity(xlen * 2);
    xa_cp.extend_from_slice(xa);
    xa_cp.extend_from_slice(xa);
    let mut seqx_cp: Vec<char> = Vec::with_capacity(xlen * 2);
    seqx_cp.extend_from_slice(seqx);
    seqx_cp.extend_from_slice(seqx);
    let mut secx_cp: Vec<char> = Vec::with_capacity(xlen * 2);
    secx_cp.extend_from_slice(secx);
    secx_cp.extend_from_slice(secx);

    // Fast TM-align on doubled structure to find CP point
    let fast_opts = AlignOptions {
        i_opt: 0,
        a_opt: 0, // not used for CP detection
        u_opt: false,
        d_opt: false,
        fast_opt: true,
        mol_type: opts.mol_type,
        tm_cut: -1.0,
        lnorm_ass: 0.0,
        d0_scale: 0.0,
        user_alignment: None,
    };

    let result_cp = tmalign(&xa_cp, ya, &seqx_cp, seqy, &secx_cp, secy, &fast_opts)?;

    // Delete gaps in seqxA_cp to get compact alignment
    let seq_x_chars: Vec<char> = result_cp.aligned_seq_x.chars().collect();
    let seq_y_chars: Vec<char> = result_cp.aligned_seq_y.chars().collect();
    let mut compact_x = Vec::new();
    let mut compact_y = Vec::new();
    for i in 0..seq_x_chars.len() {
        if seq_x_chars[i] != '-' {
            compact_x.push(seq_x_chars[i]);
            compact_y.push(seq_y_chars[i]);
        }
    }

    // Find best CP point: sliding window of size xlen
    let mut cp_point = 0usize;
    let mut cp_aln_best = 0usize;
    for r in 0..xlen.saturating_sub(1) {
        let mut cp_aln_current = 0;
        for i in r..r + xlen {
            if i < compact_y.len() && compact_y[i] != '-' {
                cp_aln_current += 1;
            }
        }
        if cp_aln_current > cp_aln_best {
            cp_aln_best = cp_aln_current;
            cp_point = r;
        }
    }

    // Run normal TM-align on original structures to compare
    let normal_result = tmalign(xa, ya, seqx, seqy, secx, secy, &fast_opts)?;
    if normal_result.n_aligned > cp_aln_best {
        cp_point = 0;
    }

    // Prepare final structure (with CP rotation if needed)
    if cp_point != 0 {
        // Rotate sequence: start from cp_point
        let mut xa_final = Vec::with_capacity(xlen);
        let mut seqx_final = Vec::with_capacity(xlen);
        let mut secx_final = Vec::with_capacity(xlen);
        for r in 0..xlen {
            xa_final.push(xa_cp[r + cp_point]);
            seqx_final.push(seqx_cp[r + cp_point]);
            secx_final.push(secx_cp[r + cp_point]);
        }

        // Full TM-align on CP-rotated structure
        let mut result = tmalign(&xa_final, ya, &seqx_final, seqy, &secx_final, secy, opts)?;

        // Insert CP marker '*' in alignment
        let aln_x_chars: Vec<char> = result.aligned_seq_x.chars().collect();
        let aln_y_chars: Vec<char> = result.aligned_seq_y.chars().collect();
        let aln_m_chars: Vec<char> = result.alignment_markers.chars().collect();

        let mut r = 0usize;
        let mut split_pos = aln_x_chars.len();
        for (i, &c) in aln_x_chars.iter().enumerate() {
            if c != '-' {
                r += 1;
            }
            if r >= xlen - cp_point {
                split_pos = i + 1;
                break;
            }
        }

        let mut new_x = String::new();
        let mut new_y = String::new();
        let mut new_m = String::new();
        for (i, &c) in aln_x_chars.iter().enumerate() {
            new_x.push(c);
            new_y.push(aln_y_chars[i]);
            new_m.push(aln_m_chars[i]);
            if i + 1 == split_pos {
                new_x.push('*');
                new_y.push('-');
                new_m.push(' ');
            }
        }

        result.aligned_seq_x = new_x;
        result.aligned_seq_y = new_y;
        result.alignment_markers = new_m;
        Ok(result)
    } else {
        // No circular permutation: run full TM-align normally
        tmalign(xa, ya, seqx, seqy, secx, secy, opts)
    }
}
