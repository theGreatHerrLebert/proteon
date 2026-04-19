//! Structure Extension (SE) refinement.
//!
//! Ported from C++ USAlign `se.h`.
//! Post-superposition re-alignment via DP on pre-rotated coordinates,
//! computing up to 5 TM-score normalizations in a single pass.

use crate::core::nwdp::nwdp_coords_no_rotation;
use crate::core::types::{dist_squared, Coord3D, DPWorkspace, MolType, TMParams};

/// Options controlling SE refinement behavior.
#[derive(Debug, Clone)]
pub struct SeOptions {
    /// Molecule type for d0 parameter calculation.
    pub mol_type: MolType,
    /// Whether to use user-provided alignment (i_opt).
    pub use_user_alignment: bool,
    /// Whether to compute average-length TM-score (a_opt).
    pub compute_avg: bool,
    /// User-specified normalization length option (0=none, 1=use, 2=use for search too).
    pub u_opt: u8,
    /// User-specified normalization length.
    pub lnorm_ass: f64,
    /// Whether to compute d_opt scaled TM-score.
    pub compute_d_scaled: bool,
    /// User-specified d0 scale.
    pub d0_scale: f64,
    /// Output format (>=2 skips sequence alignment string generation).
    pub outfmt_opt: i32,
    /// User-provided alignment sequences (for i_opt mode).
    pub sequence: Vec<String>,
}

impl Default for SeOptions {
    fn default() -> Self {
        SeOptions {
            mol_type: MolType::Protein,
            use_user_alignment: false,
            compute_avg: false,
            u_opt: 0,
            lnorm_ass: 0.0,
            compute_d_scaled: false,
            d0_scale: 0.0,
            outfmt_opt: 0,
            sequence: Vec::new(),
        }
    }
}

/// Result of SE refinement.
#[derive(Debug, Clone)]
pub struct SeResult {
    /// TM-score normalized by length of chain 2 (ylen).
    pub tm1: f64,
    /// TM-score normalized by length of chain 1 (xlen).
    pub tm2: f64,
    /// TM-score normalized by average length.
    pub tm3: f64,
    /// TM-score normalized by user-specified length.
    pub tm4: f64,
    /// TM-score normalized by d_opt scaling.
    pub tm5: f64,
    /// d0 values for each normalization.
    pub d0a: f64,
    pub d0b: f64,
    pub d0u: f64,
    pub d0a_avg: f64,
    pub d0_out: f64,
    /// RMSD of aligned pairs.
    pub rmsd: f64,
    /// Number of aligned residue pairs (total).
    pub n_ali: usize,
    /// Number of aligned pairs within distance cutoff.
    pub n_ali8: usize,
    /// Sequence identity count.
    pub liden: f64,
    /// TM-score of aligned region.
    pub tm_ali: f64,
    /// RMSD of aligned region.
    pub rmsd_ali: f64,
    /// Alignment map: `invmap[j] = i` or -1.
    pub invmap: Vec<i32>,
    /// Aligned sequence X (with gaps). Empty if outfmt_opt >= 2.
    pub seq_x_aligned: String,
    /// Aligned sequence Y (with gaps). Empty if outfmt_opt >= 2.
    pub seq_y_aligned: String,
    /// Match markers (`:` close, `.` other aligned). Empty if outfmt_opt >= 2.
    pub seq_m: String,
    /// Per-position distances. Empty if outfmt_opt >= 2.
    pub do_vec: Vec<f64>,
}

/// Main entry point for SE refinement.
///
/// Corresponds to C++ `se_main`. Takes pre-superposed coordinates (xa already rotated
/// onto ya's frame) and performs DP alignment followed by multi-normalization scoring.
///
/// `hinge` controls hinge mode: 0 = normal, >0 = append to existing invmap.
pub fn se_main(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[u8],
    seqy: &[u8],
    xlen: usize,
    ylen: usize,
    opts: &SeOptions,
    existing_invmap: Option<&[i32]>,
    hinge: u8,
) -> SeResult {
    let mol_type_int = match opts.mol_type {
        MolType::Protein => 0,
        MolType::RNA => 1,
    };

    // Set d0 parameters for various normalizations
    let search_params = TMParams::for_search(xlen, ylen);
    let score_d8 = search_params.score_d8;
    let d0 = search_params.d0;

    let params_b = TMParams::for_final(xlen as f64, opts.mol_type);
    let d0b = params_b.d0;

    let params_a = TMParams::for_final(ylen as f64, opts.mol_type);
    let d0a = params_a.d0;

    let d0a_avg = if opts.compute_avg {
        TMParams::for_final((xlen + ylen) as f64 * 0.5, opts.mol_type).d0
    } else {
        0.0
    };

    let (d0u, search_d0, search_score_d8) = if opts.u_opt > 0 {
        let pu = TMParams::for_final(opts.lnorm_ass, opts.mol_type);
        if opts.u_opt == 2 {
            let ps = TMParams::for_search(opts.lnorm_ass as usize, opts.lnorm_ass as usize);
            (pu.d0, ps.d0, ps.score_d8)
        } else {
            (pu.d0, d0, score_d8)
        }
    } else {
        (0.0, d0, score_d8)
    };

    // Use search parameters (possibly overridden by u_opt==2)
    let active_d0 = search_d0;
    let active_score_d8 = search_score_d8;

    let d0_out = if mol_type_int > 0 { 3.5 } else { 5.0 };

    // Initialize invmap
    let mut invmap = vec![-1i32; ylen];
    let mut invmap0 = vec![-1i32; ylen];

    if hinge > 0 {
        if let Some(existing) = existing_invmap {
            let n = ylen.min(existing.len());
            invmap[..n].copy_from_slice(&existing[..n]);
            invmap0[..n].copy_from_slice(&existing[..n]);
        }
    }

    // Perform alignment
    if !opts.use_user_alignment {
        // NWDP_SE: DP on pre-rotated coordinates
        let mut ws = DPWorkspace::new(xlen, ylen);
        let d02 = active_d0 * active_d0;

        if hinge == 0 {
            let new_invmap = nwdp_coords_no_rotation(&mut ws, xa, ya, d02, 0.0);
            invmap[..ylen].copy_from_slice(&new_invmap[..ylen]);
        } else {
            // Hinge mode: run DP but preserve existing alignments
            let new_invmap = nwdp_se_hinge(&mut ws, xa, ya, d02, 0.0, &invmap);
            invmap[..ylen].copy_from_slice(&new_invmap[..ylen]);
        }
    } else {
        // User-provided alignment
        if opts.sequence.len() >= 2 {
            let seq0 = opts.sequence[0].as_bytes();
            let seq1 = opts.sequence[1].as_bytes();
            let l = seq0.len().min(seq1.len());
            let mut i1: i32 = -1;
            let mut i2: i32 = -1;
            for kk in 0..l {
                if seq0[kk] != b'-' {
                    i1 += 1;
                }
                if seq1[kk] != b'-' {
                    i2 += 1;
                    if i2 >= ylen as i32 || i1 >= xlen as i32 {
                        break;
                    }
                    if seq0[kk] != b'-' {
                        invmap[i2 as usize] = i1;
                    }
                }
            }
        }
    }

    // Compute TM-scores for all normalizations in one pass
    let mut tm1 = 0.0_f64; // normalized by ylen
    let mut tm2 = 0.0_f64; // normalized by xlen
    let mut tm3 = 0.0_f64; // normalized by avg
    let mut tm4 = 0.0_f64; // normalized by user length
    let mut tm5 = 0.0_f64; // d_opt scaled
    let mut rmsd0 = 0.0_f64;
    let mut n_ali = 0usize;
    let mut n_ali8 = 0usize;

    // If hinge mode, recover previously accumulated scores
    if hinge > 0 {
        // These would have been passed in from the previous round;
        // for a clean implementation, we start from scratch for now.
        // The C++ version scales existing scores back up by length,
        // adds new contributions, then re-normalizes.
    }

    let mut m1 = Vec::new(); // aligned indices in x
    let mut m2 = Vec::new(); // aligned indices in y

    for j in 0..ylen {
        let i = invmap[j];
        if i < 0 {
            continue;
        }
        let iu = i as usize;
        n_ali += 1;
        let d = dist_squared(&xa[iu], &ya[j]).sqrt();

        if d <= active_score_d8 || opts.use_user_alignment || invmap0[j] == i {
            if opts.outfmt_opt < 2 {
                m1.push(iu);
                m2.push(j);
            }
            n_ali8 += 1;

            if invmap0[j] == i {
                continue; // already counted in previous hinge round
            }

            let d_d0b = d / d0b;
            tm2 += 1.0 / (1.0 + d_d0b * d_d0b);

            let d_d0a = d / d0a;
            tm1 += 1.0 / (1.0 + d_d0a * d_d0a);

            if opts.compute_avg {
                let d_d0avg = d / d0a_avg;
                tm3 += 1.0 / (1.0 + d_d0avg * d_d0avg);
            }

            if opts.u_opt > 0 {
                let d_d0u = d / d0u;
                tm4 += 1.0 / (1.0 + d_d0u * d_d0u);
            }

            if opts.compute_d_scaled {
                let d_d0s = d / opts.d0_scale;
                tm5 += 1.0 / (1.0 + d_d0s * d_d0s);
            }

            rmsd0 += d * d;
        } else if hinge > 0 {
            invmap[j] = -1;
        }
    }

    // Normalize
    tm2 /= xlen as f64;
    tm1 /= ylen as f64;
    if opts.compute_avg {
        tm3 /= (xlen + ylen) as f64 * 0.5;
    }
    if opts.u_opt > 0 {
        tm4 /= opts.lnorm_ass;
    }
    if opts.compute_d_scaled {
        tm5 /= ylen as f64;
    }
    if n_ali8 > 0 {
        rmsd0 = (rmsd0 / n_ali8 as f64).sqrt();
    }

    // Build aligned sequences if needed
    let mut seq_x_aligned = String::new();
    let mut seq_y_aligned = String::new();
    let mut seq_m = String::new();
    let mut do_vec = Vec::new();
    let mut liden = 0.0_f64;

    if opts.outfmt_opt < 2 {
        let ali_len = xlen + ylen;
        let mut sxa = vec![b'-'; ali_len];
        let mut sya = vec![b'-'; ali_len];
        let mut sm = vec![b' '; ali_len];
        do_vec.resize(ali_len, 0.0);

        let mut kk = 0usize;
        let mut i_old = 0usize;
        let mut j_old = 0usize;

        for k in 0..n_ali8 {
            // Gaps in x
            for i in i_old..m1[k] {
                sxa[kk] = seqx[i];
                sya[kk] = b'-';
                sm[kk] = b' ';
                kk += 1;
            }
            // Gaps in y
            for j in j_old..m2[k] {
                sxa[kk] = b'-';
                sya[kk] = seqy[j];
                sm[kk] = b' ';
                kk += 1;
            }
            // Matched pair
            sxa[kk] = seqx[m1[k]];
            sya[kk] = seqy[m2[k]];
            if sxa[kk] == sya[kk] {
                liden += 1.0;
            }
            let d = dist_squared(&xa[m1[k]], &ya[m2[k]]).sqrt();
            sm[kk] = if d < d0_out { b':' } else { b'.' };
            do_vec[kk] = d;
            kk += 1;
            i_old = m1[k] + 1;
            j_old = m2[k] + 1;
        }

        // Tail gaps
        for i in i_old..xlen {
            sxa[kk] = seqx[i];
            sya[kk] = b'-';
            sm[kk] = b' ';
            kk += 1;
        }
        for j in j_old..ylen {
            sxa[kk] = b'-';
            sya[kk] = seqy[j];
            sm[kk] = b' ';
            kk += 1;
        }

        seq_x_aligned = String::from_utf8(sxa[..kk].to_vec()).unwrap_or_default();
        seq_y_aligned = String::from_utf8(sya[..kk].to_vec()).unwrap_or_default();
        seq_m = String::from_utf8(sm[..kk].to_vec()).unwrap_or_default();
        do_vec.truncate(kk);
    }

    SeResult {
        tm1,
        tm2,
        tm3,
        tm4,
        tm5,
        d0a,
        d0b,
        d0u,
        d0a_avg,
        d0_out,
        rmsd: rmsd0,
        n_ali,
        n_ali8,
        liden,
        tm_ali: 0.0,
        rmsd_ali: 0.0,
        invmap,
        seq_x_aligned,
        seq_y_aligned,
        seq_m,
        do_vec,
    }
}

/// NWDP_SE with hinge support: DP on coordinates preserving existing alignments.
///
/// Corresponds to C++ `NWDP_SE(..., const int hinge)` variant.
/// Pre-existing invmap entries are locked: their path cells are pre-set and
/// their distance is treated as zero.
fn nwdp_se_hinge(
    ws: &mut DPWorkspace,
    x: &[Coord3D],
    y: &[Coord3D],
    d02: f64,
    gap_open: f64,
    existing_invmap: &[i32],
) -> Vec<i32> {
    let len1 = x.len();
    let len2 = y.len();
    ws.ensure_size(len1, len2);

    // Initialize
    for i in 0..=len1 {
        for j in 0..=len2 {
            ws.val[i][j] = 0.0;
            ws.path[i][j] = false;
        }
    }

    // Pre-fill existing alignment entries
    for (j, &i) in existing_invmap.iter().enumerate() {
        if i >= 0 {
            let iu = i as usize;
            ws.path[iu + 1][j + 1] = true;
            ws.val[iu + 1][j + 1] = 0.0;
        }
    }

    // DP fill
    for i in 1..=len1 {
        for j in 1..=len2 {
            let dij = if !ws.path[i][j] {
                dist_squared(&x[i - 1], &y[j - 1])
            } else {
                0.0 // locked alignment: zero distance
            };
            let d = ws.val[i - 1][j - 1] + 1.0 / (1.0 + dij / d02);

            let mut h = ws.val[i - 1][j];
            if ws.path[i - 1][j] {
                h += gap_open;
            }

            let mut v = ws.val[i][j - 1];
            if ws.path[i][j - 1] {
                v += gap_open;
            }

            if d >= h && d >= v && ws.val[i][j] == 0.0 {
                ws.path[i][j] = true;
                ws.val[i][j] = d;
            } else {
                ws.path[i][j] = false;
                ws.val[i][j] = if v >= h { v } else { h };
            }
        }
    }

    // Traceback
    let mut j2i = vec![-1_i32; len2];
    let mut i = len1;
    let mut j = len2;
    while i > 0 && j > 0 {
        if ws.path[i][j] {
            j2i[j - 1] = (i - 1) as i32;
            i -= 1;
            j -= 1;
        } else {
            let mut h = ws.val[i - 1][j];
            if ws.path[i - 1][j] {
                h += gap_open;
            }
            let mut v = ws.val[i][j - 1];
            if ws.path[i][j - 1] {
                v += gap_open;
            }
            if v >= h {
                j -= 1;
            } else {
                i -= 1;
            }
        }
    }

    j2i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_helix_coords(n: usize) -> Vec<Coord3D> {
        // Simple helix-like CA trace
        (0..n)
            .map(|i| {
                let t = i as f64 * 100.0 / 180.0 * std::f64::consts::PI;
                [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64]
            })
            .collect()
    }

    #[test]
    fn test_se_identical_structures() {
        let n = 20;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];

        let opts = SeOptions::default();
        let result = se_main(&coords, &coords, &seq, &seq, n, n, &opts, None, 0);

        // Identical structures should have TM-score = 1.0
        assert!(result.tm1 > 0.99, "tm1={}", result.tm1);
        assert!(result.tm2 > 0.99, "tm2={}", result.tm2);
        assert!(result.rmsd < 0.01, "rmsd={}", result.rmsd);
        assert_eq!(result.n_ali8, n);
    }

    #[test]
    fn test_se_sequence_alignment_output() {
        let n = 10;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = (0..n).map(|i| b'A' + (i % 20) as u8).collect();

        let opts = SeOptions::default();
        let result = se_main(&coords, &coords, &seq, &seq, n, n, &opts, None, 0);

        assert!(!result.seq_x_aligned.is_empty());
        assert!(!result.seq_y_aligned.is_empty());
        assert!(!result.seq_m.is_empty());
        // All positions should be close (`:` marker)
        assert!(result.seq_m.contains(':'));
    }

    #[test]
    fn test_se_outfmt_skips_strings() {
        let n = 10;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];

        let opts = SeOptions {
            outfmt_opt: 2,
            ..Default::default()
        };
        let result = se_main(&coords, &coords, &seq, &seq, n, n, &opts, None, 0);

        assert!(result.seq_x_aligned.is_empty());
        assert!(result.seq_y_aligned.is_empty());
        assert!(result.tm1 > 0.99);
    }
}
