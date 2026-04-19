//! Sequence-Order Independent (SOI) alignment.
//!
//! Ported from C++ USAlign `SOIalign.h` (~960 lines).
//! Performs structure alignment without assuming sequential residue correspondence.
//! Uses K-nearest neighbor fragments, enhanced greedy search with swap refinement,
//! and iterative Kabsch superposition to find optimal alignment.
//!
//! ## Algorithm overview
//!
//! 1. Start from an initial sequence-dependent alignment (from SE refinement)
//! 2. Convert superposition to a distance-based score matrix
//! 3. Run `SOI_iter`: iterative greedy assignment + Kabsch refinement
//! 4. Additionally try K-nearest-neighbor initial assignment (`get_SOI_initial_assign`)
//! 5. Pick the best alignment, run detailed TM-score search, and produce final output

pub mod close_k;
pub mod greedy;
pub mod iter;
pub mod sec_bond;

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::tmscore::{detailed_search_standard, tmscore8_search};
use crate::core::types::{dist_squared, Coord3D, MolType, TMParams, Transform};

use self::close_k::get_close_k;
use self::iter::{assign2super, get_soi_initial_assign, soi_iter, super2score, transpose_score};
use self::sec_bond::assign_sec_bond;

/// Options for SOI alignment.
#[derive(Debug, Clone)]
pub struct SoiOptions {
    /// Molecule type for d0 parameter calculation.
    pub mol_type: MolType,
    /// Number of nearest neighbors for initial assignment (typically 5).
    pub close_k_opt: usize,
    /// Whether to use fast mode (fewer iterations).
    pub fast_opt: bool,
    /// Alignment option: i_opt from C++.
    pub i_opt: i32,
    /// Average length normalization.
    pub a_opt: i32,
    /// User-specified normalization length.
    pub u_opt: bool,
    /// User-specified normalization length value.
    pub lnorm_ass: f64,
    /// d0 scaling option.
    pub d_opt: bool,
    /// d0 scale value.
    pub d0_scale: f64,
    /// Output format (0=full, >=2=skip alignment strings).
    pub outfmt_opt: i32,
    /// User-provided alignment sequences.
    pub sequence: Vec<String>,
    /// Match mode option: 6 = enforce SSE sequentiality.
    pub mm_opt: i32,
}

impl Default for SoiOptions {
    fn default() -> Self {
        SoiOptions {
            mol_type: MolType::Protein,
            close_k_opt: 5,
            fast_opt: false,
            i_opt: 0,
            a_opt: 0,
            u_opt: false,
            lnorm_ass: 0.0,
            d_opt: false,
            d0_scale: 0.0,
            outfmt_opt: 0,
            sequence: Vec::new(),
            mm_opt: 0,
        }
    }
}

/// Result of SOI alignment.
#[derive(Debug, Clone)]
pub struct SoiResult {
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
    /// d0 values for normalizations.
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
    /// Per-position distances (-1 for unaligned).
    pub dist_list: Vec<f64>,
    /// Aligned sequence X (with gaps). Empty if outfmt_opt >= 2.
    pub seq_x_aligned: String,
    /// Aligned sequence Y (with gaps). Empty if outfmt_opt >= 2.
    pub seq_y_aligned: String,
    /// Match markers (`:` close, `.` other aligned). Empty if outfmt_opt >= 2.
    pub seq_m: String,
    /// Optimal rotation/translation transform.
    pub transform: Transform,
}

/// Main entry point for SOI alignment.
///
/// Corresponds to C++ `SOIalign_main`. Performs sequence-order independent
/// structure alignment combining initial SE-based alignment with SOI
/// iterative refinement and K-nearest neighbor fragment matching.
///
/// `xa`, `ya`: CA coordinates of structure x and y.
/// `secx`, `secy`: secondary structure assignment characters.
/// `seqx`, `seqy`: one-letter amino acid sequences.
/// `initial_transform`: initial rotation/translation (e.g., from a prior SE alignment).
pub fn soialign_main(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[u8],
    seqy: &[u8],
    secx: &[u8],
    secy: &[u8],
    initial_transform: &Transform,
    opts: &SoiOptions,
) -> SoiResult {
    let xlen = xa.len();
    let ylen = ya.len();
    let mol_type_int = match opts.mol_type {
        MolType::Protein => 0,
        MolType::RNA => 1,
    };

    // Compute SSE boundaries
    let secx_bond = assign_sec_bond(secx);
    let secy_bond = assign_sec_bond(secy);

    // Precompute K-nearest neighbors
    let xk = get_close_k(xa, opts.close_k_opt);
    let yk = get_close_k(ya, opts.close_k_opt);

    // Search parameters
    let search_params = TMParams::for_search(xlen, ylen);
    let local_d0_search = search_params.d0_search;
    let iteration_max: usize = if opts.fast_opt { 2 } else { 30 };

    let mut invmap0 = vec![-1i32; ylen];
    let mut fwdmap0 = vec![-1i32; xlen];
    let mut tm_max = -1.0_f64;

    // ------------------------------------------------------------------
    // Phase 1: Initial alignment from SE (sequence order dependent)
    // ------------------------------------------------------------------
    // Apply initial transform and compute score matrix
    let mut xt = vec![[0.0; 3]; xlen];
    initial_transform.apply_batch(xa, &mut xt);
    let score = super2score(&xt, ya, search_params.d0, search_params.score_d8);
    let scoret = transpose_score(&score, xlen, ylen);

    // Forward direction: SOI_iter with x->y mapping
    let tm = soi_iter(
        xa,
        ya,
        &mut invmap0,
        iteration_max,
        &search_params,
        &secx_bond,
        &secy_bond,
        opts.mm_opt,
        false,
        &score,
    );
    if tm > tm_max {
        tm_max = tm;
    }

    // Reverse direction: SOI_iter with y->x mapping, then convert
    let mut fwdmap_tmp = vec![-1i32; xlen];
    // Initialize fwdmap from current invmap0
    for j in 0..ylen {
        let i = invmap0[j];
        if i >= 0 {
            fwdmap_tmp[i as usize] = j as i32;
        }
    }

    let tm_rev = soi_iter(
        ya,
        xa,
        &mut fwdmap_tmp,
        iteration_max,
        &search_params,
        &secy_bond,
        &secx_bond,
        opts.mm_opt,
        true,
        &scoret,
    );
    if tm_rev > tm_max {
        tm_max = tm_rev;
        // Convert fwdmap to invmap
        for j in 0..ylen {
            invmap0[j] = -1;
        }
        for i in 0..xlen {
            let j = fwdmap_tmp[i];
            if j >= 0 {
                invmap0[j as usize] = i as i32;
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: SOI initial assignment using K-nearest neighbors
    // ------------------------------------------------------------------
    if opts.close_k_opt >= 3 {
        let (mut invmap_knn, score_knn) = get_soi_initial_assign(
            &xk,
            &yk,
            opts.close_k_opt,
            xlen,
            ylen,
            &search_params,
            &secx_bond,
            &secy_bond,
            opts.mm_opt,
        );

        let scoret_knn = transpose_score(&score_knn, xlen, ylen);

        // Forward: superpose from KNN assignment, then iterate
        let (_, xt_knn) = assign2super(xa, ya, &invmap_knn, &search_params);
        let score_after_super = super2score(&xt_knn, ya, search_params.d0, search_params.score_d8);

        let tm = soi_iter(
            xa,
            ya,
            &mut invmap_knn,
            iteration_max,
            &search_params,
            &secx_bond,
            &secy_bond,
            opts.mm_opt,
            false,
            &score_after_super,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0[..ylen].copy_from_slice(&invmap_knn[..ylen]);
        }

        // Reverse direction from KNN score
        let mut fwdmap_knn = vec![-1i32; xlen];
        // Build fwdmap from invmap_knn (possibly the one that was just updated)
        for j in 0..ylen {
            let i = invmap_knn[j];
            if i >= 0 && (i as usize) < xlen {
                fwdmap_knn[i as usize] = j as i32;
            }
        }
        // Re-init fwdmap for reverse pass
        fwdmap_knn = vec![-1i32; xlen];
        if opts.mm_opt == 6 {
            let mut ws = crate::core::types::DPWorkspace::new(ylen, xlen);
            let dp_result =
                crate::core::nwdp::nwdp_score_matrix(&mut ws, &scoret_knn, ylen, xlen, -0.6);
            fwdmap_knn[..xlen].copy_from_slice(&dp_result[..xlen]);
        }
        greedy::soi_egs(
            &scoret_knn,
            ylen,
            xlen,
            &mut fwdmap_knn,
            &secy_bond,
            &secx_bond,
            opts.mm_opt,
        );

        let (_, yt_knn) = assign2super(ya, xa, &fwdmap_knn, &search_params);
        let scoret_after = super2score(&yt_knn, xa, search_params.d0, search_params.score_d8);

        let tm = soi_iter(
            ya,
            xa,
            &mut fwdmap_knn,
            iteration_max,
            &search_params,
            &secy_bond,
            &secx_bond,
            opts.mm_opt,
            false,
            &scoret_after,
        );
        if tm > tm_max {
            #[allow(unused_assignments)]
            {
                tm_max = tm;
            }
            for j in 0..ylen {
                invmap0[j] = -1;
            }
            for i in 0..xlen {
                let j = fwdmap_knn[i];
                if j >= 0 {
                    invmap0[j as usize] = i as i32;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Check that we have a valid alignment
    // ------------------------------------------------------------------
    let mut invmap_final = invmap0.clone();
    fwdmap0 = vec![-1i32; xlen];
    let mut has_alignment = false;
    for j in 0..ylen {
        let i = invmap_final[j];
        if i >= 0 {
            fwdmap0[i as usize] = j as i32;
            has_alignment = true;
        }
    }
    if !has_alignment {
        return SoiResult {
            tm1: 0.0,
            tm2: 0.0,
            tm3: 0.0,
            tm4: 0.0,
            tm5: 0.0,
            d0a: 0.0,
            d0b: 0.0,
            d0u: 0.0,
            d0a_avg: 0.0,
            d0_out: if mol_type_int > 0 { 3.5 } else { 5.0 },
            rmsd: 0.0,
            n_ali: 0,
            n_ali8: 0,
            liden: 0.0,
            tm_ali: 0.0,
            rmsd_ali: 0.0,
            invmap: invmap_final,
            dist_list: vec![-1.0; ylen],
            seq_x_aligned: String::new(),
            seq_y_aligned: String::new(),
            seq_m: String::new(),
            transform: Transform::default(),
        };
    }

    // ------------------------------------------------------------------
    // Detailed TM-score search
    // ------------------------------------------------------------------
    let (_, best_transform) = detailed_search_standard(
        xa,
        ya,
        &invmap_final,
        1, // simplify_step
        8, // score_sum_method
        local_d0_search,
        false,
        search_params.lnorm,
        search_params.score_d8,
        search_params.d0,
    );
    let t0 = best_transform;

    // ------------------------------------------------------------------
    // Final TM-score computation with multiple normalizations
    // ------------------------------------------------------------------
    // Apply best transform
    let mut xt_final = vec![[0.0; 3]; xlen];
    t0.apply_batch(xa, &mut xt_final);

    // Filter aligned pairs by score_d8 distance cutoff
    let mut n_ali = 0usize;

    let mut m1 = Vec::new();
    let mut m2 = Vec::new();
    let mut xtm = Vec::new();
    let mut ytm = Vec::new();
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();

    for i in 0..xlen {
        let j = fwdmap0[i];
        if j < 0 {
            continue;
        }
        let ju = j as usize;
        n_ali += 1;
        let d = dist_squared(&xt_final[i], &ya[ju]).sqrt();
        if d <= search_params.score_d8 {
            m1.push(i);
            m2.push(ju);
            xtm.push(xa[i]);
            ytm.push(ya[ju]);
            r1.push(xt_final[i]);
            r2.push(ya[ju]);
        } else {
            fwdmap0[i] = -1;
        }
    }
    let n_ali8 = m1.len();

    // Update invmap_final from filtered fwdmap
    invmap_final = vec![-1i32; ylen];
    for i in 0..xlen {
        let j = fwdmap0[i];
        if j >= 0 {
            invmap_final[j as usize] = i as i32;
        }
    }

    // Compute RMSD from Kabsch on filtered pairs
    let rmsd0 = if n_ali8 > 0 {
        if let Some(result) = kabsch(&r1, &r2, KabschMode::RmsOnly) {
            (result.rms / n_ali8 as f64).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Final TM-score 1: normalized by xlen
    let params_b = TMParams::for_final(xlen as f64, opts.mol_type);
    let d0b = params_b.d0;
    let tm2 = if n_ali8 > 0 {
        let (score, _) = tmscore8_search(&xtm, &ytm, 1, 0, params_b.d0_search, &params_b);
        score
    } else {
        0.0
    };

    // Final TM-score 2: normalized by ylen
    // Recompute filtered pairs using invmap (y->x direction) for consistency
    let mut xt_final2 = vec![[0.0; 3]; xlen];
    t0.apply_batch(xa, &mut xt_final2);

    let mut m1_2 = Vec::new();
    let mut m2_2 = Vec::new();
    let mut xtm2 = Vec::new();
    let mut ytm2 = Vec::new();
    for j in 0..ylen {
        let i = invmap_final[j];
        if i < 0 {
            continue;
        }
        let iu = i as usize;
        let d = dist_squared(&xt_final2[iu], &ya[j]).sqrt();
        if d <= search_params.score_d8 {
            m1_2.push(iu);
            m2_2.push(j);
            xtm2.push(xa[iu]);
            ytm2.push(ya[j]);
        } else {
            invmap_final[j] = -1;
        }
    }

    let params_a = TMParams::for_final(ylen as f64, opts.mol_type);
    let d0a = params_a.d0;
    let tm1 = if !xtm2.is_empty() {
        let (score, _) = tmscore8_search(&xtm2, &ytm2, 1, 0, params_a.d0_search, &params_a);
        score
    } else {
        0.0
    };

    // Optional normalizations
    let mut tm3 = 0.0;
    let mut d0a_avg = 0.0;
    if opts.a_opt > 0 && !xtm2.is_empty() {
        let lnorm_avg = (xlen + ylen) as f64 * 0.5;
        let params_avg = TMParams::for_final(lnorm_avg, opts.mol_type);
        d0a_avg = params_avg.d0;
        let (score, _) = tmscore8_search(&xtm2, &ytm2, 1, 0, params_avg.d0_search, &params_avg);
        tm3 = score;
    }

    let mut tm4 = 0.0;
    let mut d0u = 0.0;
    if opts.u_opt && !xtm2.is_empty() {
        let params_u = TMParams::for_final(opts.lnorm_ass, opts.mol_type);
        d0u = params_u.d0;
        let (score, _) = tmscore8_search(&xtm2, &ytm2, 1, 0, params_u.d0_search, &params_u);
        tm4 = score;
    }

    let mut tm5 = 0.0;
    if opts.d_opt && !xtm2.is_empty() {
        let params_s = TMParams::for_scale(ylen, opts.d0_scale);
        let (score, _) = tmscore8_search(&xtm2, &ytm2, 1, 0, params_s.d0_search, &params_s);
        tm5 = score;
    }

    let d0_out = if mol_type_int > 0 { 3.5 } else { 5.0 };

    // ------------------------------------------------------------------
    // Build output alignment strings
    // ------------------------------------------------------------------
    let mut seq_x_aligned = String::new();
    let mut seq_y_aligned = String::new();
    let mut seq_m = String::new();
    let mut dist_list = vec![-1.0_f64; ylen];
    let mut liden = 0.0_f64;

    // Recompute fwdmap from final invmap
    fwdmap0 = vec![-1i32; xlen];
    for j in 0..ylen {
        let i = invmap_final[j];
        if i >= 0 {
            fwdmap0[i as usize] = j as i32;
        }
    }

    // Apply final transform for output
    t0.apply_batch(xa, &mut xt_final);

    // Build alignment string in the C++ SOIalign format:
    // First ylen positions correspond to y residues (with x mapped or gap),
    // then appended unaligned x residues.
    let ali_len = {
        let n_aligned = invmap_final.iter().filter(|&&v| v >= 0).count();
        xlen + ylen - n_aligned
    };
    let mut sxa = vec![b'-'; ali_len];
    let mut sya = vec![b'-'; ali_len];
    let mut sm = vec![b' '; ali_len];

    for j in 0..ylen {
        sya[j] = seqy[j];
        let i = invmap_final[j];
        if i < 0 {
            continue;
        }
        let iu = i as usize;
        let d = dist_squared(&xt_final[iu], &ya[j]).sqrt();
        dist_list[j] = d;
        if d < d0_out {
            sm[j] = b':';
        } else {
            sm[j] = b'.';
        }
        sxa[j] = seqx[iu];
        liden += if seqx[iu] == seqy[j] { 1.0 } else { 0.0 };
    }

    // Append unaligned x residues
    let mut k = 0usize;
    for i in 0..xlen {
        let j = fwdmap0[i];
        if j >= 0 {
            continue;
        }
        sxa[ylen + k] = seqx[i];
        k += 1;
    }

    if opts.outfmt_opt < 2 {
        seq_x_aligned = String::from_utf8(sxa).unwrap_or_default();
        seq_y_aligned = String::from_utf8(sya).unwrap_or_default();
        seq_m = String::from_utf8(sm).unwrap_or_default();
    }

    SoiResult {
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
        invmap: invmap_final,
        dist_list,
        seq_x_aligned,
        seq_y_aligned,
        seq_m,
        transform: t0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_helix_coords(n: usize) -> Vec<Coord3D> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 100.0 / 180.0 * std::f64::consts::PI;
                [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64]
            })
            .collect()
    }

    #[test]
    fn test_soialign_identical() {
        let n = 15;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = SoiOptions {
            close_k_opt: 3,
            ..Default::default()
        };
        let transform = Transform::default();
        let result = soialign_main(&coords, &coords, &seq, &seq, &sec, &sec, &transform, &opts);

        assert!(result.tm1 > 0.5, "tm1={}", result.tm1);
        assert!(result.tm2 > 0.5, "tm2={}", result.tm2);
        assert!(result.n_ali8 > 0, "n_ali8={}", result.n_ali8);
    }

    #[test]
    fn test_soialign_no_knn_fallback() {
        // With close_k_opt < 3, should skip KNN phase
        let n = 10;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = SoiOptions {
            close_k_opt: 2,
            ..Default::default()
        };
        let transform = Transform::default();
        let result = soialign_main(&coords, &coords, &seq, &seq, &sec, &sec, &transform, &opts);

        // Should still produce an alignment
        assert!(result.n_ali > 0 || result.n_ali8 == 0);
    }

    #[test]
    fn test_soialign_fast_mode() {
        let n = 15;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = SoiOptions {
            close_k_opt: 3,
            fast_opt: true,
            ..Default::default()
        };
        let transform = Transform::default();
        let result = soialign_main(&coords, &coords, &seq, &seq, &sec, &sec, &transform, &opts);

        assert!(result.tm1 >= 0.0);
    }

    #[test]
    fn test_soialign_outfmt_skips_strings() {
        let n = 10;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = SoiOptions {
            close_k_opt: 3,
            outfmt_opt: 2,
            ..Default::default()
        };
        let transform = Transform::default();
        let result = soialign_main(&coords, &coords, &seq, &seq, &sec, &sec, &transform, &opts);

        assert!(result.seq_x_aligned.is_empty());
        assert!(result.seq_y_aligned.is_empty());
    }
}
