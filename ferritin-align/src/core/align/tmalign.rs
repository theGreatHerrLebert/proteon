//! TMalign_main: master alignment driver.
//!
//! Ported from C++ TMalign (lines 3923-4524).
//! Orchestrates all initialization strategies, picks the best,
//! computes final TM-scores, and builds the alignment strings.
//!
//! Reference: Zhang & Skolnick, "TM-align: a protein structure alignment
//! algorithm based on the TM-score", *Nucleic Acids Res.* 33(7),
//! 2302-2309 (2005). The original C++ reference implementation is the
//! sole oracle for this port's output.

use anyhow::{bail, Result};

use crate::core::align::dp_iter::dp_iter;
use crate::core::align::initial::get_initial;
use crate::core::align::initial_fgt::get_initial_fgt;
use crate::core::align::initial_local::get_initial5;
use crate::core::align::initial_ss::get_initial_ss;
use crate::core::align::initial_ssplus::get_initial_ssplus;
use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::tmscore::{detailed_search, detailed_search_standard, tmscore8_search};
use crate::core::types::{
    dist_squared, AlignOptions, AlignResult, Coord3D, DPWorkspace, MolType, TMParams, Transform,
};

/// Approximate TM-score given a rotation matrix and alignment map.
/// Used for TMcut early termination.
fn approx_tm(
    x: &[Coord3D],
    y: &[Coord3D],
    transform: &Transform,
    invmap: &[i32],
    a_opt: i32,
    mol_type: MolType,
) -> f64 {
    let xlen = x.len();
    let ylen = y.len();
    let lnorm_0 = match a_opt {
        -2 if xlen > ylen => xlen as f64,
        -1 if xlen < ylen => xlen as f64,
        1 => (xlen + ylen) as f64 / 2.0,
        _ => ylen as f64,
    };

    let params = TMParams::for_final(lnorm_0, mol_type);
    let mut tm = 0.0;
    for (j, &i) in invmap.iter().enumerate() {
        if i >= 0 {
            let xt = transform.apply(&x[i as usize]);
            let d = dist_squared(&xt, &y[j]).sqrt();
            tm += 1.0 / (1.0 + (d / params.d0) * (d / params.d0));
        }
    }
    tm / lnorm_0
}

/// Parse a user-provided FASTA alignment into an invmap.
fn alignment_to_invmap(sequences: &[String], xlen: usize, ylen: usize) -> Vec<i32> {
    let mut invmap = vec![-1_i32; ylen];
    let mut i1: isize = -1;
    let mut i2: isize = -1;
    let l = sequences[0].len().min(sequences[1].len());
    let seq0: Vec<char> = sequences[0].chars().collect();
    let seq1: Vec<char> = sequences[1].chars().collect();

    for kk in 0..l {
        if seq0[kk] != '-' {
            i1 += 1;
        }
        if seq1[kk] != '-' {
            i2 += 1;
            if i2 >= ylen as isize || i1 >= xlen as isize {
                break;
            }
            if seq0[kk] != '-' {
                invmap[i2 as usize] = i1 as i32;
            }
        }
    }
    invmap
}

/// Main TM-align algorithm.
///
/// Runs all initialization strategies, iteratively refines,
/// and computes final TM-scores with multiple normalizations.
pub fn tmalign(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[char],
    seqy: &[char],
    secx: &[char],
    secy: &[char],
    opts: &AlignOptions,
) -> Result<AlignResult> {
    let xlen = xa.len();
    let ylen = ya.len();
    if xlen < 3 || ylen < 3 {
        bail!("Sequence too short (< 3 residues)");
    }

    let params = TMParams::for_search(xlen, ylen);
    let simplify_step = 40;
    let score_sum_method = 8u8;
    let local_d0_search = params.d0_search;

    let mut ws = DPWorkspace::new(xlen, ylen);
    let mut invmap0 = vec![-1_i32; ylen];
    #[allow(unused_assignments)]
    let mut tm_max = -1.0_f64;
    let mut best_transform = Transform::default();

    let ddcc = if params.lnorm <= 40.0 { 0.1 } else { 0.4 };
    let iteration_max = if opts.fast_opt { 2 } else { 30 };

    let mut _tm_ali = 0.0_f64;
    let mut _l_ali = 0usize;
    let mut _rmsd_ali = 0.0_f64;

    // ------------------------------------------------------------------
    // Strategy: strict user alignment (-I)
    // ------------------------------------------------------------------
    let mut b_align_stick = false;
    if opts.i_opt == 3 {
        if let Some(ref sequences) = opts.user_alignment {
            let invmap = alignment_to_invmap(sequences, xlen, ylen);

            let (tm_score, l_ali, rmsd, _t) = crate::core::tmscore::standard_tmscore(
                xa,
                ya,
                &invmap,
                params.score_d8,
                opts.mol_type,
            );
            _tm_ali = tm_score;
            _l_ali = l_ali;
            _rmsd_ali = rmsd;

            let (tm, t) = detailed_search_standard(
                xa,
                ya,
                &invmap,
                40,
                8,
                local_d0_search,
                true,
                params.lnorm,
                params.score_d8,
                params.d0,
            );
            if tm > tm_max {
                tm_max = tm;
                invmap0 = invmap;
                best_transform = t;
            }
            b_align_stick = true;
        }
    }

    if !b_align_stick {
        // ------------------------------------------------------------------
        // 1. Gapless threading
        // ------------------------------------------------------------------
        let (_, init_map) = get_initial(xa, ya, params.d0, params.d0_search, opts.fast_opt);
        let (tm, t) = detailed_search(
            xa,
            ya,
            &init_map,
            simplify_step,
            score_sum_method,
            local_d0_search,
            &params,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0 = init_map;
            best_transform = t.clone();
        }

        let (tm, dp_map, t) = dp_iter(
            xa,
            ya,
            &mut ws,
            &t,
            0..2,
            iteration_max,
            local_d0_search,
            &params,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0 = dp_map;
            best_transform = t;
        }

        // TMcut early termination
        if opts.tm_cut > 0.0 {
            let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
            if tm_tmp < 0.5 * opts.tm_cut {
                return build_early_result(tm_tmp, &best_transform);
            }
        }

        // ------------------------------------------------------------------
        // 2. Secondary structure alignment
        // ------------------------------------------------------------------
        let init_map = get_initial_ss(&mut ws, secx, secy);
        let (tm, t) = detailed_search(
            xa,
            ya,
            &init_map,
            simplify_step,
            score_sum_method,
            local_d0_search,
            &params,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0 = init_map.clone();
            best_transform = t.clone();
        }
        if tm > tm_max * 0.2 {
            let (tm, dp_map, t2) = dp_iter(
                xa,
                ya,
                &mut ws,
                &t,
                0..2,
                iteration_max,
                local_d0_search,
                &params,
            );
            if tm > tm_max {
                tm_max = tm;
                invmap0 = dp_map;
                best_transform = t2;
            }
        }

        if opts.tm_cut > 0.0 {
            let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
            if tm_tmp < 0.52 * opts.tm_cut {
                return build_early_result(tm_tmp, &best_transform);
            }
        }

        // ------------------------------------------------------------------
        // 3. Local superposition (initial5)
        // ------------------------------------------------------------------
        let (found, init_map) = get_initial5(
            xa,
            ya,
            &mut ws,
            params.d0,
            params.d0_search,
            opts.fast_opt,
            params.d0_min,
        );
        if found {
            let (tm, t) = detailed_search(
                xa,
                ya,
                &init_map,
                simplify_step,
                score_sum_method,
                local_d0_search,
                &params,
            );
            if tm > tm_max {
                tm_max = tm;
                invmap0 = init_map;
                best_transform = t.clone();
            }
            if tm > tm_max * ddcc {
                let (tm, dp_map, t2) =
                    dp_iter(xa, ya, &mut ws, &t, 0..2, 2, local_d0_search, &params);
                if tm > tm_max {
                    tm_max = tm;
                    invmap0 = dp_map;
                    best_transform = t2;
                }
            }
        }

        if opts.tm_cut > 0.0 {
            let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
            if tm_tmp < 0.54 * opts.tm_cut {
                return build_early_result(tm_tmp, &best_transform);
            }
        }

        // ------------------------------------------------------------------
        // 4. SS + previous alignment (initial_ssplus)
        // ------------------------------------------------------------------
        let init_map = get_initial_ssplus(
            xa,
            ya,
            secx,
            secy,
            &invmap0,
            &mut ws,
            params.d0_min,
            params.d0,
        );
        let (tm, t) = detailed_search(
            xa,
            ya,
            &init_map,
            simplify_step,
            score_sum_method,
            local_d0_search,
            &params,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0 = init_map;
            best_transform = t.clone();
        }
        if tm > tm_max * ddcc {
            let (tm, dp_map, t2) = dp_iter(
                xa,
                ya,
                &mut ws,
                &t,
                0..2,
                iteration_max,
                local_d0_search,
                &params,
            );
            if tm > tm_max {
                tm_max = tm;
                invmap0 = dp_map;
                best_transform = t2;
            }
        }

        if opts.tm_cut > 0.0 {
            let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
            if tm_tmp < 0.56 * opts.tm_cut {
                return build_early_result(tm_tmp, &best_transform);
            }
        }

        // ------------------------------------------------------------------
        // 5. Fragment gapless threading (initial_fgt)
        // ------------------------------------------------------------------
        let (_, init_map) = get_initial_fgt(
            xa,
            ya,
            params.d0,
            params.d0_search,
            params.dcu0,
            opts.fast_opt,
        );
        let (tm, t) = detailed_search(
            xa,
            ya,
            &init_map,
            simplify_step,
            score_sum_method,
            local_d0_search,
            &params,
        );
        if tm > tm_max {
            tm_max = tm;
            invmap0 = init_map;
            best_transform = t.clone();
        }
        if tm > tm_max * ddcc {
            let (tm, dp_map, t2) = dp_iter(xa, ya, &mut ws, &t, 1..2, 2, local_d0_search, &params);
            if tm > tm_max {
                tm_max = tm;
                invmap0 = dp_map;
                best_transform = t2;
            }
        }

        if opts.tm_cut > 0.0 {
            let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
            if tm_tmp < 0.58 * opts.tm_cut {
                return build_early_result(tm_tmp, &best_transform);
            }
        }

        // ------------------------------------------------------------------
        // 6. User soft alignment (-i)
        // ------------------------------------------------------------------
        if opts.i_opt == 1 {
            if let Some(ref sequences) = opts.user_alignment {
                let invmap = alignment_to_invmap(sequences, xlen, ylen);

                let (tm_score, l_ali, rmsd, _t) = crate::core::tmscore::standard_tmscore(
                    xa,
                    ya,
                    &invmap,
                    params.score_d8,
                    opts.mol_type,
                );
                _tm_ali = tm_score;
                _l_ali = l_ali;
                _rmsd_ali = rmsd;

                let (tm, t) = detailed_search_standard(
                    xa,
                    ya,
                    &invmap,
                    40,
                    8,
                    local_d0_search,
                    true,
                    params.lnorm,
                    params.score_d8,
                    params.d0,
                );
                if tm > tm_max {
                    tm_max = tm;
                    invmap0 = invmap;
                    best_transform = t.clone();
                }

                let (tm, dp_map, t2) = dp_iter(
                    xa,
                    ya,
                    &mut ws,
                    &t,
                    0..2,
                    iteration_max,
                    local_d0_search,
                    &params,
                );
                if tm > tm_max {
                    #[allow(unused_assignments)]
                    {
                        tm_max = tm;
                    }
                    invmap0 = dp_map;
                    best_transform = t2;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Check: at least one aligned residue
    // ------------------------------------------------------------------
    if !invmap0.iter().any(|&i| i >= 0) {
        bail!("No alignment found between the two structures");
    }

    // Final TMcut check
    if opts.tm_cut > 0.0 {
        let tm_tmp = approx_tm(xa, ya, &best_transform, &invmap0, opts.a_opt, opts.mol_type);
        if tm_tmp < 0.6 * opts.tm_cut {
            return build_early_result(tm_tmp, &best_transform);
        }
    }

    // ------------------------------------------------------------------
    // Final detailed search
    // ------------------------------------------------------------------
    let final_simplify = if opts.fast_opt { 40 } else { 1 };
    let (_, final_t) = detailed_search_standard(
        xa,
        ya,
        &invmap0,
        final_simplify,
        8,
        local_d0_search,
        false,
        params.lnorm,
        params.score_d8,
        params.d0,
    );

    // Select pairs with distance < score_d8 for final scoring
    let mut xt = vec![[0.0; 3]; xlen];
    final_t.apply_batch(xa, &mut xt);

    let mut m1 = Vec::new(); // aligned indices in x
    let mut m2 = Vec::new(); // aligned indices in y
    let mut xtm = Vec::new();
    let mut ytm = Vec::new();
    for (j, &i) in invmap0.iter().enumerate() {
        if i >= 0 {
            let d = dist_squared(&xt[i as usize], &ya[j]).sqrt();
            if d <= params.score_d8 || opts.i_opt == 3 {
                m1.push(i as usize);
                m2.push(j);
                xtm.push(xa[i as usize]);
                ytm.push(ya[j]);
            }
        }
    }
    let n_ali8 = xtm.len();

    // Compute RMSD on final aligned pairs
    let rmsd0 = if let Some(result) = kabsch(
        &xt[..]
            .iter()
            .enumerate()
            .filter(|(idx, _)| m1.contains(idx))
            .map(|(_, c)| *c)
            .collect::<Vec<_>>(),
        &ytm,
        KabschMode::RmsOnly,
    ) {
        (result.rms / n_ali8 as f64).sqrt()
    } else {
        // Fallback: compute directly from rotated coordinates
        let r1: Vec<Coord3D> = m1.iter().map(|&i| xt[i]).collect();
        if let Some(result) = kabsch(&r1, &ytm, KabschMode::RmsOnly) {
            (result.rms / n_ali8 as f64).sqrt()
        } else {
            0.0
        }
    };

    // ------------------------------------------------------------------
    // Final TM-scores with multiple normalizations
    // ------------------------------------------------------------------
    let score_sum_method_final = 0u8;
    let simplify_final = 1;

    // TM1: normalized by ylen (chain2 = reference)
    let params_a = TMParams::for_final(ylen as f64, opts.mol_type);
    let d0a = params_a.d0;
    let (tm1, t0) = tmscore8_search(
        &xtm,
        &ytm,
        simplify_final,
        score_sum_method_final,
        params_a.d0_search,
        &params_a,
    );

    // TM2: normalized by xlen
    let params_b = TMParams::for_final(xlen as f64, opts.mol_type);
    let d0b = params_b.d0;
    let (tm2, _) = tmscore8_search(
        &xtm,
        &ytm,
        simplify_final,
        score_sum_method_final,
        params_b.d0_search,
        &params_b,
    );

    // TM3: normalized by average length (if a_opt > 0)
    let mut tm3 = 0.0;
    let mut d0_out = d0a;
    if opts.a_opt > 0 {
        let avg_len = (xlen + ylen) as f64 / 2.0;
        let params_avg = TMParams::for_final(avg_len, opts.mol_type);
        d0_out = params_avg.d0;
        let (score, _) = tmscore8_search(
            &xtm,
            &ytm,
            simplify_final,
            score_sum_method_final,
            params_avg.d0_search,
            &params_avg,
        );
        tm3 = score;
    }

    // TM4: normalized by user-specified length
    let mut tm4 = 0.0;
    if opts.u_opt {
        let params_u = TMParams::for_final(opts.lnorm_ass, opts.mol_type);
        d0_out = params_u.d0;
        let (score, _) = tmscore8_search(
            &xtm,
            &ytm,
            simplify_final,
            score_sum_method_final,
            params_u.d0_search,
            &params_u,
        );
        tm4 = score;
    }

    // TM5: scaled by user-specified d0
    let mut tm5 = 0.0;
    if opts.d_opt {
        let params_s = TMParams::for_scale(ylen, opts.d0_scale);
        d0_out = opts.d0_scale;
        let (score, _) = tmscore8_search(
            &xtm,
            &ytm,
            simplify_final,
            score_sum_method_final,
            params_s.d0_search,
            &params_s,
        );
        tm5 = score;
    }

    // ------------------------------------------------------------------
    // Build alignment strings
    // ------------------------------------------------------------------
    let mut xt_final = vec![[0.0; 3]; xlen];
    t0.apply_batch(xa, &mut xt_final);

    let mut seq_x_aln = String::new();
    let mut seq_y_aln = String::new();
    let mut seq_m = String::new();
    let mut liden = 0.0_f64;
    let mut i_old = 0usize;
    let mut j_old = 0usize;

    for k in 0..n_ali8 {
        // Gaps in x before this aligned pair
        for i in i_old..m1[k] {
            seq_x_aln.push(seqx[i]);
            seq_y_aln.push('-');
            seq_m.push(' ');
        }
        // Gaps in y before this aligned pair
        for j in j_old..m2[k] {
            seq_x_aln.push('-');
            seq_y_aln.push(seqy[j]);
            seq_m.push(' ');
        }
        // Aligned pair
        let cx = seqx[m1[k]];
        let cy = seqy[m2[k]];
        seq_x_aln.push(cx);
        seq_y_aln.push(cy);
        if cx == cy {
            liden += 1.0;
        }
        let d = dist_squared(&xt_final[m1[k]], &ya[m2[k]]).sqrt();
        if d < d0_out {
            seq_m.push(':');
        } else {
            seq_m.push('.');
        }
        i_old = m1[k] + 1;
        j_old = m2[k] + 1;
    }

    // Tail gaps
    for i in i_old..xlen {
        seq_x_aln.push(seqx[i]);
        seq_y_aln.push('-');
        seq_m.push(' ');
    }
    for j in j_old..ylen {
        seq_x_aln.push('-');
        seq_y_aln.push(seqy[j]);
        seq_m.push(' ');
    }

    Ok(AlignResult {
        tm_score_chain1: tm1,
        tm_score_chain2: tm2,
        tm_score_avg: tm3,
        tm_score_user: tm4,
        tm_score_scaled: tm5,
        rmsd: rmsd0,
        n_aligned: n_ali8,
        seq_identity: if n_ali8 > 0 {
            liden / n_ali8 as f64
        } else {
            0.0
        },
        transform: t0,
        aligned_seq_x: seq_x_aln,
        aligned_seq_y: seq_y_aln,
        alignment_markers: seq_m,
        d0a,
        d0b,
        d0_out,
    })
}

/// Build a minimal result for TMcut early termination.
fn build_early_result(tm_tmp: f64, transform: &Transform) -> Result<AlignResult> {
    Ok(AlignResult {
        tm_score_chain1: tm_tmp,
        tm_score_chain2: tm_tmp,
        tm_score_avg: tm_tmp,
        tm_score_user: tm_tmp,
        tm_score_scaled: tm_tmp,
        rmsd: 0.0,
        n_aligned: 0,
        seq_identity: 0.0,
        transform: transform.clone(),
        aligned_seq_x: String::new(),
        aligned_seq_y: String::new(),
        alignment_markers: String::new(),
        d0a: 0.0,
        d0b: 0.0,
        d0_out: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::secondary_structure::make_sec;

    #[test]
    fn tmalign_identical_structures() {
        // 10-residue "helix-like" structure aligned to itself
        let coords: Vec<Coord3D> = (0..10)
            .map(|i| {
                let t = i as f64 * 100.0_f64.to_radians();
                [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64]
            })
            .collect();
        let seq: Vec<char> = "AAAAAGGGGG".chars().collect();
        let sec = make_sec(&coords);

        let opts = AlignOptions::default();
        let result = tmalign(&coords, &coords, &seq, &seq, &sec, &sec, &opts).unwrap();

        // TM-score of identical structures should be ~1.0
        assert!(
            result.tm_score_chain1 > 0.9,
            "TM1 = {}, expected > 0.9",
            result.tm_score_chain1
        );
        assert!(result.rmsd < 0.1, "RMSD = {}, expected < 0.1", result.rmsd);
        assert!(result.n_aligned > 0, "should have aligned residues");
        assert!(
            result.seq_identity > 0.99,
            "identical sequences should have ~1.0 identity"
        );
    }

    #[test]
    fn tmalign_translated_structure() {
        // Same structure but translated by (10, 20, 30)
        let coords1: Vec<Coord3D> = (0..15)
            .map(|i| {
                let t = i as f64 * 100.0_f64.to_radians();
                [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64]
            })
            .collect();
        let coords2: Vec<Coord3D> = coords1
            .iter()
            .map(|c| [c[0] + 10.0, c[1] + 20.0, c[2] + 30.0])
            .collect();
        let seq: Vec<char> = "AAAAAGGGGGVVVVV".chars().collect();
        let sec1 = make_sec(&coords1);
        let sec2 = make_sec(&coords2);

        let opts = AlignOptions::default();
        let result = tmalign(&coords1, &coords2, &seq, &seq, &sec1, &sec2, &opts).unwrap();

        // Should still align perfectly (translation doesn't affect TM-score)
        assert!(
            result.tm_score_chain1 > 0.9,
            "TM1 = {}, expected > 0.9",
            result.tm_score_chain1
        );
        assert!(result.rmsd < 0.1, "RMSD = {}, expected < 0.1", result.rmsd);
    }

    #[test]
    fn tmalign_short_sequences_error() {
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let seq = vec!['A', 'A'];
        let sec = vec!['C', 'C'];
        let opts = AlignOptions::default();
        assert!(tmalign(&coords, &coords, &seq, &seq, &sec, &sec, &opts).is_err());
    }
}
