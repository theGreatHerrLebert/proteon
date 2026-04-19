//! TM-score computation functions.
//!
//! Ported from C++ TMalign (lines 1719-2340, 3787-3855).
//!
//! References:
//! - Zhang & Skolnick, "Scoring function for automated assessment of protein
//!   structure template quality", *Proteins* 57(4), 702-710 (2004) — the
//!   TM-score definition.
//! - Zhang & Skolnick, "TM-align: a protein structure alignment algorithm
//!   based on the TM-score", *Nucleic Acids Res.* 33(7), 2302-2309 (2005)
//!   — the iterative score-then-superpose procedure implemented here.

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::types::{dist_squared, Coord3D, MolType, TMParams, Transform};

/// Result of score_fun8: a filtered set of aligned pair indices and a TM-score.
pub struct ScoreResult {
    /// Indices of pairs within distance threshold.
    pub i_ali: Vec<usize>,
    /// TM-score (normalized by lnorm or n_ali depending on variant).
    pub score: f64,
}

/// Score aligned pairs, filtering by distance threshold.
/// Normalizes by `lnorm`.
///
/// Corresponds to C++ `score_fun8`.
pub fn score_fun8(
    xa: &[Coord3D],
    ya: &[Coord3D],
    d: f64,
    score_sum_method: u8,
    lnorm: f64,
    score_d8: f64,
    d0: f64,
) -> ScoreResult {
    let n_ali = xa.len();
    let d02 = d0 * d0;
    let score_d8_cut = score_d8 * score_d8;
    let mut inc = 0u32;
    let mut d_tmp = d * d;

    loop {
        let mut i_ali = Vec::new();
        let mut score_sum = 0.0_f64;
        for i in 0..n_ali {
            let di = dist_squared(&xa[i], &ya[i]);
            if di < d_tmp {
                i_ali.push(i);
            }
            if score_sum_method == 8 {
                if di <= score_d8_cut {
                    score_sum += 1.0 / (1.0 + di / d02);
                }
            } else {
                score_sum += 1.0 / (1.0 + di / d02);
            }
        }
        if i_ali.len() < 3 && n_ali > 3 {
            inc += 1;
            let dinc = d + inc as f64 * 0.5;
            d_tmp = dinc * dinc;
        } else {
            return ScoreResult {
                i_ali,
                score: score_sum / lnorm,
            };
        }
    }
}

/// Score aligned pairs, filtering by distance threshold.
/// Normalizes by `n_ali` (for standard TM-score).
///
/// Corresponds to C++ `score_fun8_standard`.
pub fn score_fun8_standard(
    xa: &[Coord3D],
    ya: &[Coord3D],
    d: f64,
    score_sum_method: u8,
    score_d8: f64,
    d0: f64,
) -> ScoreResult {
    let n_ali = xa.len();
    let d02 = d0 * d0;
    let score_d8_cut = score_d8 * score_d8;
    let mut inc = 0u32;
    let mut d_tmp = d * d;

    loop {
        let mut i_ali = Vec::new();
        let mut score_sum = 0.0_f64;
        for i in 0..n_ali {
            let di = dist_squared(&xa[i], &ya[i]);
            if di < d_tmp {
                i_ali.push(i);
            }
            if score_sum_method == 8 {
                if di <= score_d8_cut {
                    score_sum += 1.0 / (1.0 + di / d02);
                }
            } else {
                score_sum += 1.0 / (1.0 + di / d02);
            }
        }
        if i_ali.len() < 3 && n_ali > 3 {
            inc += 1;
            let dinc = d + inc as f64 * 0.5;
            d_tmp = dinc * dinc;
        } else {
            return ScoreResult {
                i_ali,
                score: score_sum / n_ali as f64,
            };
        }
    }
}

/// Iterative fragment-based TM-score maximization.
///
/// Takes aligned coordinate pairs `xtm`/`ytm` of length `lali`,
/// tries multiple fragment sizes and positions to find the rotation
/// that maximizes TM-score.
///
/// Returns (best_score, best_transform).
///
/// Corresponds to C++ `TMscore8_search`.
pub fn tmscore8_search(
    xtm: &[Coord3D],
    ytm: &[Coord3D],
    simplify_step: usize,
    score_sum_method: u8,
    local_d0_search: f64,
    params: &TMParams,
) -> (f64, Transform) {
    let lali = xtm.len();
    if lali == 0 {
        return (0.0, Transform::default());
    }

    let n_it = 20;
    let n_init_max = 6;
    let l_ini_min = 4.min(lali);

    // Build fragment length list: Lali, Lali/2, Lali/4, ... down to l_ini_min
    let mut l_ini = Vec::with_capacity(n_init_max);
    for i in 0..n_init_max - 1 {
        let len = lali / (1 << i); // Lali / 2^i
        if len <= l_ini_min {
            l_ini.push(l_ini_min);
            break;
        }
        l_ini.push(len);
    }
    if l_ini.len() == n_init_max - 1 {
        l_ini.push(l_ini_min);
    }

    let mut score_max = -1.0_f64;
    let mut best_transform = Transform::default();
    let mut xt = vec![[0.0; 3]; lali]; // rotated xtm

    for &l_frag in &l_ini {
        let il_max = lali - l_frag;
        let mut i = 0;
        loop {
            // Extract fragment starting from position i
            let r1: Vec<Coord3D> = (0..l_frag).map(|k| xtm[k + i]).collect();
            let r2: Vec<Coord3D> = (0..l_frag).map(|k| ytm[k + i]).collect();

            // Compute rotation from fragment
            if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                result.transform.apply_batch(xtm, &mut xt);

                // Score
                let d = local_d0_search - 1.0;
                let sr = score_fun8(
                    &xt,
                    ytm,
                    d,
                    score_sum_method,
                    params.lnorm,
                    params.score_d8,
                    params.d0,
                );
                if sr.score > score_max {
                    score_max = sr.score;
                    best_transform = result.transform.clone();
                }

                // Iterative refinement
                let d = local_d0_search + 1.0;
                let mut prev_i_ali = sr.i_ali;
                for _it in 0..n_it {
                    let n_cut = prev_i_ali.len();
                    let r1: Vec<Coord3D> = prev_i_ali.iter().map(|&m| xtm[m]).collect();
                    let r2: Vec<Coord3D> = prev_i_ali.iter().map(|&m| ytm[m]).collect();

                    if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                        result.transform.apply_batch(xtm, &mut xt);
                        let sr = score_fun8(
                            &xt,
                            ytm,
                            d,
                            score_sum_method,
                            params.lnorm,
                            params.score_d8,
                            params.d0,
                        );
                        if sr.score > score_max {
                            score_max = sr.score;
                            best_transform = result.transform.clone();
                        }
                        // Check convergence
                        if sr.i_ali.len() == n_cut && sr.i_ali == prev_i_ali {
                            break;
                        }
                        prev_i_ali = sr.i_ali;
                    } else {
                        break;
                    }
                }
            }

            if i < il_max {
                i += simplify_step;
                if i > il_max {
                    i = il_max;
                }
            } else {
                break;
            }
        }
    }

    (score_max.max(0.0), best_transform)
}

/// Same as `tmscore8_search` but normalizes by n_ali (standard mode).
///
/// Corresponds to C++ `TMscore8_search_standard`.
pub fn tmscore8_search_standard(
    xtm: &[Coord3D],
    ytm: &[Coord3D],
    simplify_step: usize,
    score_sum_method: u8,
    local_d0_search: f64,
    score_d8: f64,
    d0: f64,
) -> (f64, Transform) {
    let lali = xtm.len();
    if lali == 0 {
        return (0.0, Transform::default());
    }

    let n_it = 20;
    let n_init_max = 6;
    let l_ini_min = 4.min(lali);

    let mut l_ini = Vec::with_capacity(n_init_max);
    for i in 0..n_init_max - 1 {
        let len = lali / (1 << i);
        if len <= l_ini_min {
            l_ini.push(l_ini_min);
            break;
        }
        l_ini.push(len);
    }
    if l_ini.len() == n_init_max - 1 {
        l_ini.push(l_ini_min);
    }

    let mut score_max = -1.0_f64;
    let mut best_transform = Transform::default();
    let mut xt = vec![[0.0; 3]; lali];

    for &l_frag in &l_ini {
        let il_max = lali - l_frag;
        let mut i = 0;
        loop {
            let r1: Vec<Coord3D> = (0..l_frag).map(|k| xtm[k + i]).collect();
            let r2: Vec<Coord3D> = (0..l_frag).map(|k| ytm[k + i]).collect();

            if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                result.transform.apply_batch(xtm, &mut xt);

                let d = local_d0_search - 1.0;
                let sr = score_fun8_standard(&xt, ytm, d, score_sum_method, score_d8, d0);
                if sr.score > score_max {
                    score_max = sr.score;
                    best_transform = result.transform.clone();
                }

                let d = local_d0_search + 1.0;
                let mut prev_i_ali = sr.i_ali;
                for _it in 0..n_it {
                    let n_cut = prev_i_ali.len();
                    let r1: Vec<Coord3D> = prev_i_ali.iter().map(|&m| xtm[m]).collect();
                    let r2: Vec<Coord3D> = prev_i_ali.iter().map(|&m| ytm[m]).collect();

                    if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                        result.transform.apply_batch(xtm, &mut xt);
                        let sr = score_fun8_standard(&xt, ytm, d, score_sum_method, score_d8, d0);
                        if sr.score > score_max {
                            score_max = sr.score;
                            best_transform = result.transform.clone();
                        }
                        if sr.i_ali.len() == n_cut && sr.i_ali == prev_i_ali {
                            break;
                        }
                        prev_i_ali = sr.i_ali;
                    } else {
                        break;
                    }
                }
            }

            if i < il_max {
                i += simplify_step;
                if i > il_max {
                    i = il_max;
                }
            } else {
                break;
            }
        }
    }

    (score_max.max(0.0), best_transform)
}

/// Extract aligned pairs from invmap, then run TMscore8_search.
///
/// `invmap[j] = i` means position j in y aligns to position i in x.
/// Corresponds to C++ `detailed_search`.
pub fn detailed_search(
    x: &[Coord3D],
    y: &[Coord3D],
    invmap: &[i32],
    simplify_step: usize,
    score_sum_method: u8,
    local_d0_search: f64,
    params: &TMParams,
) -> (f64, Transform) {
    let (xtm, ytm) = extract_aligned_pairs(x, y, invmap);
    if xtm.is_empty() {
        return (0.0, Transform::default());
    }
    tmscore8_search(
        &xtm,
        &ytm,
        simplify_step,
        score_sum_method,
        local_d0_search,
        params,
    )
}

/// Extract aligned pairs from invmap, then run TMscore8_search_standard.
/// If `normalize` is true, rescales score by n_aligned / lnorm.
///
/// Corresponds to C++ `detailed_search_standard`.
pub fn detailed_search_standard(
    x: &[Coord3D],
    y: &[Coord3D],
    invmap: &[i32],
    simplify_step: usize,
    score_sum_method: u8,
    local_d0_search: f64,
    normalize: bool,
    lnorm: f64,
    score_d8: f64,
    d0: f64,
) -> (f64, Transform) {
    let (xtm, ytm) = extract_aligned_pairs(x, y, invmap);
    let k = xtm.len();
    if k == 0 {
        return (0.0, Transform::default());
    }
    let (mut score, transform) = tmscore8_search_standard(
        &xtm,
        &ytm,
        simplify_step,
        score_sum_method,
        local_d0_search,
        score_d8,
        d0,
    );
    if normalize {
        score = score * k as f64 / lnorm;
    }
    (score, transform)
}

/// Fast 3-iteration scoring without full search.
///
/// Corresponds to C++ `get_score_fast`.
pub fn get_score_fast(
    x: &[Coord3D],
    y: &[Coord3D],
    invmap: &[i32],
    d0: f64,
    d0_search: f64,
) -> (f64, Transform) {
    let (xtm, ytm) = extract_aligned_pairs(x, y, invmap);
    let n_ali = xtm.len();
    if n_ali == 0 {
        return (0.0, Transform::default());
    }

    let d02 = d0 * d0;
    let d002 = d0_search * d0_search;

    // First iteration: Kabsch on all aligned pairs
    let Some(result) = kabsch(&xtm, &ytm, KabschMode::RotationOnly) else {
        return (0.0, Transform::default());
    };
    let mut t = result.transform;

    let mut dis = vec![0.0_f64; n_ali];
    let mut tmscore = 0.0_f64;
    for k in 0..n_ali {
        let xrot = t.apply(&xtm[k]);
        dis[k] = dist_squared(&xrot, &ytm[k]);
        tmscore += 1.0 / (1.0 + dis[k] / d02);
    }

    // Second iteration: filter by d0_search
    let mut d002t = d002;
    let j = loop {
        let j: usize = (0..n_ali).filter(|&k| dis[k] <= d002t).count();
        if j < 3 && n_ali > 3 {
            d002t += 0.5;
        } else {
            break j;
        }
    };

    let mut tmscore1 = tmscore;
    let mut tmscore2 = tmscore;

    if n_ali != j {
        let r1: Vec<Coord3D> = (0..n_ali)
            .filter(|&k| dis[k] <= d002t)
            .map(|k| xtm[k])
            .collect();
        let r2: Vec<Coord3D> = (0..n_ali)
            .filter(|&k| dis[k] <= d002t)
            .map(|k| ytm[k])
            .collect();

        if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
            t = result.transform;
            tmscore1 = 0.0;
            for k in 0..n_ali {
                let xrot = t.apply(&xtm[k]);
                dis[k] = dist_squared(&xrot, &ytm[k]);
                tmscore1 += 1.0 / (1.0 + dis[k] / d02);
            }

            // Third iteration: d0_search^2 + 1
            d002t = d002 + 1.0;
            let _j = loop {
                let j: usize = (0..n_ali).filter(|&k| dis[k] <= d002t).count();
                if j < 3 && n_ali > 3 {
                    d002t += 0.5;
                } else {
                    break j;
                }
            };

            let r1: Vec<Coord3D> = (0..n_ali)
                .filter(|&k| dis[k] <= d002t)
                .map(|k| xtm[k])
                .collect();
            let r2: Vec<Coord3D> = (0..n_ali)
                .filter(|&k| dis[k] <= d002t)
                .map(|k| ytm[k])
                .collect();

            if let Some(result) = kabsch(&r1, &r2, KabschMode::RotationOnly) {
                t = result.transform;
                tmscore2 = 0.0;
                for k in 0..n_ali {
                    let xrot = t.apply(&xtm[k]);
                    let di = dist_squared(&xrot, &ytm[k]);
                    tmscore2 += 1.0 / (1.0 + di / d02);
                }
            }
        }
    }

    let best = tmscore.max(tmscore1).max(tmscore2);
    (best, t)
}

/// Standard TM-score computation for final reporting.
///
/// Returns (tmscore, l_ali, rmsd).
/// Corresponds to C++ `standard_TMscore`.
pub fn standard_tmscore(
    x: &[Coord3D],
    y: &[Coord3D],
    invmap: &[i32],
    score_d8: f64,
    mol_type: MolType,
) -> (f64, usize, f64, Transform) {
    let ylen = y.len();
    let lnorm = ylen as f64;

    // Compute d0 based on mol_type and lnorm
    let d0 = if mol_type == MolType::RNA {
        if lnorm <= 11.0 {
            0.3
        } else if lnorm <= 15.0 {
            0.4
        } else if lnorm <= 19.0 {
            0.5
        } else if lnorm <= 23.0 {
            0.6
        } else if lnorm < 30.0 {
            0.7
        } else {
            0.6 * (lnorm - 0.5).powf(1.0 / 2.0) - 2.5
        }
    } else {
        let d0_min = 0.5;
        if lnorm > 21.0 {
            (1.24 * (lnorm - 15.0).powf(1.0 / 3.0) - 1.8).max(d0_min)
        } else {
            d0_min
        }
    };

    let (xtm, ytm) = extract_aligned_pairs(x, y, invmap);
    let n_al = xtm.len();
    if n_al == 0 {
        return (0.0, 0, 0.0, Transform::default());
    }

    // Compute RMSD
    let rmsd = if let Some(result) = kabsch(&xtm, &ytm, KabschMode::RmsOnly) {
        (result.rms / n_al as f64).sqrt()
    } else {
        0.0
    };

    // Standard search
    let (tmscore, transform) = tmscore8_search_standard(&xtm, &ytm, 1, 0, d0, score_d8, d0);
    let tmscore = tmscore * n_al as f64 / lnorm;

    (tmscore, n_al, rmsd, transform)
}

/// Extract aligned coordinate pairs from an alignment map.
///
/// `invmap[j] = i` means y[j] aligns to x[i]; -1 means gap.
pub fn extract_aligned_pairs(
    x: &[Coord3D],
    y: &[Coord3D],
    invmap: &[i32],
) -> (Vec<Coord3D>, Vec<Coord3D>) {
    let mut xtm = Vec::new();
    let mut ytm = Vec::new();
    for (j, &i) in invmap.iter().enumerate() {
        if i >= 0 {
            xtm.push(x[i as usize]);
            ytm.push(y[j]);
        }
    }
    (xtm, ytm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_fun8_identical_points() {
        // Identical points should give perfect score
        let pts: Vec<Coord3D> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let sr = score_fun8(&pts, &pts, 5.0, 0, 5.0, 100.0, 1.0);
        // Each pair has distance 0, so score contribution = 1/(1+0) = 1
        // Total = 5, normalized by lnorm=5 → score = 1.0
        assert!(
            (sr.score - 1.0).abs() < 1e-10,
            "score = {}, expected 1.0",
            sr.score
        );
        assert_eq!(sr.i_ali.len(), 5);
    }

    #[test]
    fn extract_aligned_pairs_basic() {
        let x: Vec<Coord3D> = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let y: Vec<Coord3D> = vec![[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]];
        let invmap = vec![0_i32, -1]; // y[0]→x[0], y[1]→gap
        let (xtm, ytm) = extract_aligned_pairs(&x, &y, &invmap);
        assert_eq!(xtm.len(), 1);
        assert_eq!(xtm[0], [1.0, 0.0, 0.0]);
        assert_eq!(ytm[0], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn get_score_fast_identical() {
        let pts: Vec<Coord3D> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let invmap: Vec<i32> = vec![0, 1, 2, 3];
        let (score, _) = get_score_fast(&pts, &pts, &invmap, 1.0, 4.5);
        // Score should be 4.0 (4 pairs, each contributing 1.0), unnormalized
        assert!(score > 3.9, "score = {score}, expected ~4.0");
    }

    #[test]
    fn tmscore8_search_identical() {
        let pts: Vec<Coord3D> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let params = TMParams::for_search(4, 4);
        let (score, _) = tmscore8_search(&pts, &pts, 1, 8, params.d0_search, &params);
        assert!(score > 0.0, "score should be positive");
    }
}
