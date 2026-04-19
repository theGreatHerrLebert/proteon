//! Hybrid weighted RMSD iterative refinement.
//!
//! Ported from C++ USAlign `HwRMSD.h`.
//! Iteratively: sequence/SS alignment -> Kabsch superposition -> SE refinement.
//! Accepts new alignment only if both TM1 and TM2 improve.

use crate::core::kabsch::{kabsch, KabschMode};
use crate::core::types::{Coord3D, MolType, Transform};

use crate::ext::nwalign::{self, AlignMode, ReturnMode};
use crate::ext::se::{self, SeOptions, SeResult};

/// Options for HwRMSD iterative refinement.
#[derive(Debug, Clone)]
pub struct HwRMSDOptions {
    /// Molecule type.
    pub mol_type: MolType,
    /// User-provided initial alignment (i_opt: 0=none, 1=soft, 3=strict).
    pub i_opt: u8,
    /// Average length normalization.
    pub a_opt: bool,
    /// User-specified normalization length.
    pub u_opt: u8,
    /// User-specified normalization length value.
    pub lnorm_ass: f64,
    /// d0 scaling.
    pub d_opt: bool,
    /// d0 scale value.
    pub d0_scale: f64,
    /// User-provided alignment sequences.
    pub sequence: Vec<String>,
    /// Glocal mode for NWalign (0=global, 1=query, 2=both).
    pub glocal: u8,
    /// Maximum iterations (default 10).
    pub iter_opt: usize,
    /// Sequence mode: 2=secondary structure NWalign, 3=seq then SS (default 3).
    pub seq_opt: u8,
    /// Early termination threshold (default 0.01).
    pub early_opt: f64,
}

impl Default for HwRMSDOptions {
    fn default() -> Self {
        HwRMSDOptions {
            mol_type: MolType::Protein,
            i_opt: 0,
            a_opt: false,
            u_opt: 0,
            lnorm_ass: 0.0,
            d_opt: false,
            d0_scale: 0.0,
            sequence: Vec::new(),
            glocal: 0,
            iter_opt: 10,
            seq_opt: 3,
            early_opt: 0.01,
        }
    }
}

/// Result of HwRMSD refinement.
#[derive(Debug, Clone)]
pub struct HwRMSDResult {
    /// SE result from the best iteration.
    pub se: SeResult,
    /// Kabsch rotation/translation from best iteration.
    pub transform: Transform,
}

/// Parse alignment strings into an invmap.
///
/// Corresponds to C++ `parse_alignment_into_invmap`.
pub fn parse_alignment_into_invmap(
    seq_x_aligned: &str,
    seq_y_aligned: &str,
    xlen: usize,
    ylen: usize,
) -> Vec<i32> {
    let mut invmap = vec![-1i32; ylen];
    if seq_x_aligned.is_empty() {
        return invmap;
    }
    let xb = seq_x_aligned.as_bytes();
    let yb = seq_y_aligned.as_bytes();
    let l = xb.len().min(yb.len());
    let mut i1: i32 = -1;
    let mut i2: i32 = -1;
    for j in 0..l {
        if xb[j] != b'-' {
            i1 += 1;
        }
        if yb[j] != b'-' {
            i2 += 1;
            if i2 >= ylen as i32 || i1 >= xlen as i32 {
                break;
            }
            if xb[j] != b'-' {
                invmap[i2 as usize] = i1;
            }
        }
    }
    invmap
}

/// Kabsch superposition from an invmap, returning the transform and rotated coordinates.
///
/// Corresponds to C++ `Kabsch_Superpose`.
fn kabsch_superpose(
    xa: &[Coord3D],
    ya: &[Coord3D],
    invmap: &[i32],
) -> Option<(Transform, Vec<Coord3D>, f64)> {
    // Extract aligned pairs
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    for (j, &i) in invmap.iter().enumerate() {
        if i >= 0 {
            r1.push(xa[i as usize]);
            r2.push(ya[j]);
        }
    }
    if r1.is_empty() {
        return None;
    }

    let result = kabsch(&r1, &r2, KabschMode::Both)?;
    let rmsd = (result.rms / r1.len() as f64).sqrt();

    // Apply transform to all of xa
    let mut xt = vec![[0.0; 3]; xa.len()];
    result.transform.apply_batch(xa, &mut xt);

    Some((result.transform, xt, rmsd))
}

/// Main HwRMSD iterative refinement.
///
/// Corresponds to C++ `HwRMSD_main`.
pub fn hwrmsd_main(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[u8],
    seqy: &[u8],
    secx: &[u8],
    secy: &[u8],
    opts: &HwRMSDOptions,
) -> HwRMSDResult {
    let xlen = xa.len();
    let ylen = ya.len();
    let mol_type_int = match opts.mol_type {
        MolType::Protein => 0,
        MolType::RNA => 1,
    };

    let glocal_mode = match opts.glocal {
        0 => AlignMode::Global,
        1 => AlignMode::GlocalQuery,
        2 => AlignMode::GlocalBoth,
        _ => AlignMode::Global,
    };

    // Initial alignment
    let mut seq_xa_tmp;
    let mut seq_ya_tmp;

    if opts.i_opt > 0 && opts.sequence.len() >= 2 {
        seq_xa_tmp = opts.sequence[0].clone();
        seq_ya_tmp = opts.sequence[1].clone();
    } else if opts.seq_opt == 2 {
        let result = nwalign::nwalign(
            secx,
            secy,
            mol_type_int,
            glocal_mode,
            ReturnMode::AlignmentOnly,
        );
        seq_xa_tmp = result.seq_x_aligned;
        seq_ya_tmp = result.seq_y_aligned;
    } else {
        let result = nwalign::nwalign(
            seqx,
            seqy,
            mol_type_int,
            glocal_mode,
            ReturnMode::AlignmentOnly,
        );
        seq_xa_tmp = result.seq_x_aligned;
        seq_ya_tmp = result.seq_y_aligned;
    }

    let total_iter = if opts.i_opt == 3 || opts.iter_opt < 1 {
        1
    } else {
        opts.iter_opt
    };

    let se_opts = SeOptions {
        mol_type: opts.mol_type,
        use_user_alignment: opts.i_opt == 3,
        compute_avg: opts.a_opt,
        u_opt: opts.u_opt,
        lnorm_ass: opts.lnorm_ass,
        compute_d_scaled: opts.d_opt,
        d0_scale: opts.d0_scale,
        outfmt_opt: 1, // SE internally uses outfmt=1 (no alignment strings needed during iteration)
        sequence: opts.sequence.clone(),
    };

    let mut best_tm1 = -1.0_f64;
    let mut best_tm2 = -1.0_f64;
    let mut best_se: Option<SeResult> = None;
    let mut best_transform = Transform {
        t: [0.0; 3],
        u: [[0.0; 3]; 3],
    };
    let mut best_invmap = vec![-1i32; ylen];
    let mut max_tm = 0.0_f64;

    let mut invmap_tmp;

    for iter in 0..total_iter {
        // Switch to SS alignment on second iteration (if seq_opt == 3)
        if iter == 1 && opts.i_opt == 0 && opts.seq_opt == 3 {
            let result = nwalign::nwalign(
                secx,
                secy,
                mol_type_int,
                glocal_mode,
                ReturnMode::AlignmentOnly,
            );
            seq_xa_tmp = result.seq_x_aligned;
            seq_ya_tmp = result.seq_y_aligned;
        }

        // Parse alignment into invmap
        invmap_tmp = parse_alignment_into_invmap(&seq_xa_tmp, &seq_ya_tmp, xlen, ylen);

        // Kabsch superposition
        let (transform, xt, _rmsd) = match kabsch_superpose(xa, ya, &invmap_tmp) {
            Some(r) => r,
            None => break,
        };

        // SE refinement on superposed coordinates
        let se_result = se::se_main(
            &xt,
            ya,
            seqx,
            seqy,
            xlen,
            ylen,
            &se_opts,
            Some(&invmap_tmp),
            0,
        );

        if se_result.n_ali8 == 0 {
            // Zero aligned residues: try centered alignment
            let (centered_xa, centered_ya) = match xlen.cmp(&ylen) {
                std::cmp::Ordering::Less => {
                    let pad = (ylen - xlen) / 2;
                    let mut sxa = String::new();
                    for _ in 0..pad {
                        sxa.push('-');
                    }
                    sxa.push_str(&String::from_utf8_lossy(seqx));
                    while sxa.len() < ylen {
                        sxa.push('-');
                    }
                    (sxa, String::from_utf8_lossy(seqy).to_string())
                }
                std::cmp::Ordering::Greater => {
                    let pad = (xlen - ylen) / 2;
                    let mut sya = String::new();
                    for _ in 0..pad {
                        sya.push('-');
                    }
                    sya.push_str(&String::from_utf8_lossy(seqy));
                    while sya.len() < xlen {
                        sya.push('-');
                    }
                    (String::from_utf8_lossy(seqx).to_string(), sya)
                }
                std::cmp::Ordering::Equal => (
                    String::from_utf8_lossy(seqx).to_string(),
                    String::from_utf8_lossy(seqy).to_string(),
                ),
            };

            invmap_tmp = parse_alignment_into_invmap(&centered_xa, &centered_ya, xlen, ylen);

            let (transform2, xt2, _) = match kabsch_superpose(xa, ya, &invmap_tmp) {
                Some(r) => r,
                None => break,
            };

            let se_result2 = se::se_main(
                &xt2,
                ya,
                seqx,
                seqy,
                xlen,
                ylen,
                &se_opts,
                Some(&invmap_tmp),
                0,
            );

            if se_result2.tm1 > best_tm1 && se_result2.tm2 > best_tm2 {
                best_tm1 = se_result2.tm1;
                best_tm2 = se_result2.tm2;
                best_transform = transform2;
                best_invmap = se_result2.invmap.clone();
                seq_xa_tmp = se_result2.seq_x_aligned.clone();
                seq_ya_tmp = se_result2.seq_y_aligned.clone();
                best_se = Some(se_result2);
            }
            continue;
        }

        // Accept if both TM1 and TM2 improve
        if se_result.tm1 > best_tm1 && se_result.tm2 > best_tm2 {
            best_tm1 = se_result.tm1;
            best_tm2 = se_result.tm2;
            best_transform = transform;
            best_invmap = se_result.invmap.clone();
            seq_xa_tmp = se_result.seq_x_aligned.clone();
            seq_ya_tmp = se_result.seq_y_aligned.clone();
            best_se = Some(se_result);
        } else {
            if iter >= 2 {
                break;
            }
            // Revert to best alignment
            if let Some(ref best) = best_se {
                seq_xa_tmp = best.seq_x_aligned.clone();
                seq_ya_tmp = best.seq_y_aligned.clone();
            }
        }

        // Early termination
        if iter >= 2 && opts.early_opt > 0.0 {
            let cur_tm = (best_tm1 + best_tm2) / 2.0;
            if cur_tm - max_tm < opts.early_opt {
                break;
            }
            max_tm = cur_tm;
        }
    }

    // Final SE with full output (outfmt_opt=0 for strings)
    let final_se = if let Some(_se) = best_se {
        // Re-run SE with full output if we had one with outfmt=1
        let final_opts = SeOptions {
            outfmt_opt: 0,
            ..se_opts
        };
        let (_, xt, _) = kabsch_superpose(xa, ya, &best_invmap).unwrap_or_else(|| {
            (
                Transform {
                    t: [0.0; 3],
                    u: [[0.0; 3]; 3],
                },
                xa.to_vec(),
                0.0,
            )
        });
        se::se_main(
            &xt,
            ya,
            seqx,
            seqy,
            xlen,
            ylen,
            &final_opts,
            Some(&best_invmap),
            0,
        )
    } else {
        // No alignment found
        SeResult {
            tm1: 0.0,
            tm2: 0.0,
            tm3: 0.0,
            tm4: 0.0,
            tm5: 0.0,
            d0a: 0.0,
            d0b: 0.0,
            d0u: 0.0,
            d0a_avg: 0.0,
            d0_out: 5.0,
            rmsd: 0.0,
            n_ali: 0,
            n_ali8: 0,
            liden: 0.0,
            tm_ali: 0.0,
            rmsd_ali: 0.0,
            invmap: vec![-1; ylen],
            seq_x_aligned: String::new(),
            seq_y_aligned: String::new(),
            seq_m: String::new(),
            do_vec: Vec::new(),
        }
    };

    HwRMSDResult {
        se: final_se,
        transform: best_transform,
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
    fn test_hwrmsd_identical() {
        let n = 20;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = HwRMSDOptions::default();
        let result = hwrmsd_main(&coords, &coords, &seq, &seq, &sec, &sec, &opts);

        assert!(result.se.tm1 > 0.9, "tm1={}", result.se.tm1);
        assert!(result.se.tm2 > 0.9, "tm2={}", result.se.tm2);
    }

    #[test]
    fn test_hwrmsd_translated() {
        let n = 20;
        let coords = make_helix_coords(n);
        let shifted: Vec<Coord3D> = coords.iter().map(|c| [c[0] + 5.0, c[1], c[2]]).collect();
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = HwRMSDOptions::default();
        let result = hwrmsd_main(&coords, &shifted, &seq, &seq, &sec, &sec, &opts);

        // After Kabsch, should find good alignment despite translation
        assert!(result.se.tm1 > 0.8, "tm1={}", result.se.tm1);
    }

    #[test]
    fn test_parse_alignment_into_invmap() {
        let invmap = parse_alignment_into_invmap("AC-DEF", "A-GDEF", 5, 4);
        assert_eq!(invmap.len(), 4);
        assert_eq!(invmap[0], 0); // A-A
                                  // G is gapped in x, so invmap[1] = -1
        assert_eq!(invmap[1], -1);
        assert_eq!(invmap[2], 2); // D-D
        assert_eq!(invmap[3], 3); // E-E
    }
}
