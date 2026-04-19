//! Flexible hinge-based structural alignment (FlexAlign).
//!
//! Ported from C++ USAlign `flexalign.h`.
//! Iteratively detects hinge points between rigid-body domains, aligning
//! each domain with its own rotation/translation (Transform). The algorithm:
//!
//! 1. Run standard TM-align for an initial rigid-body alignment.
//! 2. Run SE refinement to identify aligned vs unaligned regions.
//! 3. For each hinge iteration (up to `hinge_opt`):
//!    a. Extract unaligned residues from both structures.
//!    b. Run TM-align on the unaligned fragment.
//!    c. Apply the new transform and run SE in hinge mode.
//!    d. Accept if >= 5 new aligned residues are found.
//! 4. Re-derive per-residue hinge assignments, smooth them, and recompute scores.

use anyhow::Result;

use crate::core::align::tmalign::tmalign;
use crate::core::types::{
    dist_squared, AlignOptions, AlignResult, Coord3D, MolType, TMParams, Transform,
};

use crate::ext::se::{se_main, SeOptions, SeResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Options controlling flexible alignment behavior.
#[derive(Debug, Clone)]
pub struct FlexOptions {
    /// Maximum number of hinge points to detect (default 9).
    pub hinge_opt: usize,
    /// Molecule type.
    pub mol_type: MolType,
    /// User-specified alignment mode (i_opt in C++).
    pub i_opt: i32,
    /// Average-length normalization (a_opt in C++).
    pub a_opt: bool,
    /// Whether to normalize by user-specified length.
    pub u_opt: bool,
    /// User-specified normalization length.
    pub lnorm_ass: f64,
    /// Whether to use d0 scaling.
    pub d_opt: bool,
    /// d0 scale value.
    pub d0_scale: f64,
    /// Use fast TM-align.
    pub fast_opt: bool,
    /// User-provided alignment sequences (for i_opt mode).
    pub sequence: Vec<String>,
}

impl Default for FlexOptions {
    fn default() -> Self {
        FlexOptions {
            hinge_opt: 9,
            mol_type: MolType::Protein,
            i_opt: 0,
            a_opt: false,
            u_opt: false,
            lnorm_ass: 0.0,
            d_opt: false,
            d0_scale: 0.0,
            fast_opt: false,
            sequence: Vec::new(),
        }
    }
}

/// Result of flexible hinge-based alignment.
#[derive(Debug, Clone)]
pub struct FlexResult {
    /// One transform per hinge segment (at least 1).
    pub transforms: Vec<Transform>,
    /// Final SE refinement result with alignment strings and scores.
    pub se_result: SeResult,
    /// Number of hinge points detected (transforms.len() - 1).
    pub hinge_count: usize,
}

// ---------------------------------------------------------------------------
// Helper: convert alignment strings into an invmap
// ---------------------------------------------------------------------------

/// Parse alignment strings (seqxA, seqyA) into an invmap[j] = i.
///
/// Corresponds to C++ `aln2invmap`.
pub fn aln2invmap(seq_x_aligned: &str, seq_y_aligned: &str, ylen: usize) -> Vec<i32> {
    let mut invmap = vec![-1i32; ylen];
    let xb = seq_x_aligned.as_bytes();
    let yb = seq_y_aligned.as_bytes();
    let len = xb.len().min(yb.len());

    let mut i: i32 = -1;
    let mut j: i32 = -1;
    for r in 0..len {
        if xb[r] != b'-' {
            i += 1;
        }
        if yb[r] != b'-' {
            j += 1;
        }
        if xb[r] != b'-' && yb[r] != b'-' && (j as usize) < ylen {
            invmap[j as usize] = i;
        }
    }
    invmap
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build `AlignOptions` for the internal TM-align calls from flex options.
fn make_align_opts(opts: &FlexOptions) -> AlignOptions {
    AlignOptions {
        i_opt: opts.i_opt,
        a_opt: i32::from(opts.a_opt),
        u_opt: opts.u_opt,
        lnorm_ass: opts.lnorm_ass,
        d_opt: opts.d_opt,
        d0_scale: opts.d0_scale,
        fast_opt: opts.fast_opt,
        mol_type: opts.mol_type,
        tm_cut: -1.0,
        user_alignment: if opts.sequence.is_empty() {
            None
        } else {
            Some(opts.sequence.clone())
        },
    }
}

/// Build `SeOptions` from flex options.
fn make_se_opts(opts: &FlexOptions) -> SeOptions {
    SeOptions {
        mol_type: opts.mol_type,
        use_user_alignment: opts.i_opt == 3,
        compute_avg: opts.a_opt,
        u_opt: u8::from(opts.u_opt),
        lnorm_ass: opts.lnorm_ass,
        compute_d_scaled: opts.d_opt,
        d0_scale: opts.d0_scale,
        outfmt_opt: 0,
        sequence: opts.sequence.clone(),
    }
}

/// Convert `&[u8]` to `&[char]` (needed because tm-align's tmalign uses `&[char]`).
fn u8_to_char(s: &[u8]) -> Vec<char> {
    s.iter().map(|&b| b as char).collect()
}

/// Extract fragment data from aligned/unaligned residues.
///
/// Walks the alignment strings and for each position where `x_pred(xb, yb)` is
/// true, copies that residue from `(coords_src, seq_src, sec_src)` using index
/// tracking from `idx_src` ('x' or 'y').
struct FragmentExtractor {
    coords: Vec<Coord3D>,
    seq: Vec<u8>,
    sec: Vec<u8>,
    /// Original index in full-length structure for each extracted position.
    orig_indices: Vec<usize>,
}

impl FragmentExtractor {
    fn new() -> Self {
        FragmentExtractor {
            coords: Vec::new(),
            seq: Vec::new(),
            sec: Vec::new(),
            orig_indices: Vec::new(),
        }
    }
}

/// Extract "x-side" fragment: residues from structure X where a predicate on the
/// alignment columns is true, using x-index tracking.
fn extract_x_fragment<F>(
    seq_x_aligned: &str,
    seq_y_aligned: &str,
    xa: &[Coord3D],
    seqx: &[u8],
    secx: &[u8],
    pred: F,
) -> FragmentExtractor
where
    F: Fn(u8, u8) -> bool,
{
    let xb = seq_x_aligned.as_bytes();
    let yb = seq_y_aligned.as_bytes();
    let len = xb.len().min(yb.len());
    let mut frag = FragmentExtractor::new();
    let mut i: i32 = -1;

    for r in 0..len {
        if xb[r] != b'-' {
            i += 1;
        }
        if pred(xb[r], yb[r]) && i >= 0 {
            let iu = i as usize;
            frag.coords.push(xa[iu]);
            frag.seq.push(seqx[iu]);
            frag.sec.push(secx[iu]);
            frag.orig_indices.push(iu);
        }
    }
    frag
}

/// Extract "y-side" fragment: residues from structure Y where a predicate on the
/// alignment columns is true, using y-index tracking.
fn extract_y_fragment<F>(
    seq_x_aligned: &str,
    seq_y_aligned: &str,
    ya: &[Coord3D],
    seqy: &[u8],
    secy: &[u8],
    pred: F,
) -> FragmentExtractor
where
    F: Fn(u8, u8) -> bool,
{
    let xb = seq_x_aligned.as_bytes();
    let yb = seq_y_aligned.as_bytes();
    let len = xb.len().min(yb.len());
    let mut frag = FragmentExtractor::new();
    let mut j: i32 = -1;

    for r in 0..len {
        if yb[r] != b'-' {
            j += 1;
        }
        if pred(xb[r], yb[r]) && j >= 0 {
            let ju = j as usize;
            frag.coords.push(ya[ju]);
            frag.seq.push(seqy[ju]);
            frag.sec.push(secy[ju]);
            frag.orig_indices.push(ju);
        }
    }
    frag
}

/// Run TM-align on two fragment sets, returning the AlignResult and transform.
/// Returns None if fragments are too short (< 3 residues).
fn tmalign_fragments(
    x_frag: &FragmentExtractor,
    y_frag: &FragmentExtractor,
    opts: &FlexOptions,
) -> Option<AlignResult> {
    if x_frag.coords.len() < 3 || y_frag.coords.len() < 3 {
        return None;
    }
    let seqx_c = u8_to_char(&x_frag.seq);
    let seqy_c = u8_to_char(&y_frag.seq);
    let secx_c = u8_to_char(&x_frag.sec);
    let secy_c = u8_to_char(&y_frag.sec);
    let align_opts = make_align_opts(opts);
    tmalign(
        &x_frag.coords,
        &y_frag.coords,
        &seqx_c,
        &seqy_c,
        &secx_c,
        &secy_c,
        &align_opts,
    )
    .ok()
}

/// Apply a transform to all coordinates of xa, returning rotated coordinates.
fn apply_transform(xa: &[Coord3D], transform: &Transform) -> Vec<Coord3D> {
    let mut xt = vec![[0.0; 3]; xa.len()];
    transform.apply_batch(xa, &mut xt);
    xt
}

// ---------------------------------------------------------------------------
// AFP smoothing (Aligned Fragment Pair assignment smoothing)
// ---------------------------------------------------------------------------

/// Smooth the hinge-to-residue assignment in seqM and seqM_char.
///
/// Four passes, matching C++ `flexalign_main` post-processing:
/// 1. Remove singleton inserts.
/// 2. Remove singletons at fragment ends.
/// 3. Remove dimer inserts.
/// 4. Remove disconnected singletons (assign to nearest neighbor).
fn smooth_afp(seq_m: &mut [u8], seq_m_char: &mut [u8], seq_y_aligned: &str, num_transforms: usize) {
    let yb = seq_y_aligned.as_bytes();
    let mlen = seq_m.len();

    // Pass 1: remove singleton inserts
    for hinge in (0..num_transforms).rev() {
        let hc = b'0' + hinge as u8;
        let mut j_idx: i32 = -1;
        for r in 0..mlen {
            if yb[r] == b'-' {
                continue;
            }
            j_idx += 1;
            if seq_m_char[j_idx as usize] != hc {
                continue;
            }
            // Check right neighbor
            if r < mlen - 1 && (seq_m[r + 1] == hc || seq_m[r + 1] == b' ') {
                continue;
            }
            // Check left neighbor
            if r > 0 && (seq_m[r - 1] == hc || seq_m[r - 1] == b' ') {
                continue;
            }
            // Both neighbors differ from hinge char: check if they agree
            if r < mlen - 1 && r > 0 && seq_m[r - 1] != seq_m[r + 1] {
                continue;
            }
            if r > 0 {
                let new_val = seq_m[r - 1];
                seq_m[r] = new_val;
                seq_m_char[j_idx as usize] = new_val;
            } else if r < mlen - 1 {
                let new_val = seq_m[r + 1];
                seq_m[r] = new_val;
                seq_m_char[j_idx as usize] = new_val;
            }
        }
    }

    // Pass 2: remove singleton at the end of fragment
    for hinge in (0..num_transforms).rev() {
        let hc = b'0' + hinge as u8;
        let mut j_idx: i32 = -1;
        for r in 0..mlen {
            if yb[r] == b'-' {
                continue;
            }
            j_idx += 1;
            if seq_m[r] != hc {
                continue;
            }
            // Skip if flanked by spaces on both sides
            if r > 0 && seq_m[r - 1] == b' ' && r < mlen - 1 && seq_m[r + 1] == b' ' {
                continue;
            }

            // Find left non-space neighbor
            let mut left_hinge = b' ';
            for ii in (0..r).rev() {
                if seq_m[ii] == b' ' {
                    continue;
                }
                left_hinge = seq_m[ii];
                break;
            }
            if left_hinge == hc {
                continue;
            }

            // Find right non-space neighbor
            let mut right_hinge = b' ';
            for ii in (r + 1)..mlen {
                if seq_m[ii] == b' ' {
                    continue;
                }
                right_hinge = seq_m[ii];
                break;
            }
            if right_hinge == hc {
                continue;
            }
            if left_hinge != right_hinge && left_hinge != b' ' && right_hinge != b' ' {
                continue;
            }

            if right_hinge != b' ' {
                seq_m[r] = right_hinge;
                seq_m_char[j_idx as usize] = right_hinge;
            } else if left_hinge != b' ' {
                seq_m[r] = left_hinge;
                seq_m_char[j_idx as usize] = left_hinge;
            }
        }
    }

    // Pass 3: remove dimer inserts
    for hinge in (0..num_transforms).rev() {
        let hc = b'0' + hinge as u8;
        let mut j_idx: i32 = -1;
        for r in 0..mlen.saturating_sub(1) {
            if yb[r] == b'-' {
                continue;
            }
            j_idx += 1;
            if seq_m[r] != hc || seq_m[r + 1] != hc {
                continue;
            }

            if r + 2 < mlen && (seq_m[r + 2] == b' ' || seq_m[r + 2] == hc) {
                continue;
            }
            if r > 0 && (seq_m[r - 1] == b' ' || seq_m[r - 1] == hc) {
                continue;
            }
            if r + 2 < mlen && r > 0 && seq_m[r - 1] != seq_m[r + 2] {
                continue;
            }

            if r > 0 {
                let new_val = seq_m[r - 1];
                seq_m[r] = new_val;
                seq_m_char[j_idx as usize] = new_val;
                seq_m[r + 1] = new_val;
                if j_idx + 1 < seq_m_char.len() as i32 {
                    seq_m_char[(j_idx + 1) as usize] = new_val;
                }
            } else if r + 2 < mlen {
                let new_val = seq_m[r + 2];
                seq_m[r] = new_val;
                seq_m_char[j_idx as usize] = new_val;
                seq_m[r + 1] = new_val;
                if j_idx + 1 < seq_m_char.len() as i32 {
                    seq_m_char[(j_idx + 1) as usize] = new_val;
                }
            }
        }
    }

    // Pass 4: remove disconnected singletons
    for hinge in (0..num_transforms).rev() {
        let hc = b'0' + hinge as u8;
        let mut j_idx: i32 = -1;
        for r in 0..mlen {
            if yb[r] == b'-' {
                continue;
            }
            j_idx += 1;
            if seq_m[r] != hc {
                continue;
            }

            // Find left non-space neighbor and distance
            let mut left_hinge = b' ';
            let mut left_dist = 0usize;
            for ii in (0..r).rev() {
                if seq_m[ii] == b' ' {
                    continue;
                }
                left_hinge = seq_m[ii];
                left_dist = r - ii;
                break;
            }
            if left_hinge == hc {
                continue;
            }

            // Find right non-space neighbor and distance
            let mut right_hinge = b' ';
            let mut right_dist = 0usize;
            for ii in (r + 1)..mlen {
                if seq_m[ii] == b' ' {
                    continue;
                }
                right_hinge = seq_m[ii];
                right_dist = ii - r;
                break;
            }
            if right_hinge == hc {
                continue;
            }

            if right_hinge == b' ' {
                seq_m[r] = left_hinge;
                seq_m_char[j_idx as usize] = left_hinge;
            } else if left_hinge == b' ' {
                seq_m[r] = right_hinge;
                seq_m_char[j_idx as usize] = right_hinge;
            } else if left_dist < right_dist {
                seq_m[r] = left_hinge;
                seq_m_char[j_idx as usize] = left_hinge;
            } else {
                seq_m[r] = right_hinge;
                seq_m_char[j_idx as usize] = right_hinge;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Main entry point for flexible hinge-based alignment.
///
/// Corresponds to C++ `flexalign_main`. The function:
/// 1. Runs rigid TM-align if no prior transforms are provided.
/// 2. Optionally tries two fragment TM-align orientations (round 2).
/// 3. Iteratively detects hinges by aligning unaligned fragments.
/// 4. Post-processes hinge assignments (smoothing) and recomputes scores.
///
/// # Arguments
/// * `xa` - Coordinates of structure 1 (chain X).
/// * `ya` - Coordinates of structure 2 (chain Y).
/// * `seqx`, `seqy` - One-letter sequences.
/// * `secx`, `secy` - Secondary structure assignments.
/// * `opts` - Algorithm options.
/// * `existing_transforms` - Previously computed transforms (empty for first call).
///
/// # Returns
/// `FlexResult` containing per-segment transforms, final SE result, and hinge count.
pub fn flexalign_main(
    xa: &[Coord3D],
    ya: &[Coord3D],
    seqx: &[u8],
    seqy: &[u8],
    secx: &[u8],
    secy: &[u8],
    opts: &FlexOptions,
    existing_transforms: &[Transform],
) -> Result<FlexResult> {
    let xlen = xa.len();
    let ylen = ya.len();
    let se_opts = make_se_opts(opts);

    let mut tu_vec: Vec<Transform> = existing_transforms.to_vec();

    // -- Round 1: initial rigid TM-align (if no prior transforms) -----------

    let round2 = !tu_vec.is_empty();
    let current_transform: Transform;

    if !round2 {
        let align_opts = make_align_opts(opts);
        let seqx_c = u8_to_char(seqx);
        let seqy_c = u8_to_char(seqy);
        let secx_c = u8_to_char(secx);
        let secy_c = u8_to_char(secy);

        let initial_result = tmalign(xa, ya, &seqx_c, &seqy_c, &secx_c, &secy_c, &align_opts)?;
        current_transform = initial_result.transform.clone();
        tu_vec.push(current_transform.clone());
    } else {
        // Use the first existing transform
        current_transform = tu_vec[0].clone();
    }

    // -- SE refinement after initial alignment ------------------------------

    let xt = apply_transform(xa, &current_transform);
    let mut se_result = se_main(&xt, ya, seqx, seqy, xlen, ylen, &se_opts, None, 1);
    let mut invmap = se_result.invmap.clone();
    let mut n_ali8 = se_result.n_ali8;

    // -- Round 2: try two fragment orientations (if had prior transforms) ----

    if round2 {
        // Try aligned-X vs unaligned-Y
        let x_aligned = extract_x_fragment(
            &se_result.seq_x_aligned,
            &se_result.seq_y_aligned,
            xa,
            seqx,
            secx,
            |x, y| x != b'-' && y != b'-', // aligned pairs -> X side
        );
        let y_unaligned = extract_y_fragment(
            &se_result.seq_x_aligned,
            &se_result.seq_y_aligned,
            ya,
            seqy,
            secy,
            |x, _y| x == b'-', // gaps in X -> unaligned Y residues
        );

        if let Some(frag_result) = tmalign_fragments(&x_aligned, &y_unaligned, opts) {
            let frag_transform = frag_result.transform;
            let xt_h = apply_transform(xa, &frag_transform);
            tu_vec[0] = frag_transform;

            let se_h = se_main(&xt_h, ya, seqx, seqy, xlen, ylen, &se_opts, None, 1);

            // Try unaligned-X vs aligned-Y
            let x_unaligned = extract_x_fragment(
                &se_result.seq_x_aligned,
                &se_result.seq_y_aligned,
                xa,
                seqx,
                secx,
                |_x, y| y == b'-', // gaps in Y -> unaligned X residues
            );
            let y_aligned = extract_y_fragment(
                &se_result.seq_x_aligned,
                &se_result.seq_y_aligned,
                ya,
                seqy,
                secy,
                |x, y| x != b'-' && y != b'-', // aligned pairs -> Y side
            );

            let se_result2;
            let transform2;

            if let Some(frag_result2) = tmalign_fragments(&x_unaligned, &y_aligned, opts) {
                transform2 = frag_result2.transform;
                let xt2 = apply_transform(xa, &transform2);
                se_result2 = se_main(&xt2, ya, seqx, seqy, xlen, ylen, &se_opts, None, 1);
            } else {
                se_result2 = se_result.clone();
                transform2 = current_transform.clone();
            }

            // Keep the better of the two
            let tm_h = se_h.tm1.max(se_h.tm2);
            let tm_2 = se_result2.tm1.max(se_result2.tm2);

            if tm_h > tm_2 {
                se_result = se_h;
                invmap = se_result.invmap.clone();
                n_ali8 = se_result.n_ali8;
                // tu_vec[0] already set
            } else {
                se_result = se_result2;
                invmap = se_result.invmap.clone();
                n_ali8 = se_result.n_ali8;
                tu_vec[0] = transform2;
            }
        }
    }

    // Replace '1' with '0' in seqM (C++ line 263)
    let mut seq_m_bytes: Vec<u8> = se_result.seq_m.bytes().collect();
    for b in &mut seq_m_bytes {
        if *b == b'1' {
            *b = b'0';
        }
    }
    se_result.seq_m = String::from_utf8(seq_m_bytes).unwrap_or_default();

    // -- Hinge detection loop -----------------------------------------------

    let minlen = xlen.min(ylen);

    for hinge in 0..opts.hinge_opt {
        if n_ali8 + 5 > minlen {
            break;
        }

        // Extract unaligned residues from X (gaps in Y alignment)
        let x_unaligned = extract_x_fragment(
            &se_result.seq_x_aligned,
            &se_result.seq_y_aligned,
            xa,
            seqx,
            secx,
            |_x, y| y == b'-',
        );
        // Extract unaligned residues from Y (gaps in X alignment)
        let y_unaligned = extract_y_fragment(
            &se_result.seq_x_aligned,
            &se_result.seq_y_aligned,
            ya,
            seqy,
            secy,
            |x, _y| x == b'-',
        );

        // TM-align the unaligned fragments
        let frag_result = match tmalign_fragments(&x_unaligned, &y_unaligned, opts) {
            Some(r) => r,
            None => break,
        };

        let frag_transform = frag_result.transform;
        let xt_h = apply_transform(xa, &frag_transform);

        // Save current state
        let prev_se = se_result.clone();
        let prev_invmap = invmap.clone();

        // Run SE in hinge mode
        let se_h = se_main(
            &xt_h,
            ya,
            seqx,
            seqy,
            xlen,
            ylen,
            &se_opts,
            Some(&invmap),
            (hinge + 1) as u8,
        );

        // Count newly aligned residues for this hinge
        let hinge_char = (hinge as u8) + b'1';
        let new_ali: usize = se_h.seq_m.bytes().filter(|&b| b == hinge_char).count();

        // Accept if enough new aligned residues and improvement in n_ali8
        if se_h.n_ali8 >= n_ali8 + 5 && new_ali >= 5 {
            se_result = se_h;
            invmap = se_result.invmap.clone();
            n_ali8 = se_result.n_ali8;
            tu_vec.push(frag_transform);
        } else {
            // Restore previous state and stop
            se_result = prev_se;
            invmap = prev_invmap;
            break;
        }
    }

    // -- If only 1 transform, return early ----------------------------------

    if tu_vec.len() <= 1 {
        return Ok(FlexResult {
            hinge_count: 0,
            transforms: tu_vec,
            se_result,
        });
    }

    // -- Re-derive per-residue hinge assignment based on best distance ------

    let mut seq_m_char: Vec<u8> = vec![b' '; ylen];
    let mut di_vec: Vec<f64> = vec![-1.0; ylen];

    for hinge_idx in (0..tu_vec.len()).rev() {
        let xt_h = apply_transform(xa, &tu_vec[hinge_idx]);
        let hc = b'0' + hinge_idx as u8;
        for j in 0..ylen {
            let i = invmap[j];
            if i < 0 {
                continue;
            }
            let d = dist_squared(&xt_h[i as usize], &ya[j]).sqrt();
            if di_vec[j] < 0.0 || d <= di_vec[j] {
                di_vec[j] = d;
                seq_m_char[j] = hc;
            }
        }
    }

    // Map seq_m_char back into the alignment string (seqM)
    let mut seq_m_bytes: Vec<u8> = se_result.seq_m.bytes().collect();
    let yb = se_result.seq_y_aligned.as_bytes();
    {
        let mut j_idx: i32 = -1;
        for r in 0..seq_m_bytes.len() {
            if r < yb.len() && yb[r] == b'-' {
                continue;
            }
            j_idx += 1;
            if (j_idx as usize) < seq_m_char.len() {
                seq_m_bytes[r] = seq_m_char[j_idx as usize];
            }
        }
    }

    // -- Smooth AFP assignment ----------------------------------------------

    smooth_afp(
        &mut seq_m_bytes,
        &mut seq_m_char,
        &se_result.seq_y_aligned,
        tu_vec.len(),
    );

    // -- Recalculate scores using best per-residue distances ----------------

    // Recompute di_vec after smoothing
    for hinge_idx in (0..tu_vec.len()).rev() {
        let xt_h = apply_transform(xa, &tu_vec[hinge_idx]);
        let hc = b'0' + hinge_idx as u8;
        for j in 0..ylen {
            let i = invmap[j];
            if i < 0 {
                continue;
            }
            if seq_m_char[j] != hc {
                continue;
            }
            let d = dist_squared(&xt_h[i as usize], &ya[j]).sqrt();
            if di_vec[j] < 0.0 || d <= di_vec[j] {
                di_vec[j] = d;
            }
        }
    }

    // Compute TM-scores
    let params_b = TMParams::for_final(xlen as f64, opts.mol_type);
    let d0b = params_b.d0;
    let params_a = TMParams::for_final(ylen as f64, opts.mol_type);
    let d0a = params_a.d0;
    let d0a_avg = if opts.a_opt {
        TMParams::for_final((xlen + ylen) as f64 * 0.5, opts.mol_type).d0
    } else {
        0.0
    };
    let d0u = if opts.u_opt {
        TMParams::for_final(opts.lnorm_ass, opts.mol_type).d0
    } else {
        0.0
    };

    let mut tm1 = 0.0_f64;
    let mut tm2 = 0.0_f64;
    let mut tm3 = 0.0_f64;
    let mut tm4 = 0.0_f64;
    let mut tm5 = 0.0_f64;
    let mut rmsd0 = 0.0_f64;

    // Recompute Liden from alignment strings
    let xab = se_result.seq_x_aligned.as_bytes();
    let yab = se_result.seq_y_aligned.as_bytes();
    let mut liden = 0.0_f64;
    for r in 0..seq_m_bytes.len().min(xab.len()).min(yab.len()) {
        if seq_m_bytes[r] != b' ' && xab[r] == yab[r] {
            liden += 1.0;
        }
    }

    for j in 0..ylen {
        let i = invmap[j];
        if i < 0 {
            continue;
        }
        let d = di_vec[j];
        if d < 0.0 {
            continue;
        }

        tm2 += 1.0 / (1.0 + (d / d0b) * (d / d0b));
        tm1 += 1.0 / (1.0 + (d / d0a) * (d / d0a));

        if opts.a_opt {
            tm3 += 1.0 / (1.0 + (d / d0a_avg) * (d / d0a_avg));
        }
        if opts.u_opt {
            tm4 += 1.0 / (1.0 + (d / d0u) * (d / d0u));
        }
        if opts.d_opt {
            tm5 += 1.0 / (1.0 + (d / opts.d0_scale) * (d / opts.d0_scale));
        }
        rmsd0 += d * d;
    }

    tm2 /= xlen as f64;
    tm1 /= ylen as f64;
    if opts.a_opt {
        tm3 /= (xlen + ylen) as f64 * 0.5;
    }
    if opts.u_opt && opts.lnorm_ass > 0.0 {
        tm4 /= opts.lnorm_ass;
    }
    if opts.d_opt {
        tm5 /= ylen as f64;
    }
    if n_ali8 > 0 {
        rmsd0 = (rmsd0 / n_ali8 as f64).sqrt();
    }

    // Prune unused trailing transforms
    while tu_vec.len() > 1 {
        let last_hinge = tu_vec.len() - 1;
        let hc = b'0' + last_hinge as u8;
        let afp_len: usize = seq_m_bytes.iter().filter(|&&b| b == hc).count();
        if afp_len > 0 {
            break;
        }
        tu_vec.pop();
    }

    // Update SE result with recomputed scores
    let final_seq_m = String::from_utf8(seq_m_bytes).unwrap_or_default();
    se_result.tm1 = tm1;
    se_result.tm2 = tm2;
    se_result.tm3 = tm3;
    se_result.tm4 = tm4;
    se_result.tm5 = tm5;
    se_result.rmsd = rmsd0;
    se_result.liden = liden;
    se_result.seq_m = final_seq_m;
    se_result.d0a = d0a;
    se_result.d0b = d0b;
    se_result.d0u = d0u;
    se_result.d0a_avg = d0a_avg;

    let hinge_count = tu_vec.len().saturating_sub(1);

    Ok(FlexResult {
        transforms: tu_vec,
        se_result,
        hinge_count,
    })
}

/// Format rotation matrices for all hinge segments.
///
/// Corresponds to C++ `output_flexalign_rotation_matrix`.
/// Returns the formatted string rather than writing to a file.
pub fn format_rotation_matrices(transforms: &[Transform]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for (hinge, tr) in transforms.iter().enumerate() {
        let _ = writeln!(
            s,
            "------ Hinge {hinge} rotation matrix (Structure_1 -> Structure_2) ------"
        );
        let _ = writeln!(
            s,
            "m {:>18} {:>14} {:>14} {:>14}",
            "t[m]", "u[m][0]", "u[m][1]", "u[m][2]"
        );
        for k in 0..3 {
            let _ = writeln!(
                s,
                "{} {:18.10} {:14.10} {:14.10} {:14.10}",
                k, tr.t[k], tr.u[k][0], tr.u[k][1], tr.u[k][2]
            );
        }
    }
    s.push_str("\nCode for rotating Structure 1 from (x,y,z) to (X,Y,Z):\n");
    s.push_str("for(i=0; i<L; i++)\n{\n");
    s.push_str("   X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];\n");
    s.push_str("   Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];\n");
    s.push_str("   Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];\n");
    s.push_str("}\n");
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a helix-like CA trace.
    fn make_helix_coords(n: usize) -> Vec<Coord3D> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 100.0 / 180.0 * std::f64::consts::PI;
                [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f64]
            })
            .collect()
    }

    #[test]
    fn test_aln2invmap_basic() {
        let invmap = aln2invmap("AC-DEF", "A-GDEF", 4);
        assert_eq!(invmap.len(), 4);
        assert_eq!(invmap[0], 0); // A-A
        assert_eq!(invmap[1], -1); // G gapped in X
        assert_eq!(invmap[2], 2); // D-D (i=2 because C counted, gap skipped)
        assert_eq!(invmap[3], 3); // E-E
    }

    #[test]
    fn test_aln2invmap_all_aligned() {
        let invmap = aln2invmap("ACDE", "ACDE", 4);
        assert_eq!(invmap, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_aln2invmap_empty() {
        let invmap = aln2invmap("", "", 5);
        assert_eq!(invmap, vec![-1; 5]);
    }

    #[test]
    fn test_flexalign_identical_structures() {
        let n = 20;
        let coords = make_helix_coords(n);
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = FlexOptions::default();
        let result = flexalign_main(&coords, &coords, &seq, &seq, &sec, &sec, &opts, &[]);

        assert!(result.is_ok());
        let result = result.unwrap();
        // Identical structures: should align well with 1 transform (0 hinges)
        assert!(!result.transforms.is_empty());
        assert!(result.se_result.tm1 > 0.9, "tm1={}", result.se_result.tm1);
        assert!(result.se_result.tm2 > 0.9, "tm2={}", result.se_result.tm2);
    }

    #[test]
    fn test_flexalign_translated_structure() {
        let n = 25;
        let coords = make_helix_coords(n);
        let shifted: Vec<Coord3D> = coords
            .iter()
            .map(|c| [c[0] + 10.0, c[1] - 5.0, c[2] + 3.0])
            .collect();
        let seq: Vec<u8> = vec![b'A'; n];
        let sec: Vec<u8> = vec![b'C'; n];

        let opts = FlexOptions::default();
        let result = flexalign_main(&coords, &shifted, &seq, &seq, &sec, &sec, &opts, &[]);

        assert!(result.is_ok());
        let result = result.unwrap();
        // Translation only: rigid alignment should suffice, TM-score should be good
        assert!(result.se_result.tm1 > 0.8, "tm1={}", result.se_result.tm1);
    }

    #[test]
    fn test_format_rotation_matrices() {
        let transforms = vec![
            Transform {
                t: [1.0, 2.0, 3.0],
                u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            Transform {
                t: [4.0, 5.0, 6.0],
                u: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            },
        ];
        let output = format_rotation_matrices(&transforms);
        assert!(output.contains("Hinge 0"));
        assert!(output.contains("Hinge 1"));
        assert!(output.contains("X[i] = t[0]"));
    }

    #[test]
    fn test_flex_result_hinge_count() {
        let result = FlexResult {
            transforms: vec![Transform::default(), Transform::default()],
            se_result: SeResult {
                tm1: 0.5,
                tm2: 0.5,
                tm3: 0.0,
                tm4: 0.0,
                tm5: 0.0,
                d0a: 1.0,
                d0b: 1.0,
                d0u: 0.0,
                d0a_avg: 0.0,
                d0_out: 5.0,
                rmsd: 2.0,
                n_ali: 10,
                n_ali8: 8,
                liden: 5.0,
                tm_ali: 0.0,
                rmsd_ali: 0.0,
                invmap: vec![-1; 10],
                seq_x_aligned: String::new(),
                seq_y_aligned: String::new(),
                seq_m: String::new(),
                do_vec: Vec::new(),
            },
            hinge_count: 1,
        };
        assert_eq!(result.hinge_count, 1);
        assert_eq!(result.transforms.len(), 2);
    }

    #[test]
    fn test_smooth_afp_no_panic() {
        // Basic smoke test: smoothing doesn't panic on small inputs
        let mut seq_m = vec![b'0', b'0', b'1', b'0', b'0'];
        let mut seq_m_char = vec![b'0', b'0', b'1', b'0', b'0'];
        let seq_y_aligned = "ACDEF"; // no gaps
        smooth_afp(&mut seq_m, &mut seq_m_char, seq_y_aligned, 2);
        // After smoothing, the singleton '1' should have been reassigned
        assert_ne!(seq_m[2], b'1');
    }

    #[test]
    fn test_extract_x_fragment() {
        let xa = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let seqx = b"ACD";
        let secx = b"CCC";

        // Both aligned (neither is gap)
        let frag = extract_x_fragment("ACD", "ACD", &xa, seqx, secx, |x, y| x != b'-' && y != b'-');
        assert_eq!(frag.coords.len(), 3);

        // Only unaligned X (gaps in Y)
        let frag = extract_x_fragment("ACD", "A-D", &xa, seqx, secx, |_x, y| y == b'-');
        assert_eq!(frag.coords.len(), 1);
        assert_eq!(frag.seq[0], b'C');
    }
}
