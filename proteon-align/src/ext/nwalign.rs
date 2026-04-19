//! Gotoh affine-gap sequence alignment (global, semi-local, and local).
//!
//! Ported from C++ USAlign `NWalign.h`.
//! Supports BLOSUM62 (protein) and BLASTN (RNA/DNA) scoring via [`crate::blosum`].

use crate::ext::blosum::BLOSUM;

/// Gap penalties for BLOSUM62 protein alignment.
const GAPOPEN_BLOSUM62: i32 = -11;
const GAPEXT_BLOSUM62: i32 = -1;

/// Gap penalties for BLASTN nucleotide alignment.
const GAPOPEN_BLASTN: i32 = -15;
const GAPEXT_BLASTN: i32 = -4;

/// Gap penalties for local BLASTN alignment.
const GAPOPEN_BLASTN_LOCAL: i32 = -5;
const GAPEXT_BLASTN_LOCAL: i32 = -2;

/// Alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Needleman-Wunsch global alignment.
    Global,
    /// Semi-local: free gaps at query (seqy) termini.
    GlocalQuery,
    /// Semi-local: free gaps at both termini.
    GlocalBoth,
    /// Smith-Waterman local alignment.
    Local,
}

impl AlignMode {
    fn as_int(self) -> i32 {
        match self {
            AlignMode::Global => 0,
            AlignMode::GlocalQuery => 1,
            AlignMode::GlocalBoth => 2,
            AlignMode::Local => 3,
        }
    }
}

/// What to return from the alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnMode {
    /// Return only aligned sequence strings.
    AlignmentOnly,
    /// Return only the invmap (j->i mapping).
    InvmapOnly,
    /// Return both aligned strings and invmap.
    Both,
}

/// Result of Gotoh alignment.
#[derive(Debug, Clone)]
pub struct NWAlignResult {
    /// Alignment score.
    pub score: i32,
    /// Aligned sequence X (with gaps). Empty if `InvmapOnly`.
    pub seq_x_aligned: String,
    /// Aligned sequence Y (with gaps). Empty if `InvmapOnly`.
    pub seq_y_aligned: String,
    /// Residue mapping: `invmap[j] = i` means residue j in seqy aligns to residue i in seqx.
    /// -1 means unaligned. Empty if `AlignmentOnly`.
    pub invmap: Vec<i32>,
}

/// Sequence identity result.
#[derive(Debug, Clone, Copy)]
pub struct SeqIdResult {
    /// Number of identical aligned residues.
    pub n_identical: f64,
    /// Number of aligned residue pairs.
    pub n_aligned: i32,
}

/// Workspace for Gotoh DP to avoid repeated allocations.
struct GotohWorkspace {
    /// Cumulative score matrix `[xlen+1][ylen+1]`.
    s: Vec<Vec<i32>>,
    /// Horizontal gap penalty matrix.
    h: Vec<Vec<i32>>,
    /// Vertical gap penalty matrix.
    v: Vec<Vec<i32>>,
    /// Path matrix (bit-encoded: 1=diag, 2=vertical, 4=horizontal).
    p: Vec<Vec<i32>>,
    /// Horizontal jump count.
    jump_h: Vec<Vec<i32>>,
    /// Vertical jump count.
    jump_v: Vec<Vec<i32>>,
}

impl GotohWorkspace {
    fn new(xlen: usize, ylen: usize) -> Self {
        let rows = xlen + 1;
        let cols = ylen + 1;
        GotohWorkspace {
            s: vec![vec![0i32; cols]; rows],
            h: vec![vec![0i32; cols]; rows],
            v: vec![vec![0i32; cols]; rows],
            p: vec![vec![0i32; cols]; rows],
            jump_h: vec![vec![0i32; cols]; rows],
            jump_v: vec![vec![0i32; cols]; rows],
        }
    }
}

/// Main entry point for Gotoh affine-gap sequence alignment.
///
/// Corresponds to C++ `NWalign_main`.
pub fn nwalign(
    seqx: &[u8],
    seqy: &[u8],
    mol_type: i32,
    mode: AlignMode,
    return_mode: ReturnMode,
) -> NWAlignResult {
    let xlen = seqx.len();
    let ylen = seqy.len();

    let (gapopen, gapext) = if mol_type > 0 {
        // RNA or DNA
        if mode == AlignMode::Local {
            (GAPOPEN_BLASTN_LOCAL, GAPEXT_BLASTN_LOCAL)
        } else {
            (GAPOPEN_BLASTN, GAPEXT_BLASTN)
        }
    } else {
        (GAPOPEN_BLOSUM62, GAPEXT_BLOSUM62)
    };

    let mut ws = GotohWorkspace::new(xlen, ylen);

    // Fill substitution scores into S matrix (1-indexed)
    for i in 0..=xlen {
        for j in 0..=ylen {
            if i == 0 || j == 0 {
                ws.s[i][j] = 0;
            } else {
                ws.s[i][j] = BLOSUM[seqx[i - 1] as usize][seqy[j - 1] as usize];
            }
        }
    }

    let aln_score = calculate_score_gotoh(&mut ws, xlen, ylen, gapopen, gapext, mode);

    let mut invmap = if return_mode != ReturnMode::AlignmentOnly {
        vec![-1i32; ylen]
    } else {
        Vec::new()
    };

    let (seq_x_aligned, seq_y_aligned) = if mode == AlignMode::Local {
        trace_back_sw(seqx, seqy, &ws, xlen, ylen, &mut invmap, return_mode)
    } else {
        trace_back_gotoh(seqx, seqy, &ws, xlen, ylen, &mut invmap, return_mode)
    };

    NWAlignResult {
        score: aln_score,
        seq_x_aligned,
        seq_y_aligned,
        invmap,
    }
}

/// Initialize the Gotoh DP matrices.
fn init_gotoh_mat(
    ws: &mut GotohWorkspace,
    xlen: usize,
    ylen: usize,
    gapopen: i32,
    gapext: i32,
    glocal: i32,
) {
    // Zero everything
    for i in 0..=xlen {
        for j in 0..=ylen {
            ws.h[i][j] = 0;
            ws.v[i][j] = 0;
            ws.p[i][j] = 0;
            ws.jump_h[i][j] = 0;
            ws.jump_v[i][j] = 0;
        }
    }

    // First column
    for i in 0..=xlen {
        if glocal < 2 {
            ws.p[i][0] = 4; // -
        }
        ws.jump_v[i][0] = i as i32;
    }

    // First row
    for j in 0..=ylen {
        if glocal < 1 {
            ws.p[0][j] = 2; // |
        }
        ws.jump_h[0][j] = j as i32;
    }

    // Initial gap scores
    if glocal < 2 {
        for i in 1..=xlen {
            ws.s[i][0] = gapopen + gapext * (i as i32 - 1);
        }
    }
    if glocal < 1 {
        for j in 1..=ylen {
            ws.s[0][j] = gapopen + gapext * (j as i32 - 1);
        }
    }

    // alt_init=1 (Wei Zheng's initialization, always used)
    if glocal < 2 {
        for i in 1..=xlen {
            ws.v[i][0] = gapopen + gapext * (i as i32 - 1);
        }
    }
    if glocal < 1 {
        for j in 1..=ylen {
            ws.h[0][j] = gapopen + gapext * (j as i32 - 1);
        }
    }
    for i in 0..=xlen {
        ws.h[i][0] = -99999;
    }
    for j in 0..=ylen {
        ws.v[0][j] = -99999;
    }
}

/// Calculate the Gotoh DP matrix.
///
/// Corresponds to C++ `calculate_score_gotoh`.
fn calculate_score_gotoh(
    ws: &mut GotohWorkspace,
    xlen: usize,
    ylen: usize,
    gapopen: i32,
    gapext: i32,
    mode: AlignMode,
) -> i32 {
    let glocal = mode.as_int();

    init_gotoh_mat(ws, xlen, ylen, gapopen, gapext, glocal);

    for i in 1..=xlen {
        for j in 1..=ylen {
            // Horizontal gap (deletion)
            if glocal < 1 || i < xlen || glocal >= 3 {
                ws.h[i][j] = (ws.s[i][j - 1] + gapopen).max(ws.h[i][j - 1] + gapext);
                ws.jump_h[i][j] = if ws.h[i][j] == ws.h[i][j - 1] + gapext {
                    ws.jump_h[i][j - 1] + 1
                } else {
                    1
                };
            } else {
                ws.h[i][j] = ws.s[i][j - 1].max(ws.h[i][j - 1]);
                ws.jump_h[i][j] = if ws.h[i][j] == ws.h[i][j - 1] {
                    ws.jump_h[i][j - 1] + 1
                } else {
                    1
                };
            }

            // Vertical gap (insertion)
            if glocal < 2 || j < ylen || glocal >= 3 {
                ws.v[i][j] = (ws.s[i - 1][j] + gapopen).max(ws.v[i - 1][j] + gapext);
                ws.jump_v[i][j] = if ws.v[i][j] == ws.v[i - 1][j] + gapext {
                    ws.jump_v[i - 1][j] + 1
                } else {
                    1
                };
            } else {
                ws.v[i][j] = ws.s[i - 1][j].max(ws.v[i - 1][j]);
                ws.jump_v[i][j] = if ws.v[i][j] == ws.v[i - 1][j] {
                    ws.jump_v[i - 1][j] + 1
                } else {
                    1
                };
            }

            let diag_score = ws.s[i - 1][j - 1] + ws.s[i][j]; // match/mismatch
            let left_score = ws.h[i][j]; // deletion
            let up_score = ws.v[i][j]; // insertion

            ws.p[i][j] = 0;

            if diag_score >= left_score && diag_score >= up_score {
                ws.s[i][j] = diag_score;
                ws.p[i][j] += 1;
            }
            if up_score >= diag_score && up_score >= left_score {
                ws.s[i][j] = up_score;
                ws.p[i][j] += 2;
            }
            if left_score >= diag_score && left_score >= up_score {
                ws.s[i][j] = left_score;
                ws.p[i][j] += 4;
            }

            // Smith-Waterman: clamp negatives to zero
            if glocal >= 3 && ws.s[i][j] < 0 {
                ws.s[i][j] = 0;
                ws.p[i][j] = 0;
                ws.h[i][j] = 0;
                ws.v[i][j] = 0;
                ws.jump_h[i][j] = 0;
                ws.jump_v[i][j] = 0;
            }
        }
    }

    let mut aln_score = ws.s[xlen][ylen];

    // Re-fill first row/column of P for backtrace
    for i in 1..=xlen {
        if glocal < 3 || ws.p[i][0] > 0 {
            ws.p[i][0] = 2; // |
        }
    }
    for j in 1..=ylen {
        if glocal < 3 || ws.p[0][j] > 0 {
            ws.p[0][j] = 4; // -
        }
    }

    // For local alignment, find highest-scoring cell
    if glocal >= 3 {
        find_highest_align_score(ws, &mut aln_score, xlen, ylen);
    }

    aln_score
}

/// Find the cell with the highest alignment score for Smith-Waterman.
/// Reset path entries after that cell.
fn find_highest_align_score(
    ws: &mut GotohWorkspace,
    aln_score: &mut i32,
    xlen: usize,
    ylen: usize,
) {
    let mut max_i = xlen;
    let mut max_j = ylen;

    for i in 0..=xlen {
        for j in 0..=ylen {
            if ws.s[i][j] >= *aln_score {
                max_i = i;
                max_j = j;
                *aln_score = ws.s[i][j];
            }
        }
    }

    // Reset path after the maximum
    for i in (max_i + 1)..=xlen {
        for j in 0..=ylen {
            ws.p[i][j] = 0;
        }
    }
    for i in 0..=xlen {
        for j in (max_j + 1)..=ylen {
            ws.p[i][j] = 0;
        }
    }
}

/// Trace back through the Gotoh DP for global/semi-local alignment.
///
/// Corresponds to C++ `trace_back_gotoh`.
fn trace_back_gotoh(
    seqx: &[u8],
    seqy: &[u8],
    ws: &GotohWorkspace,
    xlen: usize,
    ylen: usize,
    invmap: &mut [i32],
    return_mode: ReturnMode,
) -> (String, String) {
    let build_strings = return_mode != ReturnMode::InvmapOnly;

    let mut seq_x_parts: Vec<u8> = Vec::new();
    let mut seq_y_parts: Vec<u8> = Vec::new();

    let mut i = xlen;
    let mut j = ylen;

    while i + j > 0 {
        if ws.p[i][j] >= 4 {
            // Horizontal gap
            let gaplen = ws.jump_h[i][j] as usize;
            j -= gaplen;
            if build_strings {
                for k in 0..gaplen {
                    seq_x_parts.push(b'-');
                    seq_y_parts.push(seqy[j + k]);
                }
            }
        } else if ws.p[i][j] % 4 >= 2 {
            // Vertical gap
            let gaplen = ws.jump_v[i][j] as usize;
            i -= gaplen;
            if build_strings {
                for k in 0..gaplen {
                    seq_x_parts.push(seqx[i + k]);
                    seq_y_parts.push(b'-');
                }
            }
        } else {
            // Diagonal (match/mismatch) or glocal terminal
            if i == 0 && j != 0 {
                // glocal alignment: remaining y residues
                if build_strings {
                    for k in 0..j {
                        seq_x_parts.push(b'-');
                        seq_y_parts.push(seqy[k]);
                    }
                }
                break;
            }
            if i != 0 && j == 0 {
                // glocal alignment: remaining x residues
                if build_strings {
                    for k in 0..i {
                        seq_x_parts.push(seqx[k]);
                        seq_y_parts.push(b'-');
                    }
                }
                break;
            }
            i -= 1;
            j -= 1;
            if !invmap.is_empty() {
                invmap[j] = i as i32;
            }
            if build_strings {
                seq_x_parts.push(seqx[i]);
                seq_y_parts.push(seqy[j]);
            }
        }
    }

    // Reverse because we traced back from end to start
    seq_x_parts.reverse();
    seq_y_parts.reverse();

    if build_strings {
        (
            String::from_utf8(seq_x_parts).unwrap_or_default(),
            String::from_utf8(seq_y_parts).unwrap_or_default(),
        )
    } else {
        (String::new(), String::new())
    }
}

/// Trace back Smith-Waterman local alignment.
///
/// Corresponds to C++ `trace_back_sw`.
fn trace_back_sw(
    seqx: &[u8],
    seqy: &[u8],
    ws: &GotohWorkspace,
    xlen: usize,
    ylen: usize,
    invmap: &mut [i32],
    return_mode: ReturnMode,
) -> (String, String) {
    let build_strings = return_mode != ReturnMode::InvmapOnly;

    // Find the last non-zero cell in P (bottom-right of local alignment)
    let mut start_i: Option<usize> = None;
    let mut start_j: usize = 0;

    'outer: for i in (0..=xlen).rev() {
        for j in (0..=ylen).rev() {
            if ws.p[i][j] != 0 {
                start_i = Some(i);
                start_j = j;
                break 'outer;
            }
        }
    }

    let (mut i, mut j) = match start_i {
        Some(si) => (si, start_j),
        None => return (String::new(), String::new()),
    };

    // Build C-terminal unaligned tails
    let mut suffix_x = Vec::new();
    let mut suffix_y = Vec::new();

    if build_strings {
        // Gaps in x for y residues after alignment end
        for k in j..ylen {
            suffix_x.push(b'-');
            suffix_y.push(seqy[k]);
        }
        // x residues after alignment end gapped in y
        for k in i..xlen {
            suffix_x.push(seqx[k]);
            suffix_y.push(b'-');
        }
    }

    // Trace back the aligned core
    let mut core_x = Vec::new();
    let mut core_y = Vec::new();

    while ws.p[i][j] != 0 {
        if ws.p[i][j] >= 4 {
            let gaplen = ws.jump_h[i][j] as usize;
            j -= gaplen;
            if build_strings {
                for k in 0..gaplen {
                    core_x.push(b'-');
                    core_y.push(seqy[j + k]);
                }
            }
        } else if ws.p[i][j] % 4 >= 2 {
            let gaplen = ws.jump_v[i][j] as usize;
            i -= gaplen;
            if build_strings {
                for k in 0..gaplen {
                    core_x.push(seqx[i + k]);
                    core_y.push(b'-');
                }
            }
        } else {
            i -= 1;
            j -= 1;
            if !invmap.is_empty() {
                invmap[j] = i as i32;
            }
            if build_strings {
                core_x.push(seqx[i]);
                core_y.push(seqy[j]);
            }
        }
    }

    // Build N-terminal unaligned prefix
    let mut prefix_x = Vec::new();
    let mut prefix_y = Vec::new();

    if build_strings {
        // y residues before alignment start
        for k in 0..j {
            prefix_x.push(b'-');
            prefix_y.push(seqy[k]);
        }
        // x residues before alignment start
        for k in 0..i {
            prefix_x.push(seqx[k]);
            prefix_y.push(b'-');
        }
    }

    if build_strings {
        // Assemble: prefix + reversed core + suffix
        core_x.reverse();
        core_y.reverse();

        let mut final_x = prefix_x;
        final_x.extend_from_slice(&core_x);
        final_x.extend_from_slice(&suffix_x);

        let mut final_y = prefix_y;
        final_y.extend_from_slice(&core_y);
        final_y.extend_from_slice(&suffix_y);

        (
            String::from_utf8(final_x).unwrap_or_default(),
            String::from_utf8(final_y).unwrap_or_default(),
        )
    } else {
        (String::new(), String::new())
    }
}

/// Compute sequence identity from an invmap.
///
/// Corresponds to C++ `get_seqID` (invmap version).
pub fn get_seq_id_from_invmap(invmap: &[i32], seqx: &[u8], seqy: &[u8]) -> SeqIdResult {
    let mut n_identical = 0.0_f64;
    let mut n_aligned = 0i32;
    for (j, &i) in invmap.iter().enumerate() {
        if i < 0 {
            continue;
        }
        n_aligned += 1;
        if seqx[i as usize] == seqy[j] {
            n_identical += 1.0;
        }
    }
    SeqIdResult {
        n_identical,
        n_aligned,
    }
}

/// Compute sequence identity from aligned strings.
///
/// Corresponds to C++ `get_seqID` (string version).
/// Returns (identity_count, aligned_count, match_string).
pub fn get_seq_id_from_alignment(seq_x_aligned: &str, seq_y_aligned: &str) -> (f64, i32, String) {
    let mut n_identical = 0.0_f64;
    let mut n_aligned = 0i32;
    let mut seq_m = String::with_capacity(seq_x_aligned.len());

    let xb = seq_x_aligned.as_bytes();
    let yb = seq_y_aligned.as_bytes();

    for i in 0..xb.len() {
        if xb[i] == yb[i] && xb[i] != b'-' {
            n_identical += 1.0;
            seq_m.push(':');
        } else {
            seq_m.push(' ');
        }
        if xb[i] != b'-' && yb[i] != b'-' {
            n_aligned += 1;
        }
    }

    (n_identical, n_aligned, seq_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_identical() {
        let seq = b"ACDEFGHIKLMNPQRSTVWY";
        let result = nwalign(seq, seq, 0, AlignMode::Global, ReturnMode::Both);
        assert!(result.score > 0);
        assert_eq!(result.seq_x_aligned, result.seq_y_aligned);
        // Every position should be aligned to itself
        for (j, &i) in result.invmap.iter().enumerate() {
            assert_eq!(i, j as i32);
        }
    }

    #[test]
    fn test_global_with_gap() {
        let seqx = b"ACDEF";
        let seqy = b"ADEF";
        let result = nwalign(seqx, seqy, 0, AlignMode::Global, ReturnMode::AlignmentOnly);
        assert!(result.score > 0);
        assert!(result.seq_x_aligned.contains('-') || result.seq_y_aligned.contains('-'));
    }

    #[test]
    fn test_invmap_only() {
        let seqx = b"ACDEF";
        let seqy = b"ACDEF";
        let result = nwalign(seqx, seqy, 0, AlignMode::Global, ReturnMode::InvmapOnly);
        assert!(result.seq_x_aligned.is_empty());
        assert!(result.seq_y_aligned.is_empty());
        assert_eq!(result.invmap.len(), 5);
        for (j, &i) in result.invmap.iter().enumerate() {
            assert_eq!(i, j as i32);
        }
    }

    #[test]
    fn test_rna_alignment() {
        let seqx = b"acguacgu";
        let seqy = b"acguacgu";
        let result = nwalign(seqx, seqy, 1, AlignMode::Global, ReturnMode::Both);
        assert!(result.score > 0);
        assert_eq!(result.seq_x_aligned, "acguacgu");
        assert_eq!(result.seq_y_aligned, "acguacgu");
    }

    #[test]
    fn test_local_alignment() {
        let seqx = b"XXXACDEFXXX";
        let seqy = b"ACDEF";
        let result = nwalign(seqx, seqy, 0, AlignMode::Local, ReturnMode::InvmapOnly);
        let aligned: Vec<_> = result.invmap.iter().filter(|&&i| i >= 0).collect();
        assert!(!aligned.is_empty());
    }

    #[test]
    fn test_seq_id_from_invmap() {
        let seqx = b"ACDEF";
        let seqy = b"AXDEF";
        let invmap = vec![0, 1, 2, 3, 4];
        let id = get_seq_id_from_invmap(&invmap, seqx, seqy);
        assert_eq!(id.n_aligned, 5);
        assert!((id.n_identical - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_seq_id_from_alignment() {
        let (ident, ali, markers) = get_seq_id_from_alignment("ACDEF", "AXDEF");
        assert_eq!(ali, 5);
        assert!((ident - 4.0).abs() < 0.001);
        assert_eq!(markers, ": :::");
    }
}
