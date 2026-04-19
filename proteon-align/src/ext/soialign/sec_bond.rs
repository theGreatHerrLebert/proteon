//! SSE boundary detection for SOI alignment.
//!
//! Ported from C++ USAlign `SOIalign.h` `assign_sec_bond`.
//! Identifies contiguous secondary structure element (SSE) boundaries
//! so that greedy assignment can enforce intra-SSE sequentiality.

/// SSE boundary for a single residue: `[start, end)` or `[-1, -1]` if not in an SSE.
pub type SecBond = [i32; 2];

/// Assign SSE boundaries from a secondary structure string.
///
/// For each residue, `secx_bond[i]` contains the `[start, end)` half-open range
/// of the SSE it belongs to, or `[-1, -1]` if it is not part of a recognized SSE.
/// Single-residue SSEs are collapsed to `[-1, -1]`.
///
/// Recognized SSE types: `'H'`, `'E'`, `'<'`, `'>'`.
/// `'C'` and `'T'` are treated as equivalent coil.
///
/// Corresponds to C++ `assign_sec_bond`.
pub fn assign_sec_bond(secx: &[u8]) -> Vec<SecBond> {
    let xlen = secx.len();
    let mut secx_bond = vec![[-1i32, -1i32]; xlen];

    let mut starti: i32 = -1;
    let mut prev_ss: u8 = 0;

    for i in 0..xlen {
        let ss = secx[i];
        secx_bond[i] = [-1, -1];
        if ss != prev_ss && !(ss == b'C' && prev_ss == b'T') && !(ss == b'T' && prev_ss == b'C') {
            if starti >= 0 {
                // previous SSE ends at i
                let endi = i as i32;
                for j in (starti as usize)..(endi as usize) {
                    secx_bond[j][0] = starti;
                    secx_bond[j][1] = endi;
                }
            }
            if ss == b'H' || ss == b'E' || ss == b'<' || ss == b'>' {
                starti = i as i32;
            } else {
                starti = -1;
            }
        }
        prev_ss = secx[i];
    }

    // Handle trailing SSE
    if starti >= 0 {
        let endi = xlen as i32;
        for j in (starti as usize)..(endi as usize) {
            secx_bond[j][0] = starti;
            secx_bond[j][1] = endi;
        }
    }

    // Remove single-residue SSEs
    for i in 0..xlen {
        if secx_bond[i][1] - secx_bond[i][0] == 1 {
            secx_bond[i] = [-1, -1];
        }
    }

    secx_bond
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_coil() {
        let sec = b"CCCCC";
        let bonds = assign_sec_bond(sec);
        for b in &bonds {
            assert_eq!(*b, [-1, -1]);
        }
    }

    #[test]
    fn test_single_helix() {
        let sec = b"CHHHC";
        let bonds = assign_sec_bond(sec);
        assert_eq!(bonds[0], [-1, -1]);
        assert_eq!(bonds[1], [1, 4]); // H at index 1..4
        assert_eq!(bonds[2], [1, 4]);
        assert_eq!(bonds[3], [1, 4]);
        assert_eq!(bonds[4], [-1, -1]);
    }

    #[test]
    fn test_single_residue_sse_removed() {
        // A single-residue helix should be removed
        let sec = b"CHCCC";
        let bonds = assign_sec_bond(sec);
        assert_eq!(bonds[1], [-1, -1]);
    }

    #[test]
    fn test_coil_and_turn_merged() {
        // C and T are treated as equivalent
        let sec = b"CTCHH";
        let bonds = assign_sec_bond(sec);
        // C/T at 0-1 are coil, no SSE
        assert_eq!(bonds[0], [-1, -1]);
        assert_eq!(bonds[1], [-1, -1]);
        // HH at 3-4
        assert_eq!(bonds[3], [3, 5]);
        assert_eq!(bonds[4], [3, 5]);
    }

    #[test]
    fn test_trailing_sse() {
        let sec = b"CHHH";
        let bonds = assign_sec_bond(sec);
        assert_eq!(bonds[1], [1, 4]);
        assert_eq!(bonds[2], [1, 4]);
        assert_eq!(bonds[3], [1, 4]);
    }

    #[test]
    fn test_multiple_sses() {
        let sec = b"HHHCEEE";
        let bonds = assign_sec_bond(sec);
        assert_eq!(bonds[0], [0, 3]);
        assert_eq!(bonds[1], [0, 3]);
        assert_eq!(bonds[2], [0, 3]);
        assert_eq!(bonds[3], [-1, -1]);
        assert_eq!(bonds[4], [4, 7]);
        assert_eq!(bonds[5], [4, 7]);
        assert_eq!(bonds[6], [4, 7]);
    }
}
