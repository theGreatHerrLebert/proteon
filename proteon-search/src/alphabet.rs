//! Amino-acid and nucleotide alphabets with char ↔ index mappings.
//!
//! The canonical protein alphabet is the 21-letter order upstream MMseqs2 uses
//! in every `.out` matrix column header: `A C D E F G H I K L M N P Q R S T V W Y X`.
//! Ambiguous AA codes (B, Z, J, U, O) are folded onto their canonical neighbors
//! per upstream's `SubstitutionMatrix::setupLetterMapping`:
//!
//! - `B → D`, `Z → E`, `J → L`, `U → X`, `O → X`
//! - lowercase letters fold to their uppercase equivalents
//! - anything else folds to `X`

/// The 21 canonical letters in upstream's column order.
pub const PROTEIN_LETTERS: &[u8; 21] = b"ACDEFGHIKLMNPQRSTVWYX";

/// Nucleotide alphabet in upstream's `nucleotide.out` column order.
pub const NUCLEOTIDE_LETTERS: &[u8; 5] = b"ACTGX";

/// Sentinel value in `aa_to_index` for characters not in the alphabet.
const UNMAPPED: u8 = u8::MAX;

/// An alphabet over 8-bit ASCII letters.
///
/// Holds a 256-entry lookup `ascii → index` and the inverse `index → ascii`.
/// Index values are small (`<= letters.len()`), so a single `u8` is enough.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alphabet {
    pub letters: Vec<u8>,
    pub aa_to_index: [u8; 256],
}

impl Alphabet {
    /// Build an alphabet from a letter list, without any ambiguous-AA folding.
    /// Unknown letters map to `UNMAPPED` (255).
    pub fn new_raw(letters: &[u8]) -> Self {
        let mut aa_to_index = [UNMAPPED; 256];
        for (i, &c) in letters.iter().enumerate() {
            aa_to_index[c as usize] = i as u8;
            // uppercase and lowercase point at the same index
            aa_to_index[(c as char).to_ascii_lowercase() as usize] = i as u8;
        }
        Self {
            letters: letters.to_vec(),
            aa_to_index,
        }
    }

    /// Standard protein alphabet with ambiguous-AA folding matching upstream.
    ///
    /// The folding mirrors `SubstitutionMatrix::setupLetterMapping` in upstream
    /// MMseqs2 at `src/commons/SubstitutionMatrix.cpp:280-297`. In particular,
    /// every letter not in `PROTEIN_LETTERS` that is not explicitly folded
    /// (e.g. digit, punctuation) maps to the index of `X`.
    pub fn protein() -> Self {
        let mut alpha = Self::new_raw(PROTEIN_LETTERS);
        let x_idx = alpha.aa_to_index[b'X' as usize];

        // Explicit folds.
        let fold_pairs: &[(u8, u8)] = &[
            (b'B', b'D'),
            (b'Z', b'E'),
            (b'J', b'L'),
            (b'U', b'X'),
            (b'O', b'X'),
        ];
        for &(src, dst) in fold_pairs {
            let dst_idx = alpha.aa_to_index[dst as usize];
            alpha.aa_to_index[src as usize] = dst_idx;
            alpha.aa_to_index[(src as char).to_ascii_lowercase() as usize] = dst_idx;
        }

        // Anything still unmapped → X. Includes '*', digits, punctuation,
        // and any non-ASCII byte. Upstream's default branch does the same.
        for i in 0..256 {
            if alpha.aa_to_index[i] == UNMAPPED {
                alpha.aa_to_index[i] = x_idx;
            }
        }
        alpha
    }

    /// Nucleotide alphabet (ACTGX). No ambiguous folding beyond lowercase → upper.
    /// Unknown bytes fold to `X`.
    pub fn nucleotide() -> Self {
        let mut alpha = Self::new_raw(NUCLEOTIDE_LETTERS);
        let x_idx = alpha.aa_to_index[b'X' as usize];
        // Common nucleotide ambiguity codes all fold to X for MMseqs2 scoring.
        for &c in b"UNRYSWKMBDHVnryswkmbdhv" {
            if alpha.aa_to_index[c as usize] == UNMAPPED {
                alpha.aa_to_index[c as usize] = x_idx;
            }
        }
        // `U` is RNA uracil — treated as T for practical DNA/RNA mixing.
        alpha.aa_to_index[b'U' as usize] = alpha.aa_to_index[b'T' as usize];
        alpha.aa_to_index[b'u' as usize] = alpha.aa_to_index[b'T' as usize];
        for i in 0..256 {
            if alpha.aa_to_index[i] == UNMAPPED {
                alpha.aa_to_index[i] = x_idx;
            }
        }
        alpha
    }

    /// Number of letters in the alphabet.
    pub fn size(&self) -> usize {
        self.letters.len()
    }

    /// Index for an ASCII character. Out-of-range bytes are unreachable because
    /// `aa_to_index` covers the full `0..=255` range.
    pub fn encode(&self, c: u8) -> u8 {
        self.aa_to_index[c as usize]
    }

    /// Canonical ASCII character for an index.
    pub fn decode(&self, idx: u8) -> u8 {
        self.letters[idx as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protein_order_matches_upstream() {
        let alpha = Alphabet::protein();
        // Every canonical letter must map to its position in PROTEIN_LETTERS.
        for (i, &c) in PROTEIN_LETTERS.iter().enumerate() {
            assert_eq!(alpha.encode(c), i as u8, "letter {}", c as char);
            assert_eq!(alpha.decode(i as u8), c);
        }
    }

    #[test]
    fn protein_ambiguous_folding_matches_upstream() {
        let alpha = Alphabet::protein();
        assert_eq!(alpha.encode(b'B'), alpha.encode(b'D'));
        assert_eq!(alpha.encode(b'Z'), alpha.encode(b'E'));
        assert_eq!(alpha.encode(b'J'), alpha.encode(b'L'));
        assert_eq!(alpha.encode(b'U'), alpha.encode(b'X'));
        assert_eq!(alpha.encode(b'O'), alpha.encode(b'X'));
    }

    #[test]
    fn protein_lowercase_folds_to_uppercase() {
        let alpha = Alphabet::protein();
        for c in b'A'..=b'Z' {
            assert_eq!(
                alpha.encode(c),
                alpha.encode(c.to_ascii_lowercase()),
                "mismatch for {}",
                c as char,
            );
        }
    }

    #[test]
    fn protein_unknown_bytes_fold_to_x() {
        let alpha = Alphabet::protein();
        let x = alpha.encode(b'X');
        for c in [b'*', b'-', b'1', b'@', 0u8, 255u8] {
            assert_eq!(alpha.encode(c), x, "byte {c:#x} should fold to X");
        }
    }

    #[test]
    fn protein_specific_known_indices() {
        // Pin a few well-known indices so any accidental reordering breaks loudly.
        let alpha = Alphabet::protein();
        assert_eq!(alpha.encode(b'A'), 0);
        assert_eq!(alpha.encode(b'C'), 1);
        assert_eq!(alpha.encode(b'M'), 10);
        assert_eq!(alpha.encode(b'W'), 18);
        assert_eq!(alpha.encode(b'Y'), 19);
        assert_eq!(alpha.encode(b'X'), 20);
    }

    #[test]
    fn nucleotide_basic_encoding() {
        let alpha = Alphabet::nucleotide();
        assert_eq!(alpha.encode(b'A'), 0);
        assert_eq!(alpha.encode(b'C'), 1);
        assert_eq!(alpha.encode(b'T'), 2);
        assert_eq!(alpha.encode(b'G'), 3);
        assert_eq!(alpha.encode(b'X'), 4);
        assert_eq!(alpha.encode(b'U'), alpha.encode(b'T'), "U → T");
        assert_eq!(alpha.encode(b'N'), alpha.encode(b'X'), "N → X");
    }
}
