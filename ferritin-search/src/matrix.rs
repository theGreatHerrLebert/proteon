//! Substitution matrices (BLOSUM, PAM, VTML, nucleotide) in upstream's `.out`
//! file format.
//!
//! Format reference: `MMseqs2/src/commons/SubstitutionMatrix.cpp:readProbMatrix`.
//!
//! ```text
//! # optional comments starting with '#'
//! # Background (precomputed optional): f1 f2 ... fN 0.00001
//! # Lambda     (precomputed optional): f
//!    A   C   D  ...   X         <- alphabet header (letters only)
//! A  3.9 -0.4 ...  -1.0          <- row label + N floats
//! C  ...
//! ...
//! X  ...
//! ```
//!
//! The float values are half-bit log-odds scores. Downstream consumers
//! (prefilter, aligner) multiply by a `bitFactor` and round to integer
//! before use; see [`SubstitutionMatrix::to_integer_matrix`].

use std::io;
use std::path::Path;

use thiserror::Error;

use crate::alphabet::{Alphabet, NUCLEOTIDE_LETTERS, PROTEIN_LETTERS};

/// BLOSUM62 as shipped by upstream MMseqs2.
pub const BLOSUM62_OUT: &str = include_str!("builtin/blosum62.out");

/// Default nucleotide matrix as shipped by upstream MMseqs2.
pub const NUCLEOTIDE_OUT: &str = include_str!("builtin/nucleotide.out");

#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("matrix file had no alphabet header")]
    MissingAlphabetHeader,
    #[error("matrix file ended before the {expected}x{expected} score grid was complete")]
    TruncatedMatrix { expected: usize },
    #[error("row label '{row}' does not match expected letter '{expected}'")]
    RowLabelMismatch { row: char, expected: char },
    #[error("row '{row}' has {got} scores, expected {expected}")]
    WrongRowWidth { row: char, got: usize, expected: usize },
    #[error("non-numeric score '{token}' in row '{row}'")]
    BadScore { row: char, token: String },
    #[error("background frequency count {got} does not match alphabet size {expected}")]
    BackgroundMismatch { got: usize, expected: usize },
    #[error("matrix header has {header_size} letters but the provided alphabet has {alphabet_size}")]
    AlphabetSizeMismatch { header_size: usize, alphabet_size: usize },
}

pub type Result<T> = std::result::Result<T, MatrixError>;

/// A square substitution matrix keyed by alphabet indices.
///
/// Values are stored as f32 half-bit log-odds as read from the `.out` file.
/// Use [`SubstitutionMatrix::score`] to look up `score(i, j)` by alphabet
/// index, or [`SubstitutionMatrix::score_chars`] by ASCII letter.
#[derive(Debug, Clone)]
pub struct SubstitutionMatrix {
    /// Matrix name (e.g. `"blosum62"`), inferred from filename or header comment.
    pub name: String,
    /// Alphabet used for indexing; rows and columns share the same alphabet.
    pub alphabet: Alphabet,
    /// `size × size` score grid in row-major order. `size = alphabet.size()`.
    pub scores: Vec<f32>,
    /// Optional per-letter background frequencies (alphabet order), when provided.
    pub background: Option<Vec<f32>>,
    /// Optional Karlin-Altschul lambda, when provided.
    pub lambda: Option<f32>,
}

impl SubstitutionMatrix {
    /// Parse an upstream `.out` substitution matrix from a string.
    ///
    /// `alphabet` is used to resolve row/column labels to indices. It must
    /// contain every letter listed in the matrix header; extra letters (with
    /// ambiguous folding) are fine. `name` is attached to the returned matrix.
    pub fn parse(name: impl Into<String>, alphabet: Alphabet, text: &str) -> Result<Self> {
        let mut background: Option<Vec<f32>> = None;
        let mut lambda: Option<f32> = None;
        let mut alphabet_header: Option<Vec<u8>> = None;
        let mut rows: Vec<(u8, Vec<f32>)> = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix('#') {
                parse_comment(rest, &mut background, &mut lambda);
                continue;
            }

            let tokens: Vec<&str> = trimmed.split_ascii_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            // Alphabet header: first non-comment row with only single-letter tokens.
            if alphabet_header.is_none() {
                if tokens.iter().all(|t| t.len() == 1 && t.as_bytes()[0].is_ascii_alphabetic()) {
                    alphabet_header = Some(
                        tokens
                            .iter()
                            .map(|t| t.as_bytes()[0].to_ascii_uppercase())
                            .collect(),
                    );
                    continue;
                }
                // Non-header, non-comment line before header → malformed.
                return Err(MatrixError::MissingAlphabetHeader);
            }

            // Score row: single-letter label then N floats.
            let header = alphabet_header.as_ref().unwrap();
            let expected_cols = header.len();

            if tokens[0].len() != 1 || !tokens[0].as_bytes()[0].is_ascii_alphabetic() {
                continue;
            }
            let row_letter = tokens[0].as_bytes()[0].to_ascii_uppercase();

            if tokens.len() != expected_cols + 1 {
                return Err(MatrixError::WrongRowWidth {
                    row: row_letter as char,
                    got: tokens.len() - 1,
                    expected: expected_cols,
                });
            }

            let mut row = Vec::with_capacity(expected_cols);
            for &tok in &tokens[1..] {
                match tok.parse::<f32>() {
                    Ok(v) => row.push(v),
                    Err(_) => {
                        return Err(MatrixError::BadScore {
                            row: row_letter as char,
                            token: tok.to_owned(),
                        })
                    }
                }
            }
            rows.push((row_letter, row));
        }

        let header = alphabet_header.ok_or(MatrixError::MissingAlphabetHeader)?;
        if rows.len() != header.len() {
            return Err(MatrixError::TruncatedMatrix { expected: header.len() });
        }

        // Verify row label order matches header column order (upstream files all do).
        for (i, (row_letter, _)) in rows.iter().enumerate() {
            if *row_letter != header[i] {
                return Err(MatrixError::RowLabelMismatch {
                    row: *row_letter as char,
                    expected: header[i] as char,
                });
            }
        }

        if let Some(ref bg) = background {
            // Upstream writes N+1 background frequencies (N standard letters
            // followed by a 0.00001 sentinel for X). `parse_comment` strips
            // the trailing sentinel so we accept either shape.
            if bg.len() != header.len() && bg.len() + 1 != header.len() {
                return Err(MatrixError::BackgroundMismatch {
                    got: bg.len(),
                    expected: header.len(),
                });
            }
        }

        // Reorder rows+cols into the provided alphabet's index space, so
        // `scores[i*N + j]` looks up `score(alphabet.decode(i), alphabet.decode(j))`.
        let n = alphabet.size();
        if header.len() != n {
            return Err(MatrixError::AlphabetSizeMismatch {
                header_size: header.len(),
                alphabet_size: n,
            });
        }
        let mut header_idx = Vec::with_capacity(n);
        for &c in &header {
            header_idx.push(alphabet.encode(c));
        }

        let mut scores = vec![0.0f32; n * n];
        for (row_pos, (_, row_values)) in rows.iter().enumerate() {
            let row_i = header_idx[row_pos] as usize;
            for (col_pos, &v) in row_values.iter().enumerate() {
                let col_i = header_idx[col_pos] as usize;
                scores[row_i * n + col_i] = v;
            }
        }

        Ok(Self { name: name.into(), alphabet, scores, background, lambda })
    }

    /// Parse an `.out` file from disk.
    pub fn from_file(path: impl AsRef<Path>, alphabet: Alphabet) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("matrix")
            .to_owned();
        Self::parse(name, alphabet, &text)
    }

    /// BLOSUM62 from the vendored `src/builtin/blosum62.out`.
    pub fn blosum62() -> Self {
        Self::parse("blosum62", Alphabet::protein(), BLOSUM62_OUT)
            .expect("vendored blosum62.out must parse")
    }

    /// Default nucleotide matrix from the vendored `src/builtin/nucleotide.out`.
    pub fn nucleotide() -> Self {
        Self::parse("nucleotide", Alphabet::nucleotide(), NUCLEOTIDE_OUT)
            .expect("vendored nucleotide.out must parse")
    }

    pub fn size(&self) -> usize {
        self.alphabet.size()
    }

    /// `score(i, j)` for alphabet indices `i`, `j`.
    pub fn score(&self, i: u8, j: u8) -> f32 {
        self.scores[i as usize * self.size() + j as usize]
    }

    /// `score(a, b)` for ASCII letters, applying the alphabet's folding rules.
    pub fn score_chars(&self, a: u8, b: u8) -> f32 {
        self.score(self.alphabet.encode(a), self.alphabet.encode(b))
    }

    /// Convert to an integer matrix scaled by `bit_factor` with optional bias.
    ///
    /// Mirrors upstream `BaseMatrix::generateSubMatrix`'s rounding path but
    /// without any lambda recalculation — it operates directly on the float
    /// values read from the `.out` file, which are already in half-bit units.
    /// The resulting `i8` scores are what the prefilter / ungapped aligner use.
    pub fn to_integer_matrix(&self, bit_factor: f32, bias: f32) -> Vec<i8> {
        self.scores
            .iter()
            .map(|&v| clamp_round_i8(v * bit_factor + bias))
            .collect()
    }
}

fn parse_comment(rest: &str, background: &mut Option<Vec<f32>>, lambda: &mut Option<f32>) {
    // Upstream patterns:
    //   "# Background (precomputed optional): f1 f2 ... fN"
    //   "# Lambda     (precomputed optional): f"
    let rest = rest.trim_start();
    if let Some(after) = rest
        .strip_prefix("Background (precomputed optional):")
        .or_else(|| rest.strip_prefix("background (precomputed optional):"))
    {
        let values: Vec<f32> = after
            .split_ascii_whitespace()
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        if !values.is_empty() {
            // Upstream appends a sentinel 0.00001 after the N letter frequencies
            // (see any .out file). Strip it so bg matches alphabet size N, not N+1.
            let mut bg = values;
            if bg.len() > 1 && *bg.last().unwrap() <= 1e-3 {
                bg.pop();
            }
            *background = Some(bg);
        }
    } else if let Some(after) = rest.strip_prefix("Lambda") {
        if let Some(after) = after.trim_start().strip_prefix("(precomputed optional):") {
            if let Some(tok) = after.split_ascii_whitespace().next() {
                if let Ok(v) = tok.parse::<f32>() {
                    *lambda = Some(v);
                }
            }
        }
    }
}

fn clamp_round_i8(v: f32) -> i8 {
    let r = v.round();
    if r >= 127.0 {
        127
    } else if r <= -128.0 {
        -128
    } else {
        r as i8
    }
}

// Suppress unused-constant warnings when the constants are consumed only by tests.
#[allow(dead_code)]
const _CANONICAL_PROTEIN: &[u8] = PROTEIN_LETTERS;
#[allow(dead_code)]
const _CANONICAL_NUCLEOTIDE: &[u8] = NUCLEOTIDE_LETTERS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosum62_diagonal_matches_file() {
        // Hand-verified against data/blosum62.out row/col:
        //   A,A=3.9291  C,C=8.5821  D,D=5.7742  E,E=4.9028  F,F=6.0461
        //   G,G=5.5633  H,H=7.5111  I,I=3.9985  K,K=4.5046  L,L=3.8494
        //   M,M=5.3926  N,N=5.6532  P,P=7.3646  Q,Q=5.2851  R,R=5.4735
        //   S,S=3.8844  T,T=4.5453  V,V=3.7689  W,W=10.5040 Y,Y=6.5950
        //   X,X=-1.0000
        let m = SubstitutionMatrix::blosum62();
        let expected = [
            (b'A', 3.9291), (b'C', 8.5821), (b'D', 5.7742), (b'E', 4.9028),
            (b'F', 6.0461), (b'G', 5.5633), (b'H', 7.5111), (b'I', 3.9985),
            (b'K', 4.5046), (b'L', 3.8494), (b'M', 5.3926), (b'N', 5.6532),
            (b'P', 7.3646), (b'Q', 5.2851), (b'R', 5.4735), (b'S', 3.8844),
            (b'T', 4.5453), (b'V', 3.7689), (b'W', 10.5040), (b'Y', 6.5950),
            (b'X', -1.0000),
        ];
        for (c, want) in expected {
            let got = m.score_chars(c, c);
            assert!(
                (got - want).abs() < 1e-3,
                "BLOSUM62 {c}/{c}: got {got}, want {want}",
                c = c as char,
            );
        }
    }

    #[test]
    fn blosum62_known_off_diagonals() {
        // Spot-check asymmetric-looking but actually symmetric pairs that BLOSUM62
        // should have. File values: W,Y=2.1542; Y,W=2.1542 (symmetric).
        let m = SubstitutionMatrix::blosum62();
        assert!((m.score_chars(b'W', b'Y') - 2.1542).abs() < 1e-3);
        assert!((m.score_chars(b'Y', b'W') - 2.1542).abs() < 1e-3);
        // File: A,S=1.1158; S,A=1.1158
        assert!((m.score_chars(b'A', b'S') - 1.1158).abs() < 1e-3);
        assert!((m.score_chars(b'S', b'A') - 1.1158).abs() < 1e-3);
        // File: D,E=1.5103 (closely related acidic pair)
        assert!((m.score_chars(b'D', b'E') - 1.5103).abs() < 1e-3);
    }

    #[test]
    fn blosum62_is_symmetric() {
        let m = SubstitutionMatrix::blosum62();
        let n = m.size();
        for i in 0..n as u8 {
            for j in 0..n as u8 {
                assert!(
                    (m.score(i, j) - m.score(j, i)).abs() < 1e-6,
                    "BLOSUM62 not symmetric at ({i},{j}): {} vs {}",
                    m.score(i, j),
                    m.score(j, i),
                );
            }
        }
    }

    #[test]
    fn blosum62_has_background_and_lambda() {
        let m = SubstitutionMatrix::blosum62();
        // From the .out header: 21 background frequencies (20 AAs + X sentinel).
        // Our parser strips the trailing sentinel, leaving 20 AA frequencies.
        let bg = m.background.as_ref().expect("BLOSUM62 has bg");
        assert_eq!(bg.len(), 20);
        // "0.07422 0.02469 0.05363..." — A frequency.
        assert!((bg[0] - 0.07422).abs() < 1e-4);
        let lambda = m.lambda.expect("BLOSUM62 has lambda");
        assert!((lambda - 0.34657).abs() < 1e-4);
    }

    #[test]
    fn blosum62_folding_matches_diagonal_for_ambiguous() {
        let m = SubstitutionMatrix::blosum62();
        // Ambiguous folding from Alphabet::protein: B → D, Z → E, J → L, U → X, O → X.
        assert_eq!(m.score_chars(b'B', b'B'), m.score_chars(b'D', b'D'));
        assert_eq!(m.score_chars(b'Z', b'Z'), m.score_chars(b'E', b'E'));
        assert_eq!(m.score_chars(b'J', b'J'), m.score_chars(b'L', b'L'));
        assert_eq!(m.score_chars(b'U', b'U'), m.score_chars(b'X', b'X'));
        assert_eq!(m.score_chars(b'O', b'O'), m.score_chars(b'X', b'X'));
    }

    #[test]
    fn nucleotide_builtin_basic_shape() {
        let m = SubstitutionMatrix::nucleotide();
        assert_eq!(m.size(), 5);
        // A,A = 2.0, A,C = -3.0 etc. per nucleotide.out.
        assert!((m.score_chars(b'A', b'A') - 2.0).abs() < 1e-3);
        assert!((m.score_chars(b'A', b'C') - -3.0).abs() < 1e-3);
        assert!((m.score_chars(b'T', b'T') - 2.0).abs() < 1e-3);
        assert!((m.score_chars(b'G', b'G') - 2.0).abs() < 1e-3);
        assert!((m.score_chars(b'X', b'X') - -3.0).abs() < 1e-3);
    }

    #[test]
    fn to_integer_matrix_rounds_correctly() {
        // BLOSUM62 A,A = 3.9291. At bit_factor=2, rounds to round(7.8582) = 8.
        let m = SubstitutionMatrix::blosum62();
        let int_m = m.to_integer_matrix(2.0, 0.0);
        let a_idx = m.alphabet.encode(b'A') as usize;
        let aa = int_m[a_idx * m.size() + a_idx];
        assert_eq!(aa, 8);
    }

    #[test]
    fn parses_file_from_disk() {
        // Round-trip via the builtin path; filenames-as-names work.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), BLOSUM62_OUT).unwrap();
        let m = SubstitutionMatrix::from_file(tmp.path(), Alphabet::protein()).unwrap();
        // Name derives from filename stem, not the hardcoded string.
        assert!(!m.name.is_empty());
        assert!((m.score_chars(b'A', b'A') - 3.9291).abs() < 1e-3);
    }

    #[test]
    fn wrong_alphabet_returns_structured_error_not_panic() {
        // Passing the 5-letter nucleotide alphabet to a 21-column protein matrix
        // must surface as a recoverable parse error, not an assertion panic.
        let err = SubstitutionMatrix::parse("blosum62", Alphabet::nucleotide(), BLOSUM62_OUT)
            .expect_err("should fail with structured error");
        match err {
            MatrixError::AlphabetSizeMismatch { header_size, alphabet_size } => {
                assert_eq!(header_size, 21);
                assert_eq!(alphabet_size, 5);
            }
            other => panic!("expected AlphabetSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn parses_pam30_tab_separated_header() {
        // PAM30.out uses tabs between header letters; our whitespace tokenizer should handle it.
        let pam30 = "# PAM30 in 1/2 Bit\n\
                     \tA\tC\tD\tE\tF\tG\tH\tI\tK\tL\tM\tN\tP\tQ\tR\tS\tT\tV\tW\tY\tX\n\
                     A\t6\t-6\t-3\t-2\t-8\t-2\t-7\t-5\t-7\t-6\t-5\t-4\t-2\t-4\t-7\t0\t-1\t-2\t-13\t-8\t-3\n";
        // Minimal fake: only one row; we expect error because rows < header.
        let err = SubstitutionMatrix::parse("pam30", Alphabet::protein(), pam30).unwrap_err();
        assert!(matches!(err, MatrixError::TruncatedMatrix { expected: 21 }));
    }
}
