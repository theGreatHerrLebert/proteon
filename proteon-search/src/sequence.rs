//! Encoded biological sequences over an [`Alphabet`].
//!
//! Sequences are stored as alphabet indices (one `u8` per residue) rather
//! than as raw ASCII. This is the representation the prefilter and aligner
//! consume: encoding once at load time and then treating residues as small
//! integers is what lets upstream's k-mer index + SIMD scoring paths fly.

use crate::alphabet::Alphabet;

/// A biological sequence encoded as alphabet indices.
///
/// `data[i]` is the alphabet index of the `i`-th residue. For the standard
/// protein alphabet built by [`Alphabet::protein`], every byte is in `0..=20`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub alphabet: Alphabet,
    pub data: Vec<u8>,
}

impl Sequence {
    /// Encode an ASCII residue string. Unknown bytes (including `*`) fold
    /// per the alphabet (typically to `X`). Whitespace is discarded so FASTA
    /// payloads with embedded line breaks can be passed in directly.
    ///
    /// `*` is *not* dropped: in FASTA it commonly marks a terminal stop codon,
    /// but silently deleting it would shift every downstream residue's
    /// position and change k-mer / alignment coordinates. Callers that
    /// genuinely want to strip terminal stops should do so explicitly before
    /// calling this.
    pub fn from_ascii(alphabet: Alphabet, ascii: &[u8]) -> Self {
        let mut data = Vec::with_capacity(ascii.len());
        for &c in ascii {
            if c.is_ascii_whitespace() {
                continue;
            }
            data.push(alphabet.encode(c));
        }
        Self { alphabet, data }
    }

    /// Round-trip back to canonical ASCII letters. Characters that folded to
    /// a canonical letter on the way in (e.g. `B → D`, `u → X`) come out as
    /// the canonical letter, not the original input.
    pub fn to_ascii(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&idx| self.alphabet.decode(idx))
            .collect()
    }

    /// Length in residues.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over `(position, index)` pairs. Convenient for k-mer scanning.
    pub fn iter_indices(&self) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.data.iter().copied().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protein_round_trip_canonical_letters() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha.clone(), b"ACDEFGHIKLMNPQRSTVWY");
        assert_eq!(s.len(), 20);
        assert_eq!(s.to_ascii(), b"ACDEFGHIKLMNPQRSTVWY");
    }

    #[test]
    fn protein_folds_ambiguous_letters() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha, b"BZJUO");
        // B→D, Z→E, J→L, U→X, O→X
        assert_eq!(s.to_ascii(), b"DELXX");
    }

    #[test]
    fn protein_discards_whitespace_only() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha, b"A C\tG\nT\n");
        // Whitespace dropped; residues preserved.
        assert_eq!(s.len(), 4);
        assert_eq!(s.to_ascii(), b"ACGT");
    }

    #[test]
    fn protein_star_folds_to_x_without_dropping() {
        // `*` in FASTA is often a terminal stop codon. Dropping it would shift
        // every downstream position — wrong for k-mer indexing and alignment.
        // It must fold through the alphabet like any other unknown byte.
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha.clone(), b"ACGT*");
        assert_eq!(s.len(), 5, "`*` must not shift positions");
        assert_eq!(s.data[4], alpha.encode(b'X'));
        assert_eq!(s.to_ascii(), b"ACGTX");
    }

    #[test]
    fn protein_star_mid_sequence_preserves_positions() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha.clone(), b"AC*GT");
        assert_eq!(s.len(), 5);
        // Index 2 must be X (from *), not the G that would appear if * were dropped.
        assert_eq!(s.data[2], alpha.encode(b'X'));
        assert_eq!(s.data[3], alpha.encode(b'G'));
        assert_eq!(s.to_ascii(), b"ACXGT");
    }

    #[test]
    fn protein_lowercase_uppercases() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha, b"mavgtacrpa");
        assert_eq!(s.to_ascii(), b"MAVGTACRPA");
    }

    #[test]
    fn nucleotide_round_trip() {
        let alpha = Alphabet::nucleotide();
        let s = Sequence::from_ascii(alpha, b"ACGTU");
        // U → T in our nucleotide alphabet
        assert_eq!(s.to_ascii(), b"ACGTT");
    }

    #[test]
    fn encoded_indices_match_alphabet() {
        let alpha = Alphabet::protein();
        let s = Sequence::from_ascii(alpha.clone(), b"AMXW");
        let expected: Vec<u8> = b"AMXW".iter().map(|&c| alpha.encode(c)).collect();
        assert_eq!(s.data, expected);
    }
}
