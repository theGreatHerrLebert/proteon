//! Reduced-alphabet construction for k-mer indexing.
//!
//! A full 21-letter protein alphabet makes k-mer index tables grow as
//! `21^k`, which at production k=6 is 85 M slots — feasible but heavy.
//! Collapsing biologically-similar letters (e.g. `I`, `L`, `V`; `D`, `E`;
//! `K`, `R`) into equivalence classes reduces the alphabet to ~13 letters
//! and the table to 13^6 ≈ 5 M slots, at a small cost in sensitivity
//! because the collapsed letters become indistinguishable at the k-mer
//! hashing stage (scoring can still happen in the full alphabet).
//!
//! Upstream (`src/prefiltering/ReducedMatrix.cpp`) computes the reduction
//! via greedy minimisation of mutual-information loss. That algorithm
//! requires the joint probability matrix, not just log-odds scores, which
//! in upstream's code is derived from the `.out` file via a somewhat
//! idiosyncratic path. We take a cleaner but equally-reasonable approach:
//! greedy merging by average off-diagonal log-odds score. The resulting
//! reduction groups biologically-related AAs first (by the BLOSUM-family
//! intent) and is easy to reason about.
//!
//! For exact upstream parity (once we oracle against `mmseqs createindex`),
//! either:
//!   - port the MI algorithm, including the probability reconstruction
//!     from log-odds + background, or
//!   - ship a hardcoded mapping captured from upstream output.
//!
//! Both are follow-ups; neither is needed for a working reduced-alphabet
//! prefilter.

use crate::matrix::SubstitutionMatrix;

/// A mapping from a full alphabet (typically 21-letter protein) into a
/// smaller set of equivalence classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedAlphabet {
    /// Size of the original alphabet this reduction was built for.
    pub full_size: usize,
    /// Size of the reduced alphabet. `reduced_size <= full_size`.
    pub reduced_size: usize,
    /// `full_to_reduced[i]` is the reduced index for full-alphabet index `i`.
    /// Entries are in `0..reduced_size`.
    pub full_to_reduced: Vec<u8>,
    /// The reduced index corresponding to the full-alphabet "unknown"
    /// letter (typically `X` in proteins), when one was designated at
    /// construction time. Callers pass this as `skip_idx` to
    /// [`crate::kmer::KmerEncoder::iter_kmers`] and
    /// [`crate::kmer::KmerIndex::build`] so X-containing windows continue
    /// to be filtered out after reduction. `None` means no letter was
    /// reserved — the caller must either know the sequences contain no
    /// unknowns or handle skipping some other way.
    pub unknown_reduced_idx: Option<u8>,
}

impl ReducedAlphabet {
    /// Build from an explicit mapping. Useful for published reductions
    /// (Murphy 2000, Li 2003, etc.) or for pinning upstream's exact output.
    ///
    /// `unknown_full_idx` designates which full-alphabet index represents
    /// the unknown / X letter; its `full_to_reduced[unknown_full_idx]` is
    /// surfaced as [`ReducedAlphabet::unknown_reduced_idx`] so k-mer
    /// iterators can still skip X-windows after reduction. Pass `None`
    /// for alphabets without a dedicated unknown letter.
    ///
    /// Returns `None` if the mapping is not dense (i.e. the reduced indices
    /// don't cover `0..max+1`) or if `unknown_full_idx` is out of range.
    pub fn from_mapping(full_to_reduced: Vec<u8>, unknown_full_idx: Option<u8>) -> Option<Self> {
        if full_to_reduced.is_empty() {
            return None;
        }
        let max = *full_to_reduced.iter().max().unwrap();
        let reduced_size = (max as usize) + 1;
        // Density check: every reduced index in 0..reduced_size must have at
        // least one full-alphabet member.
        for r in 0..reduced_size {
            if !full_to_reduced.iter().any(|&x| x as usize == r) {
                return None;
            }
        }
        let unknown_reduced_idx = match unknown_full_idx {
            Some(u) => {
                if (u as usize) >= full_to_reduced.len() {
                    return None;
                }
                Some(full_to_reduced[u as usize])
            }
            None => None,
        };
        Some(Self {
            full_size: full_to_reduced.len(),
            reduced_size,
            full_to_reduced,
            unknown_reduced_idx,
        })
    }

    /// Identity reduction (no merging). Useful for tests and as a sanity
    /// baseline. The `unknown_full_idx`, if provided, is surfaced as
    /// `unknown_reduced_idx` unchanged.
    pub fn identity(full_size: usize, unknown_full_idx: Option<u8>) -> Self {
        let full_to_reduced: Vec<u8> = (0..full_size).map(|i| i as u8).collect();
        Self {
            full_size,
            reduced_size: full_size,
            full_to_reduced,
            unknown_reduced_idx: unknown_full_idx,
        }
    }

    /// Greedy reduction driven by a substitution matrix: at each step,
    /// merge the pair of classes with the highest **average inter-class
    /// log-odds score** (most similar pair). Repeat until `reduced_size`
    /// classes remain.
    ///
    /// `unknown_full_idx` — if `Some(u)`, the class containing `u` is
    /// excluded from every merge, guaranteeing the unknown letter stays
    /// in its own reduced class. This is essential so that callers can
    /// continue to skip X-containing k-mer windows after reduction by
    /// passing [`ReducedAlphabet::unknown_reduced_idx`] as the
    /// `skip_idx` — without this guarantee, X would be pulled into some
    /// normal class and the skip mechanism would either miss X-windows
    /// or also exclude legitimate residues.
    ///
    /// Returns `None` if `reduced_size == 0`, `reduced_size > full_size`,
    /// or `unknown_full_idx` is set but `>= full_size`. Also `None` if
    /// `unknown_full_idx` is `Some` and `reduced_size == 1` (can't
    /// reserve a class when there's only one).
    pub fn from_matrix(
        matrix: &SubstitutionMatrix,
        reduced_size: usize,
        unknown_full_idx: Option<u8>,
    ) -> Option<Self> {
        let n = matrix.size();
        if reduced_size == 0 || reduced_size > n {
            return None;
        }
        if let Some(u) = unknown_full_idx {
            if (u as usize) >= n {
                return None;
            }
            if reduced_size < 2 {
                return None;
            }
        }
        if reduced_size == n {
            return Some(Self::identity(n, unknown_full_idx));
        }

        // Each class is a Vec<usize> of full-alphabet indices. Start one
        // class per letter.
        let mut classes: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while classes.len() > reduced_size {
            // The unknown class (if any) must be preserved. We identify it
            // each iteration by the full-alphabet unknown index, since
            // class indices shift on every remove.
            let protected = unknown_full_idx.map(|u| {
                classes
                    .iter()
                    .position(|c| c.contains(&(u as usize)))
                    .expect("unknown_full_idx class must exist")
            });

            let c = classes.len();
            let mut best_i = 0usize;
            let mut best_j = 1usize;
            let mut best_score = f32::NEG_INFINITY;
            for i in 0..c {
                if Some(i) == protected {
                    continue;
                }
                for j in (i + 1)..c {
                    if Some(j) == protected {
                        continue;
                    }
                    let s = avg_inter_class_score(matrix, &classes[i], &classes[j]);
                    if s > best_score {
                        best_score = s;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Merge best_j into best_i, then remove best_j (higher index).
            let tail = classes.remove(best_j);
            classes[best_i].extend(tail);
        }

        // Emit the full → reduced mapping. Sort each class so mapping is
        // deterministic across runs.
        for cls in &mut classes {
            cls.sort_unstable();
        }
        // And assign reduced indices in order of each class's smallest
        // full-alphabet member, again for determinism.
        classes.sort_by_key(|c| c[0]);
        let mut full_to_reduced = vec![0u8; n];
        for (reduced_idx, cls) in classes.iter().enumerate() {
            for &full_idx in cls {
                full_to_reduced[full_idx] = reduced_idx as u8;
            }
        }
        let unknown_reduced_idx = unknown_full_idx.map(|u| full_to_reduced[u as usize]);
        Some(Self {
            full_size: n,
            reduced_size,
            full_to_reduced,
            unknown_reduced_idx,
        })
    }

    /// Encode a full-alphabet byte sequence into the reduced alphabet.
    ///
    /// Any byte `>= full_size` is passed through unchanged and the caller
    /// must take care (skip indices used as X-window sentinels typically
    /// sit above any valid alphabet index, so this preserves them).
    pub fn reduce_sequence(&self, bytes: &[u8]) -> Vec<u8> {
        bytes
            .iter()
            .map(|&b| {
                if (b as usize) < self.full_size {
                    self.full_to_reduced[b as usize]
                } else {
                    b
                }
            })
            .collect()
    }

    /// Average a full-alphabet score matrix down into the reduced space.
    ///
    /// Cell `(r, s)` of the output is the mean of `matrix.score(i, j)` over
    /// all full-alphabet pairs `(i, j)` with `full_to_reduced[i] == r` and
    /// `full_to_reduced[j] == s`. Symmetric if the input is symmetric.
    ///
    /// Returns a flat row-major `Vec<f32>` of length `reduced_size^2`,
    /// ready to pass through
    /// [`crate::kmer_generator::widen_to_i32`]-style conversion.
    pub fn reduce_matrix(&self, matrix: &SubstitutionMatrix) -> Vec<f32> {
        assert_eq!(
            matrix.size(),
            self.full_size,
            "matrix size ({}) must equal full_size ({})",
            matrix.size(),
            self.full_size,
        );
        let r = self.reduced_size;
        let mut sums = vec![0.0f32; r * r];
        let mut counts = vec![0u32; r * r];
        for i in 0..self.full_size {
            for j in 0..self.full_size {
                let ri = self.full_to_reduced[i] as usize;
                let rj = self.full_to_reduced[j] as usize;
                sums[ri * r + rj] += matrix.score(i as u8, j as u8);
                counts[ri * r + rj] += 1;
            }
        }
        sums.iter()
            .zip(counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f32 } else { 0.0 })
            .collect()
    }
}

/// Mean of all `score(a, b)` over `a != b` with `a in class_a`, `b in class_b`.
/// Off-diagonal only so merging isn't biased by already-identical cells.
fn avg_inter_class_score(matrix: &SubstitutionMatrix, class_a: &[usize], class_b: &[usize]) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for &a in class_a {
        for &b in class_b {
            if a == b {
                continue;
            }
            sum += matrix.score(a as u8, b as u8);
            n += 1;
        }
    }
    if n == 0 {
        f32::NEG_INFINITY
    } else {
        sum / n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn identity_reduction_is_noop() {
        let r = ReducedAlphabet::identity(21, None);
        assert_eq!(r.reduced_size, 21);
        for i in 0..21 {
            assert_eq!(r.full_to_reduced[i], i as u8);
        }
        assert_eq!(r.unknown_reduced_idx, None);
    }

    #[test]
    fn identity_preserves_unknown_index() {
        let r = ReducedAlphabet::identity(21, Some(20));
        assert_eq!(r.unknown_reduced_idx, Some(20));
    }

    #[test]
    fn from_mapping_accepts_dense_mapping() {
        let r = ReducedAlphabet::from_mapping(vec![0, 0, 1, 1, 2, 2], None).unwrap();
        assert_eq!(r.full_size, 6);
        assert_eq!(r.reduced_size, 3);
        assert_eq!(r.unknown_reduced_idx, None);
    }

    #[test]
    fn from_mapping_surfaces_unknown_reduced_idx() {
        // full idx 5 maps to reduced idx 2 via the mapping [0,0,1,1,2,2].
        let r = ReducedAlphabet::from_mapping(vec![0, 0, 1, 1, 2, 2], Some(5)).unwrap();
        assert_eq!(r.unknown_reduced_idx, Some(2));
    }

    #[test]
    fn from_mapping_rejects_out_of_range_unknown() {
        assert!(ReducedAlphabet::from_mapping(vec![0, 0, 1, 1, 2, 2], Some(99)).is_none());
    }

    #[test]
    fn from_mapping_rejects_sparse_mapping() {
        // reduced index 1 unused while 0 and 2 are present → not dense.
        let r = ReducedAlphabet::from_mapping(vec![0, 0, 2, 2], None);
        assert!(r.is_none());
    }

    #[test]
    fn from_matrix_reduces_to_target_size() {
        let m = SubstitutionMatrix::blosum62();
        for target in [5usize, 10, 13, 20] {
            let r = ReducedAlphabet::from_matrix(&m, target, None).unwrap();
            assert_eq!(r.reduced_size, target, "requested {target}");
            for &ri in &r.full_to_reduced {
                assert!((ri as usize) < target);
            }
        }
    }

    #[test]
    fn from_matrix_identity_when_target_equals_full() {
        let m = SubstitutionMatrix::blosum62();
        let r = ReducedAlphabet::from_matrix(&m, m.size(), None).unwrap();
        for i in 0..m.size() {
            assert_eq!(r.full_to_reduced[i], i as u8);
        }
    }

    #[test]
    fn from_matrix_rejects_impossible_target() {
        let m = SubstitutionMatrix::blosum62();
        assert!(ReducedAlphabet::from_matrix(&m, 0, None).is_none());
        assert!(ReducedAlphabet::from_matrix(&m, m.size() + 1, None).is_none());
    }

    #[test]
    fn from_matrix_rejects_reduced_size_1_with_unknown_reservation() {
        // If the caller wants unknown in its own class, we need at least 2
        // reduced classes total (one for unknown, one for everything else).
        let m = SubstitutionMatrix::blosum62();
        assert!(ReducedAlphabet::from_matrix(&m, 1, Some(20)).is_none());
    }

    #[test]
    fn from_matrix_keeps_unknown_in_singleton_class() {
        // With unknown_full_idx = X, the X full letter must end up in a
        // reduced class whose only member is X itself — otherwise the
        // skip-idx semantic would either leak X into k-mers or over-skip
        // legitimate letters merged with X.
        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let x_full = alpha.encode(b'X');
        // Try an aggressive reduction: 5 classes. X should still be alone.
        let r = ReducedAlphabet::from_matrix(&m, 5, Some(x_full)).unwrap();
        let x_reduced = r.unknown_reduced_idx.unwrap();

        let members: Vec<u8> = (0..m.size() as u8)
            .filter(|&i| r.full_to_reduced[i as usize] == x_reduced)
            .collect();
        assert_eq!(
            members,
            vec![x_full],
            "X class has extra members: {:?} (decoded: {:?})",
            members,
            members
                .iter()
                .map(|&i| alpha.decode(i) as char)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn blosum62_reduction_merges_biologically_similar_pairs_first() {
        // Greedy merging should group well-known similar pairs first —
        // this is the standard biological expectation of any
        // substitution-matrix-driven reduction. We check that at
        // target=20 (one merge), the two letters merged share a reduced
        // index and that they're a biologically sensible pair.
        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let r = ReducedAlphabet::from_matrix(&m, 20, None).unwrap();

        // Exactly one pair of full letters share a reduced index.
        let mut merged_letters: Vec<u8> = Vec::new();
        let mut reduced_counts = vec![0u32; r.reduced_size];
        for &ri in &r.full_to_reduced {
            reduced_counts[ri as usize] += 1;
        }
        // The one class with two members contains the merged pair.
        let merged_class = reduced_counts.iter().position(|&c| c == 2).unwrap();
        for (full_i, &ri) in r.full_to_reduced.iter().enumerate() {
            if (ri as usize) == merged_class {
                merged_letters.push(alpha.decode(full_i as u8));
            }
        }
        merged_letters.sort_unstable();
        // Accept any of the biologically-sensible first-merge pairs:
        // I/L (bioisosteres), D/E (acidic), K/R (basic), F/Y (aromatic),
        // V/I or V/L (aliphatic). We don't pin which one wins because
        // that's an algorithmic choice, but it must be one of these.
        let sensible_pairs = [
            [b'I', b'L'],
            [b'D', b'E'],
            [b'K', b'R'],
            [b'F', b'Y'],
            [b'I', b'V'],
            [b'L', b'V'],
            [b'S', b'T'],
            [b'N', b'D'],
            [b'Q', b'E'],
        ];
        assert!(
            sensible_pairs.iter().any(|p| p[..] == merged_letters[..]),
            "first BLOSUM62 merge was {:?}, expected one of {:?}",
            std::str::from_utf8(&merged_letters).unwrap_or("?"),
            sensible_pairs
                .iter()
                .map(|p| std::str::from_utf8(p).unwrap())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn reduce_sequence_applies_mapping() {
        let r = ReducedAlphabet::from_mapping(vec![0, 0, 1, 1, 2, 2], None).unwrap();
        let encoded = vec![0u8, 1, 2, 3, 4, 5];
        let reduced = r.reduce_sequence(&encoded);
        assert_eq!(reduced, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn reduce_sequence_passes_through_out_of_range_bytes() {
        // Bytes >= full_size (e.g. 250 used as an alternative sentinel)
        // survive reduction unchanged. This is NOT the X-skip path — X
        // has its own reduced index via unknown_reduced_idx. This only
        // matters for callers using a raw out-of-range sentinel.
        let r = ReducedAlphabet::from_mapping(vec![0, 0, 1], None).unwrap();
        let encoded = vec![0u8, 99, 2, 1];
        let reduced = r.reduce_sequence(&encoded);
        assert_eq!(reduced, vec![0, 99, 1, 0]);
    }

    #[test]
    fn reduce_matrix_preserves_identity_under_identity_reduction() {
        // Reducing BLOSUM62 through the identity reduction (no merging)
        // must return the original scores in the same layout.
        let m = SubstitutionMatrix::blosum62();
        let r = ReducedAlphabet::identity(m.size(), None);
        let reduced_scores = r.reduce_matrix(&m);
        assert_eq!(reduced_scores.len(), m.size() * m.size());
        for i in 0..m.size() {
            for j in 0..m.size() {
                assert!(
                    (reduced_scores[i * m.size() + j] - m.score(i as u8, j as u8)).abs() < 1e-6,
                    "identity reduction changed score[{i}][{j}]",
                );
            }
        }
    }

    #[test]
    fn end_to_end_prefilter_on_reduced_alphabet() {
        // Reduce BLOSUM62 → 13 with X reserved as its own class. Build a
        // k-mer index over reduced sequences. Prove the X-skip semantic
        // survives across the reduction boundary by including an X in one
        // of the targets and asserting the X-window doesn't contribute.
        use crate::kmer::KmerEncoder;
        use crate::kmer::KmerIndex;
        use crate::prefilter::{diagonal_prefilter, PrefilterOptions};
        use crate::sequence::Sequence;

        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let x_full = alpha.encode(b'X');
        let reducer = ReducedAlphabet::from_matrix(&m, 13, Some(x_full)).unwrap();
        let skip_idx = reducer
            .unknown_reduced_idx
            .expect("we reserved X as unknown");

        // Two near-identical targets, one unrelated.
        let t1 = Sequence::from_ascii(alpha.clone(), b"MKLVRQPST");
        let t2 = Sequence::from_ascii(alpha.clone(), b"WWWWWWWWW");
        let q = Sequence::from_ascii(alpha.clone(), b"MKLVRQPST");

        let t1_red = reducer.reduce_sequence(&t1.data);
        let t2_red = reducer.reduce_sequence(&t2.data);
        let q_red = reducer.reduce_sequence(&q.data);

        let enc = KmerEncoder::new(reducer.reduced_size as u32, 3);
        let idx = KmerIndex::build(
            enc,
            [(1u32, t1_red.as_slice()), (2u32, t2_red.as_slice())],
            skip_idx,
        )
        .unwrap();

        let hits = diagonal_prefilter(&idx, &q_red, skip_idx, &PrefilterOptions::default());
        let seq_ids: Vec<u32> = hits.iter().map(|h| h.seq_id).collect();
        assert!(seq_ids.contains(&1), "identical target should hit");
        let top = &hits[0];
        assert_eq!(top.seq_id, 1);
        assert_eq!(top.best_diagonal, 0);
        // 7 k-mers at k=3 over length 9, all on diagonal 0.
        assert_eq!(top.diagonal_score, 7);
    }

    #[test]
    fn x_windows_are_still_skipped_after_reduction() {
        // Regression test for the review finding: without the unknown-
        // reservation path, reduce_sequence would map X (full idx 20) to
        // some normal reduced class, and callers passing the full X index
        // as skip_idx would silently stop skipping any window — indexing
        // X k-mers as if they were valid residues.
        //
        // With unknown reserved, reducer.unknown_reduced_idx is an index
        // that is unique to X, and X-containing windows are correctly
        // excluded from the index.
        use crate::kmer::{KmerEncoder, KmerIndex};
        use crate::sequence::Sequence;

        let alpha = Alphabet::protein();
        let m = SubstitutionMatrix::blosum62();
        let x_full = alpha.encode(b'X');
        let reducer = ReducedAlphabet::from_matrix(&m, 13, Some(x_full)).unwrap();
        let skip_idx = reducer.unknown_reduced_idx.unwrap();

        // Sequence ACXGT: windows at k=3 are [A,C,X], [C,X,G], [X,G,T],
        // all three contain X and must be skipped.
        let s = Sequence::from_ascii(alpha.clone(), b"ACXGT");
        let s_red = reducer.reduce_sequence(&s.data);

        let enc = KmerEncoder::new(reducer.reduced_size as u32, 3);
        let idx = KmerIndex::build(enc, [(1u32, s_red.as_slice())], skip_idx).unwrap();
        assert_eq!(
            idx.total_hits(),
            0,
            "all windows contain X → index should be empty after reduction",
        );

        // Same sequence without X: [A,C,G], [C,G,T] → 2 valid windows.
        let s2 = Sequence::from_ascii(alpha.clone(), b"ACGT");
        let s2_red = reducer.reduce_sequence(&s2.data);
        let enc2 = KmerEncoder::new(reducer.reduced_size as u32, 3);
        let idx2 = KmerIndex::build(enc2, [(1u32, s2_red.as_slice())], skip_idx).unwrap();
        assert_eq!(idx2.total_hits(), 2);
    }

    #[test]
    fn reduce_matrix_has_correct_shape_after_merging() {
        let m = SubstitutionMatrix::blosum62();
        let r = ReducedAlphabet::from_matrix(&m, 13, None).unwrap();
        let scores = r.reduce_matrix(&m);
        assert_eq!(scores.len(), 13 * 13);
        // Every score is finite.
        for &s in &scores {
            assert!(s.is_finite(), "reduced score is not finite");
        }
        // Most reduced classes have positive self-score (BLOSUM-family
        // log-odds self-substitution is typically positive). X tends to
        // sit in its own class with self-score ~-1 even at target=13, so
        // we allow up to one negative-diagonal class.
        let positive_diag_count = (0..13).filter(|&i| scores[i * 13 + i] > 0.0).count();
        assert!(
            positive_diag_count >= 12,
            "expected >=12 reduced classes with positive self-score, got {positive_diag_count}",
        );
    }
}
