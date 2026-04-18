//! MSA assembly: search results → AF2-style input tensors.
//!
//! References:
//! - Steinegger & Söding, *Nat. Biotechnol.* 35, 1026-1028 (2017) — the
//!   MMseqs2 search results this module consumes.
//! - Jumper et al., "Highly accurate protein structure prediction with
//!   AlphaFold", *Nature* 596, 583-589 (2021) — the MSA feature shapes
//!   and semantics this module produces.
//!
//! Bridges [`SearchEngine`](crate::search::SearchEngine)'s output (a list
//! of [`SearchHit`] with CIGAR alignments) to the four NumPy-shaped
//! tensors AlphaFold-style training pipelines consume:
//!
//! - `aatype` — query residues, encoded.
//! - `seq_mask` — `1.0` for every valid query position.
//! - `msa` — `[n_seqs, query_len]` of alphabet indices, query at row 0,
//!   homologs projected into query coordinates with `gap_idx` for any
//!   query position the homolog doesn't cover.
//! - `deletion_matrix` — `[n_seqs, query_len]` counts of homolog
//!   residues inserted *before* each query position (the AF2 convention
//!   for representing residues that fall in the homolog but have no
//!   corresponding column in the query).
//! - `msa_mask` — `[n_seqs, query_len]` of `1.0` where the row carries
//!   information at that column.
//!
//! No tensor library here — plain `Vec<Vec<_>>` so the consumer (Python
//! via PyO3, Arrow, or raw bytes) decides the destination layout.
//! Heavy-callers can stream rows out as they're built later if memory
//! becomes a concern.

use crate::gapped::CigarOp;
use crate::search::SearchHit;
use crate::sequence::Sequence;

#[derive(Debug, Clone)]
pub struct MsaOptions {
    /// Max rows in the assembled MSA, **including** the query at row 0.
    /// Excess hits beyond `max_seqs - 1` are dropped from the bottom of
    /// the (already-sorted-by-score) hit list.
    pub max_seqs: usize,
    /// Index used to represent gaps in the encoded MSA. Convention is
    /// `alphabet.size()` (one past the last canonical letter), so
    /// downstream embedding tables size to `alphabet.size() + 1`.
    pub gap_idx: u8,
}

impl Default for MsaOptions {
    fn default() -> Self {
        Self {
            max_seqs: 256,
            gap_idx: 21,
        }
    }
}

/// Result of MSA assembly: the four tensors a downstream AF2-style
/// pipeline expects, plus shape metadata.
#[derive(Debug, Clone)]
pub struct MsaAssembly {
    pub query_len: usize,
    pub n_seqs: usize,
    pub gap_idx: u8,
    /// `[query_len]` — query residues, alphabet-encoded.
    pub aatype: Vec<u8>,
    /// `[query_len]` — `1.0` for every valid query position.
    pub seq_mask: Vec<f32>,
    /// `[n_seqs, query_len]` — row 0 is the query, subsequent rows are
    /// homologs in query coordinates with `gap_idx` for unaligned
    /// positions.
    pub msa: Vec<Vec<u8>>,
    /// `[n_seqs, query_len]` — `deletion_matrix[s][q]` is the count of
    /// homolog residues inserted in row `s` immediately before query
    /// position `q`. Row 0 (query) is all zeros by construction.
    pub deletion_matrix: Vec<Vec<u8>>,
    /// `[n_seqs, query_len]` — `1.0` where the row contributes
    /// information at that column. Row 0 is all 1.0; subsequent rows
    /// are 1.0 inside the alignment range and 0.0 outside.
    pub msa_mask: Vec<Vec<f32>>,
}

/// Assemble an AF2-style MSA from a query and a ranked list of hits.
///
/// `target_lookup(seq_id) -> Option<&[u8]>` returns the target's
/// alphabet-encoded byte slice or `None` if the id is unknown (the hit
/// is silently skipped — the engine should never produce ids it can't
/// resolve, but defensive lookup keeps this composable).
///
/// Hit order is preserved: hits earlier in the slice become earlier MSA
/// rows. Callers passing a `SearchEngine::search` result get hits sorted
/// by gapped alignment score descending, which is also a sensible MSA
/// ordering.
pub fn assemble_msa<'a, F>(
    query: &Sequence,
    hits: &[SearchHit],
    target_lookup: F,
    opts: MsaOptions,
) -> MsaAssembly
where
    F: Fn(u32) -> Option<&'a [u8]>,
{
    let q_len = query.data.len();
    let gap = opts.gap_idx;

    // Row 0 = query.
    let mut msa: Vec<Vec<u8>> = vec![query.data.clone()];
    let mut deletion_matrix: Vec<Vec<u8>> = vec![vec![0u8; q_len]];
    let mut msa_mask: Vec<Vec<f32>> = vec![vec![1.0f32; q_len]];

    let max_extra = opts.max_seqs.saturating_sub(1);
    for hit in hits.iter().take(max_extra) {
        let target = match target_lookup(hit.target_id) {
            Some(t) => t,
            None => continue,
        };
        let (row_msa, row_del, row_mask) = project_hit_into_query_frame(
            target,
            &hit.alignment.cigar,
            hit.alignment.query_start,
            hit.alignment.query_end,
            hit.alignment.target_start,
            q_len,
            gap,
        );
        msa.push(row_msa);
        deletion_matrix.push(row_del);
        msa_mask.push(row_mask);
    }

    MsaAssembly {
        query_len: q_len,
        n_seqs: msa.len(),
        gap_idx: gap,
        aatype: query.data.clone(),
        seq_mask: vec![1.0f32; q_len],
        msa,
        deletion_matrix,
        msa_mask,
    }
}

/// Walk the CIGAR for a single hit and emit the three per-row vectors.
///
/// Returns `(row_msa, row_deletion, row_mask)`, each of length `q_len`.
/// Positions outside `[query_start, query_end)` get `gap_idx` / `0` /
/// `0.0` respectively.
fn project_hit_into_query_frame(
    target: &[u8],
    cigar: &[CigarOp],
    query_start: usize,
    query_end: usize,
    target_start: usize,
    q_len: usize,
    gap_idx: u8,
) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut row_msa = vec![gap_idx; q_len];
    let mut row_del = vec![0u8; q_len];
    let mut row_mask = vec![0.0f32; q_len];

    // Mark coverage: every position the alignment touches gets mask 1,
    // including positions where the homolog has a gap (Insert op below)
    // — those are "we know this row doesn't cover here," distinct from
    // "the homolog's alignment didn't reach this column at all."
    for q in query_start..query_end {
        row_mask[q] = 1.0;
    }

    let mut q_cursor = query_start;
    let mut t_cursor = target_start;
    // Accumulator for target residues that have no query counterpart;
    // attributed to the next query position we land on (Match or Insert).
    let mut pending_deletions: u8 = 0;

    for op in cigar {
        match *op {
            CigarOp::Match(n) => {
                for i in 0..n {
                    let q = q_cursor + i;
                    let t = t_cursor + i;
                    row_msa[q] = target[t];
                    if i == 0 {
                        row_del[q] = pending_deletions;
                        pending_deletions = 0;
                    }
                }
                q_cursor += n;
                t_cursor += n;
            }
            CigarOp::Insert(n) => {
                // Query has residues here, target has nothing → gap in
                // MSA at these query columns (homolog "doesn't have" a
                // residue aligned to these query positions, even though
                // the row mask reports coverage).
                for i in 0..n {
                    let q = q_cursor + i;
                    row_msa[q] = gap_idx;
                    if i == 0 {
                        row_del[q] = pending_deletions;
                        pending_deletions = 0;
                    }
                }
                q_cursor += n;
            }
            CigarOp::Delete(n) => {
                // Target has residues here, query has nothing → these
                // are "deletions" in the AF2 sense (homolog residues
                // with no query column to land in). Saturate at u8 so
                // a homolog with a 1000-residue insert against a short
                // query stays representable; downstream consumers can
                // upcast if they want exact counts.
                pending_deletions = pending_deletions.saturating_add(if n > u8::MAX as usize {
                    u8::MAX
                } else {
                    n as u8
                });
                t_cursor += n;
            }
        }
    }

    let _ = t_cursor; // satisfy unused-var lint when the trailing ops do nothing
                      // Trailing Delete after the last Match are dropped on the floor —
                      // by construction there's no query column to attribute them to.

    (row_msa, row_del, row_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::gapped::{CigarOp, GappedAlignment};
    use crate::search::SearchHit;

    /// Build a SearchHit matching one of the hand-crafted test alignments.
    fn make_hit(
        target_id: u32,
        score: i32,
        query_start: usize,
        query_end: usize,
        target_start: usize,
        target_end: usize,
        cigar: Vec<CigarOp>,
    ) -> SearchHit {
        SearchHit {
            target_id,
            prefilter_score: 0,
            best_diagonal: 0,
            ungapped_score: score,
            alignment: GappedAlignment {
                score,
                query_start,
                query_end,
                target_start,
                target_end,
                cigar,
            },
        }
    }

    #[test]
    fn empty_hits_yields_query_only_row() {
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha, b"ACDEF");
        let msa = assemble_msa(
            &q,
            &[],
            |_| -> Option<&[u8]> { None },
            MsaOptions::default(),
        );
        assert_eq!(msa.n_seqs, 1);
        assert_eq!(msa.query_len, 5);
        assert_eq!(msa.aatype, q.data);
        assert_eq!(msa.msa[0], q.data);
        assert_eq!(msa.deletion_matrix[0], vec![0; 5]);
        assert_eq!(msa.msa_mask[0], vec![1.0; 5]);
        assert_eq!(msa.seq_mask, vec![1.0; 5]);
    }

    #[test]
    fn full_match_target_appears_as_second_row() {
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"ACDEF");
        let t = Sequence::from_ascii(alpha, b"ACDEF");
        let hit = make_hit(1, 100, 0, 5, 0, 5, vec![CigarOp::Match(5)]);
        let msa = assemble_msa(
            &q,
            &[hit],
            |id| {
                if id == 1 {
                    Some(t.data.as_slice())
                } else {
                    None
                }
            },
            MsaOptions::default(),
        );
        assert_eq!(msa.n_seqs, 2);
        assert_eq!(msa.msa[1], q.data);
        assert_eq!(msa.deletion_matrix[1], vec![0; 5]);
        assert_eq!(msa.msa_mask[1], vec![1.0; 5]);
    }

    #[test]
    fn target_with_insertion_records_deletion_count() {
        // query:   A B C D E       (5 residues)
        // target:  A B X C D E     (one extra residue X between B and C)
        // CIGAR:   2M 1D 3M
        // → query column 2 (where C lives) records deletion_count=1
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"ABCDE");
        let t = Sequence::from_ascii(alpha, b"ABXCDE");
        let hit = make_hit(
            1,
            100,
            0,
            5,
            0,
            6,
            vec![CigarOp::Match(2), CigarOp::Delete(1), CigarOp::Match(3)],
        );
        let msa = assemble_msa(
            &q,
            &[hit],
            |id| {
                if id == 1 {
                    Some(t.data.as_slice())
                } else {
                    None
                }
            },
            MsaOptions::default(),
        );
        // MSA row contains the matched target letters (ABCDE), all on the
        // query columns — the inserted X doesn't appear in the MSA matrix.
        let expected_row: Vec<u8> = b"ABCDE".iter().map(|&c| q.alphabet.encode(c)).collect();
        assert_eq!(msa.msa[1], expected_row);
        // Deletion matrix: 0 for query cols 0,1, then 1 at col 2 (the X
        // was inserted before C), then 0 for cols 3,4.
        assert_eq!(msa.deletion_matrix[1], vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn query_insertion_yields_gap_in_msa_row() {
        // query:   A B C D E         (5 residues)
        // target:  A B   D E         (3 matched + 1 gap in target → query has C with no target counterpart)
        // CIGAR:   2M 1I 2M
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"ABCDE");
        let t = Sequence::from_ascii(alpha, b"ABDE");
        let hit = make_hit(
            1,
            50,
            0,
            5,
            0,
            4,
            vec![CigarOp::Match(2), CigarOp::Insert(1), CigarOp::Match(2)],
        );
        let msa = assemble_msa(
            &q,
            &[hit],
            |id| {
                if id == 1 {
                    Some(t.data.as_slice())
                } else {
                    None
                }
            },
            MsaOptions::default(),
        );
        let gap = MsaOptions::default().gap_idx;
        let expected_row: Vec<u8> = vec![
            q.alphabet.encode(b'A'),
            q.alphabet.encode(b'B'),
            gap, // C in query has no target counterpart
            q.alphabet.encode(b'D'),
            q.alphabet.encode(b'E'),
        ];
        assert_eq!(msa.msa[1], expected_row);
        assert_eq!(msa.deletion_matrix[1], vec![0; 5]);
    }

    #[test]
    fn alignment_not_covering_full_query_pads_with_gaps_and_zero_mask() {
        // query length 7, alignment covers query[2..5] only.
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"ABCDEFG");
        let t = Sequence::from_ascii(alpha, b"CDE");
        let hit = make_hit(1, 30, 2, 5, 0, 3, vec![CigarOp::Match(3)]);
        let msa = assemble_msa(
            &q,
            &[hit],
            |id| {
                if id == 1 {
                    Some(t.data.as_slice())
                } else {
                    None
                }
            },
            MsaOptions::default(),
        );
        let gap = MsaOptions::default().gap_idx;
        // Outside [2, 5): gap in MSA, mask 0.
        assert_eq!(msa.msa[1][0], gap);
        assert_eq!(msa.msa[1][1], gap);
        assert_eq!(msa.msa[1][2], q.alphabet.encode(b'C'));
        assert_eq!(msa.msa[1][3], q.alphabet.encode(b'D'));
        assert_eq!(msa.msa[1][4], q.alphabet.encode(b'E'));
        assert_eq!(msa.msa[1][5], gap);
        assert_eq!(msa.msa[1][6], gap);
        assert_eq!(msa.msa_mask[1], vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn max_seqs_caps_total_rows_including_query() {
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"AB");
        let t = Sequence::from_ascii(alpha, b"AB");
        let hits: Vec<SearchHit> = (0..10)
            .map(|id| make_hit(id, 100, 0, 2, 0, 2, vec![CigarOp::Match(2)]))
            .collect();
        let opts = MsaOptions {
            max_seqs: 4,
            gap_idx: 21,
        };
        let msa = assemble_msa(&q, &hits, |_| Some(t.data.as_slice()), opts);
        assert_eq!(msa.n_seqs, 4); // query + 3 hits
    }

    #[test]
    fn unknown_target_id_is_silently_skipped() {
        // Hit references a target_id the lookup doesn't know — the row
        // is dropped, the rest of the MSA is unaffected.
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"AB");
        let t = Sequence::from_ascii(alpha, b"AB");
        let hits = vec![
            make_hit(1, 100, 0, 2, 0, 2, vec![CigarOp::Match(2)]),
            make_hit(99, 100, 0, 2, 0, 2, vec![CigarOp::Match(2)]), // unknown
            make_hit(2, 100, 0, 2, 0, 2, vec![CigarOp::Match(2)]),
        ];
        let lookup = |id: u32| -> Option<&[u8]> {
            match id {
                1 | 2 => Some(t.data.as_slice()),
                _ => None,
            }
        };
        let msa = assemble_msa(&q, &hits, lookup, MsaOptions::default());
        // query + 2 known hits, the unknown id silently dropped.
        assert_eq!(msa.n_seqs, 3);
    }

    #[test]
    fn deletion_count_saturates_at_u8_max() {
        // Pathological: a single Delete op larger than u8::MAX. Saturate
        // rather than wrap.
        let alpha = Alphabet::protein();
        let q = Sequence::from_ascii(alpha.clone(), b"AB");
        let t = Sequence::from_ascii(alpha, &[b'X'; 1000]);
        // CIGAR: 1M 1000D 1M (target is 1002 long: 1 + 1000 inserts + 1)
        let mut padded_target: Vec<u8> = vec![q.alphabet.encode(b'A')];
        padded_target.extend(std::iter::repeat(q.alphabet.encode(b'X')).take(1000));
        padded_target.push(q.alphabet.encode(b'B'));
        let _ = t; // we built the bytes inline above
        let hit = make_hit(
            1,
            100,
            0,
            2,
            0,
            1002,
            vec![CigarOp::Match(1), CigarOp::Delete(1000), CigarOp::Match(1)],
        );
        let msa = assemble_msa(
            &q,
            &[hit],
            |_| Some(padded_target.as_slice()),
            MsaOptions::default(),
        );
        // Saturated at u8::MAX = 255 instead of wrapping to 232.
        assert_eq!(msa.deletion_matrix[1][1], u8::MAX);
    }
}
