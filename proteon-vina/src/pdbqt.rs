// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Line parser ported from AutoDock-Vina src/lib/parse_pdbqt.cpp
// (Apache-2.0). Upstream author: Oleg Trott, Scripps Research Institute.

//! PDBQT atom-record parser.
//!
//! v0 scope: a single-pose PDBQT (receptor or ligand). The parser
//! reads `ATOM` / `HETATM` records and returns a flat `Vec<RawAtom>`
//! in input order. Tree markup (`ROOT`, `BRANCH`, `ENDBRANCH`,
//! `ENDROOT`, `TORSDOF`) is accepted silently — we flatten the
//! torsion tree for rigid scoring. Multi-pose `MODEL`/`ENDMDL`
//! blocks are rejected (matches upstream's receptor behaviour;
//! use `vina_split` to pre-split multi-pose output).
//!
//! Column mapping mirrors upstream `parse_pdbqt_atom_string`:
//! atom number `7-11`, x `31-38`, y `39-46`, z `47-54`,
//! partial charge `69-76` (blank → 0.0), AD type `78-end-of-line`
//! (matches upstream `omit_whitespace(s, 78, 79)`, which extends to
//! end-of-line per the `j < str.size()` branch).

use crate::ad_types::{parse_ad_type, AdType};

/// A single atom parsed from a PDBQT `ATOM`/`HETATM` record.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RawAtom {
    /// Atom serial number (PDBQT columns 7–11).
    pub serial: u32,
    /// Cartesian coordinates in Å (columns 31–54).
    pub coords: [f64; 3],
    /// Partial (Gasteiger) charge from columns 69–76. Blank → 0.
    pub partial_charge: f64,
    /// AutoDock atom type from columns 78 onwards.
    pub ad_type: AdType,
}

/// Parse failures with enough context to locate the offending line.
#[derive(Debug, thiserror::Error)]
pub enum PdbqtError {
    #[error("line {line}: expected at least {expected} columns, got {got}")]
    LineTooShort { line: usize, expected: usize, got: usize },

    #[error("line {line}: column {col_start}-{col_end} ({field}) is not a valid number: {raw:?}")]
    NotANumber {
        line: usize,
        col_start: usize,
        col_end: usize,
        field: &'static str,
        raw: String,
    },

    #[error("line {line}: unknown AutoDock atom type {raw:?}")]
    UnknownAdType { line: usize, raw: String },

    #[error(
        "line {line}: multi-MODEL PDBQT not supported; use vina_split to extract a single pose"
    )]
    MultiModel { line: usize },

    #[error("line {line}: unknown or inappropriate record tag: {tag:?}")]
    UnknownRecord { line: usize, tag: String },

    #[error("line {line}: malformed {tag:?} — expected two integer atom serials")]
    MalformedBranch { line: usize, tag: String },

    #[error("line {line}: ENDROOT without a matching ROOT")]
    UnmatchedEndRoot { line: usize },

    #[error("line {line}: ENDBRANCH without a matching BRANCH")]
    UnmatchedEndBranch { line: usize },
}

/// A parsed PDBQT file: the atom records plus the torsion-tree
/// structure needed for intra-ligand pair accounting.
#[derive(Debug, Clone)]
pub struct PdbqtFile {
    /// Atoms in file order.
    pub atoms: Vec<RawAtom>,
    /// Fragment ID per atom, parallel to `atoms`. Atoms in the same
    /// `ROOT` or `BRANCH` block share an ID. Receptors and other
    /// tree-free files get all-zero IDs.
    pub fragment_ids: Vec<u32>,
    /// Rotatable bonds as `(parent_serial, child_serial)` pairs from
    /// each `BRANCH p c` line. Empty for receptors.
    pub rotatable_bonds: Vec<(u32, u32)>,
    /// Per-fragment parent (index into fragments). `None` for ROOT
    /// (fragment 0) and for tree-free files.
    pub fragment_parents: Vec<Option<u32>>,
    /// Per-fragment axis-begin serial — the parent-side atom of the
    /// rotatable bond that opens this fragment. `None` for ROOT.
    pub fragment_axis_begin: Vec<Option<u32>>,
    /// Per-fragment axis-end serial — the child-side atom (first atom
    /// inside this fragment). `None` for ROOT.
    pub fragment_axis_end: Vec<Option<u32>>,
}

/// Parse every `ATOM`/`HETATM` record in `text` into a flat list.
/// Tree markup is ignored; `MODEL`/`ENDMDL` raises `MultiModel`.
/// Thin wrapper over [`parse_pdbqt`].
pub fn parse_pdbqt_atoms(text: &str) -> Result<Vec<RawAtom>, PdbqtError> {
    Ok(parse_pdbqt(text)?.atoms)
}

/// Parse a multi-pose PDBQT stream (one with repeated
/// `MODEL`/`ENDMDL` blocks, like the output of `vina` or
/// `vina_split`) into one [`PdbqtFile`] per pose. Each pose
/// block's content is parsed with [`parse_pdbqt`].
///
/// Streams without any `MODEL` tag return a single-element vector
/// (equivalent to `vec![parse_pdbqt(text)?]`).
pub fn parse_pdbqt_models(text: &str) -> Result<Vec<PdbqtFile>, PdbqtError> {
    let mut models: Vec<PdbqtFile> = Vec::new();
    let mut current: Option<String> = None;
    let mut seen_any_model = false;

    for (i, raw_line) in text.lines().enumerate() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if starts_with(line, "MODEL") {
            if current.is_some() {
                return Err(PdbqtError::UnknownRecord {
                    line: i + 1,
                    tag: "MODEL".into(),
                });
            }
            current = Some(String::new());
            seen_any_model = true;
            continue;
        }
        if starts_with(line, "ENDMDL") {
            match current.take() {
                Some(buf) => models.push(parse_pdbqt(&buf)?),
                None => return Err(PdbqtError::UnknownRecord {
                    line: i + 1,
                    tag: "ENDMDL".into(),
                }),
            }
            continue;
        }
        if let Some(buf) = current.as_mut() {
            buf.push_str(line);
            buf.push('\n');
        } else if !seen_any_model {
            // Before the first MODEL, buffer into a single-model
            // file by promoting to a synthetic MODEL immediately.
            let mut buf = String::with_capacity(text.len());
            buf.push_str(line);
            buf.push('\n');
            current = Some(buf);
            seen_any_model = false;
        }
        // Lines between ENDMDL and the next MODEL are ignored
        // (upstream vina output has none).
    }

    if let Some(buf) = current {
        if seen_any_model {
            // Unclosed MODEL — best-effort parse.
            models.push(parse_pdbqt(&buf)?);
        } else {
            // Tree-free file without any MODEL tag.
            models.push(parse_pdbqt(&buf)?);
        }
    }

    if models.is_empty() {
        // Empty input: return one empty PdbqtFile so callers can
        // handle the "no poses" case uniformly.
        models.push(parse_pdbqt("")?);
    }
    Ok(models)
}

/// Parse a PDBQT file including the torsion-tree structure.
///
/// State machine: a fragment counter increments on every `ROOT` and
/// `BRANCH`, and a stack of currently-open fragment IDs tracks nesting.
/// The top of the stack is the fragment assigned to each incoming
/// `ATOM`. Files without any tree markup (e.g. receptors) leave the
/// stack empty and get fragment-ID 0 for all atoms.
pub fn parse_pdbqt(text: &str) -> Result<PdbqtFile, PdbqtError> {
    let mut atoms: Vec<RawAtom> = Vec::new();
    let mut fragment_ids: Vec<u32> = Vec::new();
    let mut rotatable_bonds: Vec<(u32, u32)> = Vec::new();
    let mut fragment_parents: Vec<Option<u32>> = Vec::new();
    let mut fragment_axis_begin: Vec<Option<u32>> = Vec::new();
    let mut fragment_axis_end: Vec<Option<u32>> = Vec::new();

    let mut stack: Vec<u32> = Vec::new();
    let mut next_id: u32 = 0;
    // Pending axis-end capture for the most recently opened BRANCH:
    // the next ATOM record after `BRANCH p c` is the child-side atom.
    let mut pending_axis_end_fragment: Option<u32> = None;

    for (i, raw_line) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);

        if line.is_empty() {
            continue;
        }
        if starts_with(line, "MODEL") || starts_with(line, "ENDMDL") {
            return Err(PdbqtError::MultiModel { line: line_no });
        }
        if starts_with(line, "ATOM  ") || starts_with(line, "HETATM") {
            let atom = parse_atom_line(line, line_no)?;
            let serial = atom.serial;
            atoms.push(atom);
            fragment_ids.push(stack.last().copied().unwrap_or(0));
            if let Some(frag) = pending_axis_end_fragment.take() {
                fragment_axis_end[frag as usize] = Some(serial);
            }
            continue;
        }
        if starts_with(line, "ROOT") && !starts_with(line, "ROOTEND") {
            let id = next_id;
            next_id += 1;
            fragment_parents.push(None);
            fragment_axis_begin.push(None);
            fragment_axis_end.push(None);
            stack.push(id);
            continue;
        }
        if starts_with(line, "ENDROOT") {
            stack
                .pop()
                .ok_or(PdbqtError::UnmatchedEndRoot { line: line_no })?;
            continue;
        }
        if starts_with(line, "BRANCH") {
            let (p, c) = parse_branch_line(line, "BRANCH", line_no)?;
            rotatable_bonds.push((p, c));
            let id = next_id;
            next_id += 1;
            // After ENDROOT the stack is empty, but top-level
            // BRANCHes are still structurally parented by ROOT.
            // Default to fragment 0 (ROOT) when the stack is empty.
            let parent = stack.last().copied().or(Some(0));
            fragment_parents.push(parent);
            fragment_axis_begin.push(Some(p));
            fragment_axis_end.push(Some(c));
            // Upstream PDBQT convention: the next ATOM record is the
            // child-side atom. `c` from the BRANCH line is usually that
            // serial, but we also capture it from the next ATOM to
            // cope with any ordering quirks.
            pending_axis_end_fragment = Some(id);
            stack.push(id);
            continue;
        }
        if starts_with(line, "ENDBRANCH") {
            stack
                .pop()
                .ok_or(PdbqtError::UnmatchedEndBranch { line: line_no })?;
            continue;
        }
        if is_ignored_prefix(line) {
            continue;
        }
        let tag = line
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        return Err(PdbqtError::UnknownRecord { line: line_no, tag });
    }

    // Tree-free files: no ROOT/BRANCH was seen. Synthesise a single
    // fragment 0 record so fragment_ids[0..] are valid indices.
    if fragment_parents.is_empty() {
        fragment_parents.push(None);
        fragment_axis_begin.push(None);
        fragment_axis_end.push(None);
    }

    Ok(PdbqtFile {
        atoms,
        fragment_ids,
        rotatable_bonds,
        fragment_parents,
        fragment_axis_begin,
        fragment_axis_end,
    })
}

/// Parse a `BRANCH p c` or `ENDBRANCH p c` line. Both have the same
/// format — two integer atom serials separated by whitespace after the
/// tag token.
fn parse_branch_line(line: &str, tag: &'static str, line_no: usize) -> Result<(u32, u32), PdbqtError> {
    let rest = line.get(tag.len()..).unwrap_or("");
    let mut it = rest.split_whitespace();
    let p = it.next().and_then(|s| s.parse::<u32>().ok());
    let c = it.next().and_then(|s| s.parse::<u32>().ok());
    match (p, c) {
        (Some(p), Some(c)) => Ok((p, c)),
        _ => Err(PdbqtError::MalformedBranch {
            line: line_no,
            tag: tag.to_string(),
        }),
    }
}

/// Parse one `ATOM`/`HETATM` line (already validated as such).
fn parse_atom_line(line: &str, line_no: usize) -> Result<RawAtom, PdbqtError> {
    // PDBQT atoms need at least through column 78 — the first char of
    // the AD-type field. Upstream's `omit_whitespace(s, 78, 79)` with
    // the extend-to-EOL branch accepts single-char AD types like "N"
    // where column 79 is absent or whitespace.
    require_cols(line, 78, line_no)?;

    let serial = parse_u32(line, 7, 11, "Atom number", line_no)?;
    let x = parse_f64(line, 31, 38, "Coordinate", line_no)?;
    let y = parse_f64(line, 39, 46, "Coordinate", line_no)?;
    let z = parse_f64(line, 47, 54, "Coordinate", line_no)?;
    let partial_charge = if substring_is_blank(line, 69, 76) {
        0.0
    } else {
        parse_f64(line, 69, 76, "Charge", line_no)?
    };
    let ad_token = extract_ad_token(line);
    let ad_type = parse_ad_type(&ad_token).ok_or_else(|| PdbqtError::UnknownAdType {
        line: line_no,
        raw: ad_token.clone(),
    })?;

    Ok(RawAtom {
        serial,
        coords: [x, y, z],
        partial_charge,
        ad_type,
    })
}

/// Mirrors upstream `omit_whitespace(str, 78, 79)` — "columns 78 onwards,
/// trimmed". The upstream helper extends `j` to end-of-string when the
/// caller-provided `j` is short, so AD types like `CG0` (3 chars) work.
fn extract_ad_token(line: &str) -> String {
    let bytes = line.as_bytes();
    if bytes.len() < 78 {
        return String::new();
    }
    // Start at 0-indexed byte 77 = column 78.
    std::str::from_utf8(&bytes[77..])
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Tags that are silently ignored — comments, terminators, and
/// structural markers whose content we don't need.
fn is_ignored_prefix(line: &str) -> bool {
    const IGNORED: &[&str] = &[
        "REMARK", "WARNING", "TER", "END", "TORSDOF", "USER", "COMPND", "HEADER", "TITLE",
    ];
    for &tag in IGNORED {
        if starts_with(line, tag) {
            return true;
        }
    }
    false
}

/// Column-range helpers. All indices are 1-based inclusive to mirror
/// upstream. Returns `Ok(())` if the line has at least `needed` bytes.
fn require_cols(line: &str, needed: usize, line_no: usize) -> Result<(), PdbqtError> {
    if line.len() < needed {
        Err(PdbqtError::LineTooShort {
            line: line_no,
            expected: needed,
            got: line.len(),
        })
    } else {
        Ok(())
    }
}

/// Extract bytes for 1-indexed inclusive column range `[i, j]`.
fn slice_cols(line: &str, i: usize, j: usize) -> &str {
    let start = i - 1;
    let end = j.min(line.len());
    &line[start..end]
}

fn substring_is_blank(line: &str, i: usize, j: usize) -> bool {
    slice_cols(line, i, j).trim().is_empty()
}

fn parse_u32(
    line: &str,
    i: usize,
    j: usize,
    field: &'static str,
    line_no: usize,
) -> Result<u32, PdbqtError> {
    let raw = slice_cols(line, i, j).trim();
    raw.parse::<u32>().map_err(|_| PdbqtError::NotANumber {
        line: line_no,
        col_start: i,
        col_end: j,
        field,
        raw: raw.to_string(),
    })
}

fn parse_f64(
    line: &str,
    i: usize,
    j: usize,
    field: &'static str,
    line_no: usize,
) -> Result<f64, PdbqtError> {
    let raw = slice_cols(line, i, j).trim();
    raw.parse::<f64>().map_err(|_| PdbqtError::NotANumber {
        line: line_no,
        col_start: i,
        col_end: j,
        field,
        raw: raw.to_string(),
    })
}

fn starts_with(line: &str, prefix: &str) -> bool {
    line.len() >= prefix.len() && &line[..prefix.len()] == prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIGAND_HEAD: &str = "\
REMARK SMILES Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1
ROOT
ATOM      1  N   UNL     1      16.600  51.810  14.798  1.00  0.00    -0.322 N
ATOM      2  C   UNL     1      15.629  51.747  15.784  1.00  0.00     0.255 C
ATOM      3  O   UNL     1      14.789  52.612  15.983  1.00  0.00    -0.269 OA
ATOM      4  H   UNL     1      17.359  51.117  14.866  1.00  0.00     0.170 HD
ENDROOT
BRANCH   1   5
ATOM      5  C   UNL     1      16.678  52.727  13.688  1.00  0.00     0.043 A
ENDBRANCH   1   5
TORSDOF 10
";

    const RECEPTOR_SNIPPET: &str = "\
ATOM      1  C   SER A 438      11.317  66.182  33.926  1.00  0.00     0.244 C
ATOM      2  O   SER A 438      10.747  66.774  34.839  1.00  0.00    -0.272 OA
ATOM      3  CA  SER A 438      11.585  66.877  32.589  1.00  0.00     0.197 C
ATOM      4  N   SER A 438      10.415  67.633  32.165  1.00  0.00    -0.342 N
TER
END
";

    const MACROCYCLE_SNIPPET: &str = "\
ATOM      9  C9  LIG L   1      30.694   9.197  18.945  1.00  0.00     0.019 CG0
ATOM     10  *1  LIG L   1      29.197   9.201  19.164  1.00  0.00     0.000 G0
";

    #[test]
    fn ligand_head_has_five_atoms_with_correct_types() {
        let atoms = parse_pdbqt_atoms(LIGAND_HEAD).expect("parse");
        assert_eq!(atoms.len(), 5);
        assert_eq!(atoms[0].serial, 1);
        assert_eq!(atoms[0].ad_type, AdType::N);
        assert_eq!(atoms[1].ad_type, AdType::C);
        assert_eq!(atoms[2].ad_type, AdType::Oa);
        assert_eq!(atoms[3].ad_type, AdType::Hd);
        assert_eq!(atoms[4].ad_type, AdType::A);
    }

    #[test]
    fn ligand_head_coords_and_charge() {
        let atoms = parse_pdbqt_atoms(LIGAND_HEAD).unwrap();
        assert_eq!(atoms[0].coords, [16.600, 51.810, 14.798]);
        assert!((atoms[0].partial_charge - (-0.322)).abs() < 1e-12);
        assert!((atoms[2].partial_charge - (-0.269)).abs() < 1e-12);
    }

    #[test]
    fn receptor_tree_free_parses_cleanly() {
        let atoms = parse_pdbqt_atoms(RECEPTOR_SNIPPET).unwrap();
        assert_eq!(atoms.len(), 4);
        assert_eq!(atoms[0].ad_type, AdType::C);
        assert_eq!(atoms[1].ad_type, AdType::Oa);
        assert_eq!(atoms[3].ad_type, AdType::N);
    }

    #[test]
    fn macrocycle_three_char_ad_types_parse() {
        // CG0 and G0 span column 78-80, exercising the omit_whitespace
        // extend-to-eol behaviour.
        let atoms = parse_pdbqt_atoms(MACROCYCLE_SNIPPET).unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].ad_type, AdType::Cg0);
        assert_eq!(atoms[1].ad_type, AdType::G0);
    }

    #[test]
    fn multi_model_rejected() {
        let text = "MODEL 1\nATOM      1  N   UNL     1      0.0  0.0  0.0  1.00  0.00    0.0 N \n";
        let err = parse_pdbqt_atoms(text).unwrap_err();
        assert!(matches!(err, PdbqtError::MultiModel { line: 1 }));
    }

    #[test]
    fn parse_pdbqt_models_splits_by_model_endmdl() {
        let text = "\
MODEL 1
ROOT
ATOM      1  N   UNL     1      16.600  51.810  14.798  1.00  0.00    -0.322 N
ENDROOT
ENDMDL
MODEL 2
ROOT
ATOM      1  N   UNL     1      16.700  51.910  14.898  1.00  0.00    -0.322 N
ENDROOT
ENDMDL
";
        let poses = parse_pdbqt_models(text).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].atoms[0].coords, [16.600, 51.810, 14.798]);
        assert_eq!(poses[1].atoms[0].coords, [16.700, 51.910, 14.898]);
    }

    #[test]
    fn parse_pdbqt_models_passthrough_when_no_model_tags() {
        // Should return exactly one PdbqtFile for tree-flat inputs.
        let poses = parse_pdbqt_models(RECEPTOR_FIXTURE).unwrap();
        assert_eq!(poses.len(), 1);
        assert!(!poses[0].atoms.is_empty());
    }

    #[test]
    fn parse_pdbqt_models_on_real_multipose_fixture() {
        // The vendored 1iep_vina_out has four poses.
        const MULTIPOSE: &str = include_str!("../tests/fixtures/multipose/1iep/poses.pdbqt");
        let poses = parse_pdbqt_models(MULTIPOSE).unwrap();
        assert_eq!(poses.len(), 4);
        // The vina_out fixture carries 40 atoms per pose — upstream
        // elides the non-polar hydrogens it can't place in search.
        // All poses must have the same atom count since they're
        // the same ligand re-oriented.
        let expected = poses[0].atoms.len();
        assert_eq!(expected, 40, "pose atom count");
        for p in &poses {
            assert_eq!(p.atoms.len(), expected, "all poses share atom count");
        }
        // Poses are structurally distinct: at least one coord differs
        // between pose 0 and pose 1.
        assert_ne!(poses[0].atoms[0].coords, poses[1].atoms[0].coords);
    }

    #[test]
    fn unknown_tag_reports_line_and_token() {
        let text = "FOOBAR something\n";
        let err = parse_pdbqt_atoms(text).unwrap_err();
        match err {
            PdbqtError::UnknownRecord { line, tag } => {
                assert_eq!(line, 1);
                assert_eq!(tag, "FOOBAR");
            }
            _ => panic!("expected UnknownRecord"),
        }
    }

    #[test]
    fn unknown_ad_type_errors_with_line_info() {
        // AD type column "ZQ" is not a valid token.
        let text =
            "ATOM      1  N   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 ZQ\n";
        let err = parse_pdbqt_atoms(text).unwrap_err();
        match err {
            PdbqtError::UnknownAdType { line, raw } => {
                assert_eq!(line, 1);
                assert_eq!(raw, "ZQ");
            }
            _ => panic!("expected UnknownAdType"),
        }
    }

    #[test]
    fn line_shorter_than_min_cols_errors() {
        let text = "ATOM      1  N   UNL     1       0.000   0.000   0.000\n";
        let err = parse_pdbqt_atoms(text).unwrap_err();
        assert!(matches!(err, PdbqtError::LineTooShort { .. }));
    }

    #[test]
    fn blank_charge_defaults_to_zero() {
        // Hand-crafted: charge cols 69-76 all blank.
        let text =
            "ATOM      1  N   UNL     1      16.600  51.810  14.798  1.00  0.00            N \n";
        let atoms = parse_pdbqt_atoms(text).unwrap();
        assert_eq!(atoms[0].partial_charge, 0.0);
    }

    #[test]
    fn crlf_line_endings_tolerated() {
        let text = "\
ATOM      1  N   UNL     1      16.600  51.810  14.798  1.00  0.00    -0.322 N \r
END\r
";
        let atoms = parse_pdbqt_atoms(text).unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].ad_type, AdType::N);
    }

    // ---- Integration: real fixture from upstream ------------------------

    const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
    const RECEPTOR_FIXTURE: &str = include_str!("../tests/fixtures/1iep_receptor.pdbqt");

    #[test]
    fn real_ligand_1iep_parses() {
        let atoms = parse_pdbqt_atoms(LIGAND_FIXTURE).expect("1iep ligand should parse");
        // 1iep_ligand.pdbqt has 48 ATOM lines (63 total minus remarks /
        // tree markup).
        let atom_count = LIGAND_FIXTURE
            .lines()
            .filter(|l| l.starts_with("ATOM  ") || l.starts_with("HETATM"))
            .count();
        assert_eq!(atoms.len(), atom_count);
        // Partial-charge sanity: ligand charges sum approximately to
        // its net formal charge. For 1iep imatinib it's +1; we only
        // assert the sum is finite and reasonably bounded.
        let q_sum: f64 = atoms.iter().map(|a| a.partial_charge).sum();
        assert!(q_sum.is_finite());
        assert!(q_sum.abs() < 5.0, "implausible total charge {q_sum}");
    }

    #[test]
    fn real_receptor_1iep_parses() {
        let atoms = parse_pdbqt_atoms(RECEPTOR_FIXTURE).expect("1iep receptor should parse");
        let atom_count = RECEPTOR_FIXTURE
            .lines()
            .filter(|l| l.starts_with("ATOM  ") || l.starts_with("HETATM"))
            .count();
        assert_eq!(atoms.len(), atom_count);
        assert!(atoms.len() > 1000, "receptor expected to be large");
    }

    // ---- torsion-tree / PdbqtFile ---------------------------------------

    #[test]
    fn receptor_has_all_zero_fragment_ids_and_no_rotatable_bonds() {
        let file = parse_pdbqt(RECEPTOR_FIXTURE).expect("parse receptor");
        assert_eq!(file.atoms.len(), file.fragment_ids.len());
        assert!(file.fragment_ids.iter().all(|&id| id == 0));
        assert!(file.rotatable_bonds.is_empty());
    }

    #[test]
    fn tiny_ligand_head_assigns_fragments_per_branch() {
        // ROOT block → fragment 0 (atoms 1–4).
        // BRANCH 1 5 opens → fragment 1 (atom 5).
        let file = parse_pdbqt(LIGAND_HEAD).expect("parse");
        assert_eq!(file.atoms.len(), 5);
        assert_eq!(&file.fragment_ids, &[0, 0, 0, 0, 1]);
        assert_eq!(&file.rotatable_bonds, &[(1, 5)]);
    }

    #[test]
    fn real_ligand_1iep_fragment_count_matches_branch_count() {
        let file = parse_pdbqt(LIGAND_FIXTURE).expect("parse ligand");
        let n_branches = LIGAND_FIXTURE
            .lines()
            .filter(|l| l.starts_with("BRANCH"))
            .count();
        let n_frags = file
            .fragment_ids
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            + 1;
        // max fragment ID == n_branches (ROOT is id 0, each BRANCH
        // bumps the counter).
        assert_eq!(n_frags as usize, n_branches + 1);
        assert_eq!(file.rotatable_bonds.len(), n_branches);
    }

    #[test]
    fn real_ligand_root_atoms_share_fragment_zero() {
        // Atoms 1..4 are between ROOT and ENDROOT → fragment 0.
        let file = parse_pdbqt(LIGAND_FIXTURE).expect("parse");
        for (i, atom) in file.atoms.iter().enumerate().take(4) {
            assert_eq!(
                file.fragment_ids[i], 0,
                "atom serial {} should be in fragment 0",
                atom.serial
            );
        }
        // The 5th atom (BRANCH 1 5's first atom) should be in a
        // fragment != 0.
        assert_ne!(file.fragment_ids[4], 0);
    }

    #[test]
    fn unmatched_endbranch_errors() {
        let text = "\
ATOM      1  N   UNL     1      16.600  51.810  14.798  1.00  0.00    -0.322 N
ENDBRANCH   1   5
";
        let err = parse_pdbqt(text).unwrap_err();
        assert!(matches!(err, PdbqtError::UnmatchedEndBranch { line: 2 }));
    }

    #[test]
    fn malformed_branch_line_errors() {
        let text = "BRANCH abc def\n";
        let err = parse_pdbqt(text).unwrap_err();
        assert!(matches!(err, PdbqtError::MalformedBranch { line: 1, .. }));
    }
}
