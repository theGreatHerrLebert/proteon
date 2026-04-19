//! Superposition output — write the alignment transform applied to chain1
//! together with the unchanged reference chain2 as a single PDB document.
//!
//! Chain1's atoms are relabeled to chain "A", chain2's to chain "B", so
//! viewers can distinguish them regardless of what the source files used.
//!
//! Two scope limits relative to C++ `output_superpose`
//! (TMalign.cpp:3047-3660):
//!  * Atom set is whatever [`StructureData::pdb_lines`] holds — typically
//!    CA-only since the alignment loader filters to CA. Full all-atom
//!    output (with HETATMs/ligands) would require either the loader to
//!    keep raw lines for every atom, or re-reading the source file.
//!  * The four RasMol viz wrappers + five PyMOL companion scripts are not
//!    written. Modern viewers (PyMOL, ChimeraX, Mol*) open the bare PDB
//!    directly; those wrappers can be added later if anyone asks for them.

use std::io::Write;

use proteon_align::core::types::{AlignResult, StructureData, Transform};

/// Write the superposed structures to `w` as a single PDB document.
///
/// Chain1's atom coordinates have `result.transform` applied; chain2 is
/// emitted as the reference frame. Chains are separated by a `TER` record
/// and the file ends with `END`.
///
/// `pdb_lines` (the per-atom strings the loader pre-formats in
/// [`StructureData::pdb_lines`]) are 54-char fixed-column PDB records ending
/// with the x/y/z block at byte offsets 30..54 — we splice the transformed
/// coords back into that block in place. Trailing PDB columns
/// (occupancy/B-factor/element) are intentionally dropped by the loader and
/// are not reconstructed here.
pub fn output_superpose<W: Write>(
    w: &mut W,
    result: &AlignResult,
    chain1: &StructureData,
    chain2: &StructureData,
) -> std::io::Result<()> {
    write_header(w, result, chain1, chain2)?;

    for line in &chain1.pdb_lines {
        write_line(w, line, Some(&result.transform), 'A')?;
    }
    writeln!(w, "TER")?;

    for line in &chain2.pdb_lines {
        write_line(w, line, None, 'B')?;
    }
    writeln!(w, "TER")?;
    writeln!(w, "END")?;
    Ok(())
}

fn write_header<W: Write>(
    w: &mut W,
    result: &AlignResult,
    chain1: &StructureData,
    chain2: &StructureData,
) -> std::io::Result<()> {
    writeln!(w, "REMARK TM-align")?;
    writeln!(
        w,
        "REMARK Chain 1: {:<11} Size= {}",
        truncate(&chain1.source_path, 60),
        chain1.coords.len()
    )?;
    writeln!(
        w,
        "REMARK Chain 2: {:<11} Size= {} (TM-score normalized by {}, d0={:.2})",
        truncate(&chain2.source_path, 60),
        chain2.coords.len(),
        chain2.coords.len(),
        result.d0b,
    )?;
    writeln!(
        w,
        "REMARK Aligned length= {}, RMSD= {:.2}, TM-score= {:.5}",
        result.n_aligned, result.rmsd, result.tm_score_chain1,
    )?;
    Ok(())
}

/// Emit one PDB line with the chain ID overwritten to `chain` and, if
/// `transform` is `Some`, the x/y/z block transformed.
///
/// If the line is shorter than 54 bytes (malformed / missing coords) we
/// pass it through unchanged rather than panic — output formatting
/// shouldn't bring down a successful alignment.
///
/// Chain ID lives at byte offset 21 in the loader's fixed-column format
/// (see `loader.rs::extract_structure_data`).
fn write_line<W: Write>(
    w: &mut W,
    line: &str,
    transform: Option<&Transform>,
    chain: char,
) -> std::io::Result<()> {
    let bytes = line.as_bytes();
    if bytes.len() < 54 {
        writeln!(w, "{line}")?;
        return Ok(());
    }

    let head = &line[..21];
    let mid = &line[22..30];
    let parse = |range: std::ops::Range<usize>| -> Option<f64> {
        std::str::from_utf8(&bytes[range]).ok()?.trim().parse().ok()
    };
    let (Some(x), Some(y), Some(z)) = (parse(30..38), parse(38..46), parse(46..54)) else {
        // Coords unparseable — keep original tail but still relabel chain.
        writeln!(w, "{head}{chain}{}", &line[22..])?;
        return Ok(());
    };

    let [x, y, z] = match transform {
        Some(t) => apply_transform(t, [x, y, z]),
        None => [x, y, z],
    };
    writeln!(w, "{head}{chain}{mid}{x:>8.3}{y:>8.3}{z:>8.3}")?;
    Ok(())
}

fn apply_transform(t: &Transform, x: [f64; 3]) -> [f64; 3] {
    [
        t.t[0] + t.u[0][0] * x[0] + t.u[0][1] * x[1] + t.u[0][2] * x[2],
        t.t[1] + t.u[1][0] * x[0] + t.u[1][1] * x[1] + t.u[1][2] * x[2],
        t.t[2] + t.u[2][0] * x[0] + t.u[2][1] * x[1] + t.u[2][2] * x[2],
    ]
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[s.len() - max..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteon_align::core::types::{Coord3D, MolType};

    fn line(serial: u32, name: &str, resname: &str, resi: u32, x: f64, y: f64, z: f64) -> String {
        format!("ATOM  {serial:>5} {name:^4} {resname:>3} A{resi:>4}    {x:>8.3}{y:>8.3}{z:>8.3}")
    }

    fn make_result() -> AlignResult {
        AlignResult {
            tm_score_chain1: 0.0,
            tm_score_chain2: 0.0,
            tm_score_avg: 0.0,
            tm_score_user: 0.0,
            tm_score_scaled: 0.0,
            rmsd: 0.0,
            n_aligned: 0,
            seq_identity: 0.0,
            transform: Transform {
                t: [0.0; 3],
                u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            aligned_seq_x: String::new(),
            aligned_seq_y: String::new(),
            alignment_markers: String::new(),
            d0a: 0.0,
            d0b: 0.0,
            d0_out: 0.0,
        }
    }

    fn empty_struct(path: &str, lines: Vec<String>, coords: Vec<Coord3D>) -> StructureData {
        StructureData {
            coords,
            sequence: Vec::new(),
            sec_structure: Vec::new(),
            resi_ids: Vec::new(),
            chain_id: "A".to_string(),
            mol_type: MolType::Protein,
            source_path: path.to_string(),
            pdb_lines: lines,
        }
    }

    #[test]
    fn identity_transform_preserves_coords_and_relabels_chains() {
        let chain1 = empty_struct(
            "a.pdb",
            vec![line(1, "CA", "ALA", 1, 1.000, 2.000, 3.000)],
            vec![[1.0, 2.0, 3.0]],
        );
        let chain2 = empty_struct(
            "b.pdb",
            vec![line(2, "CA", "GLY", 5, 4.000, 5.000, 6.000)],
            vec![[4.0, 5.0, 6.0]],
        );
        let mut result = make_result();
        result.transform = Transform {
            t: [0.0, 0.0, 0.0],
            u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };

        let mut buf: Vec<u8> = Vec::new();
        output_superpose(&mut buf, &result, &chain1, &chain2).unwrap();
        let out = String::from_utf8(buf).unwrap();

        assert!(out.contains("   1.000   2.000   3.000"), "{out}");
        assert!(out.contains("   4.000   5.000   6.000"), "{out}");
        // chain1 → A, chain2 → B regardless of source IDs
        assert!(out.contains(" ALA A"), "chain1 should be A: {out}");
        assert!(out.contains(" GLY B"), "chain2 should be B: {out}");
        assert!(out.contains("\nTER\n"), "{out}");
        assert!(out.trim_end().ends_with("END"), "{out}");
    }

    #[test]
    fn translation_only_shifts_chain1_not_chain2() {
        let chain1 = empty_struct(
            "a.pdb",
            vec![line(1, "CA", "ALA", 1, 0.000, 0.000, 0.000)],
            vec![[0.0, 0.0, 0.0]],
        );
        let chain2 = empty_struct(
            "b.pdb",
            vec![line(2, "CA", "GLY", 5, 7.000, 8.000, 9.000)],
            vec![[7.0, 8.0, 9.0]],
        );
        let mut result = make_result();
        result.transform = Transform {
            t: [10.0, 20.0, 30.0],
            u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };

        let mut buf: Vec<u8> = Vec::new();
        output_superpose(&mut buf, &result, &chain1, &chain2).unwrap();
        let out = String::from_utf8(buf).unwrap();

        assert!(
            out.contains("  10.000  20.000  30.000"),
            "chain1 shifted: {out}"
        );
        assert!(
            out.contains("   7.000   8.000   9.000"),
            "chain2 unchanged: {out}"
        );
    }

    #[test]
    fn malformed_short_line_passes_through() {
        let chain1 = empty_struct("a.pdb", vec!["ATOM   short".to_string()], vec![]);
        let chain2 = empty_struct("b.pdb", vec![], vec![]);
        let mut result = make_result();
        result.transform = Transform {
            t: [10.0, 0.0, 0.0],
            u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };

        let mut buf: Vec<u8> = Vec::new();
        output_superpose(&mut buf, &result, &chain1, &chain2).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("ATOM   short"), "{out}");
    }
}
