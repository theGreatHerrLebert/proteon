//! Peptide backbone hydrogen placement.
//!
//! Places amide hydrogen atoms on backbone nitrogen atoms for non-N-terminal,
//! non-proline amino acid residues. Uses the DSSP bisector method:
//!
//!   H = N + 1.02 * normalize( normalize(C_prev→N) + normalize(CA→N) )
//!
//! This is the most important hydrogen for Kabsch-Sander H-bond detection
//! and DSSP secondary structure assignment.

/// N-H bond length in Ångströms.
/// Uses 1.01 to match DSSP's virtual H placement exactly,
/// ensuring identical secondary structure assignments pre/post H placement.
const NH_BOND_LENGTH: f64 = 1.01;

/// Info needed to place one backbone H atom.
struct HPlacement {
    /// Index of the chain (in first model) to insert into.
    chain_idx: usize,
    /// Index of the residue within that chain.
    residue_idx: usize,
    /// Computed (x, y, z) for the new H atom.
    pos: [f64; 3],
    /// Serial number for the new atom.
    serial: usize,
}

/// Result of peptide hydrogen placement.
pub struct AddHydrogensResult {
    /// Number of H atoms added.
    pub added: usize,
    /// Number of residues skipped (missing backbone atoms, proline, N-terminal).
    pub skipped: usize,
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Backbone atom positions for a single residue.
struct BackboneAtoms {
    n: [f64; 3],
    ca: [f64; 3],
    c: [f64; 3],
}

/// Extract backbone N, CA, C positions from a residue, if all present.
fn extract_backbone(residue: &pdbtbx::Residue) -> Option<BackboneAtoms> {
    let mut n = None;
    let mut ca = None;
    let mut c = None;

    for atom in residue.atoms() {
        match atom.name().trim() {
            "N" => n = Some(atom.pos()),
            "CA" => ca = Some(atom.pos()),
            "C" => c = Some(atom.pos()),
            _ => {}
        }
    }

    let (nx, ny, nz) = n?;
    let (cax, cay, caz) = ca?;
    let (cx, cy, cz) = c?;

    Some(BackboneAtoms {
        n: [nx, ny, nz],
        ca: [cax, cay, caz],
        c: [cx, cy, cz],
    })
}

/// Check if a residue already has a backbone H atom.
fn has_backbone_h(residue: &pdbtbx::Residue) -> bool {
    residue.atoms().any(|a| {
        let name = a.name().trim();
        name == "H" || name == "HN"
    })
}

/// Place peptide backbone hydrogen atoms on a PDB structure (in-place).
///
/// For each non-N-terminal, non-proline amino acid residue that lacks a
/// backbone H, computes the amide hydrogen position using the bisector of
/// C(i-1)→N and CA→N vectors, at 1.02 Å from N.
///
/// Returns the number of atoms added and skipped.
pub fn place_peptide_hydrogens(pdb: &mut pdbtbx::PDB) -> AddHydrogensResult {
    // --- Read pass: compute all H positions ---
    let mut placements: Vec<HPlacement> = Vec::new();
    let mut skipped = 0;

    // Find the next available serial number
    let mut max_serial: usize = pdb.atoms().map(|a| a.serial_number()).max().unwrap_or(0);

    // Only process the first model (consistent with DSSP, H-bonds, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return AddHydrogensResult { added: 0, skipped: 0 },
    };

    for (chain_idx, chain) in first_model.chains().enumerate() {
        let mut prev_backbone: Option<BackboneAtoms> = None;
        let mut residue_idx_in_chain = 0;

        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .map_or(false, |c| c.is_amino_acid());

            if !is_aa {
                prev_backbone = None;
                residue_idx_in_chain += 1;
                continue;
            }

            let backbone = extract_backbone(residue);

            if let (Some(prev), Some(ref curr)) = (&prev_backbone, &backbone) {
                let is_proline = residue.name() == Some("PRO");

                if is_proline || has_backbone_h(residue) {
                    skipped += 1;
                } else {
                    // Bisector method: H is along the average of two vectors
                    // pointing away from N: (C_prev→N) and (CA→N)
                    let v1 = normalize([
                        curr.n[0] - prev.c[0],
                        curr.n[1] - prev.c[1],
                        curr.n[2] - prev.c[2],
                    ]);
                    let v2 = normalize([
                        curr.n[0] - curr.ca[0],
                        curr.n[1] - curr.ca[1],
                        curr.n[2] - curr.ca[2],
                    ]);
                    let bisector = normalize([
                        v1[0] + v2[0],
                        v1[1] + v2[1],
                        v1[2] + v2[2],
                    ]);

                    // Degenerate case: vectors are exactly opposite
                    if bisector[0] == 0.0 && bisector[1] == 0.0 && bisector[2] == 0.0 {
                        skipped += 1;
                    } else {
                        max_serial += 1;
                        placements.push(HPlacement {
                            chain_idx,
                            residue_idx: residue_idx_in_chain,
                            pos: [
                                curr.n[0] + NH_BOND_LENGTH * bisector[0],
                                curr.n[1] + NH_BOND_LENGTH * bisector[1],
                                curr.n[2] + NH_BOND_LENGTH * bisector[2],
                            ],
                            serial: max_serial,
                        });
                    }
                }
            } else if is_aa && prev_backbone.is_none() {
                // N-terminal residue — no previous C to reference
                skipped += 1;
            }

            prev_backbone = backbone;
            residue_idx_in_chain += 1;
        }
    }

    // --- Write pass: add H atoms ---
    let added = placements.len();

    for p in placements {
        let model = match pdb.model_mut(0) {
            Some(m) => m,
            None => continue,
        };
        let chain = match model.chain_mut(p.chain_idx) {
            Some(c) => c,
            None => continue,
        };
        let residue = match chain.residue_mut(p.residue_idx) {
            Some(r) => r,
            None => continue,
        };
        let conformer = match residue.conformers_mut().next() {
            Some(c) => c,
            None => continue,
        };

        if let Some(h_atom) = pdbtbx::Atom::new(
            false,          // not HETATM
            p.serial,       // serial number
            "H",            // atom name
            "H",            // atom name (id)
            p.pos[0],
            p.pos[1],
            p.pos[2],
            1.0,            // occupancy
            0.0,            // B-factor (unknown)
            "H",            // element
            0,              // charge
        ) {
            conformer.add_atom(h_atom);
        }
    }

    AddHydrogensResult { added, skipped }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let v = normalize([3.0, 4.0, 0.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = normalize([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_h_position_geometry() {
        // Synthetic peptide bond geometry:
        // C_prev at origin, N at (1.33, 0, 0), CA at (1.33+1.47, 0, 0)
        // This is a fully linear arrangement, so bisector should be along +x

        let c_prev = [0.0, 0.0, 0.0];
        let n = [1.33, 0.0, 0.0];
        let ca = [1.33 + 1.47, 0.0, 0.0];

        let v1 = normalize([n[0] - c_prev[0], n[1] - c_prev[1], n[2] - c_prev[2]]);
        let v2 = normalize([n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]]);
        let bisector = normalize([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]);

        // v1 = [1,0,0], v2 = [-1,0,0] → bisector is [0,0,0] (degenerate)
        // This is expected — a perfectly linear C-N-CA has no defined H direction
        assert!((bisector[0].abs() + bisector[1].abs() + bisector[2].abs()) < 1e-9);
    }

    #[test]
    fn test_h_position_bent() {
        // Realistic: C_prev at origin, N at (1.33, 0, 0), CA at (2.0, 1.0, 0)
        let c_prev = [0.0, 0.0, 0.0];
        let n = [1.33, 0.0, 0.0];
        let ca = [2.0, 1.0, 0.0];

        let v1 = normalize([n[0] - c_prev[0], n[1] - c_prev[1], n[2] - c_prev[2]]);
        let v2 = normalize([n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]]);
        let bisector = normalize([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]);

        let h = [
            n[0] + NH_BOND_LENGTH * bisector[0],
            n[1] + NH_BOND_LENGTH * bisector[1],
            n[2] + NH_BOND_LENGTH * bisector[2],
        ];

        // H should be at 1.02 Å from N
        let dist = ((h[0] - n[0]).powi(2) + (h[1] - n[1]).powi(2) + (h[2] - n[2]).powi(2)).sqrt();
        assert!((dist - NH_BOND_LENGTH).abs() < 1e-10);

        // H should be below the C-N-CA plane (negative y direction)
        assert!(h[1] < n[1]);
    }

    #[test]
    fn test_empty_pdb() {
        let mut pdb = pdbtbx::PDB::new();
        let result = place_peptide_hydrogens(&mut pdb);
        assert_eq!(result.added, 0);
        assert_eq!(result.skipped, 0);
    }

    #[test]
    fn test_crambin_h_count() {
        // 1CRN: 46 residues, 1 chain
        // Residue 1 = N-terminal (no H), residues with PRO get skipped
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let n_residues: usize = pdb
            .chains()
            .flat_map(|c| c.residues())
            .filter(|r| {
                r.conformers()
                    .next()
                    .map_or(false, |c| c.is_amino_acid())
            })
            .count();

        let n_proline: usize = pdb
            .chains()
            .flat_map(|c| c.residues())
            .filter(|r| r.name() == Some("PRO"))
            .count();

        let result = place_peptide_hydrogens(&mut pdb);

        // Should place H on all non-N-terminal, non-proline residues
        // = n_residues - 1 (N-terminal) - n_proline
        let expected = n_residues - 1 - n_proline;
        assert_eq!(
            result.added, expected,
            "Expected {} H atoms (total {} AA, 1 N-term, {} PRO), got {}",
            expected, n_residues, n_proline, result.added
        );
    }

    #[test]
    fn test_h_positions_match_dssp() {
        // Verify our H positions match what DSSP computes internally
        use crate::dssp;

        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        // Get DSSP's virtual H positions first
        let dssp_residues = dssp::extract_dssp_residues(&pdb);
        let dssp_h_positions: Vec<[f64; 3]> = dssp_residues
            .iter()
            .filter(|r| r.has_h)
            .map(|r| r.h)
            .collect();

        // Now place actual H atoms
        let result = place_peptide_hydrogens(&mut pdb);
        assert!(result.added > 0);

        // Extract the placed H positions
        let mut placed_h: Vec<[f64; 3]> = Vec::new();
        for chain in pdb.chains() {
            for residue in chain.residues() {
                let is_aa = residue
                    .conformers()
                    .next()
                    .map_or(false, |c| c.is_amino_acid());
                if !is_aa {
                    continue;
                }
                for atom in residue.atoms() {
                    if atom.name().trim() == "H" {
                        let (x, y, z) = atom.pos();
                        placed_h.push([x, y, z]);
                    }
                }
            }
        }

        assert_eq!(
            placed_h.len(),
            dssp_h_positions.len(),
            "Should have same number of H atoms as DSSP virtual Hs"
        );

        // Positions should be identical (same algorithm, same bond length)
        for (i, (placed, dssp_pos)) in placed_h.iter().zip(&dssp_h_positions).enumerate() {
            let dist = ((placed[0] - dssp_pos[0]).powi(2)
                + (placed[1] - dssp_pos[1]).powi(2)
                + (placed[2] - dssp_pos[2]).powi(2))
            .sqrt();
            assert!(
                dist < 1e-6,
                "H atom {} position mismatch: dist={:.6} Å (placed={:?}, dssp={:?})",
                i,
                dist,
                placed,
                dssp_pos
            );
        }
    }

    #[test]
    fn test_idempotent() {
        // Running twice should not add duplicate H atoms
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let r1 = place_peptide_hydrogens(&mut pdb);
        let atoms_after_first = pdb.atom_count();

        let r2 = place_peptide_hydrogens(&mut pdb);
        let atoms_after_second = pdb.atom_count();

        assert!(r1.added > 0, "First pass should add hydrogens");
        assert_eq!(r2.added, 0, "Second pass should add nothing");
        assert_eq!(atoms_after_first, atoms_after_second);
    }
}
