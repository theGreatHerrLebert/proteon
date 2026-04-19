//! DSSP secondary structure assignment (Kabsch-Sander algorithm).
//!
//! Assigns H (alpha helix), G (3-10 helix), I (pi helix), E (extended strand),
//! B (isolated bridge), T (turn), S (bend), C (coil) from backbone coordinates.
//!
//! Reference: Kabsch & Sander (1983) Biopolymers 22:2577-2637.

use std::f64::consts::PI;

/// H-bond energy constant: q1 * q2 * f
const HB_CONSTANT: f64 = 0.42 * 0.20 * 332.0; // = 27.888

/// Energy cutoff for H-bond acceptance (kcal/mol)
const HB_CUTOFF: f64 = -0.5;

/// N-H bond length for virtual hydrogen placement (Å)
const NH_BOND_LENGTH: f64 = 1.01;

/// Bend angle threshold (degrees) for 'S' assignment
const BEND_THRESHOLD: f64 = 70.0;

// ---------------------------------------------------------------------------
// Backbone data extraction
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct DsspResidue {
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
    pub h: [f64; 3],
    pub has_h: bool,
    pub chain_idx: usize,
    /// True if this residue needs virtual H placement (no real H found).
    pub needs_virtual_h: bool,
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

pub(crate) fn extract_dssp_residues(pdb: &pdbtbx::PDB) -> Vec<DsspResidue> {
    let mut residues = Vec::new();
    let mut chain_idx = 0;

    // Use first model only (consistent with atom_count(), SASA, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return residues,
    };
    for chain in first_model.chains() {
        let mut chain_residues: Vec<DsspResidue> = Vec::new();

        for residue in chain.residues() {
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if !is_aa {
                continue;
            }

            let mut n = None;
            let mut ca = None;
            let mut c = None;
            let mut o = None;
            let mut h_real = None;
            let is_proline = residue.name() == Some("PRO");

            for atom in crate::altloc::residue_atoms_primary(residue) {
                let name = atom.name().trim();
                let (x, y, z) = atom.pos();
                match name {
                    "N" => n = Some([x, y, z]),
                    "CA" => ca = Some([x, y, z]),
                    "C" => c = Some([x, y, z]),
                    "O" => o = Some([x, y, z]),
                    "H" | "HN" => h_real = Some([x, y, z]),
                    _ => {}
                }
            }

            if let (Some(n), Some(ca), Some(c), Some(o)) = (n, ca, c, o) {
                let (h, has_h) = match h_real {
                    // Use real H atom if present (placed or experimental)
                    Some(pos) if !is_proline => (pos, true),
                    _ => ([0.0; 3], !is_proline),
                };
                let needs_virtual_h = h_real.is_none() && !is_proline;
                chain_residues.push(DsspResidue {
                    n,
                    ca,
                    c,
                    o,
                    h,
                    has_h,
                    chain_idx,
                    needs_virtual_h,
                });
            }
        }

        // Virtual H placement for residues without real H atoms.
        // H = N + 1.01 * normalize(normalize(C_prev→N) + normalize(CA→N))
        if !chain_residues.is_empty() && chain_residues[0].needs_virtual_h {
            chain_residues[0].has_h = false;
        }
        for i in 1..chain_residues.len() {
            if !chain_residues[i].needs_virtual_h {
                continue;
            }
            let c_prev = chain_residues[i - 1].c;
            let n_curr = chain_residues[i].n;
            let ca_curr = chain_residues[i].ca;

            // Two vectors pointing away from N: C_prev→N and CA→N
            let v1 = normalize([
                n_curr[0] - c_prev[0],
                n_curr[1] - c_prev[1],
                n_curr[2] - c_prev[2],
            ]);
            let v2 = normalize([
                n_curr[0] - ca_curr[0],
                n_curr[1] - ca_curr[1],
                n_curr[2] - ca_curr[2],
            ]);

            // Bisector
            let bisector = normalize([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]);

            chain_residues[i].h = [
                n_curr[0] + NH_BOND_LENGTH * bisector[0],
                n_curr[1] + NH_BOND_LENGTH * bisector[1],
                n_curr[2] + NH_BOND_LENGTH * bisector[2],
            ];
        }

        residues.extend(chain_residues);
        chain_idx += 1;
    }

    residues
}

// ---------------------------------------------------------------------------
// H-bond computation
// ---------------------------------------------------------------------------

#[inline]
fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Kabsch-Sander energy: CO of `acc` ← NH of `don`.
fn hbond_energy(acc: &DsspResidue, don: &DsspResidue) -> f64 {
    if !don.has_h {
        return 0.0;
    }
    if acc.chain_idx != don.chain_idx {
        return 0.0;
    }

    let r_on = dist(&acc.o, &don.n);
    if r_on > 5.2 {
        return 0.0;
    }

    let r_ch = dist(&acc.c, &don.h);
    let r_oh = dist(&acc.o, &don.h);
    let r_cn = dist(&acc.c, &don.n);

    if r_on < 0.01 || r_ch < 0.01 || r_oh < 0.01 || r_cn < 0.01 {
        return 0.0;
    }

    HB_CONSTANT * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
}

// ---------------------------------------------------------------------------
// Secondary structure assignment
// ---------------------------------------------------------------------------

/// Assign DSSP secondary structure.
pub(crate) fn assign_dssp(residues: &[DsspResidue]) -> String {
    let n = residues.len();
    if n == 0 {
        return String::new();
    }

    // --- Step 1: Compute H-bond matrix ---
    // hbond(i,j) = true means CO(i) accepts from NH(j)
    // Store as a closure for clarity
    let mut hb_matrix = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if residues[i].chain_idx != residues[j].chain_idx {
                continue;
            }
            // Skip |i-j| < 2 (no H-bond between adjacent residues)
            if (i as isize - j as isize).unsigned_abs() < 2 {
                continue;
            }
            let e = hbond_energy(&residues[i], &residues[j]);
            if e < HB_CUTOFF {
                hb_matrix[i][j] = true; // CO(i) ← NH(j)
            }
        }
    }

    let hbond = |i: usize, j: usize| -> bool {
        if i < n && j < n {
            hb_matrix[i][j]
        } else {
            false
        }
    };

    // --- Step 2: Identify n-turns ---
    let mut turn3 = vec![' '; n];
    let mut turn4 = vec![' '; n];
    let mut turn5 = vec![' '; n];

    fn mark_turn(turn: &mut [char], i: usize, span: usize, n: usize) {
        if i + span >= n {
            return;
        }
        // Start marker
        turn[i] = match turn[i] {
            '<' | 'X' => 'X',
            _ => '>',
        };
        // Interior
        for j in 1..span {
            if turn[i + j] == ' ' {
                turn[i + j] = char::from_digit(span as u32, 10).unwrap_or('?');
            }
        }
        // End marker
        turn[i + span] = match turn[i + span] {
            '>' | 'X' => 'X',
            _ => '<',
        };
    }

    for i in 0..n {
        if i + 3 < n && hbond(i, i + 3) && residues[i].chain_idx == residues[i + 3].chain_idx {
            mark_turn(&mut turn3, i, 3, n);
        }
        if i + 4 < n && hbond(i, i + 4) && residues[i].chain_idx == residues[i + 4].chain_idx {
            mark_turn(&mut turn4, i, 4, n);
        }
        if i + 5 < n && hbond(i, i + 5) && residues[i].chain_idx == residues[i + 5].chain_idx {
            mark_turn(&mut turn5, i, 5, n);
        }
    }

    // --- Step 3: Identify bridges ---
    // Bridge(i,j): two nonoverlapping stretches (i-1,i,i+1) and (j-1,j,j+1)
    #[derive(Clone, Copy, PartialEq)]
    enum BridgeType {
        None,
        Parallel,
        Antiparallel,
    }

    let mut bridge_partner: Vec<Vec<(usize, BridgeType)>> = vec![Vec::new(); n];

    for i in 1..n.saturating_sub(1) {
        for j in (i + 2)..n.saturating_sub(1) {
            if residues[i].chain_idx != residues[j].chain_idx {
                continue;
            }

            // Parallel bridge(i,j):
            //   Pattern 1: Hbond(i-1, j) AND Hbond(j, i+1)
            //   Pattern 2: Hbond(j-1, i) AND Hbond(i, j+1)
            let parallel =
                (hbond(i - 1, j) && hbond(j, i + 1)) || (hbond(j - 1, i) && hbond(i, j + 1));

            // Antiparallel bridge(i,j):
            //   Pattern 1: Hbond(i, j) AND Hbond(j, i)
            //   Pattern 2: Hbond(i-1, j+1) AND Hbond(j-1, i+1)
            let antiparallel =
                (hbond(i, j) && hbond(j, i)) || (hbond(i - 1, j + 1) && hbond(j - 1, i + 1));

            if parallel {
                bridge_partner[i].push((j, BridgeType::Parallel));
                bridge_partner[j].push((i, BridgeType::Parallel));
            }
            if antiparallel {
                bridge_partner[i].push((j, BridgeType::Antiparallel));
                bridge_partner[j].push((i, BridgeType::Antiparallel));
            }
        }
    }

    // --- Step 4: Build ladders from bridges ---
    // A ladder is a set of consecutive bridges of the same type.
    // Residues in a ladder of length >= 2 get E, isolated bridges get B.
    let mut is_ladder = vec![false; n]; // true if part of an extended ladder

    for i in 1..n.saturating_sub(1) {
        for &(j, btype) in &bridge_partner[i] {
            if btype == BridgeType::None {
                continue;
            }
            // Check if there's a consecutive bridge at i±1
            for di in [-1i32, 1] {
                let i2 = (i as i32 + di) as usize;
                if i2 >= n {
                    continue;
                }
                for &(j2, btype2) in &bridge_partner[i2] {
                    if btype2 != btype {
                        continue;
                    }
                    let dj = j2 as i32 - j as i32;
                    // Parallel: consecutive means dj == di (both move +1 or -1)
                    // Antiparallel: consecutive means dj == -di
                    let consecutive = match btype {
                        BridgeType::Parallel => dj == di,
                        BridgeType::Antiparallel => dj == -di,
                        BridgeType::None => false,
                    };
                    if consecutive {
                        is_ladder[i] = true;
                        is_ladder[j] = true;
                        is_ladder[i2] = true;
                        is_ladder[j2] = true;
                    }
                }
            }
        }
    }

    // --- Step 5: Build summary string with priority ---
    let mut summary = vec!['C'; n];

    // 5-helix (I): two consecutive 5-turns
    for i in 0..n.saturating_sub(1) {
        let s1 = turn5[i] == '>' || turn5[i] == 'X';
        let s2 = turn5[i + 1] == '>' || turn5[i + 1] == 'X';
        if s1 && s2 {
            for j in (i + 1)..=(i + 5).min(n - 1) {
                summary[j] = 'I';
            }
        }
    }

    // 3-helix (G): two consecutive 3-turns
    for i in 0..n.saturating_sub(1) {
        let s1 = turn3[i] == '>' || turn3[i] == 'X';
        let s2 = turn3[i + 1] == '>' || turn3[i + 1] == 'X';
        if s1 && s2 {
            for j in (i + 1)..=(i + 3).min(n - 1) {
                if summary[j] != 'H' {
                    summary[j] = 'G';
                }
            }
        }
    }

    // Beta: E for ladder residues, B for isolated bridges
    for i in 0..n {
        if !bridge_partner[i].is_empty() {
            if is_ladder[i] {
                summary[i] = 'E';
            } else if summary[i] == 'C' || summary[i] == 'S' || summary[i] == 'T' {
                summary[i] = 'B';
            }
        }
    }

    // 4-helix (H): two consecutive 4-turns — highest helix priority
    for i in 0..n.saturating_sub(1) {
        let s1 = turn4[i] == '>' || turn4[i] == 'X';
        let s2 = turn4[i + 1] == '>' || turn4[i + 1] == 'X';
        if s1 && s2 {
            for j in (i + 1)..=(i + 4).min(n - 1) {
                summary[j] = 'H';
            }
        }
    }

    // Clean up: isolated G or I residues (created by H overwriting) become T
    for i in 0..n {
        if summary[i] == 'G' || summary[i] == 'I' {
            let prev = if i > 0 { summary[i - 1] } else { 'C' };
            let next = if i + 1 < n { summary[i + 1] } else { 'C' };
            if prev != summary[i] && next != summary[i] {
                summary[i] = 'T';
            }
        }
    }

    // Turns (T): residues in a turn not already assigned higher
    for i in 0..n {
        if summary[i] == 'C' {
            let in_turn = (turn3[i] != ' ') || (turn4[i] != ' ') || (turn5[i] != ' ');
            if in_turn {
                summary[i] = 'T';
            }
        }
    }

    // Bend (S): angle(CA(i-2)→CA(i), CA(i)→CA(i+2)) > 70°
    for i in 2..n.saturating_sub(2) {
        if summary[i] != 'C' {
            continue;
        }
        if residues[i - 2].chain_idx != residues[i].chain_idx
            || residues[i].chain_idx != residues[i + 2].chain_idx
        {
            continue;
        }

        let v1 = [
            residues[i].ca[0] - residues[i - 2].ca[0],
            residues[i].ca[1] - residues[i - 2].ca[1],
            residues[i].ca[2] - residues[i - 2].ca[2],
        ];
        let v2 = [
            residues[i + 2].ca[0] - residues[i].ca[0],
            residues[i + 2].ca[1] - residues[i].ca[1],
            residues[i + 2].ca[2] - residues[i].ca[2],
        ];

        let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();

        if n1 > 1e-6 && n2 > 1e-6 {
            let cos_angle = (dot / (n1 * n2)).clamp(-1.0, 1.0);
            let angle = cos_angle.acos() * 180.0 / PI;
            if angle > BEND_THRESHOLD {
                summary[i] = 'S';
            }
        }
    }

    summary.into_iter().collect()
}

/// Assign DSSP from a pdbtbx PDB structure.
#[allow(dead_code)]
pub(crate) fn dssp_from_pdb(pdb: &pdbtbx::PDB) -> String {
    let residues = extract_dssp_residues(pdb);
    assign_dssp(&residues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hbond_constant() {
        let expected = 0.42 * 0.20 * 332.0;
        assert!((HB_CONSTANT - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dist() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((dist(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = normalize([3.0, 4.0, 0.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dssp_identical_with_placed_vs_virtual_h() {
        // DSSP assignments must be identical whether using virtual H
        // (no H atoms in structure) or placed H atoms.
        use crate::add_hydrogens;

        let (pdb_no_h, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        // DSSP with virtual H (original path)
        let ss_virtual = dssp_from_pdb(&pdb_no_h);

        // Now place real H atoms and run DSSP again
        let mut pdb_with_h = pdb_no_h;
        let result = add_hydrogens::place_peptide_hydrogens(&mut pdb_with_h);
        assert!(result.added > 0);

        let ss_placed = dssp_from_pdb(&pdb_with_h);

        assert_eq!(
            ss_virtual, ss_placed,
            "DSSP must be identical with virtual vs placed H.\n  virtual: {}\n  placed:  {}",
            ss_virtual, ss_placed,
        );
    }

    #[test]
    fn test_dssp_uses_real_h_when_present() {
        // Verify that extract_dssp_residues picks up real H atoms
        use crate::add_hydrogens;

        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        add_hydrogens::place_peptide_hydrogens(&mut pdb);

        let residues = extract_dssp_residues(&pdb);
        let n_with_real_h = residues
            .iter()
            .filter(|r| r.has_h && !r.needs_virtual_h)
            .count();

        assert!(
            n_with_real_h > 0,
            "Should detect placed H atoms as real (not virtual)"
        );

        // Only N-terminal residues (first in each chain) should still need virtual H
        // (they have no previous C to place H from, and we don't place H on them)
        let n_needing_virtual = residues.iter().filter(|r| r.needs_virtual_h).count();
        let n_chains = pdb.chains().count();
        assert_eq!(
            n_needing_virtual, n_chains,
            "After placing H, only N-terminal residues (one per chain) should need virtual H"
        );
    }
}
