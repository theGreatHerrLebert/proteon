//! Hydrogen bond detection.
//!
//! Two methods:
//! 1. Kabsch-Sander backbone H-bonds (electrostatic energy criterion)
//! 2. Geometric H-bonds (distance + angle criteria, works for all atoms)

use crate::dssp;

// ---------------------------------------------------------------------------
// Backbone H-bonds (Kabsch-Sander)
// ---------------------------------------------------------------------------

/// A detected backbone hydrogen bond.
#[derive(Clone, Debug)]
pub struct BackboneHBond {
    /// Acceptor residue index (CO group)
    pub acceptor: usize,
    /// Donor residue index (NH group)
    pub donor: usize,
    /// Kabsch-Sander energy (kcal/mol, more negative = stronger)
    pub energy: f64,
    /// O...N distance (Å)
    pub dist_on: f64,
}

/// Compute all backbone H-bonds using the Kabsch-Sander energy criterion.
///
/// Returns a list of (acceptor_idx, donor_idx, energy, dist_ON) for all
/// pairs where energy < cutoff (default -0.5 kcal/mol).
pub fn backbone_hbonds(
    pdb: &pdbtbx::PDB,
    energy_cutoff: f64,
) -> Vec<BackboneHBond> {
    let residues = dssp::extract_dssp_residues(pdb);
    backbone_hbonds_from_residues(&residues, energy_cutoff)
}

pub fn backbone_hbonds_from_residues(
    residues: &[dssp::DsspResidue],
    energy_cutoff: f64,
) -> Vec<BackboneHBond> {
    let n = residues.len();
    let mut result = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if residues[i].chain_idx != residues[j].chain_idx { continue; }
            if (i as isize - j as isize).unsigned_abs() < 2 { continue; }
            if !residues[j].has_h { continue; }

            let r_on = dist3(&residues[i].o, &residues[j].n);
            if r_on > 5.2 { continue; }

            let r_ch = dist3(&residues[i].c, &residues[j].h);
            let r_oh = dist3(&residues[i].o, &residues[j].h);
            let r_cn = dist3(&residues[i].c, &residues[j].n);

            if r_on < 0.01 || r_ch < 0.01 || r_oh < 0.01 || r_cn < 0.01 {
                continue;
            }

            let energy = 27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);

            if energy < energy_cutoff {
                result.push(BackboneHBond {
                    acceptor: i,
                    donor: j,
                    energy,
                    dist_on: r_on,
                });
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Geometric H-bonds (all atoms)
// ---------------------------------------------------------------------------

/// A detected geometric hydrogen bond.
#[derive(Clone, Debug)]
pub struct GeometricHBond {
    /// Donor atom index (the heavy atom, e.g., N or O)
    pub donor_atom: usize,
    /// Acceptor atom index (the heavy atom, e.g., O)
    pub acceptor_atom: usize,
    /// Donor-Acceptor distance (Å)
    pub distance: f64,
    /// Donor residue index
    #[allow(dead_code)]
    pub donor_residue: usize,
    /// Acceptor residue index
    #[allow(dead_code)]
    pub acceptor_residue: usize,
}

/// Detect geometric hydrogen bonds between donor (N, O) and acceptor (O, N, S) atoms.
///
/// Uses distance criterion only (no angle — would need H positions).
/// Default cutoff: donor-acceptor distance < 3.5 Å.
///
/// Donors: N atoms (backbone and sidechain)
/// Acceptors: O atoms (backbone and sidechain), S in Cys/Met
pub fn geometric_hbonds(
    pdb: &pdbtbx::PDB,
    dist_cutoff: f64,
) -> Vec<GeometricHBond> {
    // Collect donor and acceptor atoms with their positions
    struct AtomInfo {
        pos: [f64; 3],
        atom_idx: usize,
        res_idx: usize,
    }

    let mut donors: Vec<AtomInfo> = Vec::new();
    let mut acceptors: Vec<AtomInfo> = Vec::new();
    let mut atom_idx = 0;
    let mut res_idx = 0;

    // Use first model only (consistent with atom_count(), SASA, DSSP, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return Vec::new(),
    };
    for chain in first_model.chains() {
        for residue in chain.residues() {
            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                let pos = [x, y, z];
                let elem = atom.element().map(|e| e.symbol());

                match elem {
                    Some("N") => {
                        donors.push(AtomInfo { pos, atom_idx, res_idx });
                    }
                    Some("O") => {
                        // O is both donor and acceptor
                        donors.push(AtomInfo { pos, atom_idx, res_idx });
                        acceptors.push(AtomInfo { pos, atom_idx, res_idx });
                    }
                    Some("S") => {
                        acceptors.push(AtomInfo { pos, atom_idx, res_idx });
                    }
                    _ => {}
                }
                atom_idx += 1;
            }
            res_idx += 1;
        }
    }

    let cutoff_sq = dist_cutoff * dist_cutoff;
    let mut result = Vec::new();

    for donor in &donors {
        for acceptor in &acceptors {
            if donor.atom_idx == acceptor.atom_idx { continue; }
            // Skip same residue
            if donor.res_idx == acceptor.res_idx { continue; }

            let dx = donor.pos[0] - acceptor.pos[0];
            let dy = donor.pos[1] - acceptor.pos[1];
            let dz = donor.pos[2] - acceptor.pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq {
                result.push(GeometricHBond {
                    donor_atom: donor.atom_idx,
                    acceptor_atom: acceptor.atom_idx,
                    distance: dist_sq.sqrt(),
                    donor_residue: donor.res_idx,
                    acceptor_residue: acceptor.res_idx,
                });
            }
        }
    }

    result
}

#[inline]
fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
