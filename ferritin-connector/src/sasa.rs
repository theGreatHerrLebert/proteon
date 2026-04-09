//! Shrake-Rupley SASA (Solvent Accessible Surface Area) calculation.
//!
//! Algorithm: place test points on expanded van der Waals spheres,
//! count how many are not buried by neighboring atoms.
//! Uses cell list for O(N) neighbor lookup.

use std::f64::consts::PI;

/// Default probe radius (water molecule) in Angstroms.
#[allow(dead_code)]
pub const PROBE_RADIUS: f64 = 1.4;

/// Default number of test points per sphere (Fibonacci spiral).
/// 960 gives ~0.1 A² precision. 92 is fast but rough.
#[allow(dead_code)]
pub const DEFAULT_N_POINTS: usize = 960;

// ---------------------------------------------------------------------------
// Atomic radii sets
// ---------------------------------------------------------------------------

/// Which atomic radii table to use for SASA computation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RadiiSet {
    /// Bondi (1964) van der Waals radii, element-based.
    /// One radius per element regardless of chemical context.
    #[default]
    Bondi,
    /// ProtOr (Tsai et al. 1999) radii, atom-type-based.
    /// Different radii for the same element depending on residue context.
    /// Compatible with FreeSASA and NACCESS.
    ProtOr,
}

// ---------------------------------------------------------------------------
// Bondi radii (element-based)
// ---------------------------------------------------------------------------

/// Get Bondi van der Waals radius for an element symbol.
/// Returns None for unknown elements.
pub fn vdw_radius(element: &str) -> Option<f64> {
    match element.trim() {
        "C" => Some(1.70),
        "N" => Some(1.55),
        "O" => Some(1.52),
        "S" => Some(1.80),
        "P" => Some(1.80),
        "H" | "D" => Some(1.20),
        "F" => Some(1.47),
        "Cl" => Some(1.75),
        "Br" => Some(1.85),
        "I" => Some(1.98),
        "Se" => Some(1.90),
        "Fe" => Some(1.56), // for heme
        "Zn" => Some(1.39),
        "Cu" => Some(1.40),
        "Mg" => Some(1.73),
        "Ca" => Some(1.74),
        "Mn" => Some(1.61),
        "Co" => Some(1.52),
        "Ni" => Some(1.63),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// ProtOr radii (atom-type-based, residue-context-aware)
// From: Tsai et al. (1999) J Mol Biol 290:253-266
// Same values used by FreeSASA (--radii=protor) and NACCESS
// ---------------------------------------------------------------------------

/// Residues with aromatic rings where CG is sp2 (ring junction → 1.61).
const AROMATIC_RESIDUES: &[&str] = &["PHE", "TYR", "TRP", "HIS"];

/// Aromatic ring carbon names (sp2 → 1.76).
/// CG in aromatics is the ring junction (sp2 → 1.61).
fn is_aromatic_ring_carbon(atom_name: &str, residue_name: &str) -> bool {
    if !AROMATIC_RESIDUES.contains(&residue_name) {
        return false;
    }
    matches!(atom_name,
        "CD1" | "CD2" | "CE1" | "CE2" | "CE3" | "CZ" | "CZ2" | "CZ3" | "CH2"
    )
}

/// Get ProtOr radius for an atom given its name and residue name.
/// Values from Tsai et al. (1999), matching FreeSASA --radii=protor.
pub fn protor_radius(atom_name: &str, residue_name: &str, element: &str) -> f64 {
    let name = atom_name.trim();
    let res = residue_name.trim();
    let elem = element.trim();

    match elem {
        "C" => {
            if name == "C" {
                // Backbone carbonyl carbon (sp2)
                1.61
            } else if name == "CG" && AROMATIC_RESIDUES.contains(&res) {
                // Aromatic ring junction (sp2)
                1.61
            } else if is_aromatic_ring_carbon(name, res) {
                // Aromatic ring carbons (sp2)
                1.76
            } else {
                // All other carbons: CA, CB, CG (non-aromatic), CD, CE, etc. (sp3)
                1.88
            }
        }
        "N" => 1.64,
        "O" => {
            match name {
                "OG" | "OG1" | "OH" => 1.46,  // Hydroxyl
                _ => 1.42,                       // Carbonyl, carboxyl
            }
        }
        "S" => 1.77,
        "H" | "D" => 1.00,
        "Se" => 1.90,
        _ => vdw_radius(elem).unwrap_or(DEFAULT_RADIUS),
    }
}

/// Fallback radius for unknown elements.
pub const DEFAULT_RADIUS: f64 = 1.50;

// ---------------------------------------------------------------------------
// Fibonacci sphere — generate approximately uniform points on a unit sphere
// ---------------------------------------------------------------------------

/// Generate `n` approximately uniformly distributed points on a unit sphere
/// using the golden section spiral (same formula as Biopython/MDTraj).
fn golden_spiral_points(n: usize) -> Vec<[f64; 3]> {
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt()); // ~2.3999 rad
    let dz = 2.0 / n as f64;

    let mut points = Vec::with_capacity(n);
    let mut longitude = 0.0_f64;
    let mut z = 1.0 - dz / 2.0;

    for _ in 0..n {
        let r = (1.0 - z * z).max(0.0).sqrt();
        points.push([longitude.cos() * r, longitude.sin() * r, z]);
        z -= dz;
        longitude += golden_angle;
    }

    points
}

// ---------------------------------------------------------------------------
// Cell list for spatial neighbor search
// ---------------------------------------------------------------------------

struct CellList {
    cells: Vec<Vec<usize>>,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_size: f64,
    min_x: f64,
    min_y: f64,
    min_z: f64,
}

impl CellList {
    fn new(coords: &[[f64; 3]], radii: &[f64], probe: f64) -> Self {
        let max_r = radii.iter().cloned().fold(0.0_f64, f64::max);
        // Cell size must be >= diameter of largest expanded sphere
        // so any neighbor is in adjacent cells
        let cell_size = 2.0 * (max_r + probe) + 0.01;

        let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);
        for c in coords {
            min_x = min_x.min(c[0]);
            min_y = min_y.min(c[1]);
            min_z = min_z.min(c[2]);
            max_x = max_x.max(c[0]);
            max_y = max_y.max(c[1]);
            max_z = max_z.max(c[2]);
        }
        // Add padding
        min_x -= cell_size;
        min_y -= cell_size;
        min_z -= cell_size;

        let nx = ((max_x - min_x) / cell_size).ceil() as usize + 2;
        let ny = ((max_y - min_y) / cell_size).ceil() as usize + 2;
        let nz = ((max_z - min_z) / cell_size).ceil() as usize + 2;

        // Cap grid to avoid OOM on structures with huge bounding boxes
        // (e.g., multi-model NMR, symmetry mates, or bogus coordinates).
        // 150³ cells = 3.375M entries, ~80 MB — fits 120 threads in <10 GB.
        // A normal protein has bbox ~100 Å with cell_size ~6 Å, so nx ~17;
        // 150³ covers structures up to ~900 Å in each dimension, more than enough.
        let max_cells: usize = 150;
        let nx = nx.min(max_cells);
        let ny = ny.min(max_cells);
        let nz = nz.min(max_cells);

        let total = nx * ny * nz;
        let mut cells = vec![Vec::new(); total];

        for (i, c) in coords.iter().enumerate() {
            let cx = ((c[0] - min_x) / cell_size) as usize;
            let cy = ((c[1] - min_y) / cell_size) as usize;
            let cz = ((c[2] - min_z) / cell_size) as usize;
            let idx = cx * ny * nz + cy * nz + cz;
            if idx < total {
                cells[idx].push(i);
            }
        }

        CellList {
            cells,
            nx,
            ny,
            nz,
            cell_size,
            min_x,
            min_y,
            min_z,
        }
    }

    /// Get all atom indices in the 27 cells surrounding the given coordinate.
    /// Appends to `result` (caller should clear before calling).
    fn neighbors_into(&self, pos: &[f64; 3], result: &mut Vec<usize>) {
        let cx = ((pos[0] - self.min_x) / self.cell_size) as isize;
        let cy = ((pos[1] - self.min_y) / self.cell_size) as isize;
        let cz = ((pos[2] - self.min_z) / self.cell_size) as isize;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    let nz = cz + dz;
                    if nx >= 0
                        && ny >= 0
                        && nz >= 0
                        && (nx as usize) < self.nx
                        && (ny as usize) < self.ny
                        && (nz as usize) < self.nz
                    {
                        let idx =
                            nx as usize * self.ny * self.nz + ny as usize * self.nz + nz as usize;
                        result.extend_from_slice(&self.cells[idx]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core SASA computation
// ---------------------------------------------------------------------------

/// Compute per-atom SASA using the Shrake-Rupley algorithm.
///
/// # Arguments
/// * `coords` — Atom coordinates (N x 3)
/// * `radii` — Van der Waals radius per atom (length N)
/// * `probe` — Probe radius (default 1.4 A for water)
/// * `n_points` — Number of test points per sphere (default 960)
///
/// # Returns
/// Per-atom SASA in Angstroms², length N.
pub fn shrake_rupley(
    coords: &[[f64; 3]],
    radii: &[f64],
    probe: f64,
    n_points: usize,
) -> Vec<f64> {
    let n_atoms = coords.len();
    assert_eq!(radii.len(), n_atoms);

    if n_atoms == 0 {
        return Vec::new();
    }

    // Pre-compute expanded radii
    let expanded: Vec<f64> = radii.iter().map(|r| r + probe).collect();
    let expanded_sq: Vec<f64> = expanded.iter().map(|r| r * r).collect();

    // Generate test points on unit sphere
    let unit_points = golden_spiral_points(n_points);
    let inv_n_points = 1.0 / n_points as f64;

    // Build cell list for neighbor lookup
    let cell_list = CellList::new(coords, radii, probe);

    // For each atom, compute SASA
    let mut sasa = vec![0.0f64; n_atoms];
    let mut neighbor_buf = Vec::new();

    for i in 0..n_atoms {
        let ri = expanded[i];
        let ri_sq = expanded_sq[i];
        let xi = coords[i][0];
        let yi = coords[i][1];
        let zi = coords[i][2];

        // Find candidate neighbors (atoms whose expanded spheres could overlap)
        neighbor_buf.clear();
        cell_list.neighbors_into(&coords[i], &mut neighbor_buf);
        let mut neighbors: Vec<usize> = Vec::new();
        for &j in &neighbor_buf {
            if j == i {
                continue;
            }
            let dx = coords[j][0] - xi;
            let dy = coords[j][1] - yi;
            let dz = coords[j][2] - zi;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let sum_r = ri + expanded[j];
            if dist_sq < sum_r * sum_r {
                neighbors.push(j);
            }
        }

        // Count exposed test points
        let mut n_exposed = 0usize;

        'points: for pt in &unit_points {
            // Scale test point to atom's expanded sphere
            let px = xi + ri * pt[0];
            let py = yi + ri * pt[1];
            let pz = zi + ri * pt[2];

            // Check if this point is inside any neighbor's expanded sphere
            for &j in &neighbors {
                let dx = px - coords[j][0];
                let dy = py - coords[j][1];
                let dz = pz - coords[j][2];
                if dx * dx + dy * dy + dz * dz < expanded_sq[j] {
                    continue 'points; // buried
                }
            }

            n_exposed += 1;
        }

        // SASA = fraction_exposed * surface_area_of_expanded_sphere
        sasa[i] = n_exposed as f64 * inv_n_points * 4.0 * PI * ri_sq;
    }

    sasa
}

/// Compute SASA from a pdbtbx PDB structure.
///
/// Uses rayon to parallelize over atoms for large structures.
/// Returns per-atom SASA values in Angstroms².
pub fn sasa_from_pdb(
    pdb: &pdbtbx::PDB,
    probe: f64,
    n_points: usize,
    radii_set: RadiiSet,
) -> Vec<f64> {
    let mut coords = Vec::new();
    let mut radii = Vec::new();

    // Use first model only (consistent with atom_count(), DSSP, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return Vec::new(),
    };
    for chain in first_model.chains() {
        for residue in chain.residues() {
            let res_name = residue.name().unwrap_or("");
            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);

                let elem_str = atom
                    .element()
                    .map(|e| e.symbol())
                    .unwrap_or("");

                let r = match radii_set {
                    RadiiSet::Bondi => {
                        vdw_radius(elem_str).unwrap_or(DEFAULT_RADIUS)
                    }
                    RadiiSet::ProtOr => {
                        protor_radius(atom.name(), res_name, elem_str)
                    }
                };
                radii.push(r);
            }
        }
    }

    // Use rayon-parallel version for large structures
    if coords.len() > 500 {
        shrake_rupley_parallel(&coords, &radii, probe, n_points)
    } else {
        shrake_rupley(&coords, &radii, probe, n_points)
    }
}

/// Compute per-residue SASA by summing atom contributions.
///
/// Returns (residue_sasa, residue_names, chain_ids).
pub fn residue_sasa(pdb: &pdbtbx::PDB, atom_sasa: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    let mut atom_idx = 0;

    // Use first model only (consistent with sasa_from_pdb and atom_count)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return result,
    };
    for chain in first_model.chains() {
        for residue in chain.residues() {
            let n_atoms = residue.atom_count();
            let sum: f64 = atom_sasa[atom_idx..atom_idx + n_atoms].iter().sum();
            result.push(sum);
            atom_idx += n_atoms;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Max SASA per residue type (Tien et al. 2013, theoretical Gly-X-Gly)
// Used for RSA (relative solvent accessibility) calculation
// ---------------------------------------------------------------------------

/// Maximum SASA per residue type in Gly-X-Gly tripeptide (Tien et al. 2013).
pub fn max_sasa(residue_name: &str) -> Option<f64> {
    match residue_name.trim() {
        "ALA" => Some(129.0),
        "ARG" => Some(274.0),
        "ASN" => Some(195.0),
        "ASP" => Some(193.0),
        "CYS" => Some(167.0),
        "GLN" => Some(225.0),
        "GLU" => Some(223.0),
        "GLY" => Some(104.0),
        "HIS" => Some(224.0),
        "ILE" => Some(197.0),
        "LEU" => Some(201.0),
        "LYS" => Some(236.0),
        "MET" => Some(224.0),
        "PHE" => Some(240.0),
        "PRO" => Some(159.0),
        "SER" => Some(155.0),
        "THR" => Some(172.0),
        "TRP" => Some(285.0),
        "TYR" => Some(263.0),
        "VAL" => Some(174.0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Parallel SASA (rayon)
// ---------------------------------------------------------------------------

/// Compute SASA with rayon parallelism over atoms.
/// Same algorithm, but the per-atom loop is parallelized.
pub fn shrake_rupley_parallel(
    coords: &[[f64; 3]],
    radii: &[f64],
    probe: f64,
    n_points: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    let n_atoms = coords.len();
    assert_eq!(radii.len(), n_atoms);

    if n_atoms == 0 {
        return Vec::new();
    }

    let expanded: Vec<f64> = radii.iter().map(|r| r + probe).collect();
    let expanded_sq: Vec<f64> = expanded.iter().map(|r| r * r).collect();
    let unit_points = golden_spiral_points(n_points);
    let inv_n_points = 1.0 / n_points as f64;
    let cell_list = CellList::new(coords, radii, probe);

    // CellList is not Sync, so we wrap in a read-only accessor.
    // Safety: CellList is only read during par_iter (built before, no mutation).
    let cell_ref = &cell_list;

    (0..n_atoms)
        .into_par_iter()
        .map(|i| {
            let ri = expanded[i];
            let ri_sq = expanded_sq[i];
            let xi = coords[i][0];
            let yi = coords[i][1];
            let zi = coords[i][2];

            let mut neighbor_buf = Vec::new();
            cell_ref.neighbors_into(&coords[i], &mut neighbor_buf);
            let mut neighbors: Vec<usize> = Vec::new();
            for &j in &neighbor_buf {
                if j == i {
                    continue;
                }
                let dx = coords[j][0] - xi;
                let dy = coords[j][1] - yi;
                let dz = coords[j][2] - zi;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let sum_r = ri + expanded[j];
                if dist_sq < sum_r * sum_r {
                    neighbors.push(j);
                }
            }

            let mut n_exposed = 0usize;

            for pt in &unit_points {
                let px = xi + ri * pt[0];
                let py = yi + ri * pt[1];
                let pz = zi + ri * pt[2];

                let mut buried = false;
                for &j in &neighbors {
                    let dx = px - coords[j][0];
                    let dy = py - coords[j][1];
                    let dz = pz - coords[j][2];
                    if dx * dx + dy * dy + dz * dz < expanded_sq[j] {
                        buried = true;
                        break;
                    }
                }
                if !buried {
                    n_exposed += 1;
                }
            }

            n_exposed as f64 * inv_n_points * 4.0 * PI * ri_sq
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_atom_full_exposure() {
        // A single isolated atom should have SASA = 4π(r+probe)²
        let coords = vec![[0.0, 0.0, 0.0]];
        let radii = vec![1.7]; // carbon
        let sasa = shrake_rupley(&coords, &radii, 1.4, 960);
        let expected = 4.0 * PI * (1.7 + 1.4_f64).powi(2);
        let rel_err = (sasa[0] - expected).abs() / expected;
        assert!(rel_err < 0.01, "Single atom SASA {:.2} vs expected {:.2}", sasa[0], expected);
    }

    #[test]
    fn test_two_atoms_less_than_single() {
        // Two atoms close together should each have less SASA than isolated
        let coords = vec![[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]];
        let radii = vec![1.7, 1.7];
        let sasa = shrake_rupley(&coords, &radii, 1.4, 960);
        let single_sasa = 4.0 * PI * (1.7 + 1.4_f64).powi(2);
        assert!(sasa[0] < single_sasa);
        assert!(sasa[1] < single_sasa);
        assert!(sasa[0] > 0.0);
    }

    #[test]
    fn test_buried_atom() {
        // An atom completely enclosed by others should have ~0 SASA
        let mut coords = vec![[0.0, 0.0, 0.0]]; // center atom
        let mut radii = vec![1.0];
        // Surround with 6 atoms at ±1.5 on each axis
        for &d in &[[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0],
                     [0.0, 1.5, 0.0], [0.0, -1.5, 0.0],
                     [0.0, 0.0, 1.5], [0.0, 0.0, -1.5]] {
            coords.push(d);
            radii.push(2.0);
        }
        let sasa = shrake_rupley(&coords, &radii, 1.4, 960);
        // Center atom should be nearly fully buried
        assert!(sasa[0] < 5.0, "Buried atom SASA should be near 0, got {:.2}", sasa[0]);
    }

    #[test]
    fn test_golden_spiral_points_distribution() {
        let points = golden_spiral_points(960);
        assert_eq!(points.len(), 960);
        // All points should be on unit sphere
        for p in &points {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_matches_serial() {
        // Generate a small cluster of atoms
        let coords: Vec<[f64; 3]> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.5;
                [t.cos() * 3.0, t.sin() * 3.0, (t * 0.7).sin() * 2.0]
            })
            .collect();
        let radii = vec![1.7; 20];

        let serial = shrake_rupley(&coords, &radii, 1.4, 200);
        let parallel = shrake_rupley_parallel(&coords, &radii, 1.4, 200);

        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert!((s - p).abs() < 1e-10, "Serial {} != Parallel {}", s, p);
        }
    }
}
