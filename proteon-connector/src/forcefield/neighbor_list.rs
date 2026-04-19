//! Cell-list based neighbor list for O(N) nonbonded pair enumeration.
//!
//! Divides space into cubic cells of side length >= cutoff. For each atom,
//! only atoms in the same cell and 26 neighboring cells are checked.
//! Pair list can be reused across multiple energy evaluations and rebuilt
//! when atoms move too far.

use std::collections::HashSet;

/// A nonbonded pair with precomputed flags.
#[derive(Clone, Debug)]
pub struct NBPair {
    pub i: usize,
    pub j: usize,
    pub is_14: bool,
}

/// Neighbor list built from a cell list.
#[derive(Clone, Debug)]
pub struct NeighborList {
    pub pairs: Vec<NBPair>,
    /// Cutoff used to build this list (Å).
    #[allow(dead_code)]
    pub cutoff: f64,
    /// Buffer distance — rebuild when any atom moves more than buffer/2.
    #[allow(dead_code)]
    pub buffer: f64,
    /// Coordinates at build time (for displacement check).
    #[allow(dead_code)]
    ref_coords: Vec<[f64; 3]>,
}

impl NeighborList {
    /// Build a neighbor list from coordinates using a cell list.
    ///
    /// `cutoff` is the interaction cutoff in Å. A buffer of 2 Å is added
    /// to the cell size so the list remains valid for small displacements.
    pub fn build(
        coords: &[[f64; 3]],
        cutoff: f64,
        excluded_pairs: &HashSet<(usize, usize)>,
        pairs_14: &HashSet<(usize, usize)>,
    ) -> Self {
        let buffer = 2.0;
        let list_cutoff = cutoff + buffer;
        let list_cutoff_sq = list_cutoff * list_cutoff;
        let n = coords.len();

        if n == 0 {
            return Self {
                pairs: Vec::new(),
                cutoff,
                buffer,
                ref_coords: Vec::new(),
            };
        }

        // Find bounding box
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for c in coords {
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }

        // Cell dimensions
        let cell_size = list_cutoff;
        let mut ncells = [1usize; 3];
        for d in 0..3 {
            ncells[d] = ((hi[d] - lo[d]) / cell_size).ceil() as usize;
            ncells[d] = ncells[d].max(1);
        }

        // Cap grid to avoid OOM on structures with huge bounding boxes
        // (bogus coordinates, symmetry mates, etc.).
        // 150³ cells = 3.375M entries, ~80 MB — matches CellList in sasa.rs.
        // With cell_size ~17 Å, 150 cells covers ~2550 Å, more than enough
        // for any normal protein.
        const MAX_CELLS: usize = 150;
        for d in 0..3 {
            ncells[d] = ncells[d].min(MAX_CELLS);
        }

        let total_cells = ncells[0] * ncells[1] * ncells[2];

        // Assign atoms to cells
        let mut cell_atoms: Vec<Vec<usize>> = vec![Vec::new(); total_cells];
        let mut atom_cell = vec![0usize; n];

        for i in 0..n {
            let cx = ((coords[i][0] - lo[0]) / cell_size) as usize;
            let cy = ((coords[i][1] - lo[1]) / cell_size) as usize;
            let cz = ((coords[i][2] - lo[2]) / cell_size) as usize;
            let cx = cx.min(ncells[0] - 1);
            let cy = cy.min(ncells[1] - 1);
            let cz = cz.min(ncells[2] - 1);
            let cell_idx = cx * ncells[1] * ncells[2] + cy * ncells[2] + cz;
            cell_atoms[cell_idx].push(i);
            atom_cell[i] = cell_idx;
        }

        // Build pair list by iterating over cell pairs
        let mut pairs = Vec::new();

        for cx in 0..ncells[0] {
            for cy in 0..ncells[1] {
                for cz in 0..ncells[2] {
                    let cell_a = cx * ncells[1] * ncells[2] + cy * ncells[2] + cz;

                    // Check this cell + 13 forward neighbors (half-shell to avoid double counting)
                    // Offsets: (0,0,0) for intra-cell, then 13 forward neighbors
                    for &(dx, dy, dz) in &HALF_SHELL {
                        let nx = cx as isize + dx;
                        let ny = cy as isize + dy;
                        let nz = cz as isize + dz;

                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                        if nx >= ncells[0] || ny >= ncells[1] || nz >= ncells[2] {
                            continue;
                        }

                        let cell_b = nx * ncells[1] * ncells[2] + ny * ncells[2] + nz;

                        if cell_a == cell_b {
                            // Intra-cell pairs
                            let atoms = &cell_atoms[cell_a];
                            for ai in 0..atoms.len() {
                                for aj in (ai + 1)..atoms.len() {
                                    let (i, j) = (atoms[ai], atoms[aj]);
                                    let (lo_idx, hi_idx) = (i.min(j), i.max(j));
                                    let pair_key = (lo_idx, hi_idx);
                                    if excluded_pairs.contains(&pair_key) {
                                        continue;
                                    }
                                    let d2 = dist_sq(&coords[i], &coords[j]);
                                    if d2 <= list_cutoff_sq {
                                        pairs.push(NBPair {
                                            i: lo_idx,
                                            j: hi_idx,
                                            is_14: pairs_14.contains(&pair_key),
                                        });
                                    }
                                }
                            }
                        } else {
                            // Inter-cell pairs
                            for &i in &cell_atoms[cell_a] {
                                for &j in &cell_atoms[cell_b] {
                                    let (lo_idx, hi_idx) = (i.min(j), i.max(j));
                                    let pair_key = (lo_idx, hi_idx);
                                    if excluded_pairs.contains(&pair_key) {
                                        continue;
                                    }
                                    let d2 = dist_sq(&coords[i], &coords[j]);
                                    if d2 <= list_cutoff_sq {
                                        pairs.push(NBPair {
                                            i: lo_idx,
                                            j: hi_idx,
                                            is_14: pairs_14.contains(&pair_key),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            pairs,
            cutoff,
            buffer,
            ref_coords: coords.to_vec(),
        }
    }

    /// Check if the list needs rebuilding (any atom moved > buffer/2).
    #[allow(dead_code)]
    pub fn needs_rebuild(&self, coords: &[[f64; 3]]) -> bool {
        let threshold_sq = (self.buffer / 2.0) * (self.buffer / 2.0);
        for (i, c) in coords.iter().enumerate() {
            if i >= self.ref_coords.len() {
                return true;
            }
            let d2 = dist_sq(c, &self.ref_coords[i]);
            if d2 > threshold_sq {
                return true;
            }
        }
        false
    }
}

/// Half-shell of 14 cell neighbor offsets (self + 13 forward neighbors).
/// This ensures each pair is counted exactly once.
const HALF_SHELL: [(isize, isize, isize); 14] = [
    (0, 0, 0),   // self
    (1, 0, 0),   // +x
    (1, 1, 0),   // +x+y
    (1, -1, 0),  // +x-y
    (0, 1, 0),   // +y
    (1, 0, 1),   // +x+z
    (1, 0, -1),  // +x-z
    (0, 1, 1),   // +y+z
    (0, 1, -1),  // +y-z
    (0, 0, 1),   // +z
    (1, 1, 1),   // +x+y+z
    (1, 1, -1),  // +x+y-z
    (1, -1, 1),  // +x-y+z
    (1, -1, -1), // +x-y-z
];

#[inline]
fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
