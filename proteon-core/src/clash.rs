//! Heavy-atom steric clash detection for label-safe preparation.
//!
//! A clash metric that gates whether a prepared structure is safe to use as a
//! geometric-DL training label. It is **force-field independent on purpose**:
//! the question is "do heavy atoms physically overlap", not "is the CHARMM/AMBER
//! Lennard-Jones energy high" (the latter depends on possibly-wrong atom typing,
//! which is itself one of the hazards we are guarding against). So it uses
//! element-based Bondi van der Waals radii ([`crate::sasa::vdw_radius`]) and the
//! MolProbity 0.4 Å overlap convention.
//!
//! Bond exclusions (1-2 and 1-3 pairs) come from the REAL force-field topology
//! ([`Topology::excluded_pairs`]), not a distance-inferred bond graph — distance
//! inference can hide a genuine clash by mistaking two overlapping non-bonded
//! atoms for a bond. Because the topology is built from the primary conformer,
//! mutually-exclusive alternate-location pairs are already excluded (they never
//! coexist physically). 1-4 pairs ARE excluded too (like MolProbity's `probe`,
//! which ignores atoms ≤3 bonds apart): heavy-atom 1-4 distances in normal
//! geometry — aromatic-ring *para* carbons at ~2.8 Å, gauche backbone — fall
//! inside the Bondi contact distance, so counting them flags every aromatic ring
//! as a clash. A clash is therefore an overlap between atoms ≥4 bonds apart.
//!
//! Scope: this is a PROTEIN clash metric. Pairs touching an un-templated residue
//! (ligand / non-standard / metal — `Topology::inferred_residues`) are skipped,
//! for two reasons. Intra-ligand contacts are unreliable — the distance-inferred
//! bond graph can't be told from overlaps without bond-order chemistry. And
//! protein–ligand contacts at a binding site are EXPECTED tight contacts
//! (chemistry), not coordinate errors, so they do not corrupt a protein label.
//! The count therefore reflects protein-coordinate quality (validated: pristine
//! 1crn → 0, older / lower-resolution structures → many). Ligand geometry and
//! protein–ligand clashes are a separate ligand-chemistry hazard;
//! `Topology::inferred_bonds` flags that un-templated residues were present so
//! their contacts are known to be EXCLUDED rather than silently mis-counted
//! (those residues are also flagged by the FF-coverage signals `fully_typed` /
//! `untyped_cofactors`).

use std::collections::HashMap;

use crate::forcefield::topology::Topology;
use crate::sasa::vdw_radius;

/// MolProbity-style clash threshold: a heavy-atom pair clashes when its van der
/// Waals spheres overlap by more than this distance (Å).
pub const CLASH_OVERLAP_TOL: f64 = 0.4;

/// Fallback radius for elements absent from the Bondi table (rare heteroatoms);
/// the carbon radius is a conservative middle-of-the-road heavy-atom value.
const FALLBACK_RADIUS: f64 = 1.70;

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn cell_of(p: [f64; 3], cell: f64) -> (i32, i32, i32) {
    (
        (p[0] / cell).floor() as i32,
        (p[1] / cell).floor() as i32,
        (p[2] / cell).floor() as i32,
    )
}

/// Count heavy-atom steric clashes on `coords` (same ordering and length as
/// `topo.atoms`).
///
/// A clash is a pair of heavy atoms (both non-hydrogen) that is NOT in
/// `topo.excluded_pairs` (i.e. not 1-2 bonded and not 1-3 angle-related) and
/// whose interatomic distance `d` satisfies `d < r_i + r_j - CLASH_OVERLAP_TOL`,
/// using element-based Bondi radii. Neighbour search is a uniform grid, so the
/// cost is O(N) in the number of heavy atoms rather than O(N²).
///
/// Each clashing pair is counted once.
pub fn count_heavy_clashes(coords: &[[f64; 3]], topo: &Topology) -> usize {
    // (topology index, Bondi radius) for every heavy atom.
    let heavy: Vec<(usize, f64)> = topo
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.is_hydrogen)
        .map(|(i, a)| (i, vdw_radius(&a.element).unwrap_or(FALLBACK_RADIUS)))
        .collect();
    if heavy.len() < 2 {
        return 0;
    }

    let max_r = heavy.iter().map(|&(_, r)| r).fold(0.0_f64, f64::max);
    // Two atoms can clash only within r_i + r_j - tol <= 2*max_r - tol. A cell
    // size of the max contact distance means all clashing partners fall in the
    // 27-cell neighbourhood.
    let cell = (2.0 * max_r - CLASH_OVERLAP_TOL).max(1.0);

    // Grid: cell -> positions into `heavy`.
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (pos, &(ti, _)) in heavy.iter().enumerate() {
        grid.entry(cell_of(coords[ti], cell)).or_default().push(pos);
    }

    let mut clashes = 0usize;
    for (pos, &(ti, ri)) in heavy.iter().enumerate() {
        let ci = coords[ti];
        let (cx, cy, cz) = cell_of(ci, cell);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &other in bucket {
                        // Count each unordered pair once.
                        if other <= pos {
                            continue;
                        }
                        let (tj, rj) = heavy[other];
                        let key = (ti.min(tj), ti.max(tj));
                        // Exclude 1-2, 1-3 (excluded_pairs) and 1-4 (pairs_14):
                        // atoms ≤3 bonds apart are not clash candidates.
                        if topo.excluded_pairs.contains(&key) || topo.pairs_14.contains(&key) {
                            continue;
                        }
                        // Skip any pair touching an un-templated residue (ligand /
                        // non-standard / metal). This is a PROTEIN clash metric:
                        //   * intra-ligand contacts are unreliable (distance-inferred
                        //     bonds can't be told from overlaps);
                        //   * protein–ligand contacts at a binding site are expected
                        //     tight contacts (chemistry), not coordinate errors, and
                        //     they do not corrupt a protein-coordinate label.
                        // Ligand geometry / protein–ligand clashes are a separate
                        // ligand-chemistry hazard; un-templated residues are flagged
                        // via `clash_count_inferred` and the FF-coverage signals.
                        let (ai, aj) = (&topo.atoms[ti], &topo.atoms[tj]);
                        if topo.inferred_residues.contains(&ai.residue_idx)
                            || topo.inferred_residues.contains(&aj.residue_idx)
                        {
                            continue;
                        }
                        let contact = ri + rj - CLASH_OVERLAP_TOL;
                        if contact > 0.0 && dist2(ci, coords[tj]) < contact * contact {
                            clashes += 1;
                        }
                    }
                }
            }
        }
    }
    clashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::topology::{FFAtom, Topology};
    use std::collections::HashSet;

    /// Minimal topology with only the fields `count_heavy_clashes` reads:
    /// atoms (pos/element/is_hydrogen) and `excluded_pairs`.
    fn topo(atoms: &[([f64; 3], &str, bool)], excluded: &[(usize, usize)]) -> Topology {
        let atoms = atoms
            .iter()
            .map(|&(pos, element, is_hydrogen)| FFAtom {
                pos,
                amber_type: String::new(),
                charge: 0.0,
                residue_name: String::new(),
                atom_name: String::new(),
                element: element.to_string(),
                residue_idx: 0,
                is_hydrogen,
            })
            .collect();
        Topology {
            atoms,
            bonds: Vec::new(),
            angles: Vec::new(),
            torsions: Vec::new(),
            improper_torsions: Vec::new(),
            excluded_pairs: excluded
                .iter()
                .map(|&(i, j)| (i.min(j), i.max(j)))
                .collect(),
            pairs_14: HashSet::new(),
            lj_excluded_pairs: HashSet::new(),
            unassigned_atoms: Vec::new(),
            inferred_bonds: false,
            inferred_residues: std::collections::HashSet::new(),
        }
    }

    fn coords(t: &Topology) -> Vec<[f64; 3]> {
        t.atoms.iter().map(|a| a.pos).collect()
    }

    #[test]
    fn clean_contact_is_not_a_clash() {
        // Two carbons at 3.4 Å = exactly the Bondi contact distance (1.70+1.70):
        // overlap 0, not a clash.
        let t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([3.4, 0.0, 0.0], "C", false)],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn overlapping_pair_is_a_clash() {
        // 2.9 Å apart: overlap = 3.4 - 2.9 = 0.5 Å > 0.4 tol -> clash.
        let t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([2.9, 0.0, 0.0], "C", false)],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 1);
    }

    #[test]
    fn just_inside_tolerance_is_not_a_clash() {
        // 3.05 Å: overlap = 0.35 Å < 0.4 tol -> not a clash (boundary).
        let t = topo(
            &[
                ([0.0, 0.0, 0.0], "C", false),
                ([3.05, 0.0, 0.0], "C", false),
            ],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn bonded_pair_is_excluded() {
        // Two carbons at a bond distance (1.5 Å) would "overlap" massively, but a
        // 1-2 excluded pair must NOT count as a clash.
        let t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([1.5, 0.0, 0.0], "C", false)],
            &[(0, 1)],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn angle_1_3_pair_is_excluded() {
        // 1-3 atoms (e.g. across an angle) sit ~2.4 Å apart — inside vdW overlap
        // but geometrically constrained, not a clash. They are in excluded_pairs.
        let t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([2.4, 0.0, 0.0], "C", false)],
            &[(0, 1)],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn hydrogens_are_ignored() {
        // Two hydrogens on top of each other are not a HEAVY-atom clash.
        let t = topo(
            &[([0.0, 0.0, 0.0], "H", true), ([0.5, 0.0, 0.0], "H", true)],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn counts_each_pair_once_and_scales() {
        // Three mutually-overlapping carbons (all pairs < contact, none excluded)
        // -> 3 unordered clashing pairs.
        let t = topo(
            &[
                ([0.0, 0.0, 0.0], "C", false),
                ([2.8, 0.0, 0.0], "C", false),
                ([1.4, 1.4, 0.0], "C", false),
            ],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 3);
    }

    #[test]
    fn empty_and_single_atom() {
        let t0 = topo(&[], &[]);
        assert_eq!(count_heavy_clashes(&coords(&t0), &t0), 0);
        let t1 = topo(&[([0.0, 0.0, 0.0], "C", false)], &[]);
        assert_eq!(count_heavy_clashes(&coords(&t1), &t1), 0);
    }

    #[test]
    fn one_four_pair_is_excluded() {
        // A 1-4 pair (aromatic para carbons, gauche backbone) at clash distance
        // must NOT count — atoms ≤3 bonds apart are excluded.
        let mut t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([2.9, 0.0, 0.0], "C", false)],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 1); // control: would clash
        t.pairs_14.insert((0, 1));
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn untemplated_residue_intra_pair_is_skipped() {
        // Two overlapping atoms in the SAME un-templated residue: skipped (their
        // inferred bond graph can't tell a bond from an overlap).
        let mut t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([2.9, 0.0, 0.0], "C", false)],
            &[],
        );
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 1); // control: protein -> clash
        t.inferred_residues.insert(0); // mark residue 0 un-templated
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }

    #[test]
    fn protein_ligand_pair_is_skipped() {
        // A protein atom (res 0) overlapping a ligand atom (res 1, un-templated):
        // an expected binding contact, not a protein-coordinate error -> skipped.
        let mut t = topo(
            &[([0.0, 0.0, 0.0], "C", false), ([2.9, 0.0, 0.0], "C", false)],
            &[],
        );
        t.atoms[1].residue_idx = 1;
        t.inferred_residues.insert(1);
        assert_eq!(count_heavy_clashes(&coords(&t), &t), 0);
    }
}
