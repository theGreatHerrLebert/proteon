// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Distance-based bond inference, ported from AutoDock-Vina
// src/lib/model.cpp::assign_bonds (Apache-2.0). Upstream author:
// Oleg Trott, Scripps Research Institute.

//! Bond inference from atom coordinates and covalent radii.
//!
//! Two atoms `i`, `j` are bonded iff their distance is below
//! `1.1 × (cov_radius_i + cov_radius_j)`. This matches upstream's
//! `bond_length_allowance_factor = 1.1` in `model::assign_bonds`.
//! For v0 we skip the `atom_exists_between` collinearity guard —
//! on well-prepared structures the 1.1× threshold is tight enough
//! that spurious bonds are rare. We can add the guard later if needed.
//!
//! v0 is intentionally O(N²). Receptors of a few thousand atoms
//! finish in well under a second; a spatial partition is a trivial
//! drop-in later.

use crate::ad_types::{covalent_radius, AdType};

/// Upstream's `bond_length_allowance_factor` — the slack tolerated
/// on top of summed covalent radii before declaring a bond.
pub const BOND_LENGTH_ALLOWANCE_FACTOR: f64 = 1.1;

/// Adjacency list: `graph[i]` holds the indices of atoms bonded to `i`.
pub type BondGraph = Vec<Vec<usize>>;

/// An (atom, AD-type, coords) triple — the minimum the inference
/// needs. Works with any slice of this shape so callers can feed
/// `RawAtom`s, fully-typed `Atom`s, or lightweight test fixtures.
pub trait BondInput {
    fn ad_type(&self) -> AdType;
    fn coords(&self) -> [f64; 3];
}

impl BondInput for crate::pdbqt::RawAtom {
    #[inline]
    fn ad_type(&self) -> AdType {
        self.ad_type
    }
    #[inline]
    fn coords(&self) -> [f64; 3] {
        self.coords
    }
}

/// Infer the bond graph among `atoms`. The returned adjacency list
/// is symmetric: `graph[i]` contains `j` iff `graph[j]` contains `i`.
/// Neighbor lists are sorted for stability.
#[must_use]
pub fn infer_bonds<A: BondInput>(atoms: &[A]) -> BondGraph {
    infer_bonds_masked(atoms, None)
}

/// Infer bonds with an optional per-atom `fragment_mask`. Two atoms
/// are only bonded if their masks share at least one bit — matching
/// upstream `model::assign_bonds`, which skips pairs at
/// `DISTANCE_VARIABLE` mobility. Pass `None` for mask-free inference
/// (the natural choice for rigid receptors without a torsion tree).
///
/// Macrocycle closure: G0 and CG0 atoms placed at identical
/// coordinates would otherwise become 1-2 "ghost-bonded" in the
/// naive graph, corrupting every subsequent 1-2/1-3/1-4 exclusion
/// check. The mask filter suppresses that.
#[must_use]
pub fn infer_bonds_masked<A: BondInput>(
    atoms: &[A],
    fragment_mask: Option<&[u64]>,
) -> BondGraph {
    let n = atoms.len();
    let mut graph = vec![Vec::<usize>::new(); n];
    let cov: Vec<f64> = atoms.iter().map(|a| covalent_radius(a.ad_type())).collect();
    let coords: Vec<[f64; 3]> = atoms.iter().map(|a| a.coords()).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if let Some(mask) = fragment_mask {
                if mask[i] & mask[j] == 0 {
                    continue;
                }
            }
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let threshold = BOND_LENGTH_ALLOWANCE_FACTOR * (cov[i] + cov[j]);
            if r2 < threshold * threshold {
                graph[i].push(j);
                graph[j].push(i);
            }
        }
    }

    graph
}

/// True if atom `i` is bonded to any atom with AD type `Hd` (polar
/// hydrogen). Mirrors upstream `model::bonded_to_HD`.
#[must_use]
pub fn bonded_to_hd<A: BondInput>(graph: &BondGraph, atoms: &[A], i: usize) -> bool {
    graph[i]
        .iter()
        .any(|&j| atoms[j].ad_type() == AdType::Hd)
}

/// True if AD type `ad` counts as a heteroatom per upstream's
/// `ad_is_heteroatom`: anything other than `A` (aromatic C), `C`,
/// `H`, `HD`. Note this includes the macrocycle-closure carbons
/// `CG0`-`CG3` (they are NOT `AD_TYPE_C`), which matters for
/// carbon XS-type polarity classification near macrocycle closures.
#[inline]
#[must_use]
pub const fn ad_is_heteroatom(ad: AdType) -> bool {
    !matches!(ad, AdType::A | AdType::C | AdType::H | AdType::Hd)
}

/// True if atom `i` is bonded to any heteroatom. Mirrors upstream
/// `model::bonded_to_heteroatom`, which dispatches via
/// `ad_is_heteroatom`.
#[must_use]
pub fn bonded_to_heteroatom<A: BondInput>(graph: &BondGraph, atoms: &[A], i: usize) -> bool {
    graph[i]
        .iter()
        .any(|&j| ad_is_heteroatom(atoms[j].ad_type()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::{parse_pdbqt_atoms, RawAtom};

    /// Minimal `BondInput` used to hand-build tiny test systems
    /// without going through the PDBQT parser.
    struct TestAtom {
        ad: AdType,
        xyz: [f64; 3],
    }

    impl BondInput for TestAtom {
        fn ad_type(&self) -> AdType {
            self.ad
        }
        fn coords(&self) -> [f64; 3] {
            self.xyz
        }
    }

    fn atom(ad: AdType, x: f64, y: f64, z: f64) -> TestAtom {
        TestAtom { ad, xyz: [x, y, z] }
    }

    // --- correctness on hand-built systems -------------------------------

    #[test]
    fn two_carbons_at_cc_bond_distance_are_bonded() {
        // cov(C) = 0.77, threshold = 1.1 * 1.54 = 1.694. At 1.54 Å
        // we're well below.
        let atoms = vec![atom(AdType::C, 0.0, 0.0, 0.0), atom(AdType::C, 1.54, 0.0, 0.0)];
        let g = infer_bonds(&atoms);
        assert_eq!(g[0], vec![1]);
        assert_eq!(g[1], vec![0]);
    }

    #[test]
    fn two_carbons_far_apart_are_not_bonded() {
        // 3.0 Å >> 1.694 Å threshold.
        let atoms = vec![atom(AdType::C, 0.0, 0.0, 0.0), atom(AdType::C, 3.0, 0.0, 0.0)];
        let g = infer_bonds(&atoms);
        assert!(g[0].is_empty());
        assert!(g[1].is_empty());
    }

    #[test]
    fn c_h_at_typical_bond_distance_are_bonded() {
        // cov(C)+cov(H) = 0.77 + 0.37 = 1.14; threshold = 1.254.
        // Typical C-H bond ≈ 1.09 Å.
        let atoms = vec![atom(AdType::C, 0.0, 0.0, 0.0), atom(AdType::H, 1.09, 0.0, 0.0)];
        let g = infer_bonds(&atoms);
        assert_eq!(g[0], vec![1]);
    }

    #[test]
    fn water_oxygen_bonds_to_two_polar_hydrogens() {
        // Idealised water geometry.
        let atoms = vec![
            atom(AdType::Oa, 0.0, 0.0, 0.0),
            atom(AdType::Hd, 0.9572, 0.0, 0.0),
            atom(AdType::Hd, -0.2399, 0.9272, 0.0),
        ];
        let g = infer_bonds(&atoms);
        assert_eq!(g[0].len(), 2);
        assert!(g[0].contains(&1) && g[0].contains(&2));
        assert_eq!(g[1], vec![0]);
        assert_eq!(g[2], vec![0]);
    }

    #[test]
    fn graph_is_always_symmetric_and_sorted() {
        // Random-ish configuration.
        let atoms = vec![
            atom(AdType::C, 0.0, 0.0, 0.0),
            atom(AdType::N, 1.47, 0.0, 0.0),
            atom(AdType::Oa, 0.0, 1.4, 0.0),
            atom(AdType::Hd, 2.5, 1.0, 0.5),
        ];
        let g = infer_bonds(&atoms);
        for (i, nbrs) in g.iter().enumerate() {
            // Sorted
            assert!(nbrs.windows(2).all(|w| w[0] < w[1]));
            // Symmetric
            for &j in nbrs {
                assert!(g[j].contains(&i), "graph not symmetric at ({i},{j})");
            }
        }
    }

    // --- helpers ---------------------------------------------------------

    #[test]
    fn bonded_to_hd_detects_polar_hydrogen_neighbours() {
        // C bonded to HD: this is how upstream decides "donor-capable" N/O.
        let atoms = vec![
            atom(AdType::N, 0.0, 0.0, 0.0),
            atom(AdType::Hd, 1.0, 0.0, 0.0),
            atom(AdType::C, -1.47, 0.0, 0.0),
        ];
        let g = infer_bonds(&atoms);
        assert!(bonded_to_hd(&g, &atoms, 0));
        assert!(!bonded_to_hd(&g, &atoms, 2));
    }

    #[test]
    fn bonded_to_heteroatom_distinguishes_ch_from_cp() {
        // Central C bonded to C + H + H → no heteroatom → C_H.
        // Central C bonded to C + H + O → heteroatom → C_P.
        let ch_ch_h_h = vec![
            atom(AdType::C, 0.0, 0.0, 0.0),
            atom(AdType::C, 1.54, 0.0, 0.0),
            atom(AdType::H, -0.5, 0.9, 0.0),
            atom(AdType::H, -0.5, -0.9, 0.0),
        ];
        let g1 = infer_bonds(&ch_ch_h_h);
        assert!(!bonded_to_heteroatom(&g1, &ch_ch_h_h, 0));

        let ch_ch_h_o = vec![
            atom(AdType::C, 0.0, 0.0, 0.0),
            atom(AdType::C, 1.54, 0.0, 0.0),
            atom(AdType::H, -0.5, 0.9, 0.0),
            atom(AdType::Oa, -0.5, -0.9, 0.0),
        ];
        let g2 = infer_bonds(&ch_ch_h_o);
        assert!(bonded_to_heteroatom(&g2, &ch_ch_h_o, 0));
    }

    // --- integration on real fixtures ------------------------------------

    const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");

    fn parse_ligand() -> Vec<RawAtom> {
        parse_pdbqt_atoms(LIGAND_FIXTURE).expect("ligand parse")
    }

    #[test]
    fn real_ligand_graph_is_connected() {
        // 1iep imatinib is a single connected molecule. BFS from atom 0
        // must reach every atom.
        let atoms = parse_ligand();
        let g = infer_bonds(&atoms);
        let mut visited = vec![false; atoms.len()];
        let mut stack = vec![0];
        visited[0] = true;
        while let Some(i) = stack.pop() {
            for &j in &g[i] {
                if !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        assert!(
            visited.iter().all(|&v| v),
            "ligand bond graph is disconnected: {} isolated atoms",
            visited.iter().filter(|&&v| !v).count()
        );
    }

    #[test]
    fn real_ligand_every_heavy_atom_has_at_least_one_bond() {
        // Trivially true for any molecule with more than one atom and
        // reasonable geometry; serves as a sanity check that the
        // threshold isn't so tight we miss real bonds.
        let atoms = parse_ligand();
        let g = infer_bonds(&atoms);
        for (i, a) in atoms.iter().enumerate() {
            assert!(
                !g[i].is_empty(),
                "atom {i} ({:?}) has no bonds",
                a.ad_type
            );
        }
    }

    #[test]
    fn real_ligand_bond_count_is_reasonable() {
        // Number of edges = number of heavy-heavy + heavy-H bonds.
        // Imatinib has 37 heavy atoms + polar hydrogens → expect
        // roughly 40-55 bonds in the graph. This is a loose envelope.
        let atoms = parse_ligand();
        let g = infer_bonds(&atoms);
        let edges: usize = g.iter().map(|nbrs| nbrs.len()).sum::<usize>() / 2;
        assert!(
            (40..80).contains(&edges),
            "unexpected bond count: {edges}"
        );
    }
}
