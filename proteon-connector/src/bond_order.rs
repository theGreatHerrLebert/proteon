//! Bond order estimation and ring detection for general hydrogen placement.
//!
//! Estimates bond orders from interatomic distances and detects ring atoms
//! using BFS cycle detection. Used by Phase 3 general H placement.

use std::collections::{HashSet, VecDeque};

/// Bond order values matching BALL convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondOrder {
    pub(crate) fn as_f64(self) -> f64 {
        match self {
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Aromatic => 1.5,
        }
    }
}

/// A bond with order information.
#[derive(Clone, Debug)]
pub(crate) struct OrderedBond {
    pub i: usize,
    pub j: usize,
    pub order: BondOrder,
    pub length: f64,
}

/// Atom info for general H placement.
#[derive(Clone, Debug)]
pub(crate) struct AtomInfo {
    pub pos: [f64; 3],
    pub element: String,
    #[allow(dead_code)]
    pub name: String,
    pub neighbors: Vec<usize>,
    pub bonds: Vec<usize>, // indices into bond list
    pub is_ring_atom: bool,
}

/// Full connectivity graph with bond orders and ring info.
pub(crate) struct MolGraph {
    pub atoms: Vec<AtomInfo>,
    pub bonds: Vec<OrderedBond>,
}

// ---------------------------------------------------------------------------
// Bond distance thresholds for order estimation
// Reference: Allen et al., J. Chem. Soc. Perkin Trans. 2, 1987, S1-S19
// ---------------------------------------------------------------------------

/// Estimate bond order from element pair and distance.
fn estimate_bond_order(elem_a: &str, elem_b: &str, dist: f64) -> Option<BondOrder> {
    // Sort elements for consistent lookup
    let (e1, e2) = if elem_a <= elem_b {
        (elem_a, elem_b)
    } else {
        (elem_b, elem_a)
    };

    match (e1, e2) {
        ("C", "C") => {
            if dist < 1.25 {
                Some(BondOrder::Triple)
            } else if dist < 1.38 {
                Some(BondOrder::Double)
            } else if dist < 1.43 {
                Some(BondOrder::Aromatic)
            }
            // 1.38-1.43 aromatic
            else if dist < 1.65 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("C", "N") => {
            if dist < 1.20 {
                Some(BondOrder::Triple)
            } else if dist < 1.33 {
                Some(BondOrder::Double)
            } else if dist < 1.38 {
                Some(BondOrder::Aromatic)
            } else if dist < 1.55 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("C", "O") => {
            if dist < 1.28 {
                Some(BondOrder::Double)
            } else if dist < 1.50 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("C", "S") => {
            if dist < 1.68 {
                Some(BondOrder::Double)
            } else if dist < 1.90 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("N", "N") => {
            if dist < 1.20 {
                Some(BondOrder::Triple)
            } else if dist < 1.30 {
                Some(BondOrder::Double)
            } else if dist < 1.50 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("N", "O") => {
            if dist < 1.25 {
                Some(BondOrder::Double)
            } else if dist < 1.50 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        ("O", "P" | "S") => {
            if dist < 1.55 {
                Some(BondOrder::Double)
            } else if dist < 1.75 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
        // Default: single bond if within bonding distance
        _ => {
            let max_dist = match (e1, e2) {
                (a, _) | (_, a) if a == "H" => 1.3,
                (a, _) | (_, a) if a == "S" || a == "P" => 2.2,
                _ => 1.9,
            };
            if dist < max_dist && dist > 0.4 {
                Some(BondOrder::Single)
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ring detection (BFS-based cycle finding)
// ---------------------------------------------------------------------------

/// Detect ring atoms using BFS.
/// An atom is a ring atom if removing any of its bonds still leaves a path
/// between the bond endpoints (i.e., the bond is part of a cycle).
fn detect_ring_atoms(n_atoms: usize, bonds: &[OrderedBond]) -> Vec<bool> {
    // Build adjacency list
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_atoms];
    for bond in bonds {
        adj[bond.i].insert(bond.j);
        adj[bond.j].insert(bond.i);
    }

    let mut is_ring = vec![false; n_atoms];

    // For each bond, check if endpoints are connected without that bond
    for bond in bonds {
        let (u, v) = (bond.i, bond.j);

        // Temporarily remove this bond
        adj[u].remove(&v);
        adj[v].remove(&u);

        // BFS from u to v (with max depth 8 to avoid huge searches)
        if bfs_connected(&adj, u, v, 8) {
            is_ring[u] = true;
            is_ring[v] = true;
        }

        // Restore bond
        adj[u].insert(v);
        adj[v].insert(u);
    }

    // Also mark atoms that are ring atoms because they're between two ring atoms
    // (handles atoms on ring that weren't caught by individual bond removal)
    // This is already handled above since we check every bond.

    is_ring
}

/// BFS check: is there a path from `start` to `target` within `max_depth` steps?
fn bfs_connected(adj: &[HashSet<usize>], start: usize, target: usize, max_depth: usize) -> bool {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for &neighbor in &adj[node] {
            if neighbor == target {
                return true;
            }
            if visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Aromatic ring refinement
// ---------------------------------------------------------------------------

/// After initial bond order assignment, refine ring bonds that look aromatic.
/// If a ring has alternating single/double bonds where the "single" bonds
/// are shorter than typical single bonds, reclassify all ring bonds as aromatic.
fn refine_aromatic_bonds(atoms: &[AtomInfo], bonds: &mut [OrderedBond]) {
    // Find bonds between two ring atoms with C-C or C-N bonds
    // that have intermediate lengths (1.35-1.45 for C-C, 1.30-1.40 for C-N)
    for bond in bonds.iter_mut() {
        if !atoms[bond.i].is_ring_atom || !atoms[bond.j].is_ring_atom {
            continue;
        }
        let (e1, e2) = (&atoms[bond.i].element, &atoms[bond.j].element);
        let (e1, e2) = if e1 <= e2 {
            (e1.as_str(), e2.as_str())
        } else {
            (e2.as_str(), e1.as_str())
        };

        match (e1, e2) {
            ("C", "C") if bond.length > 1.35 && bond.length < 1.43 => {
                bond.order = BondOrder::Aromatic;
            }
            ("C", "N") if bond.length > 1.30 && bond.length < 1.38 => {
                bond.order = BondOrder::Aromatic;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Build molecular graph
// ---------------------------------------------------------------------------

/// Build a molecular graph from atom positions and elements.
/// Infers bonds from distances, estimates bond orders, detects rings.
pub(crate) fn build_mol_graph(
    positions: &[[f64; 3]],
    elements: &[String],
    names: &[String],
) -> MolGraph {
    let n = positions.len();

    // Step 1: Find bonds and estimate orders from distances
    let mut bonds = Vec::new();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut atom_bonds: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if let Some(order) = estimate_bond_order(&elements[i], &elements[j], dist) {
                let bond_idx = bonds.len();
                bonds.push(OrderedBond {
                    i,
                    j,
                    order,
                    length: dist,
                });
                neighbors[i].push(j);
                neighbors[j].push(i);
                atom_bonds[i].push(bond_idx);
                atom_bonds[j].push(bond_idx);
            }
        }
    }

    // Step 2: Detect ring atoms
    let is_ring = detect_ring_atoms(n, &bonds);

    // Step 3: Build atom info
    let atoms: Vec<AtomInfo> = (0..n)
        .map(|i| AtomInfo {
            pos: positions[i],
            element: elements[i].clone(),
            name: names[i].clone(),
            neighbors: neighbors[i].clone(),
            bonds: atom_bonds[i].clone(),
            is_ring_atom: is_ring[i],
        })
        .collect();

    // Step 4: Refine aromatic bonds in rings
    refine_aromatic_bonds(&atoms, &mut bonds);

    MolGraph { atoms, bonds }
}

// ---------------------------------------------------------------------------
// Valence / connectivity from periodic table
// ---------------------------------------------------------------------------

/// Expected number of bonds for an element (from periodic table group).
/// This is the key function from BALL's _get_connectivity.
pub(crate) fn expected_valence(element: &str) -> u8 {
    match element {
        "H" | "D" | "F" | "Cl" | "Br" | "I" => 1,
        "O" | "S" | "Se" => 2,
        "N" | "P" | "As" => 3,
        "C" | "Si" | "Ge" => 4,
        "B" => 3,
        "Fe" | "Zn" | "Cu" | "Mn" | "Co" | "Ni" | "Mg" | "Ca" => 0, // metals, skip
        _ => 0,
    }
}

/// Sum of bond orders for an atom in the molecular graph.
pub(crate) fn sum_bond_orders(graph: &MolGraph, atom_idx: usize) -> f64 {
    let atom = &graph.atoms[atom_idx];
    let mut sum = 0.0;
    for &bond_idx in &atom.bonds {
        sum += graph.bonds[bond_idx].order.as_f64();
    }
    sum
}

// ---------------------------------------------------------------------------
// MMFF94 bond length calculation (Schomaker-Stevenson rule)
// ---------------------------------------------------------------------------

/// MMFF94 covalent radii indexed by atomic number.
/// From Blom & Haaland, J. Molec. Struc. 1985, 128, 21-27.
fn mmff94_radius(element: &str) -> f64 {
    match element {
        "H" | "D" => 0.33,
        "C" => 0.77,
        "N" => 0.73,
        "O" => 0.72,
        "F" => 0.72,
        "P" => 1.09,
        "S" => 1.03,
        "Cl" => 1.01,
        "Br" => 1.14,
        "I" => 1.33,
        "Se" => 1.17,
        "Si" => 1.17,
        "B" => 0.82,
        _ => 0.77, // default to carbon
    }
}

/// MMFF94 electronegativities (Pauling scale).
fn mmff94_electronegativity(element: &str) -> f64 {
    match element {
        "H" | "D" => 2.20,
        "C" => 2.50,
        "N" => 3.07,
        "O" => 3.50,
        "F" => 4.10,
        "P" => 2.06,
        "S" => 2.44,
        "Cl" => 2.83,
        "Br" => 2.74,
        "I" => 2.21,
        "Se" => 2.48,
        "Si" => 1.74,
        "B" => 2.01,
        _ => 2.50,
    }
}

/// Calculate X-H bond length using the modified Schomaker-Stevenson rule (MMFF94).
/// Formula: r_XH = r_X + r_H - c * |χ_X - χ_H|^n
/// where c = 0.05, n = 1.4
pub(crate) fn mmff94_bond_length(element: &str) -> f64 {
    let r_x = mmff94_radius(element);
    let r_h = mmff94_radius("H");
    let chi_x = mmff94_electronegativity(element);
    let chi_h = mmff94_electronegativity("H");

    let c = 0.05;
    let n = 1.4;
    let diff = (chi_x - chi_h).abs();

    r_x + r_h - c * diff.powf(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_order_cc() {
        assert_eq!(estimate_bond_order("C", "C", 1.54), Some(BondOrder::Single));
        assert_eq!(estimate_bond_order("C", "C", 1.34), Some(BondOrder::Double));
        assert_eq!(estimate_bond_order("C", "C", 1.20), Some(BondOrder::Triple));
        assert_eq!(
            estimate_bond_order("C", "C", 1.40),
            Some(BondOrder::Aromatic)
        );
    }

    #[test]
    fn test_bond_order_co() {
        assert_eq!(estimate_bond_order("C", "O", 1.43), Some(BondOrder::Single));
        assert_eq!(estimate_bond_order("C", "O", 1.23), Some(BondOrder::Double));
    }

    #[test]
    fn test_expected_valence() {
        assert_eq!(expected_valence("C"), 4);
        assert_eq!(expected_valence("N"), 3);
        assert_eq!(expected_valence("O"), 2);
        assert_eq!(expected_valence("H"), 1);
        assert_eq!(expected_valence("S"), 2);
    }

    #[test]
    fn test_mmff94_bond_length() {
        let ch = mmff94_bond_length("C");
        assert!(ch > 1.05 && ch < 1.15, "C-H bond length: {}", ch);

        let nh = mmff94_bond_length("N");
        assert!(nh > 0.95 && nh < 1.05, "N-H bond length: {}", nh);

        let oh = mmff94_bond_length("O");
        assert!(oh > 0.90 && oh < 1.00, "O-H bond length: {}", oh);
    }

    #[test]
    fn test_ring_detection_benzene() {
        // Simple 6-membered ring: hexagonal arrangement
        let r = 1.39; // aromatic C-C distance
        let positions: Vec<[f64; 3]> = (0..6)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::PI / 3.0;
                [r * angle.cos(), r * angle.sin(), 0.0]
            })
            .collect();
        let elements = vec!["C".to_string(); 6];
        let names: Vec<String> = (0..6).map(|i| format!("C{}", i)).collect();

        let graph = build_mol_graph(&positions, &elements, &names);

        // All 6 atoms should be ring atoms
        for (i, atom) in graph.atoms.iter().enumerate() {
            assert!(atom.is_ring_atom, "Atom {} should be a ring atom", i);
        }
    }

    #[test]
    fn test_ring_detection_linear() {
        // Linear chain: no rings
        let positions = vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let elements = vec!["C".to_string(); 3];
        let names = vec!["C1".to_string(), "C2".to_string(), "C3".to_string()];

        let graph = build_mol_graph(&positions, &elements, &names);

        for atom in &graph.atoms {
            assert!(!atom.is_ring_atom);
        }
    }
}
