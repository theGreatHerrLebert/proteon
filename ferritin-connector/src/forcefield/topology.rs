//! Bond topology inference and force field atom assignment.
//!
//! Infers bonds from interatomic distances, then builds the lists
//! of bond/angle/torsion interactions needed for energy computation.

use std::collections::{HashMap, HashSet};

use super::params::ForceField;
use crate::fragment_templates;

/// Per-atom data for force field computation.
#[derive(Clone, Debug)]
pub struct FFAtom {
    pub pos: [f64; 3],
    pub amber_type: String,
    pub charge: f64,
    #[allow(dead_code)]
    pub residue_name: String,
    #[allow(dead_code)]
    pub atom_name: String,
    pub element: String,
    pub residue_idx: usize,
    pub is_hydrogen: bool,
}

/// A bond between two atoms.
#[derive(Clone, Debug)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
}

/// An angle between three atoms (i-j-k, j is central).
#[derive(Clone, Debug)]
pub struct Angle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
}

/// A torsion/dihedral between four atoms (i-j-k-l).
#[derive(Clone, Debug)]
pub struct Torsion {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
}

/// Complete topology for energy computation.
#[derive(Clone, Debug)]
pub struct Topology {
    pub atoms: Vec<FFAtom>,
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub torsions: Vec<Torsion>,
    /// Improper torsions for planarity (aromatic rings, peptide bonds)
    pub improper_torsions: Vec<Torsion>,
    /// Pairs that are 1-2 or 1-3 connected (excluded from nonbonded)
    pub excluded_pairs: HashSet<(usize, usize)>,
    /// 1-4 pairs (scaled nonbonded)
    pub pairs_14: HashSet<(usize, usize)>,
    /// Atoms that could not be assigned a force field type (residue:atom names)
    pub unassigned_atoms: Vec<String>,
}

/// Maximum bond distance for element pairs (Å).
fn max_bond_distance(elem_a: &str, elem_b: &str) -> f64 {
    match (elem_a, elem_b) {
        ("H", _) | (_, "H") => 1.3,
        ("S", _) | (_, "S") => 2.2,
        _ => 1.9, // C-C, C-N, C-O, N-O, etc.
    }
}

/// Build topology from a PDB structure.
pub fn build_topology(pdb: &pdbtbx::PDB, params: &impl ForceField) -> Topology {
    // Step 0: Pre-scan to detect terminal residues and disulfide CYS.
    // Each chain's first amino acid residue gets "-N" suffix,
    // last gets "-C" suffix. CYS with SG near another SG gets "-S" suffix.
    let mut residue_variants: HashMap<usize, String> = HashMap::new();
    let mut res_idx = 0usize;

    // Collect SG positions for disulfide detection
    let mut sg_positions: Vec<(usize, [f64; 3])> = Vec::new(); // (res_idx, pos)

    // Use first model only (consistent with atom_count(), SASA, DSSP, etc.)
    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return Topology {
            atoms: Vec::new(), bonds: Vec::new(), angles: Vec::new(),
            torsions: Vec::new(), improper_torsions: Vec::new(),
            excluded_pairs: HashSet::new(), pairs_14: HashSet::new(),
            unassigned_atoms: Vec::new(),
        },
    };
    for chain in first_model.chains() {
        let mut chain_residues: Vec<(usize, String)> = Vec::new(); // (res_idx, name)
        for residue in chain.residues() {
            let name = residue.name().unwrap_or("UNK").to_string();
            let is_aa = residue.conformers().next()
                .map_or(false, |c| c.is_amino_acid());
            if is_aa {
                chain_residues.push((res_idx, name.clone()));
            }
            // Find SG atoms for disulfide detection
            if name == "CYS" {
                for atom in residue.atoms() {
                    if atom.name().trim() == "SG" {
                        let (x, y, z) = atom.pos();
                        sg_positions.push((res_idx, [x, y, z]));
                    }
                }
            }
            res_idx += 1;
        }
        // Mark first and last amino acid in chain
        if let Some((first_idx, first_name)) = chain_residues.first() {
            residue_variants.insert(*first_idx, format!("{}-N", first_name));
        }
        if let Some((last_idx, last_name)) = chain_residues.last() {
            residue_variants.insert(*last_idx, format!("{}-C", last_name));
        }
    }

    // Note: CYS-S (disulfide) variant is NOT applied automatically.
    // BALL only applies CYS-S when explicitly flagged during preprocessing,
    // not from raw PDB geometry. Applying it here would change charges and
    // diverge from BALL's behavior.

    // Step 1: Extract atoms with type assignment using variant names
    let mut atoms = Vec::new();
    let mut unassigned_atoms = Vec::new();
    res_idx = 0;

    for chain in first_model.chains() {
        for residue in chain.residues() {
            let base_name = residue.name().unwrap_or("UNK").to_string();
            let lookup_name = residue_variants.get(&res_idx)
                .cloned()
                .unwrap_or_else(|| base_name.clone());

            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                let atom_name = atom.name().trim().to_string();
                let element = atom
                    .element()
                    .map(|e| e.symbol().to_string())
                    .unwrap_or_else(|| "C".to_string());
                let is_hydrogen = element == "H" || element == "D";

                // Look up AMBER type and charge: try variant name first, then base name
                let (amber_type, charge) = params.get_atom_type(&lookup_name, &atom_name)
                    .or_else(|| params.get_atom_type(&base_name, &atom_name))
                    .map(|e| (e.amber_type.clone(), e.charge))
                    .unwrap_or_else(|| {
                        unassigned_atoms.push(format!("{}:{}", base_name, atom_name));
                        let t = match element.as_str() {
                            "C" => "CT",
                            "N" => "N",
                            "O" => "O",
                            "S" => "S",
                            "H" => "H",
                            "P" => "P",
                            _ => "CT",
                        };
                        (t.to_string(), 0.0)
                    });

                atoms.push(FFAtom {
                    pos: [x, y, z],
                    amber_type,
                    charge,
                    residue_name: base_name.clone(),
                    atom_name,
                    element,
                    residue_idx: res_idx,
                    is_hydrogen,
                });
            }
            res_idx += 1;
        }
    }

    // Step 2: Build bonds from fragment templates + peptide/disulfide connections.
    //
    // Phase A: Intra-residue bonds from templates (chemical knowledge).
    // Phase B: Inter-residue peptide bonds (C-N within 1.8 Å between adjacent residues).
    // Phase C: Disulfide bonds (SG-SG within 2.5 Å).
    // Phase D: Distance fallback for residues without templates.
    let n = atoms.len();
    let mut bonds = Vec::new();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut bonded_set: HashSet<(usize, usize)> = HashSet::new();

    // Build an index: (residue_idx, atom_name) → global atom index
    let mut res_atom_index: HashMap<(usize, &str), usize> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        res_atom_index.insert((atom.residue_idx, &atom.atom_name), i);
    }

    // Build a template lookup
    let templates: HashMap<&str, &fragment_templates::FragmentTemplate> = fragment_templates::TEMPLATES
        .iter()
        .map(|t| (t.0, t))
        .collect();

    // Group atoms by residue_idx
    let mut residue_atoms: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        residue_atoms.entry(atom.residue_idx).or_default().push(i);
    }

    // Phase A: Intra-residue bonds from templates
    let mut residues_with_templates: HashSet<usize> = HashSet::new();
    for (&res_idx, atom_indices) in &residue_atoms {
        let res_name = &atoms[atom_indices[0]].residue_name;
        if let Some(template) = templates.get(res_name.as_str()) {
            residues_with_templates.insert(res_idx);
            let tpl_bonds = template.2;
            for &(a1_name, a2_name) in tpl_bonds {
                if let (Some(&i), Some(&j)) = (
                    res_atom_index.get(&(res_idx, a1_name)),
                    res_atom_index.get(&(res_idx, a2_name)),
                ) {
                    let pair = (i.min(j), i.max(j));
                    if bonded_set.insert(pair) {
                        bonds.push(Bond { i: pair.0, j: pair.1 });
                        neighbors[pair.0].push(pair.1);
                        neighbors[pair.1].push(pair.0);
                    }
                }
            }
        }
    }

    // Phase B: Inter-residue peptide bonds (C of res i → N of res i+1)
    let max_res_idx = atoms.iter().map(|a| a.residue_idx).max().unwrap_or(0);
    for res_idx in 0..max_res_idx {
        if let (Some(&c_idx), Some(&n_idx)) = (
            res_atom_index.get(&(res_idx, "C")),
            res_atom_index.get(&(res_idx + 1, "N")),
        ) {
            let dx = atoms[c_idx].pos[0] - atoms[n_idx].pos[0];
            let dy = atoms[c_idx].pos[1] - atoms[n_idx].pos[1];
            let dz = atoms[c_idx].pos[2] - atoms[n_idx].pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1.8 && dist > 0.5 {
                let pair = (c_idx.min(n_idx), c_idx.max(n_idx));
                if bonded_set.insert(pair) {
                    bonds.push(Bond { i: pair.0, j: pair.1 });
                    neighbors[pair.0].push(pair.1);
                    neighbors[pair.1].push(pair.0);
                }
            }
        }
    }

    // Phase B2: Terminal bonds (OXT-C for C-terminal, H1/H2/H3 for N-terminal)
    for i in 0..n {
        let name = atoms[i].atom_name.as_str();
        let res_idx = atoms[i].residue_idx;
        let parent = match name {
            "OXT" => Some("C"),
            "H1" | "H2" | "H3" => Some("N"),
            _ => None,
        };
        if let Some(parent_name) = parent {
            if let Some(&p_idx) = res_atom_index.get(&(res_idx, parent_name)) {
                let pair = (i.min(p_idx), i.max(p_idx));
                if bonded_set.insert(pair) {
                    bonds.push(Bond { i: pair.0, j: pair.1 });
                    neighbors[pair.0].push(pair.1);
                    neighbors[pair.1].push(pair.0);
                }
            }
        }
    }

    // Phase C: Disulfide bonds (SG-SG within 2.5 Å)
    let sg_atoms: Vec<usize> = atoms.iter().enumerate()
        .filter(|(_, a)| a.atom_name == "SG" && (a.residue_name == "CYS" || a.residue_name == "CYX"))
        .map(|(i, _)| i)
        .collect();
    for ai in 0..sg_atoms.len() {
        for aj in (ai + 1)..sg_atoms.len() {
            let (i, j) = (sg_atoms[ai], sg_atoms[aj]);
            let dx = atoms[i].pos[0] - atoms[j].pos[0];
            let dy = atoms[i].pos[1] - atoms[j].pos[1];
            let dz = atoms[i].pos[2] - atoms[j].pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 2.5 {
                let pair = (i.min(j), i.max(j));
                if bonded_set.insert(pair) {
                    bonds.push(Bond { i: pair.0, j: pair.1 });
                    neighbors[pair.0].push(pair.1);
                    neighbors[pair.1].push(pair.0);
                }
            }
        }
    }

    // Phase D: Distance fallback for residues without templates (ligands, non-standard)
    for (&res_idx, atom_indices) in &residue_atoms {
        if residues_with_templates.contains(&res_idx) {
            continue;
        }
        for ai in 0..atom_indices.len() {
            for aj in (ai + 1)..atom_indices.len() {
                let (i, j) = (atom_indices[ai], atom_indices[aj]);
                let dx = atoms[i].pos[0] - atoms[j].pos[0];
                let dy = atoms[i].pos[1] - atoms[j].pos[1];
                let dz = atoms[i].pos[2] - atoms[j].pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let max_dist = max_bond_distance(&atoms[i].element, &atoms[j].element);
                if dist < max_dist && dist > 0.4 {
                    let pair = (i.min(j), i.max(j));
                    if bonded_set.insert(pair) {
                        bonds.push(Bond { i: pair.0, j: pair.1 });
                        neighbors[pair.0].push(pair.1);
                        neighbors[pair.1].push(pair.0);
                    }
                }
            }
        }
    }

    // Step 3: Build angles from bonds
    let mut angles = Vec::new();
    for j in 0..n {
        let nbrs = &neighbors[j];
        for ni in 0..nbrs.len() {
            for nk in (ni + 1)..nbrs.len() {
                angles.push(Angle {
                    i: nbrs[ni],
                    j,
                    k: nbrs[nk],
                });
            }
        }
    }

    // Step 4: Build torsions from angles
    let mut torsions = Vec::new();
    let mut excluded_pairs = HashSet::new();
    let mut pairs_14 = HashSet::new();

    // Record 1-2 and 1-3 exclusions
    for bond in &bonds {
        excluded_pairs.insert((bond.i.min(bond.j), bond.i.max(bond.j)));
    }
    for angle in &angles {
        excluded_pairs.insert((angle.i.min(angle.k), angle.i.max(angle.k)));
    }

    // Build torsions: for each bond j-k, find i bonded to j and l bonded to k
    for bond in &bonds {
        let j = bond.i;
        let k = bond.j;
        for &i in &neighbors[j] {
            if i == k {
                continue;
            }
            for &l in &neighbors[k] {
                if l == j || l == i {
                    continue;
                }
                torsions.push(Torsion { i, j, k, l });
                let pair = (i.min(l), i.max(l));
                if !excluded_pairs.contains(&pair) {
                    pairs_14.insert(pair);
                }
            }
        }
    }

    // Step 5: Build improper torsions
    // For each atom with exactly 3 bonds that is listed in ResidueImproperTorsions,
    // generate improper torsion entries (central atom is position 3 in the 4-tuple).
    let mut improper_torsions = Vec::new();
    for j in 0..n {
        let nbrs = &neighbors[j];
        if nbrs.len() != 3 {
            continue;
        }
        if !params.is_improper_center(&atoms[j].residue_name, &atoms[j].atom_name) {
            continue;
        }
        // Generate all permutations of the 3 neighbors as (a1, a2, center=j, a4)
        let (n0, n1, n2) = (nbrs[0], nbrs[1], nbrs[2]);
        for &(i, k, l) in &[(n0, n1, n2), (n0, n2, n1), (n1, n0, n2),
                             (n1, n2, n0), (n2, n0, n1), (n2, n1, n0)] {
            let ti = &atoms[i].amber_type;
            let tk = &atoms[j].amber_type; // central atom
            let tj = &atoms[k].amber_type;
            let tl = &atoms[l].amber_type;
            if params.get_improper_torsion(ti, tj, tk, tl).is_some() {
                improper_torsions.push(Torsion { i, j: k, k: j, l });
                break; // only one improper per center atom per neighbor set
            }
        }
    }

    Topology {
        atoms,
        bonds,
        angles,
        torsions,
        improper_torsions,
        excluded_pairs,
        pairs_14,
        unassigned_atoms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::params;

    #[test]
    fn test_crambin_disulfides() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("read");
        let amber = params::amber96();
        let topo = build_topology(&pdb, &amber);

        // Find SG atoms
        let sg: Vec<(usize, &str, usize)> = topo.atoms.iter().enumerate()
            .filter(|(_, a)| a.atom_name == "SG")
            .map(|(i, a)| (i, a.residue_name.as_str(), a.residue_idx))
            .collect();
        println!("SG atoms: {:?}", sg);

        // Check which SG-SG pairs are bonded
        for i in 0..sg.len() {
            for j in (i+1)..sg.len() {
                let (ai, _, _) = sg[i];
                let (aj, _, _) = sg[j];
                let pair = (ai.min(aj), ai.max(aj));
                let is_bonded = topo.excluded_pairs.contains(&pair) ||
                    topo.bonds.iter().any(|b| b.i == pair.0 && b.j == pair.1);
                let dx = topo.atoms[ai].pos[0] - topo.atoms[aj].pos[0];
                let dy = topo.atoms[ai].pos[1] - topo.atoms[aj].pos[1];
                let dz = topo.atoms[ai].pos[2] - topo.atoms[aj].pos[2];
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                println!("SG[{}]-SG[{}] dist={:.3} bonded={} excluded={}",
                    ai, aj, dist, is_bonded, topo.excluded_pairs.contains(&pair));
            }
        }

        // Crambin should have 3 disulfides
        let ss_bonds: Vec<_> = topo.bonds.iter()
            .filter(|b| topo.atoms[b.i].atom_name == "SG" && topo.atoms[b.j].atom_name == "SG")
            .collect();
        println!("SS bonds found: {}", ss_bonds.len());
        assert_eq!(ss_bonds.len(), 3, "Crambin should have 3 disulfide bonds");
    }

    #[test]
    fn test_crambin_bond_count() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("read");
        let amber = params::amber96();
        let topo = build_topology(&pdb, &amber);

        println!("Atoms: {}", topo.atoms.len());
        println!("Bonds: {}", topo.bonds.len());
        println!("Angles: {}", topo.angles.len());
        println!("Torsions: {}", topo.torsions.len());
        println!("Improper torsions: {}", topo.improper_torsions.len());
        println!("Excluded pairs: {}", topo.excluded_pairs.len());
        println!("1-4 pairs: {}", topo.pairs_14.len());

        // BALL finds 337 bonds on 327-atom crambin
        // We expect ~333 template + ~45 peptide + ~3 disulfide = ~381 candidates
        // (but deduplication reduces this)
        assert!(topo.bonds.len() > 300, "Should have > 300 bonds, got {}", topo.bonds.len());
        assert!(topo.bonds.len() < 400, "Should have < 400 bonds, got {}", topo.bonds.len());

        // Debug: print all improper torsion centers
        println!("Improper torsions ({}):", topo.improper_torsions.len());
        for imp in &topo.improper_torsions {
            let a = &topo.atoms[imp.k]; // center atom (position 3)
            let ti = &topo.atoms[imp.i].amber_type;
            let tj = &topo.atoms[imp.j].amber_type;
            let tk = &a.amber_type;
            let tl = &topo.atoms[imp.l].amber_type;
            println!("  center={}:{} types={}-{}-{}-{}",
                a.residue_name, a.atom_name, ti, tj, tk, tl);
        }
    }

    #[test]
    fn test_crambin_vdw_debug() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("read");
        let amber = super::super::params::amber96();
        let topo = build_topology(&pdb, &amber);
        let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();

        // Compute VdW pair by pair, find top contributors
        let n = coords.len();
        let mut pairs: Vec<(f64, usize, usize, f64)> = Vec::new(); // (energy, i, j, dist)

        for i in 0..n {
            for j in (i+1)..n {
                let pair = (i, j);
                if topo.excluded_pairs.contains(&pair) { continue; }

                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let r2 = dx*dx + dy*dy + dz*dz;
                if r2 > 225.0 || r2 < 0.01 { continue; } // 15 Å cutoff

                let r = r2.sqrt();
                let ti = &topo.atoms[i].amber_type;
                let tj = &topo.atoms[j].amber_type;
                if let (Some(lj_i), Some(lj_j)) = (amber.get_lj(ti), amber.get_lj(tj)) {
                    let eps = (lj_i.epsilon * lj_j.epsilon).sqrt();
                    let rmin = lj_i.r + lj_j.r;
                    if eps > 1e-10 && rmin > 1e-10 {
                        let is_14 = topo.pairs_14.contains(&pair);
                        let scale = if is_14 { 0.5 } else { 1.0 };
                        let sr = rmin / r;
                        let sr6 = sr.powi(6);
                        let e = scale * eps * (sr6 * sr6 - 2.0 * sr6);
                        pairs.push((e, i, j, r));
                    }
                }
            }
        }

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending
        println!("Top 10 VdW contributors:");
        let total: f64 = pairs.iter().map(|p| p.0).sum();
        println!("Total VdW: {:.1} kcal/mol", total);
        for (e, i, j, r) in pairs.iter().take(10) {
            let is14 = topo.pairs_14.contains(&(*i, *j));
            println!("  E={:+10.1} i={} {}:{} j={} {}:{} r={:.3} 1-4={}",
                e, i, topo.atoms[*i].residue_name, topo.atoms[*i].atom_name,
                j, topo.atoms[*j].residue_name, topo.atoms[*j].atom_name,
                r, is14);
        }
    }
}
