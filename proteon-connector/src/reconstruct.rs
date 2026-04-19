//! Missing atom reconstruction using fragment templates.
//!
//! Adds missing heavy atoms and hydrogens to standard amino acid residues
//! by comparing against template structures and placing missing atoms
//! using 3-point rigid body superposition (BALL algorithm).
//!
//! Reference: BALL ReconstructFragmentProcessor (Hildebrandt et al.)

use std::collections::{HashMap, HashSet, VecDeque};

use crate::fragment_templates::{self, TplAtom, TplBond};

// ---------------------------------------------------------------------------
// 3-point rigid body alignment (match_points)
// ---------------------------------------------------------------------------

fn v_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn v_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn v_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn v_norm_sq(a: [f64; 3]) -> f64 {
    v_dot(a, a)
}

fn v_normalize(a: [f64; 3]) -> [f64; 3] {
    let n = v_norm_sq(a).sqrt();
    if n < 1e-10 {
        return [0.0; 3];
    }
    v_scale(a, 1.0 / n)
}

/// 3x3 matrix type (row-major).
type Mat3 = [[f64; 3]; 3];

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mat_neg(m: Mat3) -> Mat3 {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn mat_vec(m: Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Rotation matrix for 180° rotation around an axis (Rodrigues).
fn rotation_180(axis: [f64; 3]) -> Mat3 {
    let a = v_normalize(axis);
    // R = 2 * a*a^T - I
    [
        [
            2.0 * a[0] * a[0] - 1.0,
            2.0 * a[0] * a[1],
            2.0 * a[0] * a[2],
        ],
        [
            2.0 * a[1] * a[0],
            2.0 * a[1] * a[1] - 1.0,
            2.0 * a[1] * a[2],
        ],
        [
            2.0 * a[2] * a[0],
            2.0 * a[2] * a[1],
            2.0 * a[2] * a[2] - 1.0,
        ],
    ]
}

/// Rotation matrix for angle around axis (Rodrigues).
fn rotation_axis_angle(axis: [f64; 3], angle: f64) -> Mat3 {
    let k = v_normalize(axis);
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    [
        [
            t * k[0] * k[0] + c,
            t * k[0] * k[1] - s * k[2],
            t * k[0] * k[2] + s * k[1],
        ],
        [
            t * k[1] * k[0] + s * k[2],
            t * k[1] * k[1] + c,
            t * k[1] * k[2] - s * k[0],
        ],
        [
            t * k[2] * k[0] - s * k[1],
            t * k[2] * k[1] + s * k[0],
            t * k[2] * k[2] + c,
        ],
    ]
}

/// Compute rotation and translation that maps template points (w1,w2,w3)
/// onto actual points (v1,v2,v3).
///
/// Returns (translation, rotation) such that: v = rotation * w + translation
///
/// Port of BALL's matchPoints / BiochemicalAlgorithms.jl match_points.
fn match_points(
    w1: [f64; 3],
    w2: [f64; 3],
    w3: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> ([f64; 3], Mat3) {
    let eps = 1e-5;
    let eps2 = 1e-8;

    let mut tw2 = v_sub(w2, w1);
    let mut tw3 = v_sub(w3, w1);
    let mut tv2 = v_sub(v2, v1);
    let tv3_orig = v_sub(v3, v1);

    // Handle degenerate cases: swap if first two points too close
    if v_norm_sq(tv2) < eps2 && v_norm_sq(tv3_orig) >= eps2 {
        tv2 = tv3_orig;
    }
    if v_norm_sq(tw2) < eps2 && v_norm_sq(tw3) >= eps2 {
        std::mem::swap(&mut tw2, &mut tw3);
    }

    let mut final_translation = v_scale(w1, -1.0);
    let mut final_rotation = IDENTITY;

    if v_norm_sq(tv2) >= eps2 && v_norm_sq(tw2) >= eps2 {
        tw2 = v_normalize(tw2);
        tv2 = v_normalize(tv2);

        let rotation_axis = v_add(tw2, tv2);

        let rotation = if v_norm_sq(rotation_axis) < eps {
            // Antiparallel — negate
            mat_neg(IDENTITY)
        } else {
            // Rotate 180° around the sum axis
            rotation_180(rotation_axis)
        };

        let _tw2 = mat_vec(rotation, tw2);
        tw3 = mat_vec(rotation, tw3);
        final_rotation = mat_mul(rotation, final_rotation);
        final_translation = mat_vec(rotation, final_translation);

        // Second rotation: align tw3 onto tv3
        let tv3 = tv3_orig;
        if v_norm_sq(tw3) > eps2 && v_norm_sq(tv3) > eps2 {
            let tw3n = v_normalize(tw3);
            let tv3n = v_normalize(tv3);

            let axis_w = v_cross(tv2, tw3n);
            let axis_v = v_cross(tv2, tv3n);

            if v_norm_sq(axis_v) > eps2 && v_norm_sq(axis_w) > eps2 {
                let axis_v = v_normalize(axis_v);
                let axis_w = v_normalize(axis_w);

                let rot_axis = v_cross(axis_w, axis_v);

                let rotation = if v_norm_sq(rot_axis) < eps2 {
                    if v_dot(axis_w, axis_v) < 0.0 {
                        rotation_180(tv2)
                    } else {
                        IDENTITY
                    }
                } else {
                    let product = v_dot(axis_w, axis_v).clamp(-1.0, 1.0);
                    let angle = product.acos();
                    if angle > eps {
                        rotation_axis_angle(rot_axis, angle)
                    } else {
                        IDENTITY
                    }
                };

                final_rotation = mat_mul(rotation, final_rotation);
                final_translation = mat_vec(rotation, final_translation);
            }
        }
    }

    final_translation = v_add(final_translation, v1);
    (final_translation, final_rotation)
}

// ---------------------------------------------------------------------------
// Fragment reconstruction
// ---------------------------------------------------------------------------

/// Result of fragment reconstruction.
pub(crate) struct ReconstructResult {
    /// Number of atoms added.
    pub added: usize,
}

/// Find two reference atoms (already placed) near `center` in the template bond graph.
/// Returns (found_two, ref1_name, ref2_name).
fn get_two_reference_atoms<'a>(
    center: &'a str,
    placed: &HashSet<&str>,
    bonds: &'a [TplBond],
) -> (bool, Option<&'a str>, Option<&'a str>) {
    // BFS from center, collecting placed atoms
    let mut found: Vec<&str> = vec![center];
    let mut visited: HashSet<&str> = HashSet::new();
    visited.insert(center);

    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(center);

    while found.len() < 3 {
        let current = match queue.pop_front() {
            Some(c) => c,
            None => break,
        };

        for &(a1, a2) in bonds {
            let neighbor = if a1 == current {
                a2
            } else if a2 == current {
                a1
            } else {
                continue;
            };

            if visited.contains(neighbor) {
                continue;
            }
            visited.insert(neighbor);

            if placed.contains(neighbor) {
                found.push(neighbor);
                if found.len() >= 3 {
                    break;
                }
            }
            queue.push_back(neighbor);
        }
    }

    (
        found.len() >= 3,
        found.get(1).copied(),
        found.get(2).copied(),
    )
}

/// Reconstruct missing atoms in a single residue using a fragment template.
///
/// Returns list of (atom_name, element, position) for atoms to add.
pub(crate) fn reconstruct_residue(
    existing_atoms: &HashMap<String, [f64; 3]>,
    template_atoms: &[TplAtom],
    template_bonds: &[TplBond],
) -> Vec<(String, String, [f64; 3])> {
    let mut result = Vec::new();

    // Build template atom lookup
    let tpl_pos: HashMap<&str, [f64; 3]> = template_atoms
        .iter()
        .map(|&(name, _, pos)| (name, pos))
        .collect();

    let tpl_elem: HashMap<&str, &str> = template_atoms
        .iter()
        .map(|&(name, elem, _)| (name, elem))
        .collect();

    // Classify atoms: existing (placed) vs missing
    let mut placed: HashSet<&str> = HashSet::new();
    let mut actual_pos: HashMap<&str, [f64; 3]> = HashMap::new();

    for &(name, _, _) in template_atoms {
        if let Some(&pos) = existing_atoms.get(name) {
            placed.insert(name);
            actual_pos.insert(name, pos);
        }
    }

    if placed.is_empty() {
        return result; // nothing to anchor to
    }

    // Find missing atoms
    let missing: Vec<&str> = template_atoms
        .iter()
        .filter(|&&(name, _, _)| !placed.contains(name))
        .map(|&(name, _, _)| name)
        .collect();

    if missing.is_empty() {
        return result; // nothing to add
    }

    // BFS from placed atoms, placing missing ones as we encounter them
    let mut visited: HashSet<&str> = HashSet::new();
    let mut stack: Vec<&str> = Vec::new();

    // Start from first placed atom
    if let Some(&start) = placed.iter().next() {
        stack.push(start);
    }

    while let Some(current) = stack.pop() {
        if visited.contains(current) {
            continue;
        }
        visited.insert(current);

        // Find neighbors via bonds
        for &(a1, a2) in template_bonds {
            let neighbor = if a1 == current {
                a2
            } else if a2 == current {
                a1
            } else {
                continue;
            };

            if visited.contains(neighbor) {
                continue;
            }
            stack.push(neighbor);

            if !placed.contains(neighbor) {
                // This atom is missing — compute its position
                let (hit, ref1, ref2) = get_two_reference_atoms(current, &placed, template_bonds);

                let (translation, rotation) = if hit {
                    let r1 = ref1.unwrap();
                    let r2 = ref2.unwrap();
                    match_points(
                        tpl_pos[current],
                        tpl_pos[r1],
                        tpl_pos[r2],
                        actual_pos[current],
                        actual_pos[r1],
                        actual_pos[r2],
                    )
                } else {
                    // Only have center atom — simple translation
                    (v_sub(actual_pos[current], tpl_pos[current]), IDENTITY)
                };

                // Transform the template position of the missing atom
                let tpl = tpl_pos[neighbor];
                let new_pos = v_add(mat_vec(rotation, tpl), translation);

                actual_pos.insert(neighbor, new_pos);
                placed.insert(neighbor);

                let elem = tpl_elem.get(neighbor).copied().unwrap_or("C");
                result.push((neighbor.to_string(), elem.to_string(), new_pos));
            }
        }
    }

    result
}

/// Reconstruct missing atoms in all standard amino acid residues of a PDB.
pub(crate) fn reconstruct_fragments(pdb: &mut pdbtbx::PDB) -> ReconstructResult {
    let mut placements: Vec<(usize, usize, String, String, [f64; 3], usize)> = Vec::new();
    let mut max_serial: usize = crate::altloc::pdb_atoms_primary(pdb)
        .map(|a| a.serial_number())
        .max()
        .unwrap_or(0);

    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return ReconstructResult { added: 0 },
    };

    for (chain_idx, chain) in first_model.chains().enumerate() {
        for (residue_idx, residue) in chain.residues().enumerate() {
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if !is_aa {
                continue;
            }

            let resname = match residue.name() {
                Some(n) => n,
                None => continue,
            };

            let template = match fragment_templates::get_template(resname) {
                Some(t) => t,
                None => continue,
            };

            // Collect existing atoms (primary conformer only)
            let mut existing: HashMap<String, [f64; 3]> = HashMap::new();
            for atom in crate::altloc::residue_atoms_primary(residue) {
                let (x, y, z) = atom.pos();
                existing.insert(atom.name().trim().to_string(), [x, y, z]);
            }

            let new_atoms = reconstruct_residue(&existing, template.1, template.2);

            for (name, elem, pos) in new_atoms {
                max_serial += 1;
                placements.push((chain_idx, residue_idx, name, elem, pos, max_serial));
            }
        }
    }

    let added = placements.len();

    // Write pass
    for (chain_idx, residue_idx, name, elem, pos, serial) in placements {
        let model = match pdb.model_mut(0) {
            Some(m) => m,
            None => continue,
        };
        let chain = match model.chain_mut(chain_idx) {
            Some(c) => c,
            None => continue,
        };
        let residue = match chain.residue_mut(residue_idx) {
            Some(r) => r,
            None => continue,
        };
        let conformer = match residue.conformers_mut().next() {
            Some(c) => c,
            None => continue,
        };

        let is_h = elem == "H" || elem == "D";
        if let Some(atom) = pdbtbx::Atom::new(
            false,
            serial,
            &name,
            &name,
            pos[0],
            pos[1],
            pos[2],
            1.0,
            if is_h { 0.0 } else { 20.0 }, // default B-factor for reconstructed atoms
            &elem,
            0,
        ) {
            conformer.add_atom(atom);
        }
    }

    ReconstructResult { added }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_points_identity() {
        // Same points → identity transform
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let (t, r) = match_points(p1, p2, p3, p1, p2, p3);
        // Translation should be ~zero, rotation ~identity
        assert!(v_norm_sq(t) < 1e-6, "translation: {:?}", t);
        for i in 0..3 {
            assert!((r[i][i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_match_points_translation() {
        let w1 = [0.0, 0.0, 0.0];
        let w2 = [1.0, 0.0, 0.0];
        let w3 = [0.0, 1.0, 0.0];
        let v1 = [5.0, 3.0, 1.0];
        let v2 = [6.0, 3.0, 1.0];
        let v3 = [5.0, 4.0, 1.0];
        let (t, _r) = match_points(w1, w2, w3, v1, v2, v3);

        // Apply transform to w1
        let result = v_add(mat_vec(_r, w1), t);
        let dist = v_norm_sq(v_sub(result, v1)).sqrt();
        assert!(dist < 1e-4, "mapped w1 should be at v1, dist={}", dist);
    }

    #[test]
    fn test_reconstruct_ala_missing_cb() {
        // ALA with CB missing
        let template = fragment_templates::get_template("ALA").unwrap();
        let mut existing = HashMap::new();
        // Add backbone atoms only
        for &(name, _, pos) in template.1 {
            if name == "N" || name == "CA" || name == "C" || name == "O" {
                existing.insert(name.to_string(), pos);
            }
        }

        let added = reconstruct_residue(&existing, template.1, template.2);
        let added_names: Vec<&str> = added.iter().map(|(n, _, _)| n.as_str()).collect();

        assert!(added_names.contains(&"CB"), "Should reconstruct CB");
        assert!(added_names.contains(&"HA"), "Should reconstruct HA");
        assert!(added_names.contains(&"H"), "Should reconstruct H");
        assert!(
            added.len() > 4,
            "Should add several atoms, got {}",
            added.len()
        );

        // Check CB is at reasonable distance from CA
        let ca_pos = existing["CA"];
        let cb = added.iter().find(|(n, _, _)| n == "CB").unwrap();
        let dist = v_norm_sq(v_sub(cb.2, ca_pos)).sqrt();
        assert!(
            dist > 1.0 && dist < 2.0,
            "CB-CA distance should be ~1.5, got {}",
            dist
        );
    }

    #[test]
    fn test_reconstruct_full_residue() {
        // Give only CA — should reconstruct everything else
        let template = fragment_templates::get_template("GLY").unwrap();
        let mut existing = HashMap::new();

        // Only CA
        let ca_pos = template.1.iter().find(|a| a.0 == "CA").unwrap().2;
        existing.insert("CA".to_string(), ca_pos);

        let added = reconstruct_residue(&existing, template.1, template.2);
        // GLY has 7 atoms total, we gave 1, should add 6
        assert_eq!(added.len(), 6, "Should add 6 atoms to GLY with only CA");
    }

    #[test]
    fn test_reconstruct_crambin_pdb() {
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let before = pdb.atom_count();
        let result = reconstruct_fragments(&mut pdb);
        let after = pdb.atom_count();

        // Crambin already has all heavy atoms, so reconstruction should
        // only add H atoms (which are in the templates but not in the PDB)
        assert_eq!(after, before + result.added);
        // Should add roughly the same number as place_all_hydrogens
        // (templates include all H atoms)
        assert!(
            result.added > 200,
            "Should add >200 atoms from templates, got {}",
            result.added
        );
    }
}
