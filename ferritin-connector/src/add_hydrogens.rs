//! Hydrogen placement for protein structures.
//!
//! Two levels:
//! - `place_peptide_hydrogens`: backbone amide N-H only (Phase 1)
//! - `place_sidechain_hydrogens`: all standard amino acid H (Phase 2)
//! - `place_all_hydrogens`: both in sequence

use std::collections::{HashMap, HashSet};

/// Bond lengths in Ångströms.
const NH_BOND_LENGTH: f64 = 1.01; // matches DSSP virtual H exactly
const CH_BOND_LENGTH: f64 = 1.09; // aliphatic C-H
const AROMATIC_CH_BOND_LENGTH: f64 = 1.08; // aromatic C-H
const OH_BOND_LENGTH: f64 = 0.96; // O-H
const SH_BOND_LENGTH: f64 = 1.34; // S-H

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

// ===========================================================================
// Phase 2: Sidechain hydrogen placement (template-based)
// ===========================================================================

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rotate vector `v` around axis `axis` (must be unit) by `angle` radians (Rodrigues).
fn rotate_around_axis(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let k = axis;
    let kxv = cross(k, v);
    let kdv = dot(k, v);
    [
        v[0] * cos_a + kxv[0] * sin_a + k[0] * kdv * (1.0 - cos_a),
        v[1] * cos_a + kxv[1] * sin_a + k[1] * kdv * (1.0 - cos_a),
        v[2] * cos_a + kxv[2] * sin_a + k[2] * kdv * (1.0 - cos_a),
    ]
}

/// Get a vector perpendicular to `v`.
fn get_perpendicular(v: [f64; 3]) -> [f64; 3] {
    let candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for c in &candidates {
        let p = normalize(cross(v, *c));
        if p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0 {
            return p;
        }
    }
    [1.0, 0.0, 0.0]
}

// ---------------------------------------------------------------------------
// Geometry placement functions
// ---------------------------------------------------------------------------

/// Place 1 H on an atom with 3 heavy neighbors (tetrahedral, e.g., HA on CA).
/// H goes opposite the average of the three neighbor directions.
fn place_tet1h(parent: [f64; 3], n1: [f64; 3], n2: [f64; 3], n3: [f64; 3], bl: f64) -> [f64; 3] {
    let v1 = normalize(sub(n1, parent));
    let v2 = normalize(sub(n2, parent));
    let v3 = normalize(sub(n3, parent));
    let avg = normalize([v1[0] + v2[0] + v3[0], v1[1] + v2[1] + v3[1], v1[2] + v2[2] + v3[2]]);
    add(parent, scale(avg, -bl))
}

/// Place 2 H on an atom with 2 heavy neighbors (tetrahedral, e.g., HB2/HB3 on CB).
fn place_tet2h(
    parent: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    bl: f64,
) -> [[f64; 3]; 2] {
    let v1 = normalize(sub(n1, parent));
    let v2 = normalize(sub(n2, parent));
    let bisect = normalize(add(v1, v2));
    let anti = scale(bisect, -1.0);
    let perp = normalize(cross(v1, v2));

    // Tetrahedral angle from -bisector: ~54.75 degrees
    let half_angle = (109.5_f64 / 2.0).to_radians();
    let cos_ha = half_angle.cos();
    let sin_ha = half_angle.sin();

    let h1_dir = normalize(add(scale(anti, cos_ha), scale(perp, sin_ha)));
    let h2_dir = normalize(add(scale(anti, cos_ha), scale(perp, -sin_ha)));

    [add(parent, scale(h1_dir, bl)), add(parent, scale(h2_dir, bl))]
}

/// Place 3 H in methyl geometry around a single bond axis.
fn place_methyl3h(parent: [f64; 3], anchor: [f64; 3], bl: f64) -> [[f64; 3]; 3] {
    let axis = normalize(sub(parent, anchor));
    let tet_angle = 109.5_f64.to_radians();
    let perp = get_perpendicular(axis);

    // First H at tetrahedral angle from bond axis
    let h_dir = rotate_around_axis(axis, perp, std::f64::consts::PI - tet_angle);

    let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
    let h1 = add(parent, scale(h_dir, bl));
    let h2_dir = rotate_around_axis(h_dir, axis, two_pi_3);
    let h2 = add(parent, scale(h2_dir, bl));
    let h3_dir = rotate_around_axis(h_dir, axis, 2.0 * two_pi_3);
    let h3 = add(parent, scale(h3_dir, bl));

    [h1, h2, h3]
}

/// Place 1 H on an aromatic ring carbon with 2 ring neighbors.
fn place_aromatic1h(parent: [f64; 3], r1: [f64; 3], r2: [f64; 3], bl: f64) -> [f64; 3] {
    let v1 = normalize(sub(r1, parent));
    let v2 = normalize(sub(r2, parent));
    let avg = normalize(add(v1, v2));
    add(parent, scale(avg, -bl))
}

/// Place 1 H on an sp2 atom with 2 heavy neighbors (NH2, planar).
fn place_planar1h_2n(
    parent: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    bl: f64,
    flip: bool,
) -> [f64; 3] {
    let v1 = normalize(sub(n1, parent));
    let v2 = normalize(sub(n2, parent));
    let anti_bisect = normalize(scale(add(v1, v2), -1.0));
    let perp = normalize(cross(v1, v2));

    // Place in-plane or flip to other side
    let angle = 60.0_f64.to_radians();
    let dir = if flip {
        normalize(add(scale(anti_bisect, angle.cos()), scale(perp, -angle.sin())))
    } else {
        normalize(add(scale(anti_bisect, angle.cos()), scale(perp, angle.sin())))
    };
    add(parent, scale(dir, bl))
}

/// Place 1 H on a hydroxyl/thiol with 1 heavy neighbor.
/// Uses tetrahedral angle with an arbitrary dihedral.
fn place_oh1h(parent: [f64; 3], heavy_neighbor: [f64; 3], bl: f64) -> [f64; 3] {
    let axis = normalize(sub(parent, heavy_neighbor));
    let tet_angle = 109.5_f64.to_radians();
    let perp = get_perpendicular(axis);
    let h_dir = rotate_around_axis(axis, perp, std::f64::consts::PI - tet_angle);
    add(parent, scale(h_dir, bl))
}

// ---------------------------------------------------------------------------
// Sidechain H template definitions
// ---------------------------------------------------------------------------

/// A sidechain H atom to place, defined by geometry type.
enum SidechainH {
    /// 1 H opposite 3 neighbors (e.g., HA)
    Tet1 {
        name: &'static str,
        parent: &'static str,
        neighbors: [&'static str; 3],
        bl: f64,
    },
    /// 2 H perpendicular to 2 neighbors (e.g., HB2/HB3)
    Tet2 {
        names: [&'static str; 2],
        parent: &'static str,
        neighbors: [&'static str; 2],
        bl: f64,
    },
    /// 3 H methyl group around 1 anchor (e.g., ALA HB1/HB2/HB3)
    Methyl {
        names: [&'static str; 3],
        parent: &'static str,
        anchor: &'static str,
        bl: f64,
    },
    /// 1 H on aromatic ring carbon
    Aromatic {
        name: &'static str,
        parent: &'static str,
        ring_neighbors: [&'static str; 2],
    },
    /// 1 H on OH/SH/NH with 1 heavy neighbor
    Hydroxyl {
        name: &'static str,
        parent: &'static str,
        neighbor: &'static str,
        bl: f64,
    },
    /// 2 H on planar NH2 (ASN ND2, GLN NE2)
    PlanarNH2 {
        names: [&'static str; 2],
        parent: &'static str,
        neighbors: [&'static str; 2],
    },
}

macro_rules! ha {
    () => { SidechainH::Tet1 { name: "HA", parent: "CA", neighbors: ["N", "C", "CB"], bl: CH_BOND_LENGTH } }
}

macro_rules! hb2b3 {
    ($n1:expr, $n2:expr) => { SidechainH::Tet2 { names: ["HB2", "HB3"], parent: "CB", neighbors: [$n1, $n2], bl: CH_BOND_LENGTH } }
}

fn sidechain_templates(resname: &str) -> &'static [SidechainH] {
    match resname {
        "GLY" => &[
            SidechainH::Tet2 { names: ["HA2", "HA3"], parent: "CA", neighbors: ["N", "C"], bl: CH_BOND_LENGTH },
        ],
        "ALA" => &[
            ha!(),
            SidechainH::Methyl { names: ["HB1", "HB2", "HB3"], parent: "CB", anchor: "CA", bl: CH_BOND_LENGTH },
        ],
        "VAL" => &[
            ha!(),
            SidechainH::Tet1 { name: "HB", parent: "CB", neighbors: ["CA", "CG1", "CG2"], bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HG11", "HG12", "HG13"], parent: "CG1", anchor: "CB", bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HG21", "HG22", "HG23"], parent: "CG2", anchor: "CB", bl: CH_BOND_LENGTH },
        ],
        "LEU" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet1 { name: "HG", parent: "CG", neighbors: ["CB", "CD1", "CD2"], bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HD11", "HD12", "HD13"], parent: "CD1", anchor: "CG", bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HD21", "HD22", "HD23"], parent: "CD2", anchor: "CG", bl: CH_BOND_LENGTH },
        ],
        "ILE" => &[
            ha!(),
            SidechainH::Tet1 { name: "HB", parent: "CB", neighbors: ["CA", "CG1", "CG2"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HG12", "HG13"], parent: "CG1", neighbors: ["CB", "CD1"], bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HG21", "HG22", "HG23"], parent: "CG2", anchor: "CB", bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HD11", "HD12", "HD13"], parent: "CD1", anchor: "CG1", bl: CH_BOND_LENGTH },
        ],
        "PRO" => &[
            SidechainH::Tet1 { name: "HA", parent: "CA", neighbors: ["N", "C", "CB"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HB2", "HB3"], parent: "CB", neighbors: ["CA", "CG"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "CD"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HD2", "HD3"], parent: "CD", neighbors: ["CG", "N"], bl: CH_BOND_LENGTH },
        ],
        "PHE" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Aromatic { name: "HD1", parent: "CD1", ring_neighbors: ["CG", "CE1"] },
            SidechainH::Aromatic { name: "HD2", parent: "CD2", ring_neighbors: ["CG", "CE2"] },
            SidechainH::Aromatic { name: "HE1", parent: "CE1", ring_neighbors: ["CD1", "CZ"] },
            SidechainH::Aromatic { name: "HE2", parent: "CE2", ring_neighbors: ["CD2", "CZ"] },
            SidechainH::Aromatic { name: "HZ", parent: "CZ", ring_neighbors: ["CE1", "CE2"] },
        ],
        "TYR" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Aromatic { name: "HD1", parent: "CD1", ring_neighbors: ["CG", "CE1"] },
            SidechainH::Aromatic { name: "HD2", parent: "CD2", ring_neighbors: ["CG", "CE2"] },
            SidechainH::Aromatic { name: "HE1", parent: "CE1", ring_neighbors: ["CD1", "CZ"] },
            SidechainH::Aromatic { name: "HE2", parent: "CE2", ring_neighbors: ["CD2", "CZ"] },
            SidechainH::Hydroxyl { name: "HH", parent: "OH", neighbor: "CZ", bl: OH_BOND_LENGTH },
        ],
        "TRP" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Aromatic { name: "HD1", parent: "CD1", ring_neighbors: ["CG", "NE1"] },
            SidechainH::Hydroxyl { name: "HE1", parent: "NE1", neighbor: "CD1", bl: NH_BOND_LENGTH }, // ring NH
            SidechainH::Aromatic { name: "HE3", parent: "CE3", ring_neighbors: ["CD2", "CZ3"] },
            SidechainH::Aromatic { name: "HZ2", parent: "CZ2", ring_neighbors: ["CE2", "CH2"] },
            SidechainH::Aromatic { name: "HZ3", parent: "CZ3", ring_neighbors: ["CE3", "CH2"] },
            SidechainH::Aromatic { name: "HH2", parent: "CH2", ring_neighbors: ["CZ2", "CZ3"] },
        ],
        "SER" => &[
            ha!(),
            hb2b3!("CA", "OG"),
            SidechainH::Hydroxyl { name: "HG", parent: "OG", neighbor: "CB", bl: OH_BOND_LENGTH },
        ],
        "THR" => &[
            ha!(),
            SidechainH::Tet1 { name: "HB", parent: "CB", neighbors: ["CA", "OG1", "CG2"], bl: CH_BOND_LENGTH },
            SidechainH::Hydroxyl { name: "HG1", parent: "OG1", neighbor: "CB", bl: OH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HG21", "HG22", "HG23"], parent: "CG2", anchor: "CB", bl: CH_BOND_LENGTH },
        ],
        "CYS" => &[
            ha!(),
            hb2b3!("CA", "SG"),
            SidechainH::Hydroxyl { name: "HG", parent: "SG", neighbor: "CB", bl: SH_BOND_LENGTH },
        ],
        "MET" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "SD"], bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HE1", "HE2", "HE3"], parent: "CE", anchor: "SD", bl: CH_BOND_LENGTH },
        ],
        "ASP" => &[
            ha!(),
            hb2b3!("CA", "CG"),
        ],
        "GLU" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "CD"], bl: CH_BOND_LENGTH },
        ],
        "ASN" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::PlanarNH2 { names: ["HD21", "HD22"], parent: "ND2", neighbors: ["CG", "OD1"] },
        ],
        "GLN" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "CD"], bl: CH_BOND_LENGTH },
            SidechainH::PlanarNH2 { names: ["HE21", "HE22"], parent: "NE2", neighbors: ["CD", "OE1"] },
        ],
        "LYS" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "CD"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HD2", "HD3"], parent: "CD", neighbors: ["CG", "CE"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HE2", "HE3"], parent: "CE", neighbors: ["CD", "NZ"], bl: CH_BOND_LENGTH },
            SidechainH::Methyl { names: ["HZ1", "HZ2", "HZ3"], parent: "NZ", anchor: "CE", bl: NH_BOND_LENGTH },
        ],
        "ARG" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            SidechainH::Tet2 { names: ["HG2", "HG3"], parent: "CG", neighbors: ["CB", "CD"], bl: CH_BOND_LENGTH },
            SidechainH::Tet2 { names: ["HD2", "HD3"], parent: "CD", neighbors: ["CG", "NE"], bl: CH_BOND_LENGTH },
            SidechainH::Hydroxyl { name: "HE", parent: "NE", neighbor: "CD", bl: NH_BOND_LENGTH },
            SidechainH::PlanarNH2 { names: ["HH11", "HH12"], parent: "NH1", neighbors: ["CZ", "NE"] },
            SidechainH::PlanarNH2 { names: ["HH21", "HH22"], parent: "NH2", neighbors: ["CZ", "NH1"] },
        ],
        "HIS" => &[
            ha!(),
            hb2b3!("CA", "CG"),
            // Default: HID tautomer (H on ND1)
            SidechainH::Aromatic { name: "HD2", parent: "CD2", ring_neighbors: ["CG", "NE2"] },
            SidechainH::Aromatic { name: "HE1", parent: "CE1", ring_neighbors: ["ND1", "NE2"] },
        ],
        _ => &[],
    }
}

// ---------------------------------------------------------------------------
// Sidechain H collection logic
// ---------------------------------------------------------------------------

/// Collect atom positions from a residue into a map.
fn collect_atom_positions(residue: &pdbtbx::Residue) -> (HashMap<String, [f64; 3]>, HashSet<String>) {
    let mut positions = HashMap::new();
    let mut existing_h = HashSet::new();

    for atom in residue.atoms() {
        let name = atom.name().trim().to_string();
        let (x, y, z) = atom.pos();
        positions.insert(name.clone(), [x, y, z]);
        if name.starts_with('H') {
            existing_h.insert(name);
        }
    }

    (positions, existing_h)
}

/// Check if a CYS SG atom is in a disulfide bond (another SG within 2.5 Å).
fn is_disulfide(sg_pos: [f64; 3], all_sg_positions: &[[f64; 3]]) -> bool {
    for other in all_sg_positions {
        let dx = sg_pos[0] - other[0];
        let dy = sg_pos[1] - other[1];
        let dz = sg_pos[2] - other[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        // Another SG within 2.5 Å but not self (> 0.1)
        if dist > 0.1 && dist < 2.5 {
            return true;
        }
    }
    false
}

/// Compute sidechain H positions for a residue based on templates.
fn compute_sidechain_h(
    resname: &str,
    positions: &HashMap<String, [f64; 3]>,
    existing_h: &HashSet<String>,
    all_sg_positions: &[[f64; 3]],
) -> Vec<(String, [f64; 3])> {
    let templates = sidechain_templates(resname);
    let mut result = Vec::new();

    for tmpl in templates {
        match tmpl {
            SidechainH::Tet1 { name, parent, neighbors, bl } => {
                if existing_h.contains(*name) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let n1 = match positions.get(neighbors[0]) { Some(v) => *v, None => continue };
                let n2 = match positions.get(neighbors[1]) { Some(v) => *v, None => continue };
                let n3 = match positions.get(neighbors[2]) { Some(v) => *v, None => continue };
                result.push((name.to_string(), place_tet1h(p, n1, n2, n3, *bl)));
            }
            SidechainH::Tet2 { names, parent, neighbors, bl } => {
                if existing_h.contains(names[0]) && existing_h.contains(names[1]) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let n1 = match positions.get(neighbors[0]) { Some(v) => *v, None => continue };
                let n2 = match positions.get(neighbors[1]) { Some(v) => *v, None => continue };
                let hs = place_tet2h(p, n1, n2, *bl);
                if !existing_h.contains(names[0]) { result.push((names[0].to_string(), hs[0])); }
                if !existing_h.contains(names[1]) { result.push((names[1].to_string(), hs[1])); }
            }
            SidechainH::Methyl { names, parent, anchor, bl } => {
                if existing_h.contains(names[0]) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let a = match positions.get(*anchor) { Some(v) => *v, None => continue };
                let hs = place_methyl3h(p, a, *bl);
                for (i, h) in hs.iter().enumerate() {
                    if !existing_h.contains(names[i]) { result.push((names[i].to_string(), *h)); }
                }
            }
            SidechainH::Aromatic { name, parent, ring_neighbors } => {
                if existing_h.contains(*name) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let r1 = match positions.get(ring_neighbors[0]) { Some(v) => *v, None => continue };
                let r2 = match positions.get(ring_neighbors[1]) { Some(v) => *v, None => continue };
                result.push((name.to_string(), place_aromatic1h(p, r1, r2, AROMATIC_CH_BOND_LENGTH)));
            }
            SidechainH::Hydroxyl { name, parent, neighbor, bl } => {
                if existing_h.contains(*name) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let n = match positions.get(*neighbor) { Some(v) => *v, None => continue };
                // For CYS SG, skip if disulfide bonded
                if *parent == "SG" && is_disulfide(p, all_sg_positions) {
                    continue;
                }
                result.push((name.to_string(), place_oh1h(p, n, *bl)));
            }
            SidechainH::PlanarNH2 { names, parent, neighbors } => {
                if existing_h.contains(names[0]) && existing_h.contains(names[1]) { continue; }
                let p = match positions.get(*parent) { Some(v) => *v, None => continue };
                let n1 = match positions.get(neighbors[0]) { Some(v) => *v, None => continue };
                let n2 = match positions.get(neighbors[1]) { Some(v) => *v, None => continue };
                if !existing_h.contains(names[0]) {
                    result.push((names[0].to_string(), place_planar1h_2n(p, n1, n2, NH_BOND_LENGTH, false)));
                }
                if !existing_h.contains(names[1]) {
                    result.push((names[1].to_string(), place_planar1h_2n(p, n1, n2, NH_BOND_LENGTH, true)));
                }
            }
        }
    }

    result
}

/// Place sidechain hydrogen atoms on all standard amino acid residues.
///
/// Template-based placement for the 20 standard amino acids.
/// Handles HA, methyl, methylene, aromatic, hydroxyl, and amide H.
pub fn place_sidechain_hydrogens(pdb: &mut pdbtbx::PDB) -> AddHydrogensResult {
    let mut placements: Vec<(usize, usize, String, [f64; 3], usize)> = Vec::new();
    let mut skipped = 0;
    let mut max_serial: usize = pdb.atoms().map(|a| a.serial_number()).max().unwrap_or(0);

    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return AddHydrogensResult { added: 0, skipped: 0 },
    };

    // Collect all SG positions for disulfide detection
    let mut all_sg_positions: Vec<[f64; 3]> = Vec::new();
    for chain in first_model.chains() {
        for residue in chain.residues() {
            if residue.name() == Some("CYS") {
                for atom in residue.atoms() {
                    if atom.name().trim() == "SG" {
                        let (x, y, z) = atom.pos();
                        all_sg_positions.push([x, y, z]);
                    }
                }
            }
        }
    }

    for (chain_idx, chain) in first_model.chains().enumerate() {
        // Find first and last AA residue indices in this chain
        let aa_indices: Vec<usize> = chain
            .residues()
            .enumerate()
            .filter(|(_, r)| r.conformers().next().map_or(false, |c| c.is_amino_acid()))
            .map(|(i, _)| i)
            .collect();
        let first_aa = aa_indices.first().copied();
        let last_aa = aa_indices.last().copied();

        for (residue_idx, residue) in chain.residues().enumerate() {
            let is_aa = residue
                .conformers()
                .next()
                .map_or(false, |c| c.is_amino_acid());
            if !is_aa {
                continue;
            }

            let resname = match residue.name() {
                Some(n) => n.to_string(),
                None => continue,
            };

            let (positions, existing_h) = collect_atom_positions(residue);
            let new_hs = compute_sidechain_h(&resname, &positions, &existing_h, &all_sg_positions);

            for (h_name, pos) in &new_hs {
                max_serial += 1;
                placements.push((chain_idx, residue_idx, h_name.clone(), *pos, max_serial));
            }

            // N-terminal: add 3 H on N for NH3+ group (H1, H2, H3)
            // Phase 1 skips N-terminal, so all 3 go here
            if first_aa == Some(residue_idx) {
                if let Some(&n_pos) = positions.get("N") {
                    if let Some(&ca_pos) = positions.get("CA") {
                        let has_any = existing_h.contains("H1") || existing_h.contains("1H")
                            || existing_h.contains("H2") || existing_h.contains("2H")
                            || existing_h.contains("H") ;
                        if !has_any {
                            let hs = place_methyl3h(n_pos, ca_pos, NH_BOND_LENGTH);
                            max_serial += 1;
                            placements.push((chain_idx, residue_idx, "H1".to_string(), hs[0], max_serial));
                            max_serial += 1;
                            placements.push((chain_idx, residue_idx, "H2".to_string(), hs[1], max_serial));
                            max_serial += 1;
                            placements.push((chain_idx, residue_idx, "H3".to_string(), hs[2], max_serial));
                        }
                    }
                }
            }

            // C-terminal: add OXT hydrogen
            if last_aa == Some(residue_idx) {
                if let Some(&oxt_pos) = positions.get("OXT") {
                    if !existing_h.contains("HXT") && !existing_h.contains("HOXT") {
                        if let Some(&c_pos) = positions.get("C") {
                            let h = place_oh1h(oxt_pos, c_pos, OH_BOND_LENGTH);
                            max_serial += 1;
                            placements.push((chain_idx, residue_idx, "HXT".to_string(), h, max_serial));
                        }
                    }
                }
            }

            if new_hs.is_empty() && first_aa != Some(residue_idx) && last_aa != Some(residue_idx) {
                skipped += 1;
            }
        }
    }

    let added = placements.len();

    for (chain_idx, residue_idx, h_name, pos, serial) in placements {
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

        if let Some(h_atom) = pdbtbx::Atom::new(
            false, serial, &h_name, &h_name,
            pos[0], pos[1], pos[2],
            1.0, 0.0, "H", 0,
        ) {
            conformer.add_atom(h_atom);
        }
    }

    AddHydrogensResult { added, skipped }
}

/// Place all hydrogens: backbone amide H + sidechain H.
pub fn place_all_hydrogens(pdb: &mut pdbtbx::PDB) -> AddHydrogensResult {
    let r1 = place_peptide_hydrogens(pdb);
    let r2 = place_sidechain_hydrogens(pdb);
    AddHydrogensResult {
        added: r1.added + r2.added,
        skipped: r1.skipped + r2.skipped,
    }
}

// ===========================================================================
// Phase 3: General-purpose hydrogen placement (BALL algorithm)
// ===========================================================================

use crate::bond_order::{self, MolGraph};

/// Place hydrogens on any atom using the BALL general algorithm.
///
/// For each heavy atom: compute expected valence from periodic table,
/// subtract existing bond orders, place remaining H atoms using
/// geometry from bond count and ring membership.
///
/// This handles ligands, non-standard residues, modified amino acids,
/// and any molecule not covered by the standard AA templates.
fn general_handle_atom(
    graph: &MolGraph,
    atom_idx: usize,
) -> Vec<(String, [f64; 3])> {
    let atom = &graph.atoms[atom_idx];
    let mut results = Vec::new();

    // Skip hydrogens and metals
    let valence = bond_order::expected_valence(&atom.element);
    if valence == 0 {
        return results;
    }

    // How many H to add?
    let sum_orders = bond_order::sum_bond_orders(graph, atom_idx);
    let h_to_add = (valence as f64 - sum_orders).round() as i32;

    if h_to_add <= 0 {
        return results;
    }

    // Skip if atom already has H neighbors
    let existing_h_count = atom.neighbors.iter()
        .filter(|&&n| graph.atoms[n].element == "H" || graph.atoms[n].element == "D")
        .count() as i32;

    let h_to_add = h_to_add - existing_h_count;
    if h_to_add <= 0 {
        return results;
    }

    let bond_length = bond_order::mmff94_bond_length(&atom.element);
    let n_heavy_neighbors = atom.neighbors.iter()
        .filter(|&&n| graph.atoms[n].element != "H" && graph.atoms[n].element != "D")
        .count();

    // Get heavy neighbor positions
    let heavy_nbrs: Vec<[f64; 3]> = atom.neighbors.iter()
        .filter(|&&n| graph.atoms[n].element != "H" && graph.atoms[n].element != "D")
        .map(|&n| graph.atoms[n].pos)
        .collect();

    let atom_nr_start = results.len();

    // Dispatch based on geometry (BALL's _handle_atom! logic)
    match (h_to_add, n_heavy_neighbors) {
        // 1 H, 3 heavy neighbors → tetrahedral (HA-like)
        (1, 3) => {
            let h = place_tet1h(atom.pos, heavy_nbrs[0], heavy_nbrs[1], heavy_nbrs[2], bond_length);
            results.push((format!("H{}", atom_nr_start + 1), h));
        }
        // 1 H, 2 heavy neighbors → bent (water-like or ring)
        (1, 2) => {
            if atom.is_ring_atom {
                // Ring atom: H opposite the two ring neighbors
                let h = place_aromatic1h(atom.pos, heavy_nbrs[0], heavy_nbrs[1], bond_length);
                results.push((format!("H{}", atom_nr_start + 1), h));
            } else {
                // Bent geometry
                let h = place_aromatic1h(atom.pos, heavy_nbrs[0], heavy_nbrs[1], bond_length);
                results.push((format!("H{}", atom_nr_start + 1), h));
            }
        }
        // 1 H, 1 heavy neighbor → linear or sp2
        (1, 1) => {
            if sum_orders > 2.0 {
                // Linear (triple bond): H opposite the neighbor
                let dir = normalize(sub(atom.pos, heavy_nbrs[0]));
                let h = add(atom.pos, scale(dir, bond_length));
                results.push((format!("H{}", atom_nr_start + 1), h));
            } else {
                // sp2 or bent: place at angle
                let h = place_oh1h(atom.pos, heavy_nbrs[0], bond_length);
                results.push((format!("H{}", atom_nr_start + 1), h));
            }
        }
        // 2 H, 2 heavy neighbors → tetrahedral (CH2-like)
        (2, 2) => {
            let hs = place_tet2h(atom.pos, heavy_nbrs[0], heavy_nbrs[1], bond_length);
            results.push((format!("H{}", atom_nr_start + 1), hs[0]));
            results.push((format!("H{}", atom_nr_start + 2), hs[1]));
        }
        // 2 H, 1 heavy neighbor → NH2 or CH2 with one bond
        (2, 1) => {
            // Place first H, then recurse-like logic
            let h1 = place_oh1h(atom.pos, heavy_nbrs[0], bond_length);
            results.push((format!("H{}", atom_nr_start + 1), h1));

            // Second H: rotate 106° around normal
            let bv = normalize(sub(atom.pos, heavy_nbrs[0]));
            let perp = get_perpendicular(bv);
            let h2_dir = rotate_around_axis(bv, perp, 106.0_f64.to_radians());
            let h2 = add(atom.pos, scale(h2_dir, bond_length));
            results.push((format!("H{}", atom_nr_start + 2), h2));
        }
        // 3 H, 1 heavy neighbor → methyl (CH3 or NH3)
        (3, 1) => {
            let hs = place_methyl3h(atom.pos, heavy_nbrs[0], bond_length);
            results.push((format!("H{}", atom_nr_start + 1), hs[0]));
            results.push((format!("H{}", atom_nr_start + 2), hs[1]));
            results.push((format!("H{}", atom_nr_start + 3), hs[2]));
        }
        // 1 H, 0 heavy neighbors → isolated (e.g., H-F)
        (_, 0) => {
            let h = add(atom.pos, [bond_length, 0.0, 0.0]);
            results.push((format!("H{}", atom_nr_start + 1), h));
        }
        // 2 H, 0 neighbors (unlikely but handle)
        (n, _) if n > 0 => {
            // Fallback: methyl-like arrangement
            if n_heavy_neighbors >= 1 {
                let anchor = heavy_nbrs[0];
                let hs = place_methyl3h(atom.pos, anchor, bond_length);
                for i in 0..(n as usize).min(3) {
                    results.push((format!("H{}", atom_nr_start + i + 1), hs[i]));
                }
            }
        }
        _ => {}
    }

    results
}

/// Place hydrogens on non-standard residues using the general BALL algorithm.
///
/// First runs Phase 1+2 (backbone + sidechain templates), then applies
/// the general algorithm to any remaining unsaturated heavy atoms
/// (ligands, modified residues, cofactors).
/// Water residue names.
const WATER_NAMES: &[&str] = &["HOH", "WAT", "H2O", "TIP", "TIP3", "SPC", "DOD"];

/// Place 2 H on a water molecule's oxygen atom.
fn place_water_h(o_pos: [f64; 3]) -> [[f64; 3]; 2] {
    // Water H-O-H angle: 104.5 degrees
    // Place with arbitrary orientation (no neighbors to reference)
    let half_angle = (104.5_f64 / 2.0).to_radians();
    let bl = OH_BOND_LENGTH;

    // Pick an arbitrary direction
    let dir1 = [0.0, half_angle.sin(), half_angle.cos()];
    let dir2 = [0.0, -half_angle.sin(), half_angle.cos()];

    [
        [o_pos[0] + bl * dir1[0], o_pos[1] + bl * dir1[1], o_pos[2] + bl * dir1[2]],
        [o_pos[0] + bl * dir2[0], o_pos[1] + bl * dir2[1], o_pos[2] + bl * dir2[2]],
    ]
}

/// Place hydrogens on non-standard residues using the general BALL algorithm.
///
/// If `include_water` is true, also places 2 H on each water molecule.
pub fn place_general_hydrogens(pdb: &mut pdbtbx::PDB, include_water: bool) -> AddHydrogensResult {
    // First: place standard AA hydrogens
    let standard = place_all_hydrogens(pdb);

    // Then: find non-standard residues and apply general algorithm
    let mut placements: Vec<(usize, usize, String, [f64; 3], usize)> = Vec::new();
    let mut max_serial: usize = pdb.atoms().map(|a| a.serial_number()).max().unwrap_or(0);

    let first_model = match pdb.models().next() {
        Some(m) => m,
        None => return standard,
    };

    for (chain_idx, chain) in first_model.chains().enumerate() {
        for (residue_idx, residue) in chain.residues().enumerate() {
            let is_standard_aa = residue
                .conformers()
                .next()
                .map_or(false, |c| c.is_amino_acid());

            // Skip standard amino acids (already handled by Phase 1+2)
            if is_standard_aa {
                continue;
            }

            let resname = residue.name().unwrap_or("");

            // Water handling
            if WATER_NAMES.contains(&resname) {
                if !include_water {
                    continue;
                }
                // Find O atom, check no existing H
                let mut o_pos = None;
                let mut has_h = false;
                for atom in residue.atoms() {
                    let elem = atom.element().map(|e| e.symbol()).unwrap_or("");
                    if elem == "O" {
                        let (x, y, z) = atom.pos();
                        o_pos = Some([x, y, z]);
                    }
                    if elem == "H" || elem == "D" {
                        has_h = true;
                    }
                }
                if let Some(o) = o_pos {
                    if !has_h {
                        let hs = place_water_h(o);
                        max_serial += 1;
                        placements.push((chain_idx, residue_idx, "H1".to_string(), hs[0], max_serial));
                        max_serial += 1;
                        placements.push((chain_idx, residue_idx, "H2".to_string(), hs[1], max_serial));
                    }
                }
                continue;
            }

            // Build molecular graph for this residue
            let mut positions = Vec::new();
            let mut elements = Vec::new();
            let mut names = Vec::new();

            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                positions.push([x, y, z]);
                elements.push(
                    atom.element()
                        .map(|e| e.symbol().to_string())
                        .unwrap_or_else(|| "C".to_string()),
                );
                names.push(atom.name().trim().to_string());
            }

            if positions.is_empty() {
                continue;
            }

            let graph = bond_order::build_mol_graph(&positions, &elements, &names);

            // For each heavy atom, try to place H
            for (local_idx, atom) in graph.atoms.iter().enumerate() {
                if atom.element == "H" || atom.element == "D" {
                    continue;
                }

                let new_hs = general_handle_atom(&graph, local_idx);
                for (h_name, pos) in new_hs {
                    max_serial += 1;
                    placements.push((chain_idx, residue_idx, h_name, pos, max_serial));
                }
            }
        }
    }

    let general_added = placements.len();

    // Write pass
    for (chain_idx, residue_idx, h_name, pos, serial) in placements {
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

        if let Some(h_atom) = pdbtbx::Atom::new(
            false, serial, &h_name, &h_name,
            pos[0], pos[1], pos[2],
            1.0, 0.0, "H", 0,
        ) {
            conformer.add_atom(h_atom);
        }
    }

    AddHydrogensResult {
        added: standard.added + general_added,
        skipped: standard.skipped,
    }
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

    // --- Sidechain H tests ---

    #[test]
    fn test_sidechain_crambin_h_count() {
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let before = pdb.atom_count();
        let result = place_sidechain_hydrogens(&mut pdb);
        let after = pdb.atom_count();

        assert!(result.added > 0, "Should place sidechain H atoms");
        assert_eq!(after, before + result.added);
        // 1CRN has 46 residues, expect roughly 5-8 H per residue on average
        assert!(result.added > 150, "Expected >150 sidechain H, got {}", result.added);
        assert!(result.added < 500, "Expected <500 sidechain H, got {}", result.added);
    }

    #[test]
    fn test_sidechain_idempotent() {
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let r1 = place_sidechain_hydrogens(&mut pdb);
        assert!(r1.added > 0);

        let r2 = place_sidechain_hydrogens(&mut pdb);
        assert_eq!(r2.added, 0, "Second pass should add nothing");
    }

    #[test]
    fn test_place_all_hydrogens() {
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let before = pdb.atom_count();
        let result = place_all_hydrogens(&mut pdb);

        // Backbone (40) + sidechain (150+)
        assert!(result.added > 190, "Expected >190 total H, got {}", result.added);
        assert_eq!(pdb.atom_count(), before + result.added);

        // Idempotent
        let r2 = place_all_hydrogens(&mut pdb);
        assert_eq!(r2.added, 0);
    }

    #[test]
    fn test_place_all_vs_pdbfixer_count() {
        // PDBFixer places 315 H on 1CRN (all types including backbone)
        // We should get close
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        let result = place_all_hydrogens(&mut pdb);

        // PDBFixer: 315. We expect 250+ (missing: some terminal H, protonation-dependent H)
        assert!(
            result.added > 250,
            "Expected >250 total H (PDBFixer places 315), got {}",
            result.added
        );
    }

    #[test]
    fn test_bond_lengths() {
        // After placement, verify all H atoms are at correct distances from their parent
        let (mut pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read("../test-pdbs/1crn.pdb")
            .expect("failed to read 1crn.pdb");

        place_all_hydrogens(&mut pdb);

        // Check that every H atom is within reasonable bond length of its nearest heavy atom
        for chain in pdb.chains() {
            for residue in chain.residues() {
                let heavy: Vec<[f64; 3]> = residue.atoms()
                    .filter(|a| a.element().map_or(true, |e| e.symbol() != "H"))
                    .map(|a| { let (x,y,z) = a.pos(); [x,y,z] })
                    .collect();

                for atom in residue.atoms() {
                    if atom.element().map_or(true, |e| e.symbol() != "H") { continue; }
                    let (hx, hy, hz) = atom.pos();

                    // Find nearest heavy atom
                    let min_dist = heavy.iter()
                        .map(|h| ((hx-h[0]).powi(2) + (hy-h[1]).powi(2) + (hz-h[2]).powi(2)).sqrt())
                        .fold(f64::MAX, f64::min);

                    assert!(
                        min_dist > 0.8 && min_dist < 1.5,
                        "H atom {} in {} has nearest heavy atom at {:.3} Å (expected 0.9-1.4)",
                        atom.name(), residue.name().unwrap_or("?"), min_dist
                    );
                }
            }
        }
    }

    #[test]
    fn test_geometry_tet1h() {
        // Tetrahedral with 3 neighbors: H should be at bond_length from parent
        let parent = [0.0, 0.0, 0.0];
        let n1 = [1.5, 0.0, 0.0];
        let n2 = [0.0, 1.5, 0.0];
        let n3 = [0.0, 0.0, 1.5];
        let h = place_tet1h(parent, n1, n2, n3, 1.09);
        let dist = (h[0]*h[0] + h[1]*h[1] + h[2]*h[2]).sqrt();
        assert!((dist - 1.09).abs() < 1e-6, "dist={}", dist);
        // H should be in the -x,-y,-z direction
        assert!(h[0] < 0.0 && h[1] < 0.0 && h[2] < 0.0);
    }

    #[test]
    fn test_geometry_methyl3h() {
        let parent = [0.0, 0.0, 0.0];
        let anchor = [-1.5, 0.0, 0.0];
        let hs = place_methyl3h(parent, anchor, 1.09);
        // All 3 H should be at bond_length from parent
        for h in &hs {
            let dist = (h[0]*h[0] + h[1]*h[1] + h[2]*h[2]).sqrt();
            assert!((dist - 1.09).abs() < 1e-6, "dist={}", dist);
        }
        // H atoms should be ~120 degrees apart
        for i in 0..3 {
            for j in (i+1)..3 {
                let d = ((hs[i][0]-hs[j][0]).powi(2) + (hs[i][1]-hs[j][1]).powi(2) + (hs[i][2]-hs[j][2]).powi(2)).sqrt();
                // Expected H-H distance in methyl: ~1.78 Å
                assert!(d > 1.5 && d < 2.0, "H{}-H{} dist={:.3}", i, j, d);
            }
        }
    }
}
