//! PyO3 bindings for structure supervision extraction.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::altloc::residue_atoms_primary;
use crate::parallel::{build_pool, resolve_threads};
use crate::py_pdb::PyPDB;

const AA_ORDER: &[u8] = b"ACDEFGHIKLMNPQRSTVWYX";
const ATOM_TYPES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
    "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
    "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];

#[derive(Clone)]
struct ResidueRecord {
    name: String,
    serial_number: i32,
    atoms: Vec<(String, [f32; 3])>,
}

struct ExtractedExample {
    length: usize,
    aatype: Vec<i32>,
    residue_index: Vec<i32>,
    seq_mask: Vec<f32>,
    all_atom_positions: Vec<f32>,
    all_atom_mask: Vec<f32>,
    atom37_atom_exists: Vec<f32>,
    atom14_gt_positions: Vec<f32>,
    atom14_gt_exists: Vec<f32>,
    atom14_atom_exists: Vec<f32>,
    residx_atom14_to_atom37: Vec<i32>,
    residx_atom37_to_atom14: Vec<i32>,
    atom14_atom_is_ambiguous: Vec<f32>,
    pseudo_beta: Vec<f32>,
    pseudo_beta_mask: Vec<f32>,
    phi: Vec<f32>,
    psi: Vec<f32>,
    omega: Vec<f32>,
    phi_mask: Vec<f32>,
    psi_mask: Vec<f32>,
    omega_mask: Vec<f32>,
    chi_angles: Vec<f32>,
    chi_mask: Vec<f32>,
    rigidgroups_gt_frames: Vec<f32>,
    rigidgroups_gt_exists: Vec<f32>,
    rigidgroups_group_exists: Vec<f32>,
    rigidgroups_group_is_ambiguous: Vec<f32>,
}

fn aa_index(one_letter: u8) -> i32 {
    AA_ORDER
        .iter()
        .position(|&aa| aa == one_letter)
        .unwrap_or(AA_ORDER.len() - 1) as i32
}

fn residue_to_one_letter(name: &str) -> u8 {
    match name {
        "ALA" => b'A',
        "ARG" => b'R',
        "ASN" => b'N',
        "ASP" => b'D',
        "CYS" => b'C',
        "GLN" => b'Q',
        "GLU" => b'E',
        "GLY" => b'G',
        "HIS" => b'H',
        "ILE" => b'I',
        "LEU" => b'L',
        "LYS" => b'K',
        "MET" => b'M',
        "PHE" => b'F',
        "PRO" => b'P',
        "SER" => b'S',
        "THR" => b'T',
        "TRP" => b'W',
        "TYR" => b'Y',
        "VAL" => b'V',
        _ => b'X',
    }
}

fn atom37_index(name: &str) -> Option<usize> {
    ATOM_TYPES.iter().position(|&atom| atom == name)
}

fn atom14_names(resname: &str) -> [&'static str; 14] {
    match resname {
        "ALA" => [
            "N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", "",
        ],
        "ARG" => [
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", "",
        ],
        "ASN" => [
            "N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", "",
        ],
        "ASP" => [
            "N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", "",
        ],
        "CYS" => [
            "N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", "",
        ],
        "GLN" => [
            "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", "",
        ],
        "GLU" => [
            "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", "",
        ],
        "GLY" => ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS" => [
            "N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", "",
        ],
        "ILE" => [
            "N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", "",
        ],
        "LEU" => [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", "",
        ],
        "LYS" => [
            "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", "",
        ],
        "MET" => [
            "N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", "",
        ],
        "PHE" => [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", "",
        ],
        "PRO" => [
            "N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", "",
        ],
        "SER" => [
            "N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", "",
        ],
        "THR" => [
            "N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "",
        ],
        "TRP" => [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ],
        "TYR" => [
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", "",
        ],
        "VAL" => [
            "N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", "",
        ],
        _ => ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }
}

fn chi_angle_atoms(resname: &str) -> &'static [&'static [&'static str]] {
    match resname {
        "ALA" => &[],
        "ARG" => &[
            &["N", "CA", "CB", "CG"],
            &["CA", "CB", "CG", "CD"],
            &["CB", "CG", "CD", "NE"],
            &["CG", "CD", "NE", "CZ"],
        ],
        "ASN" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "OD1"]],
        "ASP" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "OD1"]],
        "CYS" => &[&["N", "CA", "CB", "SG"]],
        "GLN" => &[
            &["N", "CA", "CB", "CG"],
            &["CA", "CB", "CG", "CD"],
            &["CB", "CG", "CD", "OE1"],
        ],
        "GLU" => &[
            &["N", "CA", "CB", "CG"],
            &["CA", "CB", "CG", "CD"],
            &["CB", "CG", "CD", "OE1"],
        ],
        "GLY" => &[],
        "HIS" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "ND1"]],
        "ILE" => &[&["N", "CA", "CB", "CG1"], &["CA", "CB", "CG1", "CD1"]],
        "LEU" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "CD1"]],
        "LYS" => &[
            &["N", "CA", "CB", "CG"],
            &["CA", "CB", "CG", "CD"],
            &["CB", "CG", "CD", "CE"],
            &["CG", "CD", "CE", "NZ"],
        ],
        "MET" => &[
            &["N", "CA", "CB", "CG"],
            &["CA", "CB", "CG", "SD"],
            &["CB", "CG", "SD", "CE"],
        ],
        "PHE" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "CD1"]],
        "PRO" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "CD"]],
        "SER" => &[&["N", "CA", "CB", "OG"]],
        "THR" => &[&["N", "CA", "CB", "OG1"]],
        "TRP" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "CD1"]],
        "TYR" => &[&["N", "CA", "CB", "CG"], &["CA", "CB", "CG", "CD1"]],
        "VAL" => &[&["N", "CA", "CB", "CG1"]],
        _ => &[],
    }
}

fn rigidgroup_base_atoms(resname: &str) -> [(usize, [&'static str; 3]); 6] {
    let chi = chi_angle_atoms(resname);
    let mut out = [
        (0, ["C", "CA", "N"]),
        (3, ["CA", "C", "O"]),
        (8, ["", "", ""]),
        (8, ["", "", ""]),
        (8, ["", "", ""]),
        (8, ["", "", ""]),
    ];
    for (i, atom_names) in chi.iter().take(4).enumerate() {
        out[2 + i] = (4 + i, [atom_names[1], atom_names[2], atom_names[3]]);
    }
    out
}

fn is_ambiguous_atom(resname: &str, atom_name: &str) -> bool {
    match resname {
        "ASP" => matches!(atom_name, "OD1" | "OD2"),
        "GLU" => matches!(atom_name, "OE1" | "OE2"),
        "PHE" => matches!(atom_name, "CD1" | "CD2" | "CE1" | "CE2"),
        "TYR" => matches!(atom_name, "CD1" | "CD2" | "CE1" | "CE2"),
        _ => false,
    }
}

fn select_chain<'a>(pdb: &'a pdbtbx::PDB, chain_id: Option<&str>) -> PyResult<&'a pdbtbx::Chain> {
    if chain_id.is_none() && pdb.chain_count() != 1 {
        return Err(PyValueError::new_err(
            "structure_supervision_example v0 is chain-level; pass chain_id for multi-chain structures",
        ));
    }
    let wanted = chain_id.unwrap_or_else(|| pdb.chains().next().map(|c| c.id()).unwrap_or(""));
    pdb.chains()
        .find(|chain| chain.id() == wanted)
        .ok_or_else(|| PyValueError::new_err(format!("chain_id {wanted:?} not found in structure")))
}

fn extract_residue_records(chain: &pdbtbx::Chain) -> PyResult<Vec<ResidueRecord>> {
    let mut out = Vec::new();
    for residue in chain.residues() {
        let is_aa = residue
            .conformers()
            .next()
            .is_some_and(|c| c.is_amino_acid());
        if !is_aa {
            continue;
        }
        let mut atoms = Vec::new();
        for atom in residue_atoms_primary(residue) {
            let name = atom.name().trim().to_ascii_uppercase();
            if name.is_empty() || name.starts_with('H') || name.starts_with('D') {
                continue;
            }
            if atoms.iter().any(|(existing, _)| existing == &name) {
                continue;
            }
            let (x, y, z) = atom.pos();
            atoms.push((name, [x as f32, y as f32, z as f32]));
        }
        out.push(ResidueRecord {
            name: residue.name().unwrap_or("UNK").trim().to_ascii_uppercase(),
            serial_number: residue.serial_number() as i32,
            atoms,
        });
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(
            "structure_supervision_example currently requires a protein chain",
        ));
    }
    Ok(out)
}

fn get_atom_coord(residue: &ResidueRecord, name: &str) -> Option<[f64; 3]> {
    residue
        .atoms
        .iter()
        .find(|(atom_name, _)| atom_name == name)
        .map(|(_, coord)| [coord[0] as f64, coord[1] as f64, coord[2] as f64])
}

fn dihedral(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f32 {
    let b1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b3 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];

    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let n1_norm = norm(n1);
    let n2_norm = norm(n2);
    if n1_norm < 1e-8 || n2_norm < 1e-8 {
        return 0.0;
    }
    let n1 = [n1[0] / n1_norm, n1[1] / n1_norm, n1[2] / n1_norm];
    let n2 = [n2[0] / n2_norm, n2[1] / n2_norm, n2[2] / n2_norm];
    let b2_norm = norm(b2).max(1e-8);
    let b2_hat = [b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm];
    let m1 = cross(n1, b2_hat);
    let x = dot(n1, n2);
    let y = dot(m1, n2);
    (-y).atan2(x) as f32
}

fn homogeneous_frame(
    point_on_neg_x_axis: [f64; 3],
    origin: [f64; 3],
    point_on_xy_plane: [f64; 3],
    mirror_backbone: bool,
) -> [[f32; 4]; 4] {
    let mut ex = [
        origin[0] - point_on_neg_x_axis[0],
        origin[1] - point_on_neg_x_axis[1],
        origin[2] - point_on_neg_x_axis[2],
    ];
    let ex_norm = norm(ex);
    if ex_norm < 1e-8 {
        return identity4();
    }
    ex = [ex[0] / ex_norm, ex[1] / ex_norm, ex[2] / ex_norm];

    let mut ey = [
        point_on_xy_plane[0] - origin[0],
        point_on_xy_plane[1] - origin[1],
        point_on_xy_plane[2] - origin[2],
    ];
    let proj = dot(ey, ex);
    ey = [
        ey[0] - proj * ex[0],
        ey[1] - proj * ex[1],
        ey[2] - proj * ex[2],
    ];
    let ey_norm = norm(ey);
    if ey_norm < 1e-8 {
        return identity4();
    }
    ey = [ey[0] / ey_norm, ey[1] / ey_norm, ey[2] / ey_norm];

    let mut ez = cross(ex, ey);
    let ez_norm = norm(ez);
    if ez_norm < 1e-8 {
        return identity4();
    }
    ez = [ez[0] / ez_norm, ez[1] / ez_norm, ez[2] / ez_norm];

    if mirror_backbone {
        ex = [-ex[0], -ex[1], -ex[2]];
        ez = [-ez[0], -ez[1], -ez[2]];
    }

    [
        [ex[0] as f32, ey[0] as f32, ez[0] as f32, origin[0] as f32],
        [ex[1] as f32, ey[1] as f32, ez[1] as f32, origin[1] as f32],
        [ex[2] as f32, ey[2] as f32, ez[2] as f32, origin[2] as f32],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn identity4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn extract_example_from_pdb(
    pdb: &pdbtbx::PDB,
    chain_id: Option<&str>,
) -> PyResult<ExtractedExample> {
    let chain = select_chain(pdb, chain_id)?;
    let residues = extract_residue_records(chain)?;
    let n = residues.len();

    let mut aatype = vec![0_i32; n];
    let mut residue_index = vec![0_i32; n];
    let seq_mask = vec![1.0_f32; n];
    let mut all_atom_positions = vec![0.0_f32; n * 37 * 3];
    let mut all_atom_mask = vec![0.0_f32; n * 37];
    let mut atom37_atom_exists = vec![0.0_f32; n * 37];
    let mut atom14_gt_positions = vec![0.0_f32; n * 14 * 3];
    let mut atom14_gt_exists = vec![0.0_f32; n * 14];
    let mut atom14_atom_exists = vec![0.0_f32; n * 14];
    let mut residx_atom14_to_atom37 = vec![0_i32; n * 14];
    let mut residx_atom37_to_atom14 = vec![0_i32; n * 37];
    let mut atom14_atom_is_ambiguous = vec![0.0_f32; n * 14];
    let mut pseudo_beta = vec![0.0_f32; n * 3];
    let mut pseudo_beta_mask = vec![0.0_f32; n];
    let mut phi = vec![0.0_f32; n];
    let mut psi = vec![0.0_f32; n];
    let mut omega = vec![0.0_f32; n];
    let mut phi_mask = vec![0.0_f32; n];
    let mut psi_mask = vec![0.0_f32; n];
    let mut omega_mask = vec![0.0_f32; n];
    let mut chi_angles = vec![0.0_f32; n * 4];
    let mut chi_mask = vec![0.0_f32; n * 4];
    let mut rigidgroups_gt_frames = vec![0.0_f32; n * 8 * 4 * 4];
    let mut rigidgroups_gt_exists = vec![0.0_f32; n * 8];
    let mut rigidgroups_group_exists = vec![0.0_f32; n * 8];
    let mut rigidgroups_group_is_ambiguous = vec![0.0_f32; n * 8];

    for i in 0..(n * 8) {
        let base = i * 16;
        rigidgroups_gt_frames[base] = 1.0;
        rigidgroups_gt_frames[base + 5] = 1.0;
        rigidgroups_gt_frames[base + 10] = 1.0;
        rigidgroups_gt_frames[base + 15] = 1.0;
    }

    for (i, residue) in residues.iter().enumerate() {
        let one_letter = residue_to_one_letter(&residue.name);
        aatype[i] = aa_index(one_letter);
        residue_index[i] = residue.serial_number;

        for atom_name in atom14_names(&residue.name) {
            if atom_name.is_empty() {
                continue;
            }
            if let Some(a37) = atom37_index(atom_name) {
                atom37_atom_exists[i * 37 + a37] = 1.0;
            }
        }

        for (atom_name, coord) in &residue.atoms {
            if let Some(a37) = atom37_index(atom_name) {
                let pos_base = (i * 37 + a37) * 3;
                all_atom_positions[pos_base..pos_base + 3].copy_from_slice(coord);
                all_atom_mask[i * 37 + a37] = 1.0;
            }
        }

        for (a14, atom_name) in atom14_names(&residue.name).iter().enumerate() {
            if atom_name.is_empty() {
                continue;
            }
            let a37 = atom37_index(atom_name).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "internal invariant violated: atom14 atom {atom_name:?} \
                     (residue {res_name:?}) is not present in the atom37 table",
                    res_name = residue.name,
                ))
            })?;
            atom14_atom_exists[i * 14 + a14] = 1.0;
            residx_atom14_to_atom37[i * 14 + a14] = a37 as i32;
            residx_atom37_to_atom14[i * 37 + a37] = a14 as i32;
            if is_ambiguous_atom(&residue.name, atom_name) {
                atom14_atom_is_ambiguous[i * 14 + a14] = 1.0;
            }
            if all_atom_mask[i * 37 + a37] > 0.0 {
                let atom14_base = (i * 14 + a14) * 3;
                let atom37_base = (i * 37 + a37) * 3;
                atom14_gt_positions[atom14_base..atom14_base + 3]
                    .copy_from_slice(&all_atom_positions[atom37_base..atom37_base + 3]);
                atom14_gt_exists[i * 14 + a14] = 1.0;
            }
        }

        let pseudo_atom = if residue.name == "GLY" { "CA" } else { "CB" };
        if let Some(a37) = atom37_index(pseudo_atom) {
            if all_atom_mask[i * 37 + a37] > 0.0 {
                let src = (i * 37 + a37) * 3;
                let dst = i * 3;
                pseudo_beta[dst..dst + 3].copy_from_slice(&all_atom_positions[src..src + 3]);
                pseudo_beta_mask[i] = 1.0;
            }
        }

        rigidgroups_group_exists[i * 8] = 1.0;
        rigidgroups_group_exists[i * 8 + 3] = 1.0;
        for (group_idx, atom_names) in rigidgroup_base_atoms(&residue.name) {
            if group_idx >= 8 || atom_names[0].is_empty() {
                continue;
            }
            if group_idx >= 4 {
                rigidgroups_group_exists[i * 8 + group_idx] = 1.0;
            }
            if let (Some(a), Some(b), Some(c)) = (
                get_atom_coord(residue, atom_names[0]),
                get_atom_coord(residue, atom_names[1]),
                get_atom_coord(residue, atom_names[2]),
            ) {
                let frame = homogeneous_frame(a, b, c, group_idx == 0);
                let base = (i * 8 + group_idx) * 16;
                for row in 0..4 {
                    for col in 0..4 {
                        rigidgroups_gt_frames[base + row * 4 + col] = frame[row][col];
                    }
                }
                rigidgroups_gt_exists[i * 8 + group_idx] = 1.0;
            }
        }
        if matches!(residue.name.as_str(), "ASP" | "GLU" | "PHE" | "TYR") {
            let last_chi = chi_angle_atoms(&residue.name).len();
            if last_chi > 0 {
                rigidgroups_group_is_ambiguous[i * 8 + 3 + last_chi] = 1.0;
            }
        }
    }

    for i in 0..n {
        if i > 0 {
            let prev = &residues[i - 1];
            let cur = &residues[i];
            if let (Some(c_prev), Some(n_cur), Some(ca_cur), Some(c_cur)) = (
                get_atom_coord(prev, "C"),
                get_atom_coord(cur, "N"),
                get_atom_coord(cur, "CA"),
                get_atom_coord(cur, "C"),
            ) {
                phi[i] = dihedral(c_prev, n_cur, ca_cur, c_cur);
                phi_mask[i] = 1.0;
            }
            if let (Some(ca_prev), Some(c_prev), Some(n_cur), Some(ca_cur)) = (
                get_atom_coord(prev, "CA"),
                get_atom_coord(prev, "C"),
                get_atom_coord(cur, "N"),
                get_atom_coord(cur, "CA"),
            ) {
                omega[i] = dihedral(ca_prev, c_prev, n_cur, ca_cur);
                omega_mask[i] = 1.0;
            }
        }
        if i + 1 < n {
            let cur = &residues[i];
            let nxt = &residues[i + 1];
            if let (Some(n_cur), Some(ca_cur), Some(c_cur), Some(n_next)) = (
                get_atom_coord(cur, "N"),
                get_atom_coord(cur, "CA"),
                get_atom_coord(cur, "C"),
                get_atom_coord(nxt, "N"),
            ) {
                psi[i] = dihedral(n_cur, ca_cur, c_cur, n_next);
                psi_mask[i] = 1.0;
            }
        }
        for (chi_i, atom_names) in chi_angle_atoms(&residues[i].name)
            .iter()
            .enumerate()
            .take(4)
        {
            if let (Some(a), Some(b), Some(c), Some(d)) = (
                get_atom_coord(&residues[i], atom_names[0]),
                get_atom_coord(&residues[i], atom_names[1]),
                get_atom_coord(&residues[i], atom_names[2]),
                get_atom_coord(&residues[i], atom_names[3]),
            ) {
                chi_angles[i * 4 + chi_i] = dihedral(a, b, c, d);
                chi_mask[i * 4 + chi_i] = 1.0;
            }
        }
    }

    Ok(ExtractedExample {
        length: n,
        aatype,
        residue_index,
        seq_mask,
        all_atom_positions,
        all_atom_mask,
        atom37_atom_exists,
        atom14_gt_positions,
        atom14_gt_exists,
        atom14_atom_exists,
        residx_atom14_to_atom37,
        residx_atom37_to_atom14,
        atom14_atom_is_ambiguous,
        pseudo_beta,
        pseudo_beta_mask,
        phi,
        psi,
        omega,
        phi_mask,
        psi_mask,
        omega_mask,
        chi_angles,
        chi_mask,
        rigidgroups_gt_frames,
        rigidgroups_gt_exists,
        rigidgroups_group_exists,
        rigidgroups_group_is_ambiguous,
    })
}

fn example_to_dict(py: Python<'_>, ex: ExtractedExample) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("aatype", ex.aatype.into_pyarray(py))?;
    dict.set_item("residue_index", ex.residue_index.into_pyarray(py))?;
    dict.set_item("seq_mask", ex.seq_mask.into_pyarray(py))?;
    dict.set_item(
        "all_atom_positions",
        PyArray1::from_vec(py, ex.all_atom_positions)
            .reshape([ex.length, 37, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape all_atom_positions: {e}")))?,
    )?;
    dict.set_item(
        "all_atom_mask",
        PyArray1::from_vec(py, ex.all_atom_mask)
            .reshape([ex.length, 37])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape all_atom_mask: {e}")))?,
    )?;
    dict.set_item(
        "atom37_atom_exists",
        PyArray1::from_vec(py, ex.atom37_atom_exists)
            .reshape([ex.length, 37])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape atom37_atom_exists: {e}")))?,
    )?;
    dict.set_item(
        "atom14_gt_positions",
        PyArray1::from_vec(py, ex.atom14_gt_positions)
            .reshape([ex.length, 14, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape atom14_gt_positions: {e}")))?,
    )?;
    dict.set_item(
        "atom14_gt_exists",
        PyArray1::from_vec(py, ex.atom14_gt_exists)
            .reshape([ex.length, 14])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape atom14_gt_exists: {e}")))?,
    )?;
    dict.set_item(
        "atom14_atom_exists",
        PyArray1::from_vec(py, ex.atom14_atom_exists)
            .reshape([ex.length, 14])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape atom14_atom_exists: {e}")))?,
    )?;
    dict.set_item(
        "residx_atom14_to_atom37",
        PyArray1::from_vec(py, ex.residx_atom14_to_atom37)
            .reshape([ex.length, 14])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape residx_atom14_to_atom37: {e}"))
            })?,
    )?;
    dict.set_item(
        "residx_atom37_to_atom14",
        PyArray1::from_vec(py, ex.residx_atom37_to_atom14)
            .reshape([ex.length, 37])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape residx_atom37_to_atom14: {e}"))
            })?,
    )?;
    dict.set_item(
        "atom14_atom_is_ambiguous",
        PyArray1::from_vec(py, ex.atom14_atom_is_ambiguous)
            .reshape([ex.length, 14])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape atom14_atom_is_ambiguous: {e}"))
            })?,
    )?;
    dict.set_item(
        "pseudo_beta",
        PyArray1::from_vec(py, ex.pseudo_beta)
            .reshape([ex.length, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape pseudo_beta: {e}")))?,
    )?;
    dict.set_item("pseudo_beta_mask", ex.pseudo_beta_mask.into_pyarray(py))?;
    dict.set_item("phi", ex.phi.into_pyarray(py))?;
    dict.set_item("psi", ex.psi.into_pyarray(py))?;
    dict.set_item("omega", ex.omega.into_pyarray(py))?;
    dict.set_item("phi_mask", ex.phi_mask.into_pyarray(py))?;
    dict.set_item("psi_mask", ex.psi_mask.into_pyarray(py))?;
    dict.set_item("omega_mask", ex.omega_mask.into_pyarray(py))?;
    dict.set_item(
        "chi_angles",
        PyArray1::from_vec(py, ex.chi_angles)
            .reshape([ex.length, 4])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape chi_angles: {e}")))?,
    )?;
    dict.set_item(
        "chi_mask",
        PyArray1::from_vec(py, ex.chi_mask)
            .reshape([ex.length, 4])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape chi_mask: {e}")))?,
    )?;
    dict.set_item(
        "rigidgroups_gt_frames",
        PyArray1::from_vec(py, ex.rigidgroups_gt_frames)
            .reshape([ex.length, 8, 4, 4])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape rigidgroups_gt_frames: {e}")))?,
    )?;
    dict.set_item(
        "rigidgroups_gt_exists",
        PyArray1::from_vec(py, ex.rigidgroups_gt_exists)
            .reshape([ex.length, 8])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape rigidgroups_gt_exists: {e}")))?,
    )?;
    dict.set_item(
        "rigidgroups_group_exists",
        PyArray1::from_vec(py, ex.rigidgroups_group_exists)
            .reshape([ex.length, 8])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape rigidgroups_group_exists: {e}"))
            })?,
    )?;
    dict.set_item(
        "rigidgroups_group_is_ambiguous",
        PyArray1::from_vec(py, ex.rigidgroups_group_is_ambiguous)
            .reshape([ex.length, 8])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape rigidgroups_group_is_ambiguous: {e}"))
            })?,
    )?;
    Ok(dict)
}

fn batch_to_dict(py: Python<'_>, batch: Vec<ExtractedExample>) -> PyResult<Bound<'_, PyDict>> {
    let b = batch.len();
    let n_max = batch.iter().map(|ex| ex.length).max().unwrap_or(0);

    let mut aatype = vec![0_i32; b * n_max];
    let mut residue_index = vec![0_i32; b * n_max];
    let mut seq_mask = vec![0.0_f32; b * n_max];
    let mut all_atom_positions = vec![0.0_f32; b * n_max * 37 * 3];
    let mut all_atom_mask = vec![0.0_f32; b * n_max * 37];
    let mut atom37_atom_exists = vec![0.0_f32; b * n_max * 37];
    let mut atom14_gt_positions = vec![0.0_f32; b * n_max * 14 * 3];
    let mut atom14_gt_exists = vec![0.0_f32; b * n_max * 14];
    let mut atom14_atom_exists = vec![0.0_f32; b * n_max * 14];
    let mut residx_atom14_to_atom37 = vec![0_i32; b * n_max * 14];
    let mut residx_atom37_to_atom14 = vec![0_i32; b * n_max * 37];
    let mut atom14_atom_is_ambiguous = vec![0.0_f32; b * n_max * 14];
    let mut pseudo_beta = vec![0.0_f32; b * n_max * 3];
    let mut pseudo_beta_mask = vec![0.0_f32; b * n_max];
    let mut phi = vec![0.0_f32; b * n_max];
    let mut psi = vec![0.0_f32; b * n_max];
    let mut omega = vec![0.0_f32; b * n_max];
    let mut phi_mask = vec![0.0_f32; b * n_max];
    let mut psi_mask = vec![0.0_f32; b * n_max];
    let mut omega_mask = vec![0.0_f32; b * n_max];
    let mut chi_angles = vec![0.0_f32; b * n_max * 4];
    let mut chi_mask = vec![0.0_f32; b * n_max * 4];
    let mut rigidgroups_gt_frames = vec![0.0_f32; b * n_max * 8 * 4 * 4];
    let mut rigidgroups_gt_exists = vec![0.0_f32; b * n_max * 8];
    let mut rigidgroups_group_exists = vec![0.0_f32; b * n_max * 8];
    let mut rigidgroups_group_is_ambiguous = vec![0.0_f32; b * n_max * 8];

    for (bi, ex) in batch.into_iter().enumerate() {
        let n = ex.length;
        aatype[bi * n_max..bi * n_max + n].copy_from_slice(&ex.aatype);
        residue_index[bi * n_max..bi * n_max + n].copy_from_slice(&ex.residue_index);
        seq_mask[bi * n_max..bi * n_max + n].copy_from_slice(&ex.seq_mask);

        let atom37_pos_dst = bi * n_max * 37 * 3;
        let atom37_dst = bi * n_max * 37;
        let atom14_pos_dst = bi * n_max * 14 * 3;
        let atom14_dst = bi * n_max * 14;
        let pseudo_dst = bi * n_max * 3;
        all_atom_positions[atom37_pos_dst..atom37_pos_dst + n * 37 * 3]
            .copy_from_slice(&ex.all_atom_positions);
        all_atom_mask[atom37_dst..atom37_dst + n * 37].copy_from_slice(&ex.all_atom_mask);
        atom37_atom_exists[atom37_dst..atom37_dst + n * 37].copy_from_slice(&ex.atom37_atom_exists);
        atom14_gt_positions[atom14_pos_dst..atom14_pos_dst + n * 14 * 3]
            .copy_from_slice(&ex.atom14_gt_positions);
        atom14_gt_exists[atom14_dst..atom14_dst + n * 14].copy_from_slice(&ex.atom14_gt_exists);
        atom14_atom_exists[atom14_dst..atom14_dst + n * 14].copy_from_slice(&ex.atom14_atom_exists);
        residx_atom14_to_atom37[atom14_dst..atom14_dst + n * 14]
            .copy_from_slice(&ex.residx_atom14_to_atom37);
        residx_atom37_to_atom14[atom37_dst..atom37_dst + n * 37]
            .copy_from_slice(&ex.residx_atom37_to_atom14);
        atom14_atom_is_ambiguous[atom14_dst..atom14_dst + n * 14]
            .copy_from_slice(&ex.atom14_atom_is_ambiguous);
        pseudo_beta[pseudo_dst..pseudo_dst + n * 3].copy_from_slice(&ex.pseudo_beta);
        pseudo_beta_mask[bi * n_max..bi * n_max + n].copy_from_slice(&ex.pseudo_beta_mask);
        phi[bi * n_max..bi * n_max + n].copy_from_slice(&ex.phi);
        psi[bi * n_max..bi * n_max + n].copy_from_slice(&ex.psi);
        omega[bi * n_max..bi * n_max + n].copy_from_slice(&ex.omega);
        phi_mask[bi * n_max..bi * n_max + n].copy_from_slice(&ex.phi_mask);
        psi_mask[bi * n_max..bi * n_max + n].copy_from_slice(&ex.psi_mask);
        omega_mask[bi * n_max..bi * n_max + n].copy_from_slice(&ex.omega_mask);
        chi_angles[bi * n_max * 4..bi * n_max * 4 + n * 4].copy_from_slice(&ex.chi_angles);
        chi_mask[bi * n_max * 4..bi * n_max * 4 + n * 4].copy_from_slice(&ex.chi_mask);
        let rg_base = bi * n_max * 8;
        let rg_frame_base = bi * n_max * 8 * 16;
        rigidgroups_gt_frames[rg_frame_base..rg_frame_base + n * 8 * 16]
            .copy_from_slice(&ex.rigidgroups_gt_frames);
        rigidgroups_gt_exists[rg_base..rg_base + n * 8].copy_from_slice(&ex.rigidgroups_gt_exists);
        rigidgroups_group_exists[rg_base..rg_base + n * 8]
            .copy_from_slice(&ex.rigidgroups_group_exists);
        rigidgroups_group_is_ambiguous[rg_base..rg_base + n * 8]
            .copy_from_slice(&ex.rigidgroups_group_is_ambiguous);
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "aatype",
        PyArray1::from_vec(py, aatype)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch aatype: {e}")))?,
    )?;
    dict.set_item(
        "residue_index",
        PyArray1::from_vec(py, residue_index)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch residue_index: {e}")))?,
    )?;
    dict.set_item(
        "seq_mask",
        PyArray1::from_vec(py, seq_mask)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch seq_mask: {e}")))?,
    )?;
    dict.set_item(
        "all_atom_positions",
        PyArray1::from_vec(py, all_atom_positions)
            .reshape([b, n_max, 37, 3])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch all_atom_positions: {e}"))
            })?,
    )?;
    dict.set_item(
        "all_atom_mask",
        PyArray1::from_vec(py, all_atom_mask)
            .reshape([b, n_max, 37])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch all_atom_mask: {e}")))?,
    )?;
    dict.set_item(
        "atom37_atom_exists",
        PyArray1::from_vec(py, atom37_atom_exists)
            .reshape([b, n_max, 37])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch atom37_atom_exists: {e}"))
            })?,
    )?;
    dict.set_item(
        "atom14_gt_positions",
        PyArray1::from_vec(py, atom14_gt_positions)
            .reshape([b, n_max, 14, 3])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch atom14_gt_positions: {e}"))
            })?,
    )?;
    dict.set_item(
        "atom14_gt_exists",
        PyArray1::from_vec(py, atom14_gt_exists)
            .reshape([b, n_max, 14])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch atom14_gt_exists: {e}")))?,
    )?;
    dict.set_item(
        "atom14_atom_exists",
        PyArray1::from_vec(py, atom14_atom_exists)
            .reshape([b, n_max, 14])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch atom14_atom_exists: {e}"))
            })?,
    )?;
    dict.set_item(
        "residx_atom14_to_atom37",
        PyArray1::from_vec(py, residx_atom14_to_atom37)
            .reshape([b, n_max, 14])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch residx_atom14_to_atom37: {e}"))
            })?,
    )?;
    dict.set_item(
        "residx_atom37_to_atom14",
        PyArray1::from_vec(py, residx_atom37_to_atom14)
            .reshape([b, n_max, 37])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch residx_atom37_to_atom14: {e}"))
            })?,
    )?;
    dict.set_item(
        "atom14_atom_is_ambiguous",
        PyArray1::from_vec(py, atom14_atom_is_ambiguous)
            .reshape([b, n_max, 14])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch atom14_atom_is_ambiguous: {e}"))
            })?,
    )?;
    dict.set_item(
        "pseudo_beta",
        PyArray1::from_vec(py, pseudo_beta)
            .reshape([b, n_max, 3])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch pseudo_beta: {e}")))?,
    )?;
    dict.set_item(
        "pseudo_beta_mask",
        PyArray1::from_vec(py, pseudo_beta_mask)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch pseudo_beta_mask: {e}")))?,
    )?;
    dict.set_item(
        "phi",
        PyArray1::from_vec(py, phi)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch phi: {e}")))?,
    )?;
    dict.set_item(
        "psi",
        PyArray1::from_vec(py, psi)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch psi: {e}")))?,
    )?;
    dict.set_item(
        "omega",
        PyArray1::from_vec(py, omega)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch omega: {e}")))?,
    )?;
    dict.set_item(
        "phi_mask",
        PyArray1::from_vec(py, phi_mask)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch phi_mask: {e}")))?,
    )?;
    dict.set_item(
        "psi_mask",
        PyArray1::from_vec(py, psi_mask)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch psi_mask: {e}")))?,
    )?;
    dict.set_item(
        "omega_mask",
        PyArray1::from_vec(py, omega_mask)
            .reshape([b, n_max])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch omega_mask: {e}")))?,
    )?;
    dict.set_item(
        "chi_angles",
        PyArray1::from_vec(py, chi_angles)
            .reshape([b, n_max, 4])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch chi_angles: {e}")))?,
    )?;
    dict.set_item(
        "chi_mask",
        PyArray1::from_vec(py, chi_mask)
            .reshape([b, n_max, 4])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape batch chi_mask: {e}")))?,
    )?;
    dict.set_item(
        "rigidgroups_gt_frames",
        PyArray1::from_vec(py, rigidgroups_gt_frames)
            .reshape([b, n_max, 8, 4, 4])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch rigidgroups_gt_frames: {e}"))
            })?,
    )?;
    dict.set_item(
        "rigidgroups_gt_exists",
        PyArray1::from_vec(py, rigidgroups_gt_exists)
            .reshape([b, n_max, 8])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch rigidgroups_gt_exists: {e}"))
            })?,
    )?;
    dict.set_item(
        "rigidgroups_group_exists",
        PyArray1::from_vec(py, rigidgroups_group_exists)
            .reshape([b, n_max, 8])
            .map_err(|e| {
                PyRuntimeError::new_err(format!("reshape batch rigidgroups_group_exists: {e}"))
            })?,
    )?;
    dict.set_item(
        "rigidgroups_group_is_ambiguous",
        PyArray1::from_vec(py, rigidgroups_group_is_ambiguous)
            .reshape([b, n_max, 8])
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "reshape batch rigidgroups_group_is_ambiguous: {e}"
                ))
            })?,
    )?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (structure, chain_id=None))]
fn extract_structure_supervision_chain<'py>(
    py: Python<'py>,
    structure: &PyPDB,
    chain_id: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let ex = extract_example_from_pdb(&structure.inner, chain_id.as_deref())?;
    example_to_dict(py, ex)
}

#[pyfunction]
#[pyo3(signature = (structures, chain_ids=None, n_threads=None))]
fn batch_extract_structure_supervision(
    py: Python<'_>,
    structures: Vec<Py<PyPDB>>,
    chain_ids: Option<Vec<Option<String>>>,
    n_threads: Option<i32>,
) -> PyResult<Bound<'_, PyDict>> {
    let n = structures.len();
    let chain_ids = match chain_ids {
        Some(ids) if ids.len() != n => {
            return Err(PyValueError::new_err(format!(
                "expected {n} chain_ids, got {}",
                ids.len()
            )))
        }
        Some(ids) => ids,
        None => vec![None; n],
    };

    let inputs: Vec<(pdbtbx::PDB, Option<String>)> = structures
        .into_iter()
        .zip(chain_ids)
        .map(|(structure, chain_id)| {
            let pdb = structure.borrow(py).inner.clone();
            (pdb, chain_id)
        })
        .collect();

    let pool = build_pool(resolve_threads(n_threads));
    let batch: Vec<PyResult<ExtractedExample>> = pool.install(|| {
        inputs
            .into_par_iter()
            .map(|(pdb, chain_id)| extract_example_from_pdb(&pdb, chain_id.as_deref()))
            .collect()
    });

    let mut extracted = Vec::with_capacity(n);
    for item in batch {
        extracted.push(item?);
    }
    batch_to_dict(py, extracted)
}

#[pymodule]
pub(crate) fn py_supervision(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_structure_supervision_chain, m)?)?;
    m.add_function(wrap_pyfunction!(batch_extract_structure_supervision, m)?)?;
    Ok(())
}
