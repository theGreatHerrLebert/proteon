//! Bond topology inference and force field atom assignment.
//!
//! Infers bonds from interatomic distances, then builds the lists
//! of bond/angle/torsion interactions needed for energy computation.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::params::ForceField;
use crate::altloc::residue_atoms_primary;
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

/// Return alternate hydrogen atom-name spellings, in priority order.
///
/// Background: proteon's `amber96.ini` / `charmm19_eef1.ini` templates
/// inherit from BALL (old wwPDB convention) where a methyl/methylene
/// prefixes the digit: `1HB`, `2HG1`. PDBFixer and modern AMBER/OpenMM
/// use the PDB v3 convention: `HB1`, `HG12`.
///
/// Two cases to bridge:
///   1. *Methyl* (3 H's on one carbon): names `HB1/HB2/HB3` in PDB v3
///      correspond to `1HB/2HB/3HB` in BALL. Simple "move the last digit
///      to the front" rotation suffices.
///   2. *Methylene* (2 H's on one carbon): PDB v3 uses `HB2/HB3`,
///      BALL uses `1HB/2HB`. `HB2 → 2HB` works by rotation, but
///      `HB3 → 3HB` misses because BALL has no `3HB`. We therefore
///      also offer the "decrement-then-rotate" candidate `2HB` as a
///      fallback for any trailing-digit name.
///
/// Returned candidates are tried in order; the first FF hit wins. Since
/// methylene H's share atom type, mapping `HB3` to `2HB` gives the right
/// AMBER type even though `HB2`/`HB3` are chemically indistinguishable.
/// For methyl H's the rotation always hits before the decrement does.
///
/// Names without a leading or trailing digit (`HA`, `HN`, `H`) return
/// an empty vec — nothing to rotate.
pub(crate) fn alt_h_name_candidates(name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(2);
    if name.is_empty() {
        return out;
    }
    let bytes = name.as_bytes();
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;
    if first.is_ascii_digit() {
        // BALL → PDB v3: `1HB` → `HB1`. (Decrement variant unnecessary
        // here — BALL is the convention we're normalizing *from* in the
        // input, not the one we're normalizing *to*.)
        let rest = &name[1..];
        out.push(format!("{rest}{first}"));
    } else if last.is_ascii_digit() {
        let head = &name[..name.len() - 1];
        let d = last.to_digit(10).unwrap();
        // Methyl rotation (PDB v3 → BALL): HB1 → 1HB.
        out.push(format!("{last}{head}"));
        // Methylene decrement-then-rotate (PDB v3 → BALL): HB3 → 2HB.
        // Only meaningful for d > 1.
        if d > 1 {
            out.push(format!("{}{head}", (d - 1)));
        }
    } else if name == "H" {
        // PDBFixer names the first NH3+ proton on the N-terminal residue
        // just `H` (alongside `H2`, `H3`). BALL's amber96.ini keys that
        // atom as `1H` in the per-residue N-terminal template (e.g.
        // `THR-N:1H`). With no trailing digit to rotate, the generic
        // PDB v3 ↔ BALL alt-name machinery misses it and the lookup falls
        // back to the interior amide `*:H` charge (+0.272e instead of
        // +0.193e on crambin), which blows up GB by ~0.08e of excess
        // charge on the N-terminus. Mapping `H` → `1H` restores parity.
        out.push("1H".to_string());
    }
    out
}

/// Look up an atom type, trying the original name and (for hydrogens) the
/// alternate PDB v3 <-> BALL naming convention. Returns the first match.
fn get_atom_type_with_aliases<'a, F: ForceField + ?Sized>(
    params: &'a F,
    residue: &str,
    atom: &str,
) -> Option<&'a super::params::AtomTypeEntry> {
    if let Some(e) = params.get_atom_type(residue, atom) {
        return Some(e);
    }
    for alt in alt_h_name_candidates(atom) {
        if let Some(e) = params.get_atom_type(residue, &alt) {
            return Some(e);
        }
    }
    None
}

/// Shared predicate: should this (residue_name, atom_name, element) triple
/// be included in the force-field topology?
///
/// Returns `false` in two cases that must be mirrored across
/// `build_topology` and `apply_coords_to_pdb` (coordinate write-back
/// after minimization) so the two stay in lockstep:
///
/// 1. **Water residue** (HOH/WAT/...) — waters have no type in a vacuum
///    protein FF and were previously catastrophically clashing when
///    fallback-typed as CT. See the water-skip commit message for
///    numerical impact.
///
/// 2. **Non-polar hydrogen with no FF entry** — under polar-H united-
///    atom force fields (CHARMM19+EEF1), C-H atoms don't exist at all;
///    they're absorbed into CH1E/CH2E/CH3E united carbon types. Adding
///    them with a fallback "H" type produced ~40% unassigned atoms and
///    wrong-signed totals on every v1 PDB except 1crn. See the related
///    commit for the BALL + GROMACS cross-oracle confirmation.
///
/// If you change this function, update `apply_coords_to_pdb` in
/// `crate::py_add_hydrogens` in the same commit.
pub(crate) fn should_include_atom<F: ForceField + ?Sized>(
    residue_name: &str,
    atom_name: &str,
    element: &str,
    params: &F,
    // The residue lookup name used by build_topology — for terminal
    // variants this is "ALA-N" / "VAL-C" / etc. so the FF has a chance
    // to resolve N-terminal H, OXT, etc. apply_coords_to_pdb doesn't
    // know the variants (it has no topology context), so it passes the
    // base name and relies on the or_else fallback in get_atom_type.
    lookup_name: &str,
) -> bool {
    // Waters: skip entirely.
    if crate::add_hydrogens::is_water_residue(residue_name) {
        return false;
    }

    // For hydrogens: skip if the FF has no entry for this
    // (residue, atom_name) pair. Non-H atoms with no entry fall
    // through to the topology-builder fallback for visibility.
    //
    // Also try the alternate PDB v3 <-> BALL naming convention — proteon's
    // .ini templates use the BALL convention (`1HB`, `2HG1`) but PDBFixer
    // and modern AMBER emit PDB v3 (`HB1`, `HG12`). Without the alias
    // lookup, PDBFixer-prepped structures silently lose ~30% of their
    // hydrogens from the topology. See `alt_h_name` for the rotation rule
    // and memory/project_amber96_broken.md for the 2026-04-13 diagnosis.
    let is_hydrogen = element == "H" || element == "D";
    if is_hydrogen {
        let nterm = format!("{residue_name}-N");
        let cterm = format!("{residue_name}-C");
        let found = get_atom_type_with_aliases(params, lookup_name, atom_name).is_some()
            || get_atom_type_with_aliases(params, residue_name, atom_name).is_some()
            // Also try terminal variants. build_topology computes these
            // from chain context and passes them via `lookup_name`; but
            // `apply_coords_to_pdb` has no chain context and passes the
            // bare residue name, so it would silently skip N-terminal H1
            // / C-terminal OXT hydrogens and drift out of lockstep with
            // build_topology, causing a coord-array vs atom-count panic
            // downstream. Trying the variants here keeps the predicate
            // symmetric between callers. In practice these H names only
            // appear at true termini in well-formed PDBs, so we don't
            // produce false positives.
            || get_atom_type_with_aliases(params, &nterm, atom_name).is_some()
            || get_atom_type_with_aliases(params, &cterm, atom_name).is_some();
        if !found {
            return false;
        }
    }
    true
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
        None => {
            return Topology {
                atoms: Vec::new(),
                bonds: Vec::new(),
                angles: Vec::new(),
                torsions: Vec::new(),
                improper_torsions: Vec::new(),
                excluded_pairs: HashSet::new(),
                pairs_14: HashSet::new(),
                unassigned_atoms: Vec::new(),
            }
        }
    };
    for chain in first_model.chains() {
        let mut chain_residues: Vec<(usize, String)> = Vec::new(); // (res_idx, name)
        for residue in chain.residues() {
            let name = residue.name().unwrap_or("UNK").to_string();
            let is_aa = residue
                .conformers()
                .next()
                .is_some_and(|c| c.is_amino_acid());
            if is_aa {
                chain_residues.push((res_idx, name.clone()));
            }
            // Find SG atoms for disulfide detection (primary conformer only —
            // see crate::altloc for the altloc-duplication background).
            if name == "CYS" {
                for atom in residue_atoms_primary(residue) {
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

    // Auto-apply the CYS-S (disulfide) variant when two SG atoms are within
    // 2.5 Å (same threshold used for disulfide bond detection in Phase C
    // below). AMBER/OpenMM's `amber96.xml` detects disulfides from PDB
    // geometry and uses CYX charges in this case (SG q=-0.108 vs
    // CYS-SH q=-0.312) — without this, the 6 cysteines in crambin carry a
    // total of ~1.2e excess negative charge, which is invisible to the
    // vacuum oracle (charges mostly cancel in long-range Coulomb) but
    // blows up GB by ~16% on crambin. If a disulfide CYS is also chain-
    // terminal, the terminal variant wins (we don't carry a combined
    // CYS-S-N / CYS-S-C template). This differs from BALL, which only
    // applies CYS-S when explicitly flagged during preprocessing.
    for i in 0..sg_positions.len() {
        for j in (i + 1)..sg_positions.len() {
            let (ri, pi) = &sg_positions[i];
            let (rj, pj) = &sg_positions[j];
            let dx = pi[0] - pj[0];
            let dy = pi[1] - pj[1];
            let dz = pi[2] - pj[2];
            if dx * dx + dy * dy + dz * dz < 2.5_f64 * 2.5_f64 {
                residue_variants
                    .entry(*ri)
                    .or_insert_with(|| "CYS-S".to_string());
                residue_variants
                    .entry(*rj)
                    .or_insert_with(|| "CYS-S".to_string());
            }
        }
    }

    // Step 1: Extract atoms with type assignment using variant names.
    //
    // Waters, non-polar hydrogens under polar-H force fields, and other
    // "skipped atoms" are filtered out via `should_include_atom`. That
    // predicate is the single source of truth for the include/skip
    // decision, and `apply_coords_to_pdb` must call the same function
    // so the coord write-back iterates in lockstep with the topology.
    // See the module comment on `should_include_atom` for rationale.
    let mut atoms = Vec::new();
    let mut unassigned_atoms = Vec::new();
    res_idx = 0;

    for chain in first_model.chains() {
        for residue in chain.residues() {
            let base_name = residue.name().unwrap_or("UNK").to_string();
            let lookup_name = residue_variants
                .get(&res_idx)
                .cloned()
                .unwrap_or_else(|| base_name.clone());

            // Primary conformer only: pdbtbx duplicates non-altloc backbone
            // atoms into every altloc conformer, which would otherwise yield
            // a zero-distance atom pair and blow up 1/r^12 vdW terms.
            // See crate::altloc for the full story.
            for atom in residue_atoms_primary(residue) {
                let (x, y, z) = atom.pos();
                let atom_name = atom.name().trim().to_string();
                let element = atom
                    .element()
                    .map(|e| e.symbol().to_string())
                    .unwrap_or_else(|| "C".to_string());
                let is_hydrogen = element == "H" || element == "D";

                // Shared include/skip predicate — identical in
                // apply_coords_to_pdb. See `should_include_atom` above.
                //
                // Skipped atoms are NOT added to `unassigned_atoms`. That
                // field tracks "atoms we wanted to put into the topology
                // but couldn't figure out a type for" — it's a signal
                // for the `skipped_no_protein` heuristic in batch_prepare
                // and a diagnostic for users. Atoms skipped here are
                // skipped by DESIGN (waters because the FF is vacuum,
                // non-polar H because the FF is polar-H united-atom), not
                // because of a typing failure. Conflating the two would
                // re-break the heuristic — it would count expected skips
                // against the "is this a protein?" threshold and falsely
                // reject valid inputs.
                if !should_include_atom(&base_name, &atom_name, &element, params, &lookup_name) {
                    continue;
                }

                // At this point the predicate said "include". For hydrogens
                // this means the FF has a type for (residue, atom_name).
                // For non-H it means "not water, maybe unknown ligand —
                // include with whatever the lookup gives us, falling back
                // to a default type if unknown". The fallback path is
                // retained for visibility of unknown ligand atoms.
                let lookup = get_atom_type_with_aliases(params, &lookup_name, &atom_name)
                    .or_else(|| get_atom_type_with_aliases(params, &base_name, &atom_name));
                let (amber_type, charge) = if let Some(e) = lookup {
                    (e.amber_type.clone(), e.charge)
                } else {
                    // Only non-H atoms reach here — H would have been
                    // skipped by the predicate above.
                    unassigned_atoms.push(format!("{}:{}", base_name, atom_name));
                    let t = match element.as_str() {
                        "C" => "CT",
                        "N" => "N",
                        "O" => "O",
                        "S" => "S",
                        "P" => "P",
                        _ => "CT",
                    };
                    (t.to_string(), 0.0)
                };

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

    // Build an index: (residue_idx, atom_name) → global atom index.
    //
    // Owned String keys (not &str) so we can also store the alternate
    // PDB v3 <-> BALL hydrogen-naming convention next to the original.
    // Template bond lists in proteon use BALL convention (`1HB`), but
    // PDBFixer-prepped inputs use PDB v3 (`HB1`); inserting both aliases
    // here lets Phase A find the atom regardless of which convention the
    // source PDB used.
    // Classify methylene carbons per residue so the PDB v3 → BALL
    // name mapping is unambiguous. AMBER's PDB v3 convention numbers
    // methylene hydrogens as `HB2, HB3` (no `HB1`); BALL numbers the
    // same two atoms as `1HB, 2HB`. We need the subtract-then-rotate
    // mapping `HB2→1HB, HB3→2HB` for methylenes, and the plain
    // rotation `HB1→1HB, HB2→2HB, HB3→3HB` for methyls. Without this
    // classification we'd guess wrong for one case or the other and
    // silently drop a bond per methylene. Detect methylenes by
    // counting: exactly two hydrogens on one carbon-prefix, numbered 2
    // and 3 (no 1) ⇒ methylene.
    let mut is_methylene: HashSet<(usize, String)> = HashSet::new();
    {
        let mut by_prefix: HashMap<(usize, String), Vec<u32>> = HashMap::new();
        for atom in &atoms {
            if !atom.is_hydrogen {
                continue;
            }
            let name = atom.atom_name.as_bytes();
            if name.len() < 2 {
                continue;
            }
            let first = name[0];
            let last = name[name.len() - 1];
            // Only PDB v3 style names have a non-digit leading char and
            // a trailing digit.
            if !first.is_ascii_digit() && last.is_ascii_digit() {
                let d = (last as char).to_digit(10).unwrap();
                let prefix = atom.atom_name[..atom.atom_name.len() - 1].to_string();
                by_prefix
                    .entry((atom.residue_idx, prefix))
                    .or_default()
                    .push(d);
            }
        }
        for ((res_idx, prefix), mut digits) in by_prefix {
            digits.sort_unstable();
            if digits.len() == 2 && digits[0] == 2 && digits[1] == 3 {
                is_methylene.insert((res_idx, prefix));
            }
        }
    }

    let mut res_atom_index: HashMap<(usize, String), usize> = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        res_atom_index.insert((atom.residue_idx, atom.atom_name.clone()), i);
        if atom.is_hydrogen {
            // Derive the alt name based on methyl-vs-methylene classification.
            let name = atom.atom_name.as_bytes();
            if name.is_empty() {
                continue;
            }
            let first = name[0];
            let last = name[name.len() - 1];
            if first.is_ascii_digit() {
                // BALL → PDB v3 rotation (for inputs that already use BALL).
                let rest = &atom.atom_name[1..];
                let alt = format!("{rest}{}", first as char);
                res_atom_index.entry((atom.residue_idx, alt)).or_insert(i);
            } else if last.is_ascii_digit() {
                let head = &atom.atom_name[..atom.atom_name.len() - 1];
                let d = (last as char).to_digit(10).unwrap();
                let methylene_key = (atom.residue_idx, head.to_string());
                if is_methylene.contains(&methylene_key) && d >= 2 {
                    // PDB v3 methylene HB2→1HB, HB3→2HB.
                    let alt = format!("{}{head}", d - 1);
                    res_atom_index.entry((atom.residue_idx, alt)).or_insert(i);
                } else {
                    // Methyl (or other): simple rotation HB1→1HB etc.
                    let alt = format!("{last}{head}", last = last as char);
                    res_atom_index.entry((atom.residue_idx, alt)).or_insert(i);
                }
            }
        }
    }

    // Build a template lookup
    let templates: HashMap<&str, &fragment_templates::FragmentTemplate> =
        fragment_templates::TEMPLATES
            .iter()
            .map(|t| (t.0, t))
            .collect();

    // Group atoms by residue_idx.
    //
    // BTreeMap (not HashMap): this map is iterated below in Phase A and
    // Phase D, and HashMap iteration order is non-deterministic across
    // instances (RandomState reseeded per-map in std). That would push
    // bonds into `bonds` in a different order on every build_topology
    // call, which makes the serial bond_stretch sum FP-reassociate
    // differently per call — visible as 1 ULP drift in
    // TestDeterminism.test_two_calls_identical under --release. A
    // BTreeMap sorts by key, so iteration is deterministic by type and
    // future edits can't accidentally re-introduce the bug.
    let mut residue_atoms: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
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
                    res_atom_index.get(&(res_idx, a1_name.to_string())),
                    res_atom_index.get(&(res_idx, a2_name.to_string())),
                ) {
                    let pair = (i.min(j), i.max(j));
                    if bonded_set.insert(pair) {
                        bonds.push(Bond {
                            i: pair.0,
                            j: pair.1,
                        });
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
            res_atom_index.get(&(res_idx, "C".to_string())),
            res_atom_index.get(&(res_idx + 1, "N".to_string())),
        ) {
            let dx = atoms[c_idx].pos[0] - atoms[n_idx].pos[0];
            let dy = atoms[c_idx].pos[1] - atoms[n_idx].pos[1];
            let dz = atoms[c_idx].pos[2] - atoms[n_idx].pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1.8 && dist > 0.5 {
                let pair = (c_idx.min(n_idx), c_idx.max(n_idx));
                if bonded_set.insert(pair) {
                    bonds.push(Bond {
                        i: pair.0,
                        j: pair.1,
                    });
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
            if let Some(&p_idx) = res_atom_index.get(&(res_idx, parent_name.to_string())) {
                let pair = (i.min(p_idx), i.max(p_idx));
                if bonded_set.insert(pair) {
                    bonds.push(Bond {
                        i: pair.0,
                        j: pair.1,
                    });
                    neighbors[pair.0].push(pair.1);
                    neighbors[pair.1].push(pair.0);
                }
            }
        }
    }

    // Phase C: Disulfide bonds (SG-SG within 2.5 Å)
    let sg_atoms: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            a.atom_name == "SG" && (a.residue_name == "CYS" || a.residue_name == "CYX")
        })
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
                    bonds.push(Bond {
                        i: pair.0,
                        j: pair.1,
                    });
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
                        bonds.push(Bond {
                            i: pair.0,
                            j: pair.1,
                        });
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

    // Build torsions: for each bond j-k, find i bonded to j and l bonded to k.
    //
    // CHARMM filter: AmberParams's `is_canonical_torsion` default returns
    // `true`, preserving AMBER's "every bonded 4-atom path is a torsion"
    // semantics. CharmmParams overrides to consult `[ResidueTorsions]` —
    // the 4-tuple is matched against the per-residue template, with
    // cross-residue atom names prefixed `-`/`+` based on the offset from
    // the anchor (= residue containing atom J, BALL's atom2 convention,
    // see charmmTorsion.C:188). 1-4 pair list inclusion is gated by the
    // same canonical check, so non-canonical paths (which proteon
    // previously over-counted on CHARMM, contributing the observed 2.66×
    // factor on crambin) also stop adding LJ/Coulomb 1-4 contributions.
    let residue_prefix = |atom_res_idx: usize, anchor_res_idx: usize| -> Option<&'static str> {
        if atom_res_idx == anchor_res_idx {
            Some("")
        } else if atom_res_idx + 1 == anchor_res_idx {
            Some("-")
        } else if atom_res_idx == anchor_res_idx + 1 {
            Some("+")
        } else {
            None
        }
    };
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
                // 1-4 pair list is a TOPOLOGICAL fact (atoms separated
                // by exactly 3 bonds), independent of whether this
                // particular 4-atom path is a canonical FF torsion.
                // Insert unconditionally so vdw / electrostatic 1-4
                // scaling applies regardless of whether the torsion
                // contributes to the proper-torsion energy.
                let pair = (i.min(l), i.max(l));
                if !excluded_pairs.contains(&pair) {
                    pairs_14.insert(pair);
                }
                // Torsion enumeration is filtered by the FF's
                // canonical-torsion predicate. AMBER's default returns
                // true (every 4-atom path is a torsion). CHARMM's
                // override consults [ResidueTorsions] with anchor =
                // residue containing atom J and `-`/`+` prefixes for
                // cross-residue atoms.
                let anchor_res_idx = atoms[j].residue_idx;
                let anchor_res_name = &atoms[j].residue_name;
                // Compute cross-residue prefixes when possible. For
                // 4-atom paths whose residue spread is greater than
                // ±1 (e.g. disulfide-bridge torsions on crambin's
                // CYS pairs), the prefix is undefined — pass the
                // raw atom names through. AMBER's default
                // `is_canonical_torsion` returns true regardless, so
                // those torsions are kept (correct). CHARMM's
                // override won't match the unprefixed names against
                // its `[ResidueTorsions]` template and will return
                // false (acceptable for v0 — disulfide-bridge
                // torsions in CHARMM use `=` prefixes which proteon
                // does not yet emit).
                let pi = residue_prefix(atoms[i].residue_idx, anchor_res_idx).unwrap_or("");
                let pk = residue_prefix(atoms[k].residue_idx, anchor_res_idx).unwrap_or("");
                let pl = residue_prefix(atoms[l].residue_idx, anchor_res_idx).unwrap_or("");
                let name_a = format!("{}{}", pi, atoms[i].atom_name);
                let name_c = format!("{}{}", pk, atoms[k].atom_name);
                let name_d = format!("{}{}", pl, atoms[l].atom_name);
                let canonical = params.is_canonical_torsion(
                    anchor_res_name,
                    &name_a,
                    &atoms[j].atom_name,
                    &name_c,
                    &name_d,
                );
                if !canonical {
                    continue;
                }
                torsions.push(Torsion { i, j, k, l });
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
        for &(i, k, l) in &[
            (n0, n1, n2),
            (n0, n2, n1),
            (n1, n0, n2),
            (n1, n2, n0),
            (n2, n0, n1),
            (n2, n1, n0),
        ] {
            let ti = &atoms[i].amber_type;
            let tk = &atoms[j].amber_type; // central atom
            let tj = &atoms[k].amber_type;
            let tl = &atoms[l].amber_type;
            // Cosine impropers (AMBER) and harmonic impropers (CHARMM)
            // use DIFFERENT atom-slot conventions in their parameter
            // tables:
            //   - AMBER cosine table keys central at slot 3 (matches
            //     proteon's stored Torsion.k position).
            //   - CHARMM harmonic table keys central at slot 1 — A is
            //     central, B/C/D are the 3 neighbors.
            // Pass each FF the convention it expects.
            let has_cosine = params.get_improper_torsion(ti, tj, tk, tl).is_some();
            let has_harmonic = params.get_harmonic_improper(tk, ti, tj, tl).is_some();
            if has_cosine || has_harmonic {
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
        let sg: Vec<(usize, &str, usize)> = topo
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.atom_name == "SG")
            .map(|(i, a)| (i, a.residue_name.as_str(), a.residue_idx))
            .collect();
        println!("SG atoms: {:?}", sg);

        // Check which SG-SG pairs are bonded
        for i in 0..sg.len() {
            for j in (i + 1)..sg.len() {
                let (ai, _, _) = sg[i];
                let (aj, _, _) = sg[j];
                let pair = (ai.min(aj), ai.max(aj));
                let is_bonded = topo.excluded_pairs.contains(&pair)
                    || topo.bonds.iter().any(|b| b.i == pair.0 && b.j == pair.1);
                let dx = topo.atoms[ai].pos[0] - topo.atoms[aj].pos[0];
                let dy = topo.atoms[ai].pos[1] - topo.atoms[aj].pos[1];
                let dz = topo.atoms[ai].pos[2] - topo.atoms[aj].pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                println!(
                    "SG[{}]-SG[{}] dist={:.3} bonded={} excluded={}",
                    ai,
                    aj,
                    dist,
                    is_bonded,
                    topo.excluded_pairs.contains(&pair)
                );
            }
        }

        // Crambin should have 3 disulfides
        let ss_bonds: Vec<_> = topo
            .bonds
            .iter()
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
        assert!(
            topo.bonds.len() > 300,
            "Should have > 300 bonds, got {}",
            topo.bonds.len()
        );
        assert!(
            topo.bonds.len() < 400,
            "Should have < 400 bonds, got {}",
            topo.bonds.len()
        );

        // Debug: print all improper torsion centers
        println!("Improper torsions ({}):", topo.improper_torsions.len());
        for imp in &topo.improper_torsions {
            let a = &topo.atoms[imp.k]; // center atom (position 3)
            let ti = &topo.atoms[imp.i].amber_type;
            let tj = &topo.atoms[imp.j].amber_type;
            let tk = &a.amber_type;
            let tl = &topo.atoms[imp.l].amber_type;
            println!(
                "  center={}:{} types={}-{}-{}-{}",
                a.residue_name, a.atom_name, ti, tj, tk, tl
            );
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
            for j in (i + 1)..n {
                let pair = (i, j);
                if topo.excluded_pairs.contains(&pair) {
                    continue;
                }

                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if !(0.01..=225.0).contains(&r2) {
                    continue;
                } // 15 Å cutoff

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
            println!(
                "  E={:+10.1} i={} {}:{} j={} {}:{} r={:.3} 1-4={}",
                e,
                i,
                topo.atoms[*i].residue_name,
                topo.atoms[*i].atom_name,
                j,
                topo.atoms[*j].residue_name,
                topo.atoms[*j].atom_name,
                r,
                is14
            );
        }
    }
}
