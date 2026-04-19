//! Convert between pdbtbx structures and Arrow RecordBatches.
//!
//! Two directions:
//! - `pdb_to_atom_batch` — PDB → Arrow (export)
//! - `atom_batch_to_pdb` — Arrow → PDB (import)

use std::collections::BTreeMap;

use arrow::array::{
    Array, AsArray, BooleanArray, Float64Array, Int64Array, RecordBatch, UInt32Array,
};

use crate::atom::AtomBatchBuilder;
use crate::structure::StructureBatchBuilder;

const BACKBONE_ATOMS: &[&str] = &["N", "CA", "C", "O"];

/// Convert a pdbtbx PDB structure into a per-atom Arrow RecordBatch.
///
/// Iterates over all models, chains, residues, and atoms, emitting
/// one row per atom with full metadata.
pub fn pdb_to_atom_batch(pdb: &pdbtbx::PDB, structure_id: &str) -> anyhow::Result<RecordBatch> {
    let n_atoms = pdb.atom_count();
    let mut builder = AtomBatchBuilder::new(n_atoms);

    for (model_idx, model) in pdb.models().enumerate() {
        for chain in model.chains() {
            let chain_id = chain.id();
            for residue in chain.residues() {
                let res_name = residue.name().unwrap_or("UNK");
                let res_serial = residue.serial_number() as i64;

                let insertion_code = residue.insertion_code();

                for conformer in residue.conformers() {
                    let conf_id = conformer.alternative_location();
                    for atom in conformer.atoms() {
                        let atom_name = atom.name();
                        let element = atom.element().map(|e| e.symbol());
                        let pos = atom.pos();
                        let is_backbone = BACKBONE_ATOMS.contains(&atom_name);

                        builder.append(
                            structure_id,
                            model_idx as u32,
                            chain_id,
                            res_name,
                            res_serial,
                            insertion_code,
                            conf_id,
                            atom_name,
                            atom.serial_number() as i64,
                            element,
                            pos.0,
                            pos.1,
                            pos.2,
                            atom.b_factor(),
                            atom.occupancy(),
                            atom.hetero(),
                            is_backbone,
                        );
                    }
                }
            }
        }
    }

    builder.finish()
}

/// Convert a pdbtbx PDB structure into a per-structure summary Arrow RecordBatch.
pub fn pdb_to_structure_batch(
    pdb: &pdbtbx::PDB,
    structure_id: &str,
) -> anyhow::Result<RecordBatch> {
    let mut builder = StructureBatchBuilder::new(1);

    let model = pdb.models().next();
    let (chain_count, chains_str) = match model {
        Some(m) => {
            let chains: Vec<&str> = m.chains().map(|c| c.id()).collect();
            (chains.len() as u32, chains.join(","))
        }
        None => (0, String::new()),
    };

    builder.append(
        structure_id,
        pdb.atom_count() as i64,
        pdb.residue_count() as i64,
        chain_count,
        pdb.model_count() as u32,
        &chains_str,
    );

    builder.finish()
}

/// Convert multiple PDB structures into a single atom RecordBatch.
///
/// Each structure is identified by its `structure_id` in the batch.
pub fn pdbs_to_atom_batch(pdbs: &[(&pdbtbx::PDB, &str)]) -> anyhow::Result<RecordBatch> {
    let total_atoms: usize = pdbs.iter().map(|(pdb, _)| pdb.atom_count()).sum();
    let mut builder = AtomBatchBuilder::new(total_atoms);

    for (pdb, structure_id) in pdbs {
        for (model_idx, model) in pdb.models().enumerate() {
            for chain in model.chains() {
                let chain_id = chain.id();
                for residue in chain.residues() {
                    let res_name = residue.name().unwrap_or("UNK");
                    let res_serial = residue.serial_number() as i64;
                    let insertion_code = residue.insertion_code();

                    for conformer in residue.conformers() {
                        let conf_id = conformer.alternative_location();
                        for atom in conformer.atoms() {
                            let atom_name = atom.name();
                            let element = atom.element().map(|e| e.symbol());
                            let pos = atom.pos();
                            let is_backbone = BACKBONE_ATOMS.contains(&atom_name);

                            builder.append(
                                structure_id,
                                model_idx as u32,
                                chain_id,
                                res_name,
                                res_serial,
                                insertion_code,
                                conf_id,
                                atom_name,
                                atom.serial_number() as i64,
                                element,
                                pos.0,
                                pos.1,
                                pos.2,
                                atom.b_factor(),
                                atom.occupancy(),
                                atom.hetero(),
                                is_backbone,
                            );
                        }
                    }
                }
            }
        }
    }

    builder.finish()
}

// ============================================================================
// Arrow → PDB (from_arrow)
// ============================================================================

/// Convert an atom-schema Arrow RecordBatch back into pdbtbx PDB structures.
///
/// Groups rows by `structure_id`, then by model/chain/residue/conformer
/// to rebuild the full hierarchy including insertion codes and alt conformers.
///
/// Returns an error if the batch does not have the expected atom schema.
pub fn atom_batch_to_pdbs(batch: &RecordBatch) -> anyhow::Result<Vec<(String, pdbtbx::PDB)>> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(vec![]);
    }

    // Schema validation: check required columns exist by name
    let schema = batch.schema();
    let get_col = |name: &str| -> anyhow::Result<usize> {
        schema.index_of(name).map_err(|_| {
            anyhow::anyhow!(
                "Missing column '{}' — expected atom schema (got columns: {:?})",
                name,
                schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect::<Vec<_>>()
            )
        })
    };

    let idx_sid = get_col("structure_id")?;
    let idx_model = get_col("model")?;
    let idx_chain = get_col("chain_id")?;
    let idx_resname = get_col("residue_name")?;
    let idx_resser = get_col("residue_serial")?;
    let idx_icode = schema.index_of("insertion_code").ok();
    let idx_confid = schema.index_of("conformer_id").ok();
    let idx_atomname = get_col("atom_name")?;
    let idx_atomser = get_col("atom_serial")?;
    let idx_elem = get_col("element")?;
    let idx_x = get_col("x")?;
    let idx_y = get_col("y")?;
    let idx_z = get_col("z")?;
    let idx_bfac = get_col("b_factor")?;
    let idx_occ = get_col("occupancy")?;
    let idx_het = get_col("is_hetero")?;

    let structure_ids = batch.column(idx_sid).as_string::<i32>();
    let models = batch
        .column(idx_model)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let chain_ids = batch.column(idx_chain).as_string::<i32>();
    let residue_names = batch.column(idx_resname).as_string::<i32>();
    let residue_serials = batch
        .column(idx_resser)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let insertion_codes = idx_icode.map(|i| batch.column(i).as_string::<i32>());
    let conformer_ids = idx_confid.map(|i| batch.column(i).as_string::<i32>());
    let atom_names = batch.column(idx_atomname).as_string::<i32>();
    let atom_serials = batch
        .column(idx_atomser)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let elements = batch.column(idx_elem).as_string::<i32>();
    let xs = batch
        .column(idx_x)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let ys = batch
        .column(idx_y)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let zs = batch
        .column(idx_z)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let b_factors = batch
        .column(idx_bfac)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let occupancies = batch
        .column(idx_occ)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let is_hetero = batch
        .column(idx_het)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    // Helpers to read optional nullable string columns
    let get_icode = |row: usize| -> Option<&str> {
        insertion_codes.and_then(|ic| {
            if ic.is_null(row) {
                None
            } else {
                Some(ic.value(row))
            }
        })
    };
    let get_confid = |row: usize| -> Option<&str> {
        conformer_ids.and_then(|cf| {
            if cf.is_null(row) {
                None
            } else {
                Some(cf.value(row))
            }
        })
    };

    // Group rows by structure_id, preserving order
    let mut structure_order: Vec<String> = Vec::new();
    let mut structure_rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let sid = structure_ids.value(i).to_string();
        structure_rows
            .entry(sid.clone())
            .or_insert_with(|| {
                structure_order.push(sid.clone());
                Vec::new()
            })
            .push(i);
    }

    let mut results = Vec::new();

    for sid in &structure_order {
        let rows = &structure_rows[sid];
        let mut pdb = pdbtbx::PDB::new();

        // Group by model
        let mut model_rows: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &row in rows {
            model_rows.entry(models.value(row)).or_default().push(row);
        }

        for (model_serial, m_rows) in &model_rows {
            let mut model = pdbtbx::Model::new(*model_serial as usize);

            // Group by chain
            let mut chain_rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            let mut chain_order: Vec<String> = Vec::new();
            for &row in m_rows {
                let cid = chain_ids.value(row).to_string();
                chain_rows
                    .entry(cid.clone())
                    .or_insert_with(|| {
                        chain_order.push(cid.clone());
                        Vec::new()
                    })
                    .push(row);
            }

            for cid in &chain_order {
                let c_rows = &chain_rows[cid];
                let mut chain = pdbtbx::Chain::new(cid).unwrap();

                // Group by residue: (serial, insertion_code) is the unique key
                type ResKey = (i64, Option<String>);
                let mut res_rows: BTreeMap<ResKey, Vec<usize>> = BTreeMap::new();
                let mut res_order: Vec<ResKey> = Vec::new();
                for &row in c_rows {
                    let key: ResKey = (
                        residue_serials.value(row),
                        get_icode(row).map(|s| s.to_string()),
                    );
                    res_rows
                        .entry(key.clone())
                        .or_insert_with(|| {
                            res_order.push(key.clone());
                            Vec::new()
                        })
                        .push(row);
                }

                for rkey in &res_order {
                    let r_rows = &res_rows[rkey];
                    let (rs, ref icode) = *rkey;

                    let mut residue =
                        pdbtbx::Residue::new(rs as isize, icode.as_deref(), None).unwrap();

                    // Group by conformer (alt-loc ID)
                    let mut conf_rows: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
                    let mut conf_order: Vec<Option<String>> = Vec::new();
                    for &row in r_rows {
                        let cfid = get_confid(row).map(|s| s.to_string());
                        conf_rows
                            .entry(cfid.clone())
                            .or_insert_with(|| {
                                conf_order.push(cfid.clone());
                                Vec::new()
                            })
                            .push(row);
                    }

                    for cfid in &conf_order {
                        let cf_rows = &conf_rows[cfid];
                        let first = cf_rows[0];
                        let res_name = residue_names.value(first);

                        let mut conformer =
                            pdbtbx::Conformer::new(res_name, cfid.as_deref(), None).unwrap();

                        for &row in cf_rows {
                            let element_str = if elements.is_null(row) {
                                "C"
                            } else {
                                elements.value(row)
                            };

                            let atom = pdbtbx::Atom::new(
                                is_hetero.value(row),
                                atom_serials.value(row) as usize,
                                "",
                                atom_names.value(row),
                                xs.value(row),
                                ys.value(row),
                                zs.value(row),
                                occupancies.value(row),
                                b_factors.value(row),
                                element_str,
                                0,
                            )
                            .unwrap();

                            conformer.add_atom(atom);
                        }

                        residue.add_conformer(conformer);
                    }

                    chain.add_residue(residue);
                }

                model.add_chain(chain);
            }

            pdb.add_model(model);
        }

        results.push((sid.clone(), pdb));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_pdb() -> Option<pdbtbx::PDB> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-pdbs/1crn.pdb");
        if !std::path::Path::new(path).exists() {
            return None;
        }
        let (pdb, _) = pdbtbx::ReadOptions::new()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(path)
            .ok()?;
        Some(pdb)
    }

    #[test]
    fn test_pdb_to_atom_batch() {
        let pdb = match load_test_pdb() {
            Some(p) => p,
            None => return,
        };

        let batch = pdb_to_atom_batch(&pdb, "1crn").unwrap();
        assert!(batch.num_rows() > 300, "crambin should have >300 atoms");
        assert_eq!(batch.num_columns(), 17);

        let ids = batch.column(0).as_string::<i32>();
        assert_eq!(ids.value(0), "1crn");
    }

    #[test]
    fn test_pdb_to_structure_batch() {
        let pdb = match load_test_pdb() {
            Some(p) => p,
            None => return,
        };

        let batch = pdb_to_structure_batch(&pdb, "1crn").unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_roundtrip() {
        let pdb = match load_test_pdb() {
            Some(p) => p,
            None => return,
        };

        let original_atoms = pdb.atom_count();

        // PDB → Arrow
        let batch = pdb_to_atom_batch(&pdb, "1crn").unwrap();
        assert_eq!(batch.num_rows(), original_atoms);

        // Arrow → PDB
        let rebuilt = atom_batch_to_pdbs(&batch).unwrap();
        assert_eq!(rebuilt.len(), 1);

        let (sid, rebuilt_pdb) = &rebuilt[0];
        assert_eq!(sid, "1crn");
        assert_eq!(rebuilt_pdb.atom_count(), original_atoms);

        // Spot-check: first atom coordinates should match
        let orig_atom = pdb
            .models()
            .next()
            .unwrap()
            .chains()
            .next()
            .unwrap()
            .residues()
            .next()
            .unwrap()
            .conformers()
            .next()
            .unwrap()
            .atoms()
            .next()
            .unwrap();
        let rebuilt_atom = rebuilt_pdb
            .models()
            .next()
            .unwrap()
            .chains()
            .next()
            .unwrap()
            .residues()
            .next()
            .unwrap()
            .conformers()
            .next()
            .unwrap()
            .atoms()
            .next()
            .unwrap();

        let (ox, oy, oz) = orig_atom.pos();
        let (rx, ry, rz) = rebuilt_atom.pos();
        assert!((ox - rx).abs() < 1e-10, "x mismatch: {ox} vs {rx}");
        assert!((oy - ry).abs() < 1e-10, "y mismatch: {oy} vs {ry}");
        assert!((oz - rz).abs() < 1e-10, "z mismatch: {oz} vs {rz}");
    }

    #[test]
    fn test_rejects_wrong_schema() {
        let pdb = match load_test_pdb() {
            Some(p) => p,
            None => return,
        };

        // Structure schema batch should be rejected by atom_batch_to_pdbs
        let structure_batch = pdb_to_structure_batch(&pdb, "1crn").unwrap();
        let result = atom_batch_to_pdbs(&structure_batch);
        assert!(result.is_err(), "should reject structure schema");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Missing column"),
            "error should mention missing column: {err_msg}"
        );
    }
}
