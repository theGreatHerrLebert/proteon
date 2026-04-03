//! Edge-case tests for Arrow round-trip correctness.
//!
//! Covers: insertion codes, alternate conformers, multi-model structures,
//! schema mismatch, empty structures, and single-atom edge cases.

use ferritin_arrow::atom::AtomBatchBuilder;
use ferritin_arrow::convert::{atom_batch_to_pdbs, pdb_to_atom_batch, pdb_to_structure_batch};

/// Build a minimal PDB with insertion codes (residue 1 and 1A).
fn pdb_with_insertion_codes() -> pdbtbx::PDB {
    let mut pdb = pdbtbx::PDB::new();
    let mut model = pdbtbx::Model::new(0);
    let mut chain = pdbtbx::Chain::new("A").unwrap();

    // Residue 1 (no insertion code)
    let mut res1 = pdbtbx::Residue::new(1, None, None).unwrap();
    let mut conf1 = pdbtbx::Conformer::new("ALA", None, None).unwrap();
    conf1.add_atom(
        pdbtbx::Atom::new(false, 1, "", "CA", 1.0, 2.0, 3.0, 1.0, 10.0, "C", 0).unwrap(),
    );
    conf1.add_atom(
        pdbtbx::Atom::new(false, 2, "", "N", 1.5, 2.5, 3.5, 1.0, 12.0, "N", 0).unwrap(),
    );
    res1.add_conformer(conf1);
    chain.add_residue(res1);

    // Residue 1A (insertion code "A")
    let mut res1a = pdbtbx::Residue::new(1, Some("A"), None).unwrap();
    let mut conf1a = pdbtbx::Conformer::new("GLY", None, None).unwrap();
    conf1a.add_atom(
        pdbtbx::Atom::new(false, 3, "", "CA", 4.0, 5.0, 6.0, 1.0, 11.0, "C", 0).unwrap(),
    );
    conf1a.add_atom(
        pdbtbx::Atom::new(false, 4, "", "N", 4.5, 5.5, 6.5, 1.0, 13.0, "N", 0).unwrap(),
    );
    res1a.add_conformer(conf1a);
    chain.add_residue(res1a);

    model.add_chain(chain);
    pdb.add_model(model);
    pdb
}

#[test]
fn test_insertion_codes_preserved() {
    let pdb = pdb_with_insertion_codes();

    // Should have 2 residues, 4 atoms
    let n_residues: usize = pdb
        .models()
        .next()
        .unwrap()
        .chains()
        .next()
        .unwrap()
        .residue_count();
    assert_eq!(n_residues, 2, "original should have 2 residues");

    // Round-trip
    let batch = pdb_to_atom_batch(&pdb, "test").unwrap();
    assert_eq!(batch.num_rows(), 4);

    let rebuilt = atom_batch_to_pdbs(&batch).unwrap();
    assert_eq!(rebuilt.len(), 1);
    let (_, rebuilt_pdb) = &rebuilt[0];

    let rebuilt_residues: usize = rebuilt_pdb
        .models()
        .next()
        .unwrap()
        .chains()
        .next()
        .unwrap()
        .residue_count();
    assert_eq!(
        rebuilt_residues, 2,
        "round-trip must preserve 2 distinct residues (1 and 1A)"
    );

    // Check insertion code is preserved
    let residues: Vec<_> = rebuilt_pdb
        .models()
        .next()
        .unwrap()
        .chains()
        .next()
        .unwrap()
        .residues()
        .collect();
    assert_eq!(residues[0].insertion_code(), None);
    assert_eq!(residues[1].insertion_code(), Some("A"));

    // Check residue names
    assert_eq!(
        residues[0].conformers().next().unwrap().name(),
        "ALA"
    );
    assert_eq!(
        residues[1].conformers().next().unwrap().name(),
        "GLY"
    );
}

/// Build a PDB with alternate conformers (A and B for same residue).
fn pdb_with_alt_conformers() -> pdbtbx::PDB {
    let mut pdb = pdbtbx::PDB::new();
    let mut model = pdbtbx::Model::new(0);
    let mut chain = pdbtbx::Chain::new("A").unwrap();

    let mut res = pdbtbx::Residue::new(1, None, None).unwrap();

    // Conformer A
    let mut conf_a = pdbtbx::Conformer::new("ALA", Some("A"), None).unwrap();
    conf_a.add_atom(
        pdbtbx::Atom::new(false, 1, "", "CA", 1.0, 2.0, 3.0, 0.6, 10.0, "C", 0).unwrap(),
    );
    res.add_conformer(conf_a);

    // Conformer B
    let mut conf_b = pdbtbx::Conformer::new("ALA", Some("B"), None).unwrap();
    conf_b.add_atom(
        pdbtbx::Atom::new(false, 2, "", "CA", 1.1, 2.1, 3.1, 0.4, 11.0, "C", 0).unwrap(),
    );
    res.add_conformer(conf_b);

    chain.add_residue(res);
    model.add_chain(chain);
    pdb.add_model(model);
    pdb
}

#[test]
fn test_alt_conformers_preserved() {
    let pdb = pdb_with_alt_conformers();

    let batch = pdb_to_atom_batch(&pdb, "test").unwrap();
    assert_eq!(batch.num_rows(), 2, "should have 2 atoms (one per conformer)");

    let rebuilt = atom_batch_to_pdbs(&batch).unwrap();
    let (_, rebuilt_pdb) = &rebuilt[0];

    let res: Vec<_> = rebuilt_pdb
        .models()
        .next()
        .unwrap()
        .chains()
        .next()
        .unwrap()
        .residues()
        .collect();
    assert_eq!(res.len(), 1, "should be 1 residue");

    let conformers: Vec<_> = res[0].conformers().collect();
    assert_eq!(
        conformers.len(),
        2,
        "should have 2 conformers (A and B)"
    );
}

/// Build a multi-model structure (like NMR ensemble).
fn pdb_multi_model() -> pdbtbx::PDB {
    let mut pdb = pdbtbx::PDB::new();
    for model_serial in 0..5 {
        let mut model = pdbtbx::Model::new(model_serial);
        let mut chain = pdbtbx::Chain::new("A").unwrap();
        let mut res = pdbtbx::Residue::new(1, None, None).unwrap();
        let mut conf = pdbtbx::Conformer::new("ALA", None, None).unwrap();
        conf.add_atom(
            pdbtbx::Atom::new(
                false,
                1,
                "",
                "CA",
                model_serial as f64 * 1.0,
                0.0,
                0.0,
                1.0,
                10.0,
                "C",
                0,
            )
            .unwrap(),
        );
        res.add_conformer(conf);
        chain.add_residue(res);
        model.add_chain(chain);
        pdb.add_model(model);
    }
    pdb
}

#[test]
fn test_multi_model_serials_preserved() {
    let pdb = pdb_multi_model();
    assert_eq!(pdb.model_count(), 5);

    let batch = pdb_to_atom_batch(&pdb, "test").unwrap();
    assert_eq!(batch.num_rows(), 5);

    let rebuilt = atom_batch_to_pdbs(&batch).unwrap();
    let (_, rebuilt_pdb) = &rebuilt[0];
    assert_eq!(rebuilt_pdb.model_count(), 5);

    // Check model serials are 0,1,2,3,4 (not all 0)
    let serials: Vec<usize> = rebuilt_pdb.models().map(|m| m.serial_number()).collect();
    assert_eq!(serials, vec![0, 1, 2, 3, 4], "model serials must be preserved");

    // Check coordinates differ per model (each model has x = model_serial)
    for (i, model) in rebuilt_pdb.models().enumerate() {
        let atom = model
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
        let (x, _, _) = atom.pos();
        assert!(
            (x - i as f64).abs() < 1e-10,
            "model {i}: expected x={i}, got x={x}"
        );
    }
}

#[test]
fn test_structure_schema_rejected_by_from_arrow() {
    let pdb = pdb_with_insertion_codes();
    let structure_batch = pdb_to_structure_batch(&pdb, "test").unwrap();

    let result = atom_batch_to_pdbs(&structure_batch);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Missing column"),
        "should explain what's wrong: {msg}"
    );
}

#[test]
fn test_empty_batch() {
    let builder = AtomBatchBuilder::new(0);
    let batch = builder.finish().unwrap();
    let result = atom_batch_to_pdbs(&batch).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_hetero_atoms_preserved() {
    let mut pdb = pdbtbx::PDB::new();
    let mut model = pdbtbx::Model::new(0);
    let mut chain = pdbtbx::Chain::new("A").unwrap();
    let mut res = pdbtbx::Residue::new(1, None, None).unwrap();
    let mut conf = pdbtbx::Conformer::new("HOH", None, None).unwrap();
    conf.add_atom(
        pdbtbx::Atom::new(true, 1, "", "O", 1.0, 2.0, 3.0, 1.0, 20.0, "O", 0).unwrap(),
    );
    res.add_conformer(conf);
    chain.add_residue(res);
    model.add_chain(chain);
    pdb.add_model(model);

    let batch = pdb_to_atom_batch(&pdb, "test").unwrap();
    let rebuilt = atom_batch_to_pdbs(&batch).unwrap();
    let (_, rebuilt_pdb) = &rebuilt[0];

    let atom = rebuilt_pdb
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
    assert!(atom.hetero(), "HETATM flag must be preserved");
}
