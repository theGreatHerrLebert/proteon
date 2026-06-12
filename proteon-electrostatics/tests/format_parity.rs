//! Mesh / charge file-format parity vs NESSie's readers + round-trip gates.
//!
//! OFF and PQR are checked against fixtures copied from NESSie's `data/born/`
//! with reference counts/values taken from NESSie's own loaders. HMO and MSMS
//! are gated by synthetic round-trips (write a known mesh in the format's exact
//! layout, read it back, assert the topology survives the 1-based↔0-based and
//! header-skipping conventions).

use std::io::Write;

use proteon_electrostatics::{read_hmo, read_msms, read_off, read_pqr, write_off};

fn fixture(name: &str) -> String {
    format!(
        "{}/tests/fixtures/format/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-7
}

#[test]
fn read_off_matches_nessie() {
    let mesh = read_off(fixture("na.off")).expect("read na.off");
    // NESSie reports nodes=258 elems=512 for this surface.
    assert_eq!(mesh.verts.len(), 258, "node count");
    assert_eq!(mesh.tris.len(), 512, "element count");
    // First vertex per NESSie: [0.0, 0.0, 1.0049999952].
    let v0 = mesh.verts[0];
    assert!(
        approx(v0.x, 0.0) && approx(v0.y, 0.0) && approx(v0.z, 1.0049999952),
        "v0 = {v0:?}"
    );
    // Indices must be in range and 0-based (NESSie's OFF faces are 0-based).
    for t in &mesh.tris {
        for &idx in t {
            assert!((idx as usize) < mesh.verts.len());
        }
    }
}

#[test]
fn read_pqr_matches_nessie() {
    let charges = read_pqr(fixture("na.pqr")).expect("read na.pqr");
    assert_eq!(charges.len(), 1, "one non-zero charge");
    let c = &charges[0];
    assert!(
        approx(c.pos.x, 0.0) && approx(c.pos.y, 0.0) && approx(c.pos.z, 0.0),
        "pos = {:?}",
        c.pos
    );
    assert!(approx(c.val, 1.0), "val = {}", c.val);
}

#[test]
fn off_round_trips() {
    let mesh = read_off(fixture("na.off")).expect("read na.off");
    let tmp = std::env::temp_dir().join("proteon_off_roundtrip.off");
    write_off(&mesh, &tmp).expect("write off");
    let back = read_off(&tmp).expect("re-read off");
    assert_eq!(back.verts.len(), mesh.verts.len());
    assert_eq!(back.tris.len(), mesh.tris.len());
    for (a, b) in mesh.verts.iter().zip(&back.verts) {
        assert!(approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z));
    }
    assert_eq!(mesh.tris, back.tris);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn read_hmo_round_trip() {
    // A single triangle + one charge, in HMO's exact block layout (1-based ids,
    // one count/comment line after each BEG marker).
    let tmp = std::env::temp_dir().join("proteon_hmo_test.hmo");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "BEG_NODL_DATA").unwrap();
        writeln!(f, "3").unwrap();
        writeln!(f, "1 0.0 0.0 0.0").unwrap();
        writeln!(f, "2 1.0 0.0 0.0").unwrap();
        writeln!(f, "3 0.0 1.0 0.0").unwrap();
        writeln!(f, "END_NODL_DATA").unwrap();
        writeln!(f, "BEG_ELEM_DATA").unwrap();
        writeln!(f, "1").unwrap();
        writeln!(f, "1 0 0 1 2 3").unwrap();
        writeln!(f, "END_ELEM_DATA").unwrap();
        writeln!(f, "BEG_CHARGE_DATA").unwrap();
        writeln!(f, "1").unwrap();
        writeln!(f, "1 0.3 0.3 0.0 -0.5").unwrap();
        writeln!(f, "END_CHARGE_DATA").unwrap();
    }
    let (mesh, charges) = read_hmo(&tmp).expect("read hmo");
    assert_eq!(mesh.verts.len(), 3);
    assert_eq!(mesh.tris.len(), 1);
    assert_eq!(mesh.tris[0], [0, 1, 2], "1-based ids normalised to 0-based");
    assert_eq!(charges.len(), 1);
    assert!(approx(charges[0].val, -0.5));
    assert!(approx(charges[0].pos.x, 0.3));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn read_pqr_excludes_hetatm_like_nessie() {
    // NESSie's readpqr takes only lines beginning with "ATOM"; a charged HETATM
    // water record must be dropped. Mirror that exactly.
    let tmp = std::env::temp_dir().join("proteon_pqr_hetatm.pqr");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(
            f,
            "ATOM      1  N   THR     1   -17.108  25.866  23.850  0.1812 1.8240"
        )
        .unwrap();
        writeln!(
            f,
            "HETATM    2  O   HOH     2   -21.160  40.444  40.509 -0.8340 1.6612"
        )
        .unwrap();
        writeln!(
            f,
            "ATOM      3  CA  THR     3   -16.775  27.193  23.310  0.0034 1.9080"
        )
        .unwrap();
    }
    let charges = read_pqr(&tmp).expect("read pqr");
    assert_eq!(charges.len(), 2, "HETATM water excluded");
    assert!(approx(charges[0].val, 0.1812));
    assert!(approx(charges[1].val, 0.0034));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn read_hmo_keeps_zero_charges() {
    // Unlike PQR, NESSie's HMO loader constructs every listed charge, zero or not.
    let tmp = std::env::temp_dir().join("proteon_hmo_zero.hmo");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(
            f,
            "BEG_NODL_DATA\n3\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\nEND_NODL_DATA"
        )
        .unwrap();
        writeln!(f, "BEG_ELEM_DATA\n1\n1 0 0 1 2 3\nEND_ELEM_DATA").unwrap();
        writeln!(
            f,
            "BEG_CHARGE_DATA\n2\n1 0.1 0.1 0.0 0.0\n2 0.2 0.2 0.0 -0.5\nEND_CHARGE_DATA"
        )
        .unwrap();
    }
    let (_, charges) = read_hmo(&tmp).expect("read hmo");
    assert_eq!(charges.len(), 2, "zero charge retained");
    assert!(approx(charges[0].val, 0.0));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn malformed_rows_error_not_skip() {
    // A skipped node row would shift every subsequent 1-based element index;
    // an undersized non-blank row must be a hard error instead.
    let tmp = std::env::temp_dir().join("proteon_hmo_bad.hmo");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "BEG_NODL_DATA\n2\n1 0.0 0.0 0.0\n2 1.0\nEND_NODL_DATA").unwrap();
        writeln!(f, "BEG_ELEM_DATA\n0\nEND_ELEM_DATA").unwrap();
    }
    assert!(read_hmo(&tmp).is_err(), "undersized node row must error");
    let _ = std::fs::remove_file(&tmp);

    // HMO 1-based element id out of range must error (not wrap on the -1).
    let tmp2 = std::env::temp_dir().join("proteon_hmo_oor.hmo");
    {
        let mut f = std::fs::File::create(&tmp2).unwrap();
        writeln!(f, "BEG_NODL_DATA\n1\n1 0.0 0.0 0.0\nEND_NODL_DATA").unwrap();
        writeln!(f, "BEG_ELEM_DATA\n1\n1 0 0 1 2 3\nEND_ELEM_DATA").unwrap();
    }
    assert!(
        read_hmo(&tmp2).is_err(),
        "out-of-range element id must error"
    );
    let _ = std::fs::remove_file(&tmp2);
}

#[test]
fn read_msms_round_trip() {
    // .vert / .face pair, three header lines each, 1-based face indices.
    let vert = std::env::temp_dir().join("proteon_msms_test.vert");
    let face = std::env::temp_dir().join("proteon_msms_test.face");
    {
        let mut f = std::fs::File::create(&vert).unwrap();
        writeln!(f, "# header 1").unwrap();
        writeln!(f, "# header 2").unwrap();
        writeln!(f, "4 0 1.5 1.0").unwrap();
        writeln!(f, "0.0 0.0 0.0 0.0 0.0 1.0 1 1 2").unwrap();
        writeln!(f, "1.0 0.0 0.0 0.0 0.0 1.0 1 1 2").unwrap();
        writeln!(f, "0.0 1.0 0.0 0.0 0.0 1.0 1 1 2").unwrap();
        writeln!(f, "1.0 1.0 0.0 0.0 0.0 1.0 1 1 2").unwrap();
    }
    {
        let mut f = std::fs::File::create(&face).unwrap();
        writeln!(f, "# header 1").unwrap();
        writeln!(f, "# header 2").unwrap();
        writeln!(f, "2 0 1.5 1.0").unwrap();
        writeln!(f, "1 2 3 1 0").unwrap();
        writeln!(f, "2 4 3 1 0").unwrap();
    }
    let mesh = read_msms(&vert, &face).expect("read msms");
    assert_eq!(mesh.verts.len(), 4);
    assert_eq!(mesh.tris.len(), 2);
    assert_eq!(mesh.tris[0], [0, 1, 2]);
    assert_eq!(mesh.tris[1], [1, 3, 2]);
    let _ = std::fs::remove_file(&vert);
    let _ = std::fs::remove_file(&face);
}
