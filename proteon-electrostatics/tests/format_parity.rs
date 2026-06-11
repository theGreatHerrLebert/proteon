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
    format!("{}/tests/fixtures/format/{name}", env!("CARGO_MANIFEST_DIR"))
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
    assert!(approx(v0.x, 0.0) && approx(v0.y, 0.0) && approx(v0.z, 1.0049999952), "v0 = {v0:?}");
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
    assert!(approx(c.pos.x, 0.0) && approx(c.pos.y, 0.0) && approx(c.pos.z, 0.0), "pos = {:?}", c.pos);
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
