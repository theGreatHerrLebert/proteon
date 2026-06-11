//! Mesh / charge file-format readers and writers (NESSie `format/` layer port).
//!
//! NESSie ships a small I/O layer for the surface meshes and charge sets that
//! drive BEM: OFF (Geomview), PQR (atomic charges + radii), HMO (BEM mesh +
//! charges in one file), and MSMS (`.vert` / `.face` pair). proteon's BEM core
//! is I/O-free by design (`Tri`/`Charge` in-memory), so this module is the thin
//! bridge that turns those on-disk formats into [`Mesh`](proteon_core::surface::mesh::Mesh)
//! + [`Charge`] and back.
//!
//! Index convention: OFF and MSMS-internal face indices are 0-based and 1-based
//! respectively in the *files*; NESSie's HMO/MSMS use 1-based node ids. proteon's
//! [`Mesh::tris`](proteon_core::surface::mesh::Mesh) are always 0-based `u32`, so
//! every reader normalises to 0-based on the way in and OFF's writer restores the
//! file convention on the way out.
//!
//! Gated against NESSie's own readers via committed fixtures in
//! `tests/fixtures/format/` (see `tests/format_parity.rs`).

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;

use crate::model::Charge;

fn parse_err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn f64_field(s: &str) -> io::Result<f64> {
    s.parse::<f64>()
        .map_err(|_| parse_err(format!("expected float, got {s:?}")))
}

fn usize_field(s: &str) -> io::Result<usize> {
    s.parse::<usize>()
        .map_err(|_| parse_err(format!("expected integer, got {s:?}")))
}

/// Read a Geomview **OFF** surface mesh.
///
/// Layout: line 1 = `OFF`; line 2 = `n_nodes n_elem [n_edges]`; then `n_nodes`
/// vertex lines `x y z`; then `n_elem` face lines `count i j k …` where `count`
/// is the polygon valence (3 for triangles) followed by **0-based** vertex
/// indices. Only triangles are supported (NESSie's surface meshes are triangular).
/// Blank lines and `#` comments are skipped.
pub fn read_off(path: impl AsRef<Path>) -> io::Result<Mesh> {
    let reader = BufReader::new(File::open(path)?);
    let mut lines = reader
        .lines()
        .map(|l| l.map(|s| s.trim().to_string()))
        .filter(|l| match l {
            Ok(s) => !s.is_empty() && !s.starts_with('#'),
            Err(_) => true,
        });

    let header = lines
        .next()
        .ok_or_else(|| parse_err("empty OFF file"))??;
    if header != "OFF" {
        return Err(parse_err(format!("expected OFF magic, got {header:?}")));
    }

    let counts = lines
        .next()
        .ok_or_else(|| parse_err("missing OFF counts line"))??;
    let counts: Vec<&str> = counts.split_whitespace().collect();
    if counts.len() < 2 {
        return Err(parse_err("OFF counts line needs n_nodes n_elem"));
    }
    let n_nodes = usize_field(counts[0])?;
    let n_elem = usize_field(counts[1])?;

    let mut verts = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let line = lines
            .next()
            .ok_or_else(|| parse_err("OFF truncated in node block"))??;
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            return Err(parse_err(format!("OFF node line needs x y z: {line:?}")));
        }
        verts.push(Vec3::new(f64_field(f[0])?, f64_field(f[1])?, f64_field(f[2])?));
    }

    let mut tris = Vec::with_capacity(n_elem);
    for _ in 0..n_elem {
        let line = lines
            .next()
            .ok_or_else(|| parse_err("OFF truncated in element block"))??;
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            return Err(parse_err("empty OFF element line"));
        }
        let valence = usize_field(f[0])?;
        if valence != 3 {
            return Err(parse_err(format!(
                "OFF reader supports triangles only, got valence {valence}"
            )));
        }
        if f.len() < 4 {
            return Err(parse_err(format!("OFF triangle needs 3 indices: {line:?}")));
        }
        let i = usize_field(f[1])?;
        let j = usize_field(f[2])?;
        let k = usize_field(f[3])?;
        for &idx in &[i, j, k] {
            if idx >= n_nodes {
                return Err(parse_err(format!("OFF face index {idx} out of range")));
            }
        }
        tris.push([i as u32, j as u32, k as u32]);
    }

    Ok(Mesh { verts, normals: Vec::new(), tris })
}

/// Write a [`Mesh`](proteon_core::surface::mesh::Mesh) as Geomview **OFF**.
///
/// Round-trips [`read_off`]: `OFF` magic, `n_nodes n_elem 0` counts, vertex
/// block, then `3 i j k` triangle lines with 0-based indices.
pub fn write_off(mesh: &Mesh, path: impl AsRef<Path>) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "OFF")?;
    writeln!(w, "{} {} 0", mesh.verts.len(), mesh.tris.len())?;
    for v in &mesh.verts {
        writeln!(w, "{:.10e} {:.10e} {:.10e}", v.x, v.y, v.z)?;
    }
    for t in &mesh.tris {
        writeln!(w, "3 {} {} {}", t[0], t[1], t[2])?;
    }
    w.flush()
}

/// Read a **PQR** charge set (atomic charges + radii).
///
/// Each `ATOM`/`HETATM` record carries, in its **last five** whitespace fields,
/// `x y z charge radius`. NESSie ignores everything else and drops zero-charge
/// atoms; we mirror both. (Splitting on the trailing five fields is robust to
/// the ragged column packing real PQR writers emit.)
pub fn read_pqr(path: impl AsRef<Path>) -> io::Result<Vec<Charge>> {
    let reader = BufReader::new(File::open(path)?);
    let mut charges = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let t = line.trim_start();
        if !(t.starts_with("ATOM") || t.starts_with("HETATM")) {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let n = f.len();
        if n < 5 {
            return Err(parse_err(format!("PQR record too short: {line:?}")));
        }
        let x = f64_field(f[n - 5])?;
        let y = f64_field(f[n - 4])?;
        let z = f64_field(f[n - 3])?;
        let val = f64_field(f[n - 2])?;
        // f[n - 1] is the radius — unused by the BEM charge set.
        if val == 0.0 {
            continue;
        }
        charges.push(Charge { pos: Vec3::new(x, y, z), val });
    }
    Ok(charges)
}

/// Read an **HMO** file (BEM mesh + charges in one document).
///
/// HMO groups data in `BEG_*_DATA` / `END_*_DATA` blocks, each preceded by one
/// count/comment line we skip:
/// - `NODL` — `id x y z` (node ids ignored; positions taken).
/// - `ELEM` — `id ? ? i j k` (**1-based** node ids; we normalise to 0-based).
/// - `CHARGE` — `id x y z val` (charge value in the last field).
///
/// Returns the mesh and its charge set together, matching NESSie's combined HMO
/// loader.
pub fn read_hmo(path: impl AsRef<Path>) -> io::Result<(Mesh, Vec<Charge>)> {
    let reader = BufReader::new(File::open(path)?);
    let raw: Vec<String> = reader.lines().collect::<io::Result<_>>()?;

    let mut verts = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut charges = Vec::new();

    let mut i = 0;
    while i < raw.len() {
        let line = raw[i].trim();
        if line == "BEG_NODL_DATA" {
            i += 2; // skip the count line after the marker
            while i < raw.len() && raw[i].trim() != "END_NODL_DATA" {
                let f: Vec<&str> = raw[i].split_whitespace().collect();
                if f.len() >= 4 {
                    verts.push(Vec3::new(
                        f64_field(f[1])?,
                        f64_field(f[2])?,
                        f64_field(f[3])?,
                    ));
                }
                i += 1;
            }
        } else if line == "BEG_ELEM_DATA" {
            i += 2;
            while i < raw.len() && raw[i].trim() != "END_ELEM_DATA" {
                let f: Vec<&str> = raw[i].split_whitespace().collect();
                if f.len() >= 6 {
                    let a = usize_field(f[3])?;
                    let b = usize_field(f[4])?;
                    let c = usize_field(f[5])?;
                    if a == 0 || b == 0 || c == 0 {
                        return Err(parse_err("HMO element ids are 1-based; found 0"));
                    }
                    tris.push([(a - 1) as u32, (b - 1) as u32, (c - 1) as u32]);
                }
                i += 1;
            }
        } else if line == "BEG_CHARGE_DATA" {
            i += 2;
            while i < raw.len() && raw[i].trim() != "END_CHARGE_DATA" {
                let f: Vec<&str> = raw[i].split_whitespace().collect();
                if f.len() >= 5 {
                    let val = f64_field(f[4])?;
                    if val != 0.0 {
                        charges.push(Charge {
                            pos: Vec3::new(
                                f64_field(f[1])?,
                                f64_field(f[2])?,
                                f64_field(f[3])?,
                            ),
                            val,
                        });
                    }
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    for t in &tris {
        for &idx in t {
            if idx as usize >= verts.len() {
                return Err(parse_err(format!("HMO face index {idx} out of range")));
            }
        }
    }

    Ok((Mesh { verts, normals: Vec::new(), tris }, charges))
}

/// Read an **MSMS** surface from its `.vert` / `.face` pair.
///
/// Both files carry **three header lines** (comment, comment, counts) that we
/// skip. Vertex lines start `x y z …`; face lines start `i j k …` with
/// **1-based** vertex indices (we normalise to 0-based). Trailing per-row
/// metadata (normals, sphere ids, face type) is ignored — the BEM core needs
/// only positions and connectivity.
pub fn read_msms(vert_path: impl AsRef<Path>, face_path: impl AsRef<Path>) -> io::Result<Mesh> {
    let verts = {
        let reader = BufReader::new(File::open(vert_path)?);
        let mut verts = Vec::new();
        for line in reader.lines().skip(3) {
            let line = line?;
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 3 {
                continue;
            }
            verts.push(Vec3::new(f64_field(f[0])?, f64_field(f[1])?, f64_field(f[2])?));
        }
        verts
    };

    let tris = {
        let reader = BufReader::new(File::open(face_path)?);
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for line in reader.lines().skip(3) {
            let line = line?;
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 3 {
                continue;
            }
            let a = usize_field(f[0])?;
            let b = usize_field(f[1])?;
            let c = usize_field(f[2])?;
            if a == 0 || b == 0 || c == 0 {
                return Err(parse_err("MSMS face ids are 1-based; found 0"));
            }
            for &idx in &[a, b, c] {
                if idx > verts.len() {
                    return Err(parse_err(format!("MSMS face index {idx} out of range")));
                }
            }
            tris.push([(a - 1) as u32, (b - 1) as u32, (c - 1) as u32]);
        }
        tris
    };

    Ok(Mesh { verts, normals: Vec::new(), tris })
}
