//! Mesh / charge file-format readers and writers (NESSie `format/` layer port).
//!
//! NESSie ships a small I/O layer for the surface meshes and charge sets that
//! drive BEM: OFF (Geomview), PQR (atomic charges + radii), HMO (combined mesh +
//! charges in one file), and MSMS (`.vert` / `.face` pair). proteon's BEM core
//! is I/O-free by design (`Tri`/`Charge` in-memory), so this module is the thin
//! bridge that turns those on-disk formats into [`Mesh`](proteon_core::surface::mesh::Mesh)
//! + [`Charge`] and back.
//!
//! Index convention: OFF face indices are 0-based in the file; NESSie's HMO and
//! MSMS node ids are 1-based. proteon's [`Mesh::tris`](proteon_core::surface::mesh::Mesh)
//! are always 0-based `u32`, so every reader normalises to 0-based on the way in
//! (validating the *original* id before the `-1` to avoid wrap-around), and OFF's
//! writer restores the file convention on the way out.
//!
//! Fail-closed posture: a non-blank but malformed data row is an error, never a
//! silent skip — skipping a node row would shift every subsequent 1-based element
//! index and corrupt connectivity without warning. Blank lines are tolerated.
//!
//! Gated against NESSie's own readers via committed fixtures in
//! `tests/fixtures/format/` (see `tests/format_parity.rs`).

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;

use crate::model::Charge;

/// Cap on count-driven pre-allocation, so a hostile/corrupt header count cannot
/// trigger an allocation-failure panic before any data is read.
const PREALLOC_CAP: usize = 1 << 20;

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

/// Parse a 1-based node id and normalise to a 0-based `u32` index, validating the
/// **original** id against `n_verts` first so a huge id cannot wrap on the cast.
fn idx_1based(s: &str, n_verts: usize) -> io::Result<u32> {
    let id = usize_field(s)?;
    if id == 0 || id > n_verts {
        return Err(parse_err(format!(
            "1-based node id {id} out of range 1..={n_verts}"
        )));
    }
    u32::try_from(id - 1).map_err(|_| parse_err(format!("node index {} exceeds u32", id - 1)))
}

/// Validate a 0-based index and convert to `u32`.
fn idx_0based(id: usize, n_verts: usize) -> io::Result<u32> {
    if id >= n_verts {
        return Err(parse_err(format!("index {id} out of range 0..{n_verts}")));
    }
    u32::try_from(id).map_err(|_| parse_err(format!("index {id} exceeds u32")))
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
    let mut lines = reader.lines().filter_map(|l| match l {
        Ok(s) => {
            let t = s.trim().to_string();
            if t.is_empty() || t.starts_with('#') {
                None
            } else {
                Some(Ok(t))
            }
        }
        Err(e) => Some(Err(e)),
    });

    let header = lines.next().ok_or_else(|| parse_err("empty OFF file"))??;
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

    let mut verts = Vec::with_capacity(n_nodes.min(PREALLOC_CAP));
    for _ in 0..n_nodes {
        let line = lines
            .next()
            .ok_or_else(|| parse_err("OFF truncated in node block"))??;
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            return Err(parse_err(format!("OFF node line needs x y z: {line:?}")));
        }
        verts.push(Vec3::new(
            f64_field(f[0])?,
            f64_field(f[1])?,
            f64_field(f[2])?,
        ));
    }

    let mut tris = Vec::with_capacity(n_elem.min(PREALLOC_CAP));
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
        let i = idx_0based(usize_field(f[1])?, n_nodes)?;
        let j = idx_0based(usize_field(f[2])?, n_nodes)?;
        let k = idx_0based(usize_field(f[3])?, n_nodes)?;
        tris.push([i, j, k]);
    }

    Ok(Mesh {
        verts,
        normals: Vec::new(),
        tris,
    })
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
/// Matches NESSie's `readpqr` exactly: a record is taken only when the line
/// begins with `ATOM` (so `HETATM` water records are excluded), and its **last
/// five** whitespace fields are `x y z charge radius`. The radius is unused by
/// the BEM charge set and zero-charge atoms are dropped — both as NESSie does.
/// (Splitting on the trailing fields is robust to ragged PQR column packing.)
pub fn read_pqr(path: impl AsRef<Path>) -> io::Result<Vec<Charge>> {
    let reader = BufReader::new(File::open(path)?);
    let mut charges = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("ATOM") {
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
        charges.push(Charge {
            pos: Vec3::new(x, y, z),
            val,
        });
    }
    Ok(charges)
}

/// Read an **HMO** file (BEM mesh + charges in one document).
///
/// HMO groups data in `BEG_*_DATA` / `END_*_DATA` blocks, each preceded by one
/// count/comment line we skip:
/// - `NODL` — `id x y z` (node ids ignored; positions taken).
/// - `ELEM` — `id ? ? i j k` (**1-based** node ids; normalised to 0-based).
/// - `CHARGE` — `id x y z val` (charge value in the last field; **kept even when
///   zero**, matching NESSie — only PQR drops zero charges).
///
/// Returns the mesh and its charge set together, matching NESSie's combined HMO
/// loader. Malformed (non-blank, undersized) rows error rather than skip.
pub fn read_hmo(path: impl AsRef<Path>) -> io::Result<(Mesh, Vec<Charge>)> {
    let reader = BufReader::new(File::open(path)?);
    let raw: Vec<String> = reader.lines().collect::<io::Result<_>>()?;

    let mut verts = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut charges = Vec::new();
    let (mut saw_nodl, mut saw_elem, mut saw_charge) = (false, false, false);

    let mut i = 0;
    while i < raw.len() {
        let line = raw[i].trim();
        if line == "BEG_NODL_DATA" {
            saw_nodl = true;
            i += 2; // skip the count line after the marker
            while i < raw.len() && raw[i].trim() != "END_NODL_DATA" {
                let row = raw[i].trim();
                if !row.is_empty() {
                    let f: Vec<&str> = row.split_whitespace().collect();
                    if f.len() < 4 {
                        return Err(parse_err(format!("HMO node row needs id x y z: {row:?}")));
                    }
                    verts.push(Vec3::new(
                        f64_field(f[1])?,
                        f64_field(f[2])?,
                        f64_field(f[3])?,
                    ));
                }
                i += 1;
            }
        } else if line == "BEG_ELEM_DATA" {
            saw_elem = true;
            i += 2;
            while i < raw.len() && raw[i].trim() != "END_ELEM_DATA" {
                let row = raw[i].trim();
                if !row.is_empty() {
                    let f: Vec<&str> = row.split_whitespace().collect();
                    if f.len() < 6 {
                        return Err(parse_err(format!(
                            "HMO element row needs id ? ? i j k: {row:?}"
                        )));
                    }
                    let a = idx_1based(f[3], verts.len())?;
                    let b = idx_1based(f[4], verts.len())?;
                    let c = idx_1based(f[5], verts.len())?;
                    tris.push([a, b, c]);
                }
                i += 1;
            }
        } else if line == "BEG_CHARGE_DATA" {
            saw_charge = true;
            i += 2;
            while i < raw.len() && raw[i].trim() != "END_CHARGE_DATA" {
                let row = raw[i].trim();
                if !row.is_empty() {
                    let f: Vec<&str> = row.split_whitespace().collect();
                    if f.len() < 5 {
                        return Err(parse_err(format!(
                            "HMO charge row needs id x y z val: {row:?}"
                        )));
                    }
                    charges.push(Charge {
                        pos: Vec3::new(f64_field(f[1])?, f64_field(f[2])?, f64_field(f[3])?),
                        val: f64_field(f[4])?,
                    });
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    if !saw_nodl || !saw_elem {
        return Err(parse_err(
            "HMO file missing BEG_NODL_DATA or BEG_ELEM_DATA block",
        ));
    }
    let _ = saw_charge; // charge block is optional

    Ok((
        Mesh {
            verts,
            normals: Vec::new(),
            tris,
        },
        charges,
    ))
}

/// Read an **MSMS** surface from its `.vert` / `.face` pair.
///
/// Both files carry **three header lines** (comment, comment, counts) that are
/// skipped; the counts line's first field is the declared vertex/face count,
/// which we validate against what we read. Vertex lines start `x y z …`; face
/// lines start `i j k …` with **1-based** vertex indices (normalised to 0-based).
/// Trailing per-row metadata (normals, sphere ids, face type) is ignored. A
/// non-blank but undersized row errors rather than corrupting the index mapping.
pub fn read_msms(vert_path: impl AsRef<Path>, face_path: impl AsRef<Path>) -> io::Result<Mesh> {
    let vert_lines: Vec<String> = BufReader::new(File::open(vert_path)?)
        .lines()
        .collect::<io::Result<_>>()?;
    let face_lines: Vec<String> = BufReader::new(File::open(face_path)?)
        .lines()
        .collect::<io::Result<_>>()?;
    if vert_lines.len() < 3 || face_lines.len() < 3 {
        return Err(parse_err("MSMS .vert/.face must have 3 header lines"));
    }

    let declared = |lines: &[String]| -> Option<usize> {
        lines[2]
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<usize>().ok())
    };
    let want_v = declared(&vert_lines);
    let want_f = declared(&face_lines);

    let mut verts = Vec::new();
    for line in &vert_lines[3..] {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            return Err(parse_err(format!("MSMS vertex row needs x y z: {line:?}")));
        }
        verts.push(Vec3::new(
            f64_field(f[0])?,
            f64_field(f[1])?,
            f64_field(f[2])?,
        ));
    }

    let mut tris: Vec<[u32; 3]> = Vec::new();
    for line in &face_lines[3..] {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            return Err(parse_err(format!("MSMS face row needs i j k: {line:?}")));
        }
        let a = idx_1based(f[0], verts.len())?;
        let b = idx_1based(f[1], verts.len())?;
        let c = idx_1based(f[2], verts.len())?;
        tris.push([a, b, c]);
    }

    if let Some(n) = want_v {
        if n != verts.len() {
            return Err(parse_err(format!(
                "MSMS .vert declares {n} vertices, found {}",
                verts.len()
            )));
        }
    }
    if let Some(n) = want_f {
        if n != tris.len() {
            return Err(parse_err(format!(
                "MSMS .face declares {n} faces, found {}",
                tris.len()
            )));
        }
    }

    Ok(Mesh {
        verts,
        normals: Vec::new(),
        tris,
    })
}
