//! Kabsch algorithm for optimal rotation superposition of two point sets.
//!
//! Ported from the C++ TMalign implementation (lines 983-1303).
//! Computes the optimal rotation matrix U and translation vector t
//! that minimizes RMSD between two sets of 3D coordinates.
//!
//! Reference: Kabsch, "A solution for the best rotation to relate two sets
//! of vectors", *Acta Crystallogr. A* 32, 922-923 (1976).

use crate::core::types::{Coord3D, Transform};

/// Controls what the Kabsch algorithm computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KabschMode {
    /// Compute RMS only (no rotation matrix).
    RmsOnly,
    /// Compute rotation matrix and translation only (no RMS).
    RotationOnly,
    /// Compute both RMS and rotation/translation.
    Both,
}

impl KabschMode {
    fn as_int(self) -> i32 {
        match self {
            KabschMode::RmsOnly => 0,
            KabschMode::RotationOnly => 1,
            KabschMode::Both => 2,
        }
    }
}

/// Result of the Kabsch algorithm.
#[derive(Debug, Clone)]
pub struct KabschResult {
    /// Sum of weighted squared distances (not normalized — divide by n for RMSD²).
    pub rms: f64,
    /// Optimal rotation + translation to superpose x onto y.
    pub transform: Transform,
}

/// Compute the optimal rotation/translation to superpose point set `x` onto `y`.
///
/// Both slices must have the same length (n >= 1).
/// Returns `None` if n < 1.
///
/// This is a literal port of the C++ Kabsch function preserving operation order
/// for numerical fidelity.
pub fn kabsch(x: &[Coord3D], y: &[Coord3D], mode: KabschMode) -> Option<KabschResult> {
    let n = x.len();
    if n < 1 || y.len() != n {
        return None;
    }

    let mode_int = mode.as_int();
    let sqrt3: f64 = 1.732_050_807_568_88;
    let tol: f64 = 0.01;
    let ip: [usize; 9] = [0, 1, 3, 1, 2, 4, 3, 4, 5];
    let ip2312: [usize; 4] = [1, 2, 0, 1];
    let epsilon: f64 = 0.000_000_01;

    let mut a_failed = false;
    let mut b_failed = false;

    // Initialize outputs
    let mut rms: f64 = 0.0;
    let mut e0: f64 = 0.0;
    let mut t = [0.0_f64; 3];
    let mut u = [[0.0_f64; 3]; 3];
    let mut r = [[0.0_f64; 3]; 3];
    let mut a = [[0.0_f64; 3]; 3];

    // Identity for u and a
    for i in 0..3 {
        u[i][i] = 1.0;
        a[i][i] = 1.0;
    }

    // Compute centers for vector sets x, y
    let mut s1 = [0.0_f64; 3];
    let mut s2 = [0.0_f64; 3];
    let mut sx = [0.0_f64; 3];
    let mut sy = [0.0_f64; 3];
    let mut sz = [0.0_f64; 3];

    let nf = n as f64;

    for i in 0..n {
        let c1 = x[i];
        let c2 = y[i];

        for j in 0..3 {
            s1[j] += c1[j];
            s2[j] += c2[j];
        }

        for j in 0..3 {
            sx[j] += c1[0] * c2[j];
            sy[j] += c1[1] * c2[j];
            sz[j] += c1[2] * c2[j];
        }
    }

    let mut xc = [0.0_f64; 3];
    let mut yc = [0.0_f64; 3];
    for i in 0..3 {
        xc[i] = s1[i] / nf;
        yc[i] = s2[i] / nf;
    }

    if mode_int == 2 || mode_int == 0 {
        for mm in 0..n {
            for nn in 0..3 {
                e0 += (x[mm][nn] - xc[nn]) * (x[mm][nn] - xc[nn])
                    + (y[mm][nn] - yc[nn]) * (y[mm][nn] - yc[nn]);
            }
        }
    }

    for j in 0..3 {
        r[j][0] = sx[j] - s1[0] * s2[j] / nf;
        r[j][1] = sy[j] - s1[1] * s2[j] / nf;
        r[j][2] = sz[j] - s1[2] * s2[j] / nf;
    }

    // Compute determinant of matrix r
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    let sigma = det;

    // Compute trans(r)*r
    let mut rr = [0.0_f64; 6];
    let mut m = 0;
    for j in 0..3 {
        for i in 0..=j {
            rr[m] = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
            m += 1;
        }
    }

    let spur = (rr[0] + rr[2] + rr[5]) / 3.0;
    let cof = (((((rr[2] * rr[5] - rr[4] * rr[4]) + rr[0] * rr[5]) - rr[3] * rr[3])
        + rr[0] * rr[2])
        - rr[1] * rr[1])
        / 3.0;
    let det_sq = det * det;

    let mut e = [spur; 3];

    if spur > 0.0 {
        let d = spur * spur;
        let h = d - cof;
        let g = (spur * cof - det_sq) / 2.0 - spur * h;

        if h > 0.0 {
            let sqrth = h.sqrt();
            let mut d = h * h * h - g * g;
            if d < 0.0 {
                d = 0.0;
            }
            d = (d.sqrt()).atan2(-g) / 3.0;
            let cth = sqrth * d.cos();
            let sth = sqrth * sqrt3 * d.sin();
            e[0] = (spur + cth) + cth;
            e[1] = (spur - cth) + sth;
            e[2] = (spur - cth) - sth;

            if mode_int != 0 {
                // Compute eigenvectors a
                for l in (0..3).step_by(2) {
                    let d = e[l];
                    let mut ss = [0.0_f64; 6];
                    ss[0] = (d - rr[2]) * (d - rr[5]) - rr[4] * rr[4];
                    ss[1] = (d - rr[5]) * rr[1] + rr[3] * rr[4];
                    ss[2] = (d - rr[0]) * (d - rr[5]) - rr[3] * rr[3];
                    ss[3] = (d - rr[2]) * rr[3] + rr[1] * rr[4];
                    ss[4] = (d - rr[0]) * rr[4] + rr[1] * rr[3];
                    ss[5] = (d - rr[0]) * (d - rr[2]) - rr[1] * rr[1];

                    for val in &mut ss {
                        if val.abs() <= epsilon {
                            *val = 0.0;
                        }
                    }

                    let j = if ss[0].abs() >= ss[2].abs() {
                        if ss[0].abs() < ss[5].abs() {
                            2
                        } else {
                            0
                        }
                    } else if ss[2].abs() >= ss[5].abs() {
                        1
                    } else {
                        2
                    };

                    let mut d = 0.0_f64;
                    let j3 = 3 * j;
                    for i in 0..3 {
                        let k = ip[i + j3];
                        a[i][l] = ss[k];
                        d += ss[k] * ss[k];
                    }

                    if d > epsilon {
                        d = 1.0 / d.sqrt();
                    } else {
                        d = 0.0;
                    }
                    for i in 0..3 {
                        a[i][l] *= d;
                    }
                }

                let _d = a[0][0] * a[0][2] + a[1][0] * a[1][2] + a[2][0] * a[2][2];
                let (m1, m_col) = if (e[0] - e[1]) > (e[1] - e[2]) {
                    (2_usize, 0_usize)
                } else {
                    (0_usize, 2_usize)
                };
                let mut p = 0.0_f64;
                for i in 0..3 {
                    a[i][m1] -= _d * a[i][m_col];
                    p += a[i][m1] * a[i][m1];
                }
                if p <= tol {
                    p = 1.0;
                    let mut j = 0_usize;
                    for i in 0..3 {
                        if p < a[i][m_col].abs() {
                            continue;
                        }
                        p = a[i][m_col].abs();
                        j = i;
                    }
                    let k = ip2312[j];
                    let l = ip2312[j + 1];
                    p = (a[k][m_col] * a[k][m_col] + a[l][m_col] * a[l][m_col]).sqrt();
                    if p > tol {
                        a[j][m1] = 0.0;
                        a[k][m1] = -a[l][m_col] / p;
                        a[l][m1] = a[k][m_col] / p;
                    } else {
                        a_failed = true;
                    }
                } else {
                    p = 1.0 / p.sqrt();
                    for i in 0..3 {
                        a[i][m1] *= p;
                    }
                }
                if !a_failed {
                    a[0][1] = a[1][2] * a[2][0] - a[1][0] * a[2][2];
                    a[1][1] = a[2][2] * a[0][0] - a[2][0] * a[0][2];
                    a[2][1] = a[0][2] * a[1][0] - a[0][0] * a[1][2];
                }
            }
        }

        // Compute b
        if mode_int != 0 && !a_failed {
            let mut b = [[0.0_f64; 3]; 3];
            for l in 0..2 {
                let mut d = 0.0_f64;
                for i in 0..3 {
                    b[i][l] = r[i][0] * a[0][l] + r[i][1] * a[1][l] + r[i][2] * a[2][l];
                    d += b[i][l] * b[i][l];
                }
                if d > epsilon {
                    d = 1.0 / d.sqrt();
                } else {
                    d = 0.0;
                }
                for i in 0..3 {
                    b[i][l] *= d;
                }
            }
            let d = b[0][0] * b[0][1] + b[1][0] * b[1][1] + b[2][0] * b[2][1];
            let mut p = 0.0_f64;
            for i in 0..3 {
                b[i][1] -= d * b[i][0];
                p += b[i][1] * b[i][1];
            }

            if p <= tol {
                p = 1.0;
                let mut j = 0_usize;
                for i in 0..3 {
                    if p < b[i][0].abs() {
                        continue;
                    }
                    p = b[i][0].abs();
                    j = i;
                }
                let k = ip2312[j];
                let l = ip2312[j + 1];
                p = (b[k][0] * b[k][0] + b[l][0] * b[l][0]).sqrt();
                if p > tol {
                    b[j][1] = 0.0;
                    b[k][1] = -b[l][0] / p;
                    b[l][1] = b[k][0] / p;
                } else {
                    b_failed = true;
                }
            } else {
                p = 1.0 / p.sqrt();
                for i in 0..3 {
                    b[i][1] *= p;
                }
            }
            if !b_failed {
                b[0][2] = b[1][0] * b[2][1] - b[1][1] * b[2][0];
                b[1][2] = b[2][0] * b[0][1] - b[2][1] * b[0][0];
                b[2][2] = b[0][0] * b[1][1] - b[0][1] * b[1][0];
                // Compute u
                for i in 0..3 {
                    for j in 0..3 {
                        u[i][j] = b[i][0] * a[j][0] + b[i][1] * a[j][1] + b[i][2] * a[j][2];
                    }
                }
            }

            // Compute t
            for i in 0..3 {
                t[i] = ((yc[i] - u[i][0] * xc[0]) - u[i][1] * xc[1]) - u[i][2] * xc[2];
            }
        }
    } else {
        // spur <= 0: just compute t
        for i in 0..3 {
            t[i] = ((yc[i] - u[i][0] * xc[0]) - u[i][1] * xc[1]) - u[i][2] * xc[2];
        }
    }

    // Compute rms
    for val in &mut e {
        if *val < 0.0 {
            *val = 0.0;
        }
        *val = val.sqrt();
    }
    let mut d = e[2];
    if sigma < 0.0 {
        d = -d;
    }
    d = (d + e[1]) + e[0];

    if mode_int == 2 || mode_int == 0 {
        rms = (e0 - d) - d;
        if rms < 0.0 {
            rms = 0.0;
        }
    }

    Some(KabschResult {
        rms,
        transform: Transform { t, u },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rotation() {
        let pts: Vec<Coord3D> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let result = kabsch(&pts, &pts, KabschMode::Both).unwrap();
        assert!(result.rms < 1e-10, "rms should be ~0, got {}", result.rms);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.transform.u[i][j] - expected).abs() < 1e-6,
                    "u[{i}][{j}] = {}, expected {expected}",
                    result.transform.u[i][j]
                );
            }
        }
    }

    #[test]
    fn known_rotation_90_deg_z() {
        let x: Vec<Coord3D> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let y: Vec<Coord3D> = vec![
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let result = kabsch(&x, &y, KabschMode::Both).unwrap();
        assert!(result.rms < 1e-10, "rms should be ~0, got {}", result.rms);

        for (xi, yi) in x.iter().zip(y.iter()) {
            let mapped = result.transform.apply(xi);
            for k in 0..3 {
                assert!(
                    (mapped[k] - yi[k]).abs() < 1e-6,
                    "mismatch at coord {k}: mapped={}, expected={}",
                    mapped[k],
                    yi[k]
                );
            }
        }
    }

    #[test]
    fn translation_only() {
        let x: Vec<Coord3D> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let y: Vec<Coord3D> = vec![
            [5.0, -3.0, 2.0],
            [6.0, -3.0, 2.0],
            [5.0, -2.0, 2.0],
            [5.0, -3.0, 3.0],
        ];
        let result = kabsch(&x, &y, KabschMode::Both).unwrap();
        assert!(result.rms < 1e-10);
        assert!((result.transform.t[0] - 5.0).abs() < 1e-6);
        assert!((result.transform.t[1] - (-3.0)).abs() < 1e-6);
        assert!((result.transform.t[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn single_point() {
        let x = vec![[1.0, 2.0, 3.0]];
        let y = vec![[4.0, 5.0, 6.0]];
        let result = kabsch(&x, &y, KabschMode::Both).unwrap();
        let mapped = result.transform.apply(&x[0]);
        for k in 0..3 {
            assert!(
                (mapped[k] - y[0][k]).abs() < 1e-6,
                "single point mapping failed"
            );
        }
    }

    #[test]
    fn empty_returns_none() {
        let empty: Vec<Coord3D> = vec![];
        assert!(kabsch(&empty, &empty, KabschMode::Both).is_none());
    }

    #[test]
    fn rms_only_mode() {
        let x: Vec<Coord3D> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = kabsch(&x, &x, KabschMode::RmsOnly).unwrap();
        assert!(result.rms < 1e-10);
    }
}
