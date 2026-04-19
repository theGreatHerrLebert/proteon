//! Secondary structure assignment from CA coordinates.
//!
//! Ported from C++ TMalign (lines 2392-2491).
//! Assigns H (helix), E (strand), T (turn), or C (coil) based on
//! CA-CA distance patterns.

use crate::core::types::{dist_squared, Coord3D};

/// Classify secondary structure from six pairwise CA distances.
///
/// Corresponds to C++ `sec_str`.
fn sec_str(dis13: f64, dis14: f64, dis15: f64, dis24: f64, dis25: f64, dis35: f64) -> char {
    // Helix check
    let delta = 2.1;
    if (dis15 - 6.37).abs() < delta
        && (dis14 - 5.18).abs() < delta
        && (dis25 - 5.18).abs() < delta
        && (dis13 - 5.45).abs() < delta
        && (dis24 - 5.45).abs() < delta
        && (dis35 - 5.45).abs() < delta
    {
        return 'H';
    }

    // Strand check
    let delta = 1.42;
    if (dis15 - 13.0).abs() < delta
        && (dis14 - 10.4).abs() < delta
        && (dis25 - 10.4).abs() < delta
        && (dis13 - 6.1).abs() < delta
        && (dis24 - 6.1).abs() < delta
        && (dis35 - 6.1).abs() < delta
    {
        return 'E';
    }

    // Turn check
    if dis15 < 8.0 {
        return 'T';
    }

    'C'
}

/// Assign secondary structure for each residue from CA coordinates.
///
/// Returns a Vec of chars: 'H' (helix), 'E' (strand), 'T' (turn), 'C' (coil).
/// Corresponds to C++ `make_sec`.
pub fn make_sec(coords: &[Coord3D]) -> Vec<char> {
    let len = coords.len();
    let mut sec = vec!['C'; len];

    for i in 0..len {
        let j1 = i as isize - 2;
        let j5 = i + 2;

        if j1 >= 0 && j5 < len {
            let j1 = j1 as usize;
            let j2 = i - 1;
            let j3 = i;
            let j4 = i + 1;

            let d13 = dist_squared(&coords[j1], &coords[j3]).sqrt();
            let d14 = dist_squared(&coords[j1], &coords[j4]).sqrt();
            let d15 = dist_squared(&coords[j1], &coords[j5]).sqrt();
            let d24 = dist_squared(&coords[j2], &coords[j4]).sqrt();
            let d25 = dist_squared(&coords[j2], &coords[j5]).sqrt();
            let d35 = dist_squared(&coords[j3], &coords[j5]).sqrt();

            sec[i] = sec_str(d13, d14, d15, d24, d25, d35);
        }
    }

    smooth(&mut sec);
    sec
}

/// Smooth secondary structure assignments to remove isolated elements.
///
/// Corresponds to C++ `smooth`. Uses integer codes internally:
/// 1=C, 2=H, 3=T, 4=E.
fn smooth(sec: &mut [char]) {
    let len = sec.len();
    if len < 5 {
        return;
    }

    // Convert to integer codes for smoothing
    let mut s: Vec<i32> = sec
        .iter()
        .map(|&c| match c {
            'H' => 2,
            'T' => 3,
            'E' => 4,
            _ => 1,
        })
        .collect();

    // Smooth single: --x-- => -----
    for i in 2..len - 2 {
        if s[i] == 2 || s[i] == 4 {
            let j = s[i];
            if s[i - 2] != j && s[i - 1] != j && s[i + 1] != j && s[i + 2] != j {
                s[i] = 1;
            }
        }
    }

    // Smooth double: --xx-- => ------
    for i in 0..len.saturating_sub(5) {
        // helix
        if s[i] != 2
            && s[i + 1] != 2
            && s[i + 2] == 2
            && s[i + 3] == 2
            && s[i + 4] != 2
            && s[i + 5] != 2
        {
            s[i + 2] = 1;
            s[i + 3] = 1;
        }
        // strand
        if s[i] != 4
            && s[i + 1] != 4
            && s[i + 2] == 4
            && s[i + 3] == 4
            && s[i + 4] != 4
            && s[i + 5] != 4
        {
            s[i + 2] = 1;
            s[i + 3] = 1;
        }
    }

    // Smooth connect: fill single gaps between same type
    for i in 0..len.saturating_sub(2) {
        if s[i] == 2 && s[i + 1] != 2 && s[i + 2] == 2 {
            s[i + 1] = 2;
        } else if s[i] == 4 && s[i + 1] != 4 && s[i + 2] == 4 {
            s[i + 1] = 4;
        }
    }

    // Convert back to chars
    for (i, &val) in s.iter().enumerate() {
        sec[i] = match val {
            2 => 'H',
            3 => 'T',
            4 => 'E',
            _ => 'C',
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_sequence_all_coil() {
        // Fewer than 5 residues: all should be coil
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]];
        let sec = make_sec(&coords);
        assert_eq!(sec, vec!['C', 'C']);
    }

    #[test]
    fn helix_like_geometry() {
        // Approximate alpha helix: ~3.8A rise per residue, ~1.5A rise along axis,
        // 100 degrees turn. Using idealized helix coordinates.
        let coords: Vec<Coord3D> = vec![
            [2.30, 0.00, 0.00],
            [1.38, 2.09, 1.50],
            [-1.01, 2.15, 3.00],
            [-2.30, 0.13, 4.50],
            [-0.72, -1.96, 6.00],
        ];
        let sec = make_sec(&coords);
        // Middle residue (index 2) should be classified based on geometry
        // At minimum it should return a 5-element result
        assert_eq!(sec.len(), 5);
        // First two and last two are always 'C' since they lack ±2 neighbors
        assert_eq!(sec[0], 'C');
        assert_eq!(sec[1], 'C');
        assert_eq!(sec[3], 'C');
        assert_eq!(sec[4], 'C');
    }

    #[test]
    fn smooth_removes_isolated_helix() {
        let mut sec = vec!['C', 'C', 'H', 'C', 'C'];
        smooth(&mut sec);
        assert_eq!(sec[2], 'C'); // isolated H removed
    }

    #[test]
    fn smooth_removes_isolated_pair() {
        let mut sec = vec!['C', 'C', 'H', 'H', 'C', 'C'];
        smooth(&mut sec);
        assert_eq!(sec[2], 'C');
        assert_eq!(sec[3], 'C');
    }

    #[test]
    fn smooth_fills_gap() {
        let mut sec = vec!['H', 'C', 'H', 'H', 'H'];
        smooth(&mut sec);
        assert_eq!(sec[1], 'H'); // gap filled between helices
    }

    #[test]
    fn empty_input() {
        let sec = make_sec(&[]);
        assert!(sec.is_empty());
    }
}
