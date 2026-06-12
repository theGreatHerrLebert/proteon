//! 1-D Chebyshev (second-kind / Chebyshev–Lobatto) interpolation in barycentric form.
//!
//! The treecode cluster expansion interpolates the kernel on a tensor grid of these
//! nodes (Wang–Krasny–Tlupova 2020). Second-kind nodes `x_k = cos(kπ/p)` have the
//! simple barycentric weights `w_k = (−1)^k δ_k` (`δ_0 = δ_p = 1/2`, else `1`), and
//! the barycentric Lagrange formula is the numerically stable way to evaluate the
//! interpolant and the basis functions `L_k` (whose panel integrals are the treecode
//! moments).

/// Chebyshev second-kind nodes on `[a, b]`: `p + 1` points, returned in **ascending**
/// order. (`x_k = cos(kπ/p)` runs `+1 → −1`; we map and reverse so `nodes[0] = a`.)
#[must_use]
pub fn nodes(p: usize, a: f64, b: f64) -> Vec<f64> {
    if p == 0 {
        return vec![(a + b) * 0.5];
    }
    let mid = (a + b) * 0.5;
    let half = (b - a) * 0.5;
    let mut v: Vec<f64> = (0..=p)
        .map(|k| {
            let t = (std::f64::consts::PI * k as f64 / p as f64).cos(); // +1 .. -1
            mid + half * t
        })
        .collect();
    v.reverse(); // ascending: a .. b
    v
}

/// Barycentric weights for the second-kind nodes in **ascending** order:
/// `w_k = (−1)^k δ_k` with the endpoints halved. (Reversing the node order from the
/// canonical `cos` ordering negates every weight, an overall sign that cancels in the
/// barycentric quotient — so the standard alternating pattern is used directly.)
#[must_use]
pub fn bary_weights(p: usize) -> Vec<f64> {
    if p == 0 {
        return vec![1.0];
    }
    (0..=p)
        .map(|k| {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            let d = if k == 0 || k == p { 0.5 } else { 1.0 };
            s * d
        })
        .collect()
}

/// The Lagrange basis values `L_k(y)`, `k = 0..=p`, at `y` (barycentric form).
///
/// `nodes`/`weights` come from [`nodes`]/[`bary_weights`] (same `p`, same ordering).
/// When `y` coincides with node `k` the basis is the unit vector `e_k` (the
/// barycentric quotient is singular there and `L_k(x_m) = δ_{km}`).
#[must_use]
pub fn lagrange_basis(nodes: &[f64], weights: &[f64], y: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut l = vec![0.0; n];
    // Scale-aware on-node threshold: within this of a node, `weights/diff` can overflow
    // and yield `∞/∞ → NaN`, so snap to the cardinal basis (`L_k(x_m) = δ_km`). The
    // threshold is relative to the node span so it tracks the coordinate scale.
    let span = (nodes[n - 1] - nodes[0]).abs().max(f64::MIN_POSITIVE);
    let on_node_tol = span * 1e-13;
    let mut denom = 0.0;
    for k in 0..n {
        let diff = y - nodes[k];
        if diff.abs() <= on_node_tol {
            // On (or within rounding of) a node: clear any partial `t` values already
            // written for earlier nodes and return the unit vector there.
            l.fill(0.0);
            l[k] = 1.0;
            return l;
        }
        let t = weights[k] / diff;
        l[k] = t;
        denom += t;
    }
    for v in &mut l {
        *v /= denom;
    }
    l
}

/// Evaluate the interpolant of samples `f` (one per node) at `y` (barycentric form).
#[must_use]
pub fn interpolate(nodes: &[f64], weights: &[f64], f: &[f64], y: f64) -> f64 {
    let basis = lagrange_basis(nodes, weights, y);
    basis.iter().zip(f).map(|(b, v)| b * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_span_endpoints_ascending() {
        let n = nodes(6, -2.0, 3.0);
        assert_eq!(n.len(), 7);
        assert!((n[0] - (-2.0)).abs() < 1e-12, "first node = a");
        assert!((n[6] - 3.0).abs() < 1e-12, "last node = b");
        for w in n.windows(2) {
            assert!(w[1] > w[0], "ascending");
        }
    }

    #[test]
    fn basis_is_partition_of_unity_and_cardinal() {
        let p = 8;
        let nd = nodes(p, 0.0, 1.0);
        let w = bary_weights(p);
        // Partition of unity at arbitrary points.
        for &y in &[0.07, 0.31, 0.5, 0.83, 0.99] {
            let l = lagrange_basis(&nd, &w, y);
            let s: f64 = l.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "Σ L_k(y) = 1, got {s}");
        }
        // Cardinality L_k(x_m) = δ_km.
        for (m, &xm) in nd.iter().enumerate() {
            let l = lagrange_basis(&nd, &w, xm);
            for (k, &lk) in l.iter().enumerate() {
                let want = f64::from(u8::from(k == m));
                assert!((lk - want).abs() < 1e-12, "L_{k}(x_{m}) = {lk}, want {want}");
            }
        }
    }

    #[test]
    fn reproduces_polynomials_up_to_degree_p() {
        // A degree-p interpolant is exact on any polynomial of degree ≤ p.
        let p = 6;
        let (a, b) = (-1.5, 2.5);
        let nd = nodes(p, a, b);
        let w = bary_weights(p);
        // f(x) = 1 - 2x + 0.5x^2 - x^3 + 0.2x^4 - 0.1x^5 + 0.03x^6 (degree 6 = p).
        let f = |x: f64| 1.0 - 2.0 * x + 0.5 * x * x - x.powi(3) + 0.2 * x.powi(4)
            - 0.1 * x.powi(5) + 0.03 * x.powi(6);
        let samples: Vec<f64> = nd.iter().map(|&x| f(x)).collect();
        for &y in &[-1.2, 0.0, 0.4, 1.7, 2.3] {
            let got = interpolate(&nd, &w, &samples, y);
            assert!((got - f(y)).abs() < 1e-9, "poly reproduction at {y}: {got} vs {}", f(y));
        }
    }

    #[test]
    fn converges_spectrally_on_smooth_function() {
        // Error on a smooth (analytic) function falls fast with p.
        let f = |x: f64| (1.3 * x).sin() * (-0.2 * x).exp();
        let err = |p: usize| {
            let nd = nodes(p, -1.0, 1.0);
            let w = bary_weights(p);
            let s: Vec<f64> = nd.iter().map(|&x| f(x)).collect();
            let mut e: f64 = 0.0;
            for i in 0..50 {
                let y = -1.0 + 2.0 * i as f64 / 49.0;
                e = e.max((interpolate(&nd, &w, &s, y) - f(y)).abs());
            }
            e
        };
        let e4 = err(4);
        let e10 = err(10);
        assert!(e10 < e4, "error should drop with p: {e4} -> {e10}");
        assert!(e10 < 1e-8, "p=10 should be tight on a smooth fn, got {e10}");
    }
}
