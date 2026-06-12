//! Xie multi-charge dielectric-sphere analytic test models (port of NESSie's
//! `TestModel/xie`). Closed-form local + two nonlocal Poisson solutions for several point
//! charges in an origin-centred dielectric sphere — the analytic oracle for the BEM
//! beyond the single-charge Born case. Faithful port of `testmodel/xie/*.jl`, gated
//! bit-exact against NESSie's own output (`tests/xie_parity.rs`).
//!
//! References: Xie et al., *Commun. Comput. Phys.* (2016). The series coefficients
//! (`A_in` / `C_in`) and the potential evaluation are transcribed verbatim; the spherical
//! modified Bessel functions use Miller's downward recursion (first kind) and stable
//! upward recursion (second kind).

use crate::model::Charge;
use crate::post::{ENERGY_FACTOR, POTPREFACTOR};
use proteon_core::surface::geom::Vec3;

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;
/// NESSie's reaction-potential factor `ec/ε0 = potprefactor · 4π`.
const RF_FACTOR: f64 = POTPREFACTOR * FOUR_PI;
/// `molpotential` distance floor (NESSie `_molpotential` tolerance).
const MOL_TOL: f64 = 1e-10;

// ---- special functions ------------------------------------------------------------

/// Legendre polynomials `P₀(x) … P_{maxn-1}(x)` (Bonnet's recursion).
fn legendre(maxn: usize, x: f64) -> Vec<f64> {
    let mut p = vec![0.0; maxn];
    if maxn >= 1 {
        p[0] = 1.0;
    }
    if maxn >= 2 {
        p[1] = x;
        for n in 1..maxn - 1 {
            let nf = n as f64;
            p[n + 1] = ((2.0 * nf + 1.0) * x * p[n] - nf * p[n - 1]) / (nf + 1.0);
        }
    }
    p
}

/// Modified spherical Bessel function of the **first** kind `iₙ(x)` for `n = −1 … maxn`,
/// returned with `vec[n+1] = iₙ(x)`. Miller's downward recursion + normalisation by the
/// exact `i₀ = sinh(x)/x` (the upward recursion is unstable for small `x`).
fn spherical_besseli(maxn: usize, x: f64) -> Vec<f64> {
    let start = maxn + 25; // buffer above maxn for the downward sweep
    let mut vals = vec![0.0; maxn + 2]; // index n+1, n = -1 … maxn
    let mut next = 0.0; // iₙ₊₁ (unnormalised)
    let mut cur = 1.0; // iₙ at n = start
    for n in (0..=start).rev() {
        if n <= maxn {
            vals[n + 1] = cur; // iₙ
        }
        let prev = next + (2.0 * n as f64 + 1.0) / x * cur; // i_{n-1}
        next = cur;
        cur = prev;
    }
    vals[0] = cur; // i₋₁
    let scale = (x.sinh() / x) / vals[1];
    for v in &mut vals {
        *v *= scale;
    }
    vals
}

/// Modified spherical Bessel function of the **second** kind `kₙ(x)` for `n = −1 … maxn`,
/// returned with `vec[n+1] = kₙ(x)`. `k₋₁ = k₀ = (π/2x)e^{-x}`, stable upward recursion.
fn spherical_besselk(maxn: usize, x: f64) -> Vec<f64> {
    let mut vals = vec![0.0; maxn + 2];
    let k0 = std::f64::consts::PI / (2.0 * x) * (-x).exp();
    vals[0] = k0; // k₋₁
    vals[1] = k0; // k₀
    for idx in 2..maxn + 2 {
        let n = (idx - 2) as f64;
        vals[idx] = vals[idx - 2] + (2.0 * n + 1.0) / x * vals[idx - 1];
    }
    vals
}

// ---- model ------------------------------------------------------------------------

/// Which Xie model: local Poisson, or the two nonlocal variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XieKind {
    /// Local Poisson dielectric model (`LocalXieModel`).
    Local,
    /// First nonlocal model (`NonlocalXieModel1`).
    Nonlocal1,
    /// Second nonlocal model (`NonlocalXieModel2`).
    Nonlocal2,
}

/// A Xie analytic test model: a dielectric sphere with point charges, plus the
/// pre-computed series coefficients. `charges` are already scaled/centred into the sphere.
#[derive(Debug, Clone)]
pub struct XieModel {
    kind: XieKind,
    radius: f64,
    charges: Vec<Charge>,
    eps_omega: f64,
    eps_sigma: f64,
    eps_inf: f64,
    lambda: f64,
    len: usize,
    /// `m[charge][n]` coefficient matrices `M₁, M₂, M₃` (`A_in` / `C_in`).
    m1: Vec<Vec<f64>>,
    m2: Vec<Vec<f64>>,
    m3: Vec<Vec<f64>>,
}

/// Center and rescale the charges so the outermost sits at 80 % of `radius` (NESSie
/// `scalemodel`). `compat` rounds the scale factor up, matching the Xie reference.
#[must_use]
pub fn scalemodel(charges: &[Charge], radius: f64, compat: bool) -> Vec<Charge> {
    if charges.is_empty() {
        return Vec::new();
    }
    // Center at the midpoint of the per-axis extrema.
    let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in charges {
        lo = Vec3::new(lo.x.min(c.pos.x), lo.y.min(c.pos.y), lo.z.min(c.pos.z));
        hi = Vec3::new(hi.x.max(c.pos.x), hi.y.max(c.pos.y), hi.z.max(c.pos.z));
    }
    let center = (lo + hi) * 0.5;
    let mut newpos: Vec<Vec3> = charges.iter().map(|c| c.pos - center).collect();
    let max_norm = newpos.iter().map(|p| p.norm()).fold(0.0_f64, f64::max);
    let sf = if compat { max_norm.ceil() } else { max_norm };
    if sf > 0.0 {
        let factor = 0.8 * radius / sf;
        for p in &mut newpos {
            *p = *p * factor;
        }
    }
    newpos
        .into_iter()
        .zip(charges)
        .map(|(pos, c)| Charge { pos, val: c.val })
        .collect()
}

impl XieModel {
    /// Build a Xie model (compat scaling, as the reference) from raw charges.
    #[must_use]
    pub fn new(
        kind: XieKind,
        radius: f64,
        charges: &[Charge],
        eps_omega: f64,
        eps_sigma: f64,
        eps_inf: f64,
        lambda: f64,
        len: usize,
    ) -> Self {
        let charges = scalemodel(charges, radius, true);
        let mut m = XieModel {
            kind,
            radius,
            charges,
            eps_omega,
            eps_sigma,
            eps_inf,
            lambda,
            len,
            m1: Vec::new(),
            m2: Vec::new(),
            m3: Vec::new(),
        };
        let (m1, m2, m3) = match kind {
            XieKind::Local => m.coeff_local(),
            XieKind::Nonlocal1 => m.coeff_nonlocal1(),
            XieKind::Nonlocal2 => m.coeff_nonlocal2(),
        };
        m.m1 = m1;
        m.m2 = m2;
        m.m3 = m3;
        m
    }

    fn coeff_local(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (a, lam, eo, es) = (self.radius, self.lambda, self.eps_omega, self.eps_sigma);
        let len = self.len;
        let ia = spherical_besseli(len, a / lam);
        let ka = spherical_besselk(len, a / lam);
        // s[n] = iₐ(n+1)·kₐ(n) + iₐ(n)·kₐ(n+1);  e[n] = (εΣ(n+1) + n·εΩ)·s[n].
        let s: Vec<f64> = (0..len)
            .map(|n| ia[n + 2] * ka[n + 1] + ia[n + 1] * ka[n + 2])
            .collect();
        let e: Vec<f64> = (0..len)
            .map(|n| (es * (n as f64 + 1.0) + n as f64 * eo) * s[n])
            .collect();
        self.fill_c(&ia, &s, &e, a, lam, eo)
    }

    fn coeff_nonlocal2(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (a, lam, eo, es, ei) = (
            self.radius,
            self.lambda,
            self.eps_omega,
            self.eps_sigma,
            self.eps_inf,
        );
        let len = self.len;
        let kappa = (es / ei).sqrt() / lam;
        let ia = spherical_besseli(len, a / lam);
        let ka = spherical_besselk(len, a / lam);
        let kk = spherical_besselk(len, a * kappa);
        // u, s, w, e (Eqs. 57–60).
        let u: Vec<f64> = (0..len)
            .map(|n| {
                let nf = n as f64;
                -nf * eo
                    + (es - ei) * nf * (nf + 1.0) * 2.0 * a / std::f64::consts::PI / lam
                        * ia[n + 1]
                        * ka[n + 1]
            })
            .collect();
        let s: Vec<f64> = (0..len)
            .map(|n| {
                (es - ei) / ei * (2.0 * n as f64 + 1.0) * lam / a * ia[n + 1] * kk[n + 1]
                    + es / ei * ia[n + 2] * kk[n + 1]
                    + kappa * lam * ia[n + 1] * kk[n + 2]
            })
            .collect();
        let w: Vec<f64> = (0..len)
            .map(|n| {
                (es - ei) / ei * n as f64 * lam / a * ia[n + 1] * kk[n + 1]
                    + es / ei * ia[n + 2] * kk[n + 1]
                    + kappa * lam * ia[n + 1] * kk[n + 2]
            })
            .collect();
        let e: Vec<f64> = (0..len)
            .map(|n| es * (n as f64 + 1.0) * w[n] - u[n] * s[n])
            .collect();
        // C₂ uses `w` in the e/s sense but C₁/C₃ use s and w per the local pattern; the
        // nonlocal2 fill matches local except `w` replaces `s` in C₁'s `wₙ` and `e` differs.
        self.fill_c2(&ia, &s, &w, &e, a, lam, eo)
    }

    /// C-coefficient fill shared by Local (`w = s`) — `wₙ` is the per-`n` weight in `C₁`.
    fn fill_c(
        &self,
        ia: &[f64],
        s: &[f64],
        e: &[f64],
        a: f64,
        lam: f64,
        eo: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        self.fill_c2(ia, s, s, e, a, lam, eo) // local: wₙ = sₙ
    }

    /// General C-fill: `C₁` uses `wₙ`, `C₃` uses `sₙ` (Local passes `w = s`).
    fn fill_c2(
        &self,
        ia: &[f64],
        s: &[f64],
        w: &[f64],
        e: &[f64],
        a: f64,
        lam: f64,
        eo: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let len = self.len;
        let mut c1 = Vec::with_capacity(self.charges.len());
        let mut c2 = Vec::with_capacity(self.charges.len());
        let mut c3 = Vec::with_capacity(self.charges.len());
        for q in &self.charges {
            let r = q.pos.norm();
            let (mut q1, mut q2, mut q3) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
            for n in 0..len {
                let nf = n as f64;
                let rn = r.powi(n as i32);
                q1[n] = (2.0 * nf + 1.0) / FOUR_PI / e[n] * rn * w[n];
                q2[n] = -(2.0 * nf + 1.0) * (nf + 1.0) / FOUR_PI / a.powi(n as i32 + 2) / e[n]
                    * lam
                    * rn
                    * ia[n + 1];
                q3[n] = rn / FOUR_PI / a.powi(2 * n as i32 + 1)
                    * ((2.0 * nf + 1.0) * s[n] / e[n] - 1.0 / eo);
            }
            c1.push(q1);
            c2.push(q2);
            c3.push(q3);
        }
        (c1, c2, c3)
    }

    fn coeff_nonlocal1(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (a, lam, eo, es, ei) = (
            self.radius,
            self.lambda,
            self.eps_omega,
            self.eps_sigma,
            self.eps_inf,
        );
        let len = self.len;
        let kappa = (es / ei).sqrt() / lam;
        let c = lam * (es - ei) / (a * a * ei);
        let ia = spherical_besseli(len, a / lam);
        let ka = spherical_besselk(len, a * kappa);
        // d[n] (Eq. 21), w[n] (Eq. 22).
        let d: Vec<f64> = (0..len)
            .map(|n| {
                let nf = n as f64;
                nf * (2.0 * nf + 1.0) * eo * c / a.powi(n as i32) * ia[n + 1] * ka[n + 1]
                    + (nf * eo + (nf + 1.0) * es) / a.powi(n as i32 + 1)
                        * (es / ei * ia[n + 2] * ka[n + 1] + kappa * lam * ia[n + 1] * ka[n + 2])
            })
            .collect();
        let w: Vec<f64> = (0..len)
            .map(|n| {
                let nf = n as f64;
                es / ei * ka[n + 1] * (-lam * (nf + 1.0) / a * ia[n + 1] + ia[n])
                    + kappa * lam * ia[n + 1] * ((nf + 1.0) / (kappa * a) * ka[n + 1] + ka[n])
            })
            .collect();
        let mut a1 = Vec::with_capacity(self.charges.len());
        let mut a2 = Vec::with_capacity(self.charges.len());
        let mut a3 = Vec::with_capacity(self.charges.len());
        for q in &self.charges {
            let r = q.pos.norm();
            let ir = spherical_besseli(len, (r / lam).max(1e-10));
            let (mut q1, mut q2, mut q3) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
            for n in 0..len {
                let nf = n as f64;
                let rn = r.powi(n as i32);
                q1[n] = (2.0 * nf + 1.0) / (FOUR_PI * d[n])
                    * (rn / a.powi(n as i32 + 1) * w[n] - c * nf * ir[n + 1] * ka[n + 1]);
                q2[n] = lam * (2.0 * nf + 1.0) / (FOUR_PI * eo * d[n]) / a.powi(n as i32 + 3)
                    * ((es - eo) * (nf + 1.0) * rn / a.powi(n as i32) * ia[n + 1]
                        - (nf * (eo + es) + es) * ir[n + 1]);
                q3[n] = q1[n] / a.powi(2 * n as i32 + 1)
                    + q2[n] * (ei - es) / (ei * a.powi(n as i32)) * ka[n + 1]
                    - rn / (FOUR_PI * eo * a.powi(2 * n as i32 + 1));
            }
            a1.push(q1);
            a2.push(q2);
            a3.push(q3);
        }
        (a1, a2, a3)
    }

    /// Reaction-field energy `W*` (kJ/mol): `Σ_c q_c · φ_rf(Ω, r_c)` × energy factor.
    #[must_use]
    pub fn rfenergy(&self) -> f64 {
        let sum: f64 = self
            .charges
            .iter()
            .map(|c| c.val * self.rfpotential_omega(c.pos))
            .sum();
        sum * ENERGY_FACTOR
    }

    /// Electrostatic potential (V) at `xi` = reaction field + molecular potential.
    #[must_use]
    pub fn espotential(&self, xi: Vec3) -> f64 {
        if xi.norm() <= self.radius {
            self.rfpotential_omega(xi) + self.molpotential(xi)
        } else {
            self.espotential_sigma(xi)
        }
    }

    /// Molecular (point-charge) potential (V) at `xi`.
    #[must_use]
    pub fn molpotential(&self, xi: Vec3) -> f64 {
        let raw: f64 = self
            .charges
            .iter()
            .map(|c| c.val / (c.pos - xi).norm().max(MOL_TOL))
            .sum();
        POTPREFACTOR * raw / self.eps_omega
    }

    /// Reaction-field potential inside Ω (the `ec/ε0`-scaled series; `_rfpotential_Ω`).
    fn rfpotential_omega(&self, xi: Vec3) -> f64 {
        let (a, lam, eo, es, ei) = (
            self.radius,
            self.lambda,
            self.eps_omega,
            self.eps_sigma,
            self.eps_inf,
        );
        let r = xi.norm();
        let mut phi = 0.0;
        for (qi, q) in self.charges.iter().enumerate() {
            if q.pos.norm() < 1e-10 {
                // Central charge: the nonlocal Born potential (NESSie's branch).
                let t1 = (a * es + lam * (eo - es) * (a / lam).sinh())
                    / ((a * (ei * es).sqrt() + lam * (ei - es)) * (a / lam).sinh()
                        + a * es * (a / lam).cosh());
                let t2 = (eo - es - (ei - es) * t1) / a / es;
                phi += q.val / FOUR_PI / eo * t2;
                continue;
            }
            if r < 1e-10 {
                phi += self.m3[qi][0] * q.val;
                continue;
            }
            let cos = xi.dot(q.pos) / (r * q.pos.norm());
            let p = legendre(self.len, cos);
            let mut phij = 0.0;
            for n in 0..self.len {
                phij += self.m3[qi][n] * r.powi(n as i32) * p[n];
            }
            phi += phij * q.val;
        }
        phi * RF_FACTOR
    }

    /// Electrostatic potential outside Σ (`_espotential_Σ`).
    fn espotential_sigma(&self, xi: Vec3) -> f64 {
        let (a, lam, eo, es, ei) = (
            self.radius,
            self.lambda,
            self.eps_omega,
            self.eps_sigma,
            self.eps_inf,
        );
        let kappa = (es / ei).sqrt() / lam;
        let r = xi.norm();
        let kr = spherical_besselk(self.len, kappa * r);
        let mut phi = 0.0;
        for (qi, q) in self.charges.iter().enumerate() {
            if q.pos.norm() < 1e-10 {
                let t1 = (kappa * a).exp() * (es - ei) / eo
                    * (a * es + lam * (eo - es) * (a / lam).sinh());
                let t2 = (a * (ei * es).sqrt() + lam * (ei - es)) * (a / lam).sinh()
                    + a * es * (a / lam).cosh();
                let t3 = t1 / t2 * (-kappa * r).exp();
                phi += (1.0 + t3) / es * q.val / FOUR_PI / r.max(MOL_TOL);
                continue;
            }
            let cos = xi.dot(q.pos) / (r * q.pos.norm());
            let p = legendre(self.len, cos);
            let mut phij = 0.0;
            for n in 0..self.len {
                phij += (ei - es) / ei * self.m2[qi][n] * kr[n + 1] * p[n]
                    + self.m1[qi][n] / r.powi(n as i32 + 1) * p[n];
            }
            phi += phij * q.val;
        }
        phi * RF_FACTOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_functions_match_nessie() {
        // Reference values dumped from NESSie (legendre(0.3), bessel(0.25)).
        let p = legendre(6, 0.3);
        let want_p = [1.0, 0.3, -0.365, -0.3825, 0.0729375, 0.34538625];
        for (a, b) in p.iter().zip(want_p) {
            assert!((a - b).abs() < 1e-12, "legendre {a} vs {b}");
        }
        let i = spherical_besseli(5, 0.25); // n = -1 … 5
        let want_i = [
            4.1256523995182945,
            1.0104492672326746,
            0.08385533058759989,
            0.0041853001814759015,
            0.00014932695808186772,
            4.145355183603734e-6,
            9.417147213346324e-8,
        ];
        for (a, b) in i.iter().zip(want_i) {
            assert!((a - b).abs() / b.abs() < 1e-9, "besseli {a} vs {b}");
        }
        let k = spherical_besselk(5, 0.25);
        let want_k = [
            4.893349637414207,
            4.893349637414207,
            24.466748187071037,
            298.4943278822666,
            5994.353305832403,
            168140.38689118956,
            6.059048281388656e6,
        ];
        for (a, b) in k.iter().zip(want_k) {
            assert!((a - b).abs() / b.abs() < 1e-9, "besselk {a} vs {b}");
        }
    }
}
