//! Plant dynamics: a `Dynamics` trait + native Rust impls for the benchmark
//! plants, ported EXACTLY from the Python `dyn` functions in `examples/*`.
//!
//! Bug-fix #1 from bench/RUST_PLAN.md is folded in by construction: `eval`
//! writes the derivative into `&mut [f64]` (no per-eval `X.copy()` allocation).

/// A plant's right-hand side dX/dt = f(t, x, u).
///
/// `out` is a caller-owned scratch buffer of length `self.dim()`; `eval` writes
/// the derivative into it (zero allocation in the integration inner loop).
pub trait Dynamics {
    fn eval(&self, t: f64, x: &[f64], u: &[f64], out: &mut [f64]);
    fn dim(&self) -> usize;
}

/// Per-plant integrator configuration mirroring the Python `SIM.__init__`
/// scipy `ode('dopri5')` settings.
#[derive(Clone, Copy)]
pub struct IntegratorCfg {
    pub rtol: f64,
    pub atol: f64,
    /// Maximum step size. `None` => unbounded (h_max = horizon), matching the
    /// plants that do not pass `max_step` to scipy.
    pub max_step: Option<f64>,
}

/// The natively-ported benchmark plants, keyed by their Python module name
/// (the `.tst` plant_path stem: `vanDerPol.py` -> "vanDerPol", etc.).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plant {
    VanDerPol,
    Brusselator,
    Lorenz,
}

impl Plant {
    pub fn from_name(name: &str) -> Option<Plant> {
        match name {
            "vanDerPol" => Some(Plant::VanDerPol),
            "brusselator" => Some(Plant::Brusselator),
            "lorenz" => Some(Plant::Lorenz),
            _ => None,
        }
    }

    pub fn all_names() -> &'static [&'static str] {
        &["vanDerPol", "brusselator", "lorenz"]
    }

    pub fn cfg(&self) -> IntegratorCfg {
        match self {
            // examples/vdp/vanDerPol.py: rtol=1e-6, max_step=1e-2 (property_checker on)
            Plant::VanDerPol => IntegratorCfg {
                rtol: 1e-6,
                atol: 1e-12,
                max_step: Some(1e-2),
            },
            // examples/brusselator/brusselator.py: rtol=1e-6, no max_step
            Plant::Brusselator => IntegratorCfg {
                rtol: 1e-6,
                atol: 1e-12,
                max_step: None,
            },
            // examples/lorenz/lorenz.py: rtol=1e-6, no max_step
            Plant::Lorenz => IntegratorCfg {
                rtol: 1e-6,
                atol: 1e-12,
                max_step: None,
            },
        }
    }
}

impl Dynamics for Plant {
    #[inline]
    fn dim(&self) -> usize {
        match self {
            Plant::VanDerPol => 2,
            Plant::Brusselator => 2,
            Plant::Lorenz => 4,
        }
    }

    #[inline]
    fn eval(&self, _t: f64, x: &[f64], _u: &[f64], out: &mut [f64]) {
        match self {
            // vanDerPol.py:
            //   X[0], X[1] = (X[1], 5.0 * (1 - X[0]**2) * X[1] - X[0])
            Plant::VanDerPol => {
                out[0] = x[1];
                out[1] = 5.0 * (1.0 - x[0] * x[0]) * x[1] - x[0];
            }
            // brusselator.py:  a = 1.0, b = 2.5
            //   Y[0] = 1 + a*(X[0]**2)*X[1] - (b+1)*X[0]
            //   Y[1] = b*X[0] - a*(X[0]**2)*X[1]
            Plant::Brusselator => {
                let a = 1.0;
                let b = 2.5;
                let x0sq = x[0] * x[0];
                out[0] = 1.0 + a * x0sq * x[1] - (b + 1.0) * x[0];
                out[1] = b * x[0] - a * x0sq * x[1];
            }
            // lorenz.py:  x, y, z, t = X
            //   Y[0] = 10*(y - x)
            //   Y[1] = x*(28 - z) - y
            //   Y[2] = x*y - 2.6667*z
            //   Y[3] = 1
            Plant::Lorenz => {
                let (px, py, pz) = (x[0], x[1], x[2]);
                out[0] = 10.0 * (py - px);
                out[1] = px * (28.0 - pz) - py;
                out[2] = px * py - 2.6667 * pz;
                out[3] = 1.0;
            }
        }
    }
}
