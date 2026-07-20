//! Single-sample horizon integration with early-stop on unsafe-set entry.
//!
//! Uses `ode_solvers::Dopri5` (Dormand-Prince 5(4), the same method scipy's
//! `dopri5` uses). The `System::solout` hook is scipy's `set_solout` early-stop
//! (S3CAMR returns -1 to stop; here we return `true`) — this is the property
//! checker: integration halts as soon as the trajectory enters the unsafe box.

use ode_solvers::dop_shared::{OutputType, System};
use ode_solvers::{DVector, Dopri5};

use crate::dynamics::{Dynamics, IntegratorCfg};

type State = DVector<f64>;

/// Membership test `x in [lo, hi]` — a single short-circuiting loop, no
/// allocation (bug-fix #4 from the plan: no intermediate boolean arrays +
/// reduce). `+/-inf` bounds are handled naturally by the comparisons, matching
/// `constraints.IntervalCons.__contains__` on the unsafe (error) set.
#[inline]
fn in_box(x: &[f64], lo: &[f64], hi: &[f64]) -> bool {
    for i in 0..x.len() {
        if !(x[i] >= lo[i] && x[i] <= hi[i]) {
            return false;
        }
    }
    true
}

/// The ODE problem handed to Dopri5: dynamics + the unsafe box for solout.
struct Prob<'a, D: Dynamics> {
    dynamics: &'a D,
    u: &'a [f64],
    lo: &'a [f64],
    hi: &'a [f64],
}

impl<D: Dynamics> System<f64, State> for Prob<'_, D> {
    #[inline]
    fn system(&self, x: f64, y: &State, dy: &mut State) {
        // Zero-alloc RHS: write straight into dy (bug-fix #1).
        self.dynamics.eval(x, y.as_slice(), self.u, dy.as_mut_slice());
    }

    #[inline]
    fn solout(&mut self, _x: f64, y: &State, _dy: &State) -> bool {
        // Stop integration the moment the trajectory enters the unsafe set.
        in_box(y.as_slice(), self.lo, self.hi)
    }
}

/// Integrate one sample over `[0, delta_t]` (matching the Python plants, which
/// `set_initial_value(X0, t=0.0)` then `integrate(delta_t)`), with early-stop.
///
/// Returns the reached state (the early-stop point if the unsafe set was
/// entered, else the state at `delta_t`) and whether it lies in the unsafe box
/// (== Python's `property_checker.check(Tf, X_)`).
pub fn simulate_one<D: Dynamics>(
    dynamics: &D,
    x0: &[f64],
    u: &[f64],
    delta_t: f64,
    lo: &[f64],
    hi: &[f64],
    cfg: IntegratorCfg,
    out: &mut [f64],
) -> bool {
    let dim = x0.len();
    let y0 = State::from_row_slice(x0);
    let h_max = cfg.max_step.unwrap_or(delta_t);

    let prob = Prob {
        dynamics,
        u,
        lo,
        hi,
    };

    // from_param with the crate's own default controller constants (safety 0.9,
    // beta 0.04, fac_min 0.2, fac_max 10.0) so behaviour == Dopri5::new(), but
    // with a configurable h_max (== scipy's max_step for vdp).
    let mut stepper = Dopri5::from_param(
        prob,
        0.0,      // x
        delta_t,  // x_end
        delta_t,  // dx (unused: Sparse output)
        y0.clone(),
        cfg.rtol,
        cfg.atol,
        0.9,   // safety_factor
        0.04,  // beta
        0.2,   // fac_min
        10.0,  // fac_max
        h_max,
        0.0,       // h = 0 => auto initial step
        100_000,   // n_max
        1000,      // n_stiff
        OutputType::Sparse,
    );

    // Solout is called at each accepted step; on error (stiffness / max steps)
    // we still use whatever was integrated, mirroring scipy returning its last
    // state rather than aborting the whole falsification run.
    let _ = stepper.integrate();

    let reached: &State = stepper.y_out().last().unwrap_or(&y0);
    let rs = reached.as_slice();
    out[..dim].copy_from_slice(&rs[..dim]);

    in_box(rs, lo, hi)
}
