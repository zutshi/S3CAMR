//! s3cam_rs — Rust simulation backend for S3CAMR.
//!
//! Replaces the body of `psim.Simulator.simulate` (the batched Δ-step) for the
//! `--sim-engine rust` path. Integrates each sample over one horizon with
//! Dormand-Prince 5(4) (`ode_solvers::Dopri5`) and the scipy-`set_solout`
//! early-stop (the property checker), returning next states + violated flags.
//!
//! Two entry points:
//!   * `simulate_batch_native`  — natively-ported benchmark plants (fast path,
//!     releases the GIL, rayon-parallel over the independent samples).
//!   * `simulate_batch_py`      — black-box fallback that calls the user's
//!     Python `dyn` per eval (correct, slow; keeps the black-box contract).

mod dynamics;
mod integrate;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use dynamics::{Dynamics, IntegratorCfg, Plant};
use integrate::simulate_one;

/// Fast path: a natively-ported benchmark plant selected by module name.
///
/// Parameters
/// ----------
/// x        : (n_samples, n_dims) initial states, row-major
/// delta_t  : the Δ step (horizon) to integrate each sample
/// plant    : plant module name ("vanDerPol" | "brusselator" | "lorenz")
/// lo, hi   : (n_dims,) unsafe-box bounds (property checker's final_cons.l/.h);
///            pass lo>hi (e.g. +inf / -inf) to disable early-stop (never-detect)
/// u        : (n_samples, n_u) per-sample plant inputs (may be n×0; ignored by
///            the native benchmark dynamics, forwarded for generality)
///
/// Returns (x_next: (n_samples, n_dims), violated: (n_samples,) bool).
#[pyfunction]
fn simulate_batch_native<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    delta_t: f64,
    plant: &str,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<bool>>)> {
    let plant_enum = Plant::from_name(plant).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unregistered native plant '{plant}'; known: {:?}",
            Plant::all_names()
        ))
    })?;
    let cfg = plant_enum.cfg();
    let dim = plant_enum.dim();

    let xv = x.as_array();
    let (n, ncols) = (xv.nrows(), xv.ncols());
    if ncols != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "plant '{plant}' expects {dim} dims, got x with {ncols} columns"
        )));
    }

    // Copy inputs into owned buffers so we can drop the GIL for the compute.
    let x_flat: Vec<f64> = xv.iter().copied().collect();
    let lo_v: Vec<f64> = lo.as_array().iter().copied().collect();
    let hi_v: Vec<f64> = hi.as_array().iter().copied().collect();
    let uv = u.as_array();
    let n_u = uv.ncols();
    let u_flat: Vec<f64> = uv.iter().copied().collect();

    let mut out = vec![0.0f64; n * dim];
    let mut violated = vec![false; n];

    py.allow_threads(|| {
        run_batch_parallel(
            &plant_enum, &x_flat, &u_flat, n_u, dim, delta_t, &lo_v, &hi_v, cfg, &mut out,
            &mut violated,
        );
    });

    let arr = Array2::from_shape_vec((n, dim), out).unwrap();
    Ok((
        arr.to_pyarray(py).into(),
        PyArray1::from_vec(py, violated).into(),
    ))
}

/// Rayon-parallel over samples (samples are independent — bug-fix #2/#4/M4).
#[allow(clippy::too_many_arguments)]
fn run_batch_parallel<D: Dynamics + Sync>(
    dynamics: &D,
    x_flat: &[f64],
    u_flat: &[f64],
    n_u: usize,
    dim: usize,
    delta_t: f64,
    lo: &[f64],
    hi: &[f64],
    cfg: IntegratorCfg,
    out: &mut [f64],
    violated: &mut [bool],
) {
    out.par_chunks_mut(dim)
        .zip(violated.par_iter_mut())
        .enumerate()
        .for_each(|(i, (out_row, vio))| {
            let x0 = &x_flat[i * dim..i * dim + dim];
            let u = if n_u > 0 {
                &u_flat[i * n_u..i * n_u + n_u]
            } else {
                &[][..]
            };
            *vio = simulate_one(dynamics, x0, u, delta_t, lo, hi, cfg, out_row);
        });
}

/// Black-box fallback: a `Dynamics` that calls the user's Python `dyn` per eval.
/// Holds the GIL (the whole batch loop stays on the Python thread), so this is
/// the correct-but-slow path for plants not natively ported.
struct PyDynamics {
    dyn_fn: Py<PyAny>,
    dim: usize,
    takes_u: bool,
}

impl Dynamics for PyDynamics {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, t: f64, x: &[f64], u: &[f64], out: &mut [f64]) {
        Python::with_gil(|py| {
            let x_py = PyList::new(py, x.iter().copied())
                .expect("failed to build x list for Python dyn");
            let res = if self.takes_u {
                let u_py = PyList::new(py, u.iter().copied())
                    .expect("failed to build u list for Python dyn");
                self.dyn_fn.call1(py, (t, x_py, u_py))
            } else {
                self.dyn_fn.call1(py, (t, x_py))
            };
            let res = res.expect("Python dyn(t, X[, u]) call failed");
            // dyn returns an array-like of length dim; extract elementwise.
            let seq = res.bind(py);
            for i in 0..self.dim {
                out[i] = seq
                    .get_item(i)
                    .and_then(|v| v.extract::<f64>())
                    .expect("Python dyn returned a non-float / short result");
            }
        });
    }
}

/// Fallback path: integrate the batch calling a Python `dyn` callable per eval.
/// Runs serially under the GIL (no `allow_threads`, no rayon).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn simulate_batch_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    delta_t: f64,
    dyn_fn: Py<PyAny>,
    dim: usize,
    takes_u: bool,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray2<'py, f64>,
    rtol: f64,
    atol: f64,
    max_step: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<bool>>)> {
    let xv = x.as_array();
    let (n, ncols) = (xv.nrows(), xv.ncols());
    if ncols != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fallback plant expects {dim} dims, got x with {ncols} columns"
        )));
    }
    let lo_v: Vec<f64> = lo.as_array().iter().copied().collect();
    let hi_v: Vec<f64> = hi.as_array().iter().copied().collect();
    let uv = u.as_array();
    let n_u = uv.ncols();

    let cfg = IntegratorCfg {
        rtol,
        atol,
        max_step: if max_step > 0.0 { Some(max_step) } else { None },
    };

    let dynamics = PyDynamics {
        dyn_fn,
        dim,
        takes_u,
    };

    let mut out = vec![0.0f64; n * dim];
    let mut violated = vec![false; n];

    for i in 0..n {
        let x0: Vec<f64> = xv.row(i).iter().copied().collect();
        let u_row: Vec<f64> = if n_u > 0 {
            uv.row(i).iter().copied().collect()
        } else {
            Vec::new()
        };
        let out_row = &mut out[i * dim..i * dim + dim];
        violated[i] = simulate_one(
            &dynamics, &x0, &u_row, delta_t, &lo_v, &hi_v, cfg, out_row,
        );
    }

    let arr = Array2::from_shape_vec((n, dim), out).unwrap();
    Ok((
        arr.to_pyarray(py).into(),
        PyArray1::from_vec(py, violated).into(),
    ))
}

/// Names of the natively-ported plants (so Python can decide native vs fallback).
#[pyfunction]
fn native_plants() -> Vec<String> {
    Plant::all_names().iter().map(|s| s.to_string()).collect()
}

/// Evaluate a native plant's `dyn` (dX/dt) at one point. Exposed purely for the
/// plant-parity gate test (native Rust `dyn` == Python `dyn` on a grid).
#[pyfunction]
fn eval_dyn(plant: &str, t: f64, x: Vec<f64>, u: Vec<f64>) -> PyResult<Vec<f64>> {
    let plant_enum = Plant::from_name(plant).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unregistered native plant '{plant}'"))
    })?;
    let mut out = vec![0.0f64; plant_enum.dim()];
    plant_enum.eval(t, &x, &u, &mut out);
    Ok(out)
}

#[pymodule]
fn s3cam_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(native_plants, m)?)?;
    m.add_function(wrap_pyfunction!(eval_dyn, m)?)?;
    Ok(())
}
