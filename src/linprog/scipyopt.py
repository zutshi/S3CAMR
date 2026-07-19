import numpy as np
import scipy.optimize as spopt

import linprog.spec as spec
import settings


def linprog(obj, A_ub, b_ub):
    """linprog

    DECOMMISSIONED. scipy.optimize.linprog is no longer used as an LP backend:
    its input validation rejects +inf in b_ub (produced by unbounded/half-space
    unsafe sets), and it hardcoded the deprecated method='simplex'. Use the
    'highs' (native HiGHS) or 'glpk' backends instead, both of which accept inf
    natively. The original implementation is preserved below for reference.

    Parameters
    ----------
    c : obj as a list
    A_ub : as an numpy array
    b_ub : as a numpy array

    Returns
    -------
    s

    Notes
    ------
    """
    raise NotImplementedError(
        "scipy LP backend is decommissioned; use --opt-engine highs or glpk")

    # --- preserved (unreachable) original implementation ---
    # columns of A_ub
    num_opt_vars = A_ub.shape[1]
    bounds = [(-np.inf, np.inf)] * num_opt_vars

    disp_opt = True if settings.debug else False

    res = spopt.linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        method='simplex', options={'disp': disp_opt})

    return spec.OPTRES(res.fun, res.x, res.status, res.success)
