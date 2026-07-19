import numpy as np
import highspy

import linprog.spec as spec
import settings


def linprog(obj, A_ub, b_ub):
    """LP feasibility/optimization via the native HiGHS API (highspy).

    Minimize  obj . x   subject to   A_ub x <= b_ub,  x free.

    Unlike scipy.optimize.linprog, the native HiGHS API accepts +inf in the
    row bounds (its native "no upper bound" marker, kHighsInf), so the vacuous
    rows produced by unbounded (half-space) unsafe sets need no pre-filtering.

    Parameters
    ----------
    obj  : objective coefficients (1-D, length = #vars)
    A_ub : constraint matrix (2-D)
    b_ub : constraint upper bounds (1-D)

    Returns
    -------
    spec.OPTRES(fun, x, status, success)  -- follows the scipy convention.
    """
    obj = np.asarray(obj, dtype=float)
    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)

    num_rows, num_vars = A_ub.shape
    INF = highspy.kHighsInf

    h = highspy.Highs()
    h.setOptionValue('output_flag', bool(settings.debug))

    # Free variables (lb = -inf, ub = +inf) with the given linear objective.
    lb = np.full(num_vars, -INF)
    ub = np.full(num_vars, INF)
    h.addVars(num_vars, lb, ub)
    h.changeColsCost(num_vars,
                     np.arange(num_vars, dtype=np.int32),
                     obj)

    # Each constraint row:  -inf <= A_ub[i] . x <= b_ub[i]
    # (b_ub[i] may be +inf, which HiGHS treats as "no upper bound").
    idx = np.arange(num_vars, dtype=np.int32)
    for i in range(num_rows):
        h.addRow(-INF, float(b_ub[i]), num_vars, idx, A_ub[i])

    h.run()

    status = h.getModelStatus()
    success = (status == highspy.HighsModelStatus.kOptimal)
    sol = h.getSolution()
    x = np.array(sol.col_value) if success else None
    fun = h.getObjectiveValue() if success else None

    return spec.OPTRES(fun, x, str(status), success)
