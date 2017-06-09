from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.optimize as spopt
import sympy as sym

import nonlinprog.spec as spec
import settings

from IPython import embed

METHOD = 'SLSQP'


def nlinprog(obj, cons, Vars):
    """nlinprog

    Parameters
    ----------
    obj :
    cons :

    Returns
    -------

    Notes
    ------
    """
    # ignoring objective, will check only feasibilit for now
    if obj != 0:
        raise NotImplementedError
    obj_f = return_zero

    # incase Vars is an unordered object, freeze the order
    all_vars = tuple(Vars)
    # constraints are encoded as g(x) >= 0, hence, reverse the sign
    #cons_f = tuple(sym.lambdify(all_vars, -c) for c in cons)

    # The below constraint encoding assumes all cons
    # (constraint exprs) are of the form f(x) <= 0
    lambdafied = []
    for c in cons:
        assert(isinstance(c, sym.LessThan))
        assert(c.args[1] == 0)
        lambdafied.append(sym.lambdify(all_vars, -c.args[0]))

    # more concise but can not put in asserts
    #lambdafied = tuple(sym.lambdify(all_vars, -c.args[0]) for c in cons)
    #cons_f = tuple({'type': 'ineq', 'fun': lambda x: sym.lambdify(all_vars, -c)(*x)} for c in cons)

    # Must pass l as an arguement, else late binding will make sure
    # that all functions are the same: the last one
    cons_f = tuple({'type': 'ineq', 'fun': lambda x, l=l: l(*x)} for l in lambdafied)

    #cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},

    bounds = None
    x0 = np.zeros(len(all_vars))

# Refer:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# Signature:
# scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
#                          bounds=None, constraints=(), tol=None, callback=None, options=None)

    res = spopt.minimize(obj_f, x0, method=METHOD, jac=None,
                         hess=None, hessp=None, bounds=bounds,
                         constraints=cons_f,
                         tol=None, callback=None, options=None)

    print(res.message)
    varval_map = {var: val for var, val in zip(all_vars, res.x)}
    print(varval_map)
    return res.success, varval_map
    #return res.fun, res.x, res.status, res.success


def return_zero(x):
    return 0.0
