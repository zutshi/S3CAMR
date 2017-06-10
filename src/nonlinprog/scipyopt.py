from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.optimize as spopt
import sympy as sym
import functools as ft

import nonlinprog.spec as spec
import settings

from utils import print_function

from IPython import embed


METHOD = 'COBYLA'#SLSQP' #'COBYLA'
TOL = None#1e-5


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
    cons = list(cons)

    # incase Vars is an unordered object, freeze the order
    all_vars = tuple(Vars)
    # constraints are encoded as g(x) >= 0, hence, reverse the sign
    #cons_f = tuple(sym.lambdify(all_vars, -c) for c in cons)

    def debugf(f, e, x):
        y = f(*x)
        #print(x, ':', -e.args[0] >= 0, ':', f(*x))
        print(-e.args[0] >= 0, ':', f(*x))
        return y

    # The below constraint encoding assumes all cons
    # (constraint exprs) are of the form f(x) <= 0
    lambdafied = []
    for c in cons:
        assert(isinstance(c, sym.LessThan))
        assert(c.args[1] == 0)
        lambdafied.append(sym.lambdify(all_vars, -c.args[0], str('numpy')))

    # more concise but can not put in asserts
    #lambdafied = tuple(sym.lambdify(all_vars, -c.args[0]) for c in cons)
    #cons_f = tuple({'type': 'ineq', 'fun': lambda x: sym.lambdify(all_vars, -c)(*x)} for c in cons)

    # Must pass l as an arguement, else late binding will make sure
    # that all functions are the same: the last one
    cons_f = tuple({'type': 'ineq', 'fun': lambda x, l=l: l(*x)} for l in lambdafied)
    #cons_f = tuple({'type': 'ineq', 'fun': ft.partial(debugf, l, e)} for l, e in zip(lambdafied, cons))
    #cons_f2 = [ft.partial(debugf, l, e) for l, e in zip(lambdafied, cons)]

    #cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},

    bounds = None#[(-100, 100) for v in Vars]
    x0 = np.zeros(len(all_vars))

# Refer:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# Signature:
# scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
#                          bounds=None, constraints=(), tol=None, callback=None, options=None)

    res = spopt.minimize(obj, x0, method=METHOD, jac=None,
                         hess=None, hessp=None, bounds=bounds,
                         constraints=cons_f,
                         tol=TOL, callback=None,
                         options={'disp': True, 'maxiter': 10000})


#     res_ = spopt.fmin_slsqp(obj_f, x0, ieqcons=cons_f2, bounds=(),
#                             iter=1000, acc=1e-06, iprint=2, disp=True,
#                             full_output=True)

#     embed()

    print(res.message)
    #varval_map = {var: val for var, val in zip(all_vars, res.x)}
    #print(varval_map)
    return spec.OPTRES(res.fun, res.x, res.status, res.success)
    #return res.success, varval_map
    #return res.fun, res.x, res.status, res.success
