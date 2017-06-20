from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from IPython import embed

import numpy as np
import z3

from sympy2z3.sympy2z3 import sympy2z3

import err

from globalopts import opts as gopts

from polynomial.poly import Poly
import nonlinprog.spec as spec


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

    solver = z3.Solver()
    solver = z3.Solver()
    # Objective and constraints should be polynomials
    # assert(isinstance(obj, Poly))
    sym2Z3_varmap, z3_cons = sympy2z3(cons)
    solver.add(z3_cons)

    #smt_vars = z3.Reals(','.join('x{}'.format(v) for v in nvars))
#     for c in cons:
#         solver.add(poly2z3(c, smt_vars))

    res = solver.check()
    if res == z3.sat:
        model = solver.model()
        #varval_map = {sv: real2float(model[zv]) for sv, zv in sym2Z3_varmap.iteritems()}
        res_x = np.array([real2float(model[sym2Z3_varmap[v]]) for v in Vars])
    elif res == z3.unsat:
        #varval_map = None
        res_x = None
    else:
        raise RuntimeError(solver.reason_unknown())

    #return (res == z3.sat), varval_map
    return spec.OPTRES(0, res_x, 'OK', res == z3.sat)


def real2float(r):
    assert(r.is_real())
    assert(isinstance(gopts.bmc_prec, float))
    # ############### slow!
    #n = float(r.numerator().as_long())
    #d = float(r.denominator().as_long())
    #return n/d
    # ############### optimized, a magnitude faster
    return float(r.as_decimal(gopts.bmc_prec).replace('?', ''))

# def poly2z3(poly, smt_vars):
#     z3_expr = 0
#     for powers, coeff in poly.as_dict():
#         for v, p in zip(smt_vars, powers):
#             z3_expr += v**p
#         z3_expr *= coeff
#     return z3_expr
