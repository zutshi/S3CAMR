from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from IPython import embed

import z3

from sympy2z3.sympy2z3 import sympy2z3

import err


from polynomial.poly import Poly
import nonlinprog.spec as spec


def nonlinprog(obj, cons, nvars):

    err.warn('ignoring objective, will check only feasibilit for now.')
    return polyprog(obj, cons, nvars)


def polyprog(obj, cons, Vars):
    if obj != 0:
        raise NotImplementedError

    nvars = len(Vars)
    solver = z3.Solver()
    # Objective and constraints should be polynomials
    # assert(isinstance(obj, Poly))
    z3_vars, z3_cons = sympy2z3(cons)
    solver.add(z3_cons)

    #smt_vars = z3.Reals(','.join('x{}'.format(v) for v in nvars))
#     for c in cons:
#         solver.add(poly2z3(c, smt_vars))

    res = solver.check()
    if res == z3.sat:
        #TODa
        model = solver.model
    elif res == z3.unsat:
        model = None
    else:
        raise RuntimeError(solver.reason_unknown())

    embed()
    return (res == z3.sat), model


# def poly2z3(poly, smt_vars):
#     z3_expr = 0
#     for powers, coeff in poly.as_dict():
#         for v, p in zip(smt_vars, powers):
#             z3_expr += v**p
#         z3_expr *= coeff
#     return z3_expr
