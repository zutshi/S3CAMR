# Refer example: /home/zutshi/software/pyipopt/examples/hs071.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy as sp
import sympy as sym
import collections
import operator as op
import functools as ft

import pyipopt

import utils as U

import nonlinprog.spec as spec
import settings

from IPython import embed

CONS = collections.namedtuple('constraint', ('expr', 'lb', 'ub'))

debug = False


def jac(Vars, exprs):
    e_mat = sym.Matrix(exprs)
    j = e_mat.jacobian(Vars)
    js = sp.sparse.coo_matrix(j)
    return js


def grad(Vars, expr):
    return tuple(sym.diff(expr, v) for v in Vars)


def eval_jac_cons((jac_row, jac_col, jac_f), x, flag, user_data=None):
    if flag:
        ret = jac_row, jac_col
    else:
        ret = np.array(jac_f(*x))
    if debug:
        print('eval_jac_cons')
        print(flag, ret)
    return ret


def eval_grad_obj(grad_f, x, user_data=None):
    ret = np.array(grad_f(*x))
    if debug:
        print('eval_grad_obj')
        print(ret)
    return ret


def list2array_wrap(f, x):
    ret = np.array(f(*x))
    if debug:
        print('eval_g')
        print('cons:', ret)
    return ret


def eval_expr(f, x):
    y = f(*x)
    if debug:
        print('eval_f')
        print(y)
    return y


def nlinprog(obj, cons, Vars, x0=None):
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
    pyipopt.set_loglevel(0) # increasing verbosity -> 0, 1, 2
    cons = list(cons)

    if x0 is None:
        x0 = np.zeros(len(Vars))

    return cons_opt(obj, cons, Vars, x0)


# Wrong
def uncons_opt(obj, cons, Vars, x0):
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
    raise NotImplementedError

    assert(obj == 0)
    for c in cons:
        assert(isinstance(c, sym.LessThan))
        assert(c.rhs == 0)
        obj += c.lhs**2

    eval_f = ft.partial(eval_expr, sym.lambdify(Vars, obj))
    eval_grad_f = ft.partial(eval_grad_obj, sym.lambdify(Vars, grad(Vars, obj)))
    eval_hessian_f = ft.partial(eval_expr, sym.lambdify(Vars, sym.hessian(obj, Vars)))

    # Return codes in IpReturnCodes_inc.h
    res_x, mL, mU, Lambda, res_obj, status = pyipopt.fmin_unconstrained(
            eval_f,
            x0,
            fprime=eval_grad_f,
            fhess=eval_hessian_f,
            )

    return spec.OPTRES(res_obj, res_x, 'OK', res_obj <= 0)


def cons_opt(obj, cons, Vars, x0):
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
    nvars = len(Vars)

    x_L = np.array((pyipopt.NLP_LOWER_BOUND_INF,)*nvars)
    x_U = np.array((pyipopt.NLP_UPPER_BOUND_INF,)*nvars)
    #x_L = -20.*np.ones(nvars)
    #x_U = 20.*np.ones(nvars)

    g_L, g_U, g = [], [], []
    for gc in group_cons_by_ub_lb(cons):
        g_L.append(gc.lb)
        g_U.append(gc.ub)
        g.append(gc.expr)
    ncon = len(g)

    g_L, g_U = np.array(g_L), np.array(g_U)
    eval_g = ft.partial(list2array_wrap, sym.lambdify(Vars, g))

    js = jac(Vars, g)
    jrow, jcol, jdata = np.asarray(js.row, dtype=int), np.asarray(js.col, dtype=int), js.data
    eval_jac_g = ft.partial(eval_jac_cons, (jrow, jcol, sym.lambdify(Vars, jdata.tolist())))

    eval_f = ft.partial(eval_expr, sym.lambdify(Vars, obj))
    eval_grad_f = ft.partial(eval_grad_obj, sym.lambdify(Vars, grad(Vars, obj)))
    #eval_hessian_f = ft.partial(eval_expr, sym.lambdify(Vars, sym.hessian(obj, Vars)))

    nnzj = js.nnz
    nnzh = 0

    if debug:
        for gi, lb, ub in zip(g, g_L, g_U):
            print('{} \in [{}, {}]'.format(gi, lb, ub))

    nlp = pyipopt.create(nvars, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
    # Verbosity level \in [0, 12]
    nlp.int_option('print_level', 5)
    res_x, zl, zu, constraint_multipliers, res_obj, status = nlp.solve(x0)
    nlp.close()

    if debug:

        def print_variable(variable_name, value):
            for i in xrange(len(value)):
                print(variable_name + "["+str(i)+"] =", value[i])

        print()
        print("Solution of the primal variables, x")
        print_variable("x", res_x)
        print()
        print("Solution of the bound multipliers, z_L and z_U")
        print_variable("z_L", zl)
        print_variable("z_U", zu)
        print()
        print("Solution of the constraint multipliers, lambda")
        print_variable("lambda", constraint_multipliers)
        print()
        print("Objective value")
        print("f(x*) =", res_obj)

    # Return codes in IpReturnCodes_inc.h
    print('status:', status)
    return spec.OPTRES(res_obj, res_x, 'OK', status in (0, 1))


def group_cons_by_ub_lb(cons):
    """
    Groups constraints as follows:
        a*x0 + bx1 <= 3
        a*x0 + bx1 >= 2

    Parameters
    ----------
    cons : iterable of constraints

    Returns
    -------
    ((c0 <= a0, c0 >= b0), (c0 <= a0, c0 >= b0), ...)

    Notes
    ------
    """
    # Brute force search, extremely inefficient
    cons = list(cons)
    len_cons = len(cons)
    grouped_cons = []
    #cons_set = set(cons)
#     for c in cons:
#         print(c)
#     print('='*20)
    while len(cons) > 0:
        ci = cons.pop()
#         print(ci)
        assert(ci.rhs.is_Number and ci.rhs == 0)
        found = False
        for cj in cons:
            assert(cj.rhs.is_Number and cj.rhs == 0)
            ci_lhs, cj_lhs = ci.lhs, cj.lhs
            if (ci_lhs + cj_lhs).is_Number:
                di, dj = ci_lhs.as_coefficients_dict(), cj_lhs.as_coefficients_dict()
                try:
                    b1 = di.pop(1)
                except KeyError:
                    b1 = 0
                try:
                    b2 = dj.pop(1)
                except KeyError:
                    b2 = 0
                if -b1 >= -b2:
                    # swap
                    b1, b2 = b2, b1
                    di, dj = dj, di
                else:
                    pass
                lb, ub = (float(b1), float(-b2))
                expr = expr_from_dict(dj)
                c = CONS(expr=expr, lb=lb, ub=ub)
                grouped_cons.append(c)
                found = True
                break
        assert(found)
        cons.remove(cj)

    # every cons has a pair
    assert(2*len(grouped_cons) == len_cons)
    return grouped_cons


def expr_from_dict(d):
    return reduce(op.add, (expr*coeff for expr, coeff in d.iteritems()))
