from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.linalg as linalg
import sympy as sym

from globalopts import opts as gopts
import nonlinprog.z3opt as z3opt

import constraints as cons

import settings
import linprog.linprog as linprog

from IPython import embed

import err


class SolverFaliure(Exception):
    pass


def part_constraints(partition_trace, vars_grouped_by_model, next_vars_grouped_by_model):
    """constraints due to partitions of the pwa model

    Parameters
    ----------
    pwa_trace : pwa trace form the bmc module

    Returns
    -------
    cons corresponding to Cx - d <= 0

    Notes
    ------
    """
    C, d = [], []
    # for each sub_model
    for p in partition_trace:
        C.append(p.C)
        d.append(p.d)

    A = linalg.block_diag(*C)
    b = np.hstack(d)

    assert(A.shape[0] == b.shape[0])

    part_cons = np.dot(A, next_var) + b
    return 


def dyn_constraints(models, vars_grouped_by_model, next_vars_grouped_by_model):
    """constraints due to dynamics of the pwa model

    Parameters
    ----------
    pwa_trace : pwa trace form the bmc module

    Returns
    -------
    polynomial constraints corresponding to x' \in p(x) + [error.l, error.h]

    Notes
    ------
    """

    # for each sub_model
    # x' >= p(x) + e.l
    # x' <= p(x) + e.h

    dyn_cons = []

    for m, Vars, next_vars in zip(models, vars_grouped_by_model, next_vars_grouped_by_model):
        for p, el, eh, x, x_ in zip(m.poly, m.error.l, m.error.h, Vars, next_vars):
            assert(len(p.vars) == ndimx)
            old2new_var_map = {v: v_ for v, v_ in zip(p.vars, Vars)}
            poly = p.subs(old2new_var_map)
            # xi' <= p(x) + eli
            cons_ub = x_ - (poly + el)
            # xi' >= p(x) + ehi
            cons_lb = poly + eh - x_
            dyn_cons.append(cons_ub)
            dyn_cons.append(cons_lb)

    return dyn_cons


def prop_constraints(num_dims, prop, num_partitions):
    raise NotImplementedError
    """constraints due to the initial set, final set and ci/pi

    Parameters
    ----------
    prop :

    Returns
    -------

    Notes
    ------
    """
    #trace_len = len(pwa_trace)
    iA, ib = prop.init_cons.poly()
    fA, fb = prop.final_cons.poly()

    assert(iA.shape == fA.shape)

    # find the dim(x) using any submodel's b vector
    # This is the dim of total variables: dimX = dimW
    #dimX = pwa_trace[0].m.b.size

    dimX = num_dims.x + num_dims.pi
    dimW = num_dims.pi

    # pad iA, ib, fA, fb with 0's to accomodate w/pi
    # Remember: the state vector for pwa is: [x]
    #                                        [w]
    # Of course this much code is not required, and the intended
    # operation can be done much succintly. But this is much easier to
    # understand IMO.

    # append 0s for wi: A -> [A 0 .. 0]
    padding_scheme = ((0, 0), (0, dimW))

    iA = np.pad(iA, padding_scheme, 'constant')
    fA = np.pad(fA, padding_scheme, 'constant')

    # number of rows and coloumns of iA/fA
    nr, nc = iA.shape
    num_cons = 2 * nr

    # Add 1, because trace captures transitions,
    # and num_states = num_trans + 1
    #A = np.zeros((num_cons, dimX * (trace_len)))
    A = np.zeros((num_cons, dimX * (num_partitions)))
    A[0:nr, 0:nc] = iA
    A[-nr:, -nc:] = fA
    b = np.hstack((ib, fb))
    #print(A)
    #print(b)
    # Each constraint expression in A has a value in b
    assert(A.shape[0] == b.size)
    return A, b


def truncate(*args):
    raise NotImplementedError
    assert(isinstance(gopts.bmc_prec, int))
    prec = gopts.bmc_prec

    # Round off to the same amount as the bmc query
    def arr2str(n): return '{n:0.{p}f}'.format(n=n, p=prec)
    trunc_array = np.vectorize(lambda x: np.float(arr2str(x)))

    return (trunc_array(X) for X in args)


def pwatrace2cons(pwa_trace, num_dims, prop):
    # find the dim(x) using any submodel's b vector
    m = models[0]
    ndimx = m.error.l.size

    nvars = len(models) * ndimx
    all_vars = sym.var(','.join(('x{}'.format(i) for i in range(nvars))))
    all_next_vars = sym.var(','.join(('x{}_'.format(i) for i in range(nvars))))

    vars_grouped_by_model = zip(*[all_vars[i::ndimx] for i in range(ndimx)])
    next_vars_grouped_by_model = zip(*[all_next_vars[i::ndimx] for i in range(ndimx)])

    part_cons = part_constraints(pwa_trace.partitions, vars_grouped_by_model, next_vars_grouped_by_model)
    dyn_cons = dyn_constraints(pwa_trace.models, vars_grouped_by_model, next_vars_grouped_by_model)

    print(dyn_cons)
    exit()

    prop_cons = prop_constraints(num_dims, prop, len(pwa_trace.partitions))

    all_cons = part_cons + dyn_cons + prop_cons
    return all_cons


def lpsoln2x(x, trace_len):
    raise NotImplementedError
    '''converts solution of lp: a vector to a concrete trace: numpy
    array'''
    x = np.array(x)
    assert(x.ndim == 1)
    num_vars = len(x)//trace_len
    return np.reshape(x, (trace_len, num_vars))


#TODO: getting lpsolver every time a linprog is executed!
#@memoize
def lpfun(solver):
    raise NotImplementedError
    return linprog.factory(solver)


def feasible(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    cons = pwatrace2cons(pwa_trace, num_dims, prop)

    num_opt_vars = A_ub.shape[1]
    #nvars = num_dims.x + num_dims.pi

    A_ub, b_ub = truncate(A_ub, b_ub)
    obj = 0

    res = z3opt.polyprog(obj, cons, nvars)

    raise NotImplementedError
    #return lpsoln2x(res.x, len(pwa_trace)) if res.success else None


def overapprox_x0(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    raise NotImplementedError
