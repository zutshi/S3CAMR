from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.linalg as linalg
import sympy as sym
import itertools as it

from globalopts import opts as gopts
import nonlinprog.z3opt as z3opt

import settings
import linprog.linprog as linprog

from IPython import embed

import err
import utils as U


class SolverFaliure(Exception):
    pass


def part_constraints(partition_trace):
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
    C_, d_ = [], []
    # for each sub_model
    for p in partition_trace:
        C_.append(p.C)
        d_.append(p.d)

    C = linalg.block_diag(*C_)
    d = np.hstack(d_)

    assert(C.shape[0] == d.shape[0])

    return C, d



def dyn_constraints(models, vars_grouped_by_states):
    """constraints due to dynamics of the pwa model

    Parameters
    ----------
    models : list of pwa models
    vars_grouped_by_states: variables for state vectors grouped
    together. e.g. for a 2dim/state vector (x1,x2):

        k=0 (init)     k=1       k=2
        [(v0, v1),  (v2, v3), (v4, v5), ...]

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

    for m, (Vars, next_vars) in zip(models, U.pairwise(vars_grouped_by_states)):
        for p, el, eh, x, x_ in zip(m.poly, m.error.l, m.error.h, Vars, next_vars):
            p.truncate_coeffs(gopts.bmc_prec)
            #assert(len(p.vars) == ndimx)
            old2new_var_map = {v: v_ for v, v_ in zip(p.vars, Vars)}
            tpoly = p.subs_vars(old2new_var_map)
            # xi' <= p(x) + ehi
            cons_ub = x_ - (tpoly.as_expr() + eh)
            # xi' >= p(x) + eli
            cons_lb = tpoly.as_expr() + el - x_
            dyn_cons.append(cons_ub <= 0)
            dyn_cons.append(cons_lb <= 0)

    return dyn_cons


def param_constraints(num_dims, prop, num_partitions):
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
    assert(isinstance(gopts.bmc_prec, int))
    prec = gopts.bmc_prec

    # Round off to the same amount as the bmc query
    def arr2str(n): return '{n:0.{p}f}'.format(n=n, p=prec)
    trunc_array = np.vectorize(lambda x: np.float(arr2str(x)))

    return (trunc_array(X) for X in args)


def pwatrace2cons(pwa_trace, num_dims, prop):

    # find the dim(x) using any submodel's b vector
    m = pwa_trace.models[0]
    ndimx = m.error.l.size

    nvars = (len(pwa_trace.models)+1) * ndimx
    all_vars = sym.var(','.join(('x{}'.format(i) for i in range(nvars))))
    vars_grouped_by_state = zip(*[all_vars[i::ndimx] for i in range(ndimx)])

    C, d = part_constraints(pwa_trace.partitions)
    dyn_cons = dyn_constraints(pwa_trace.models, vars_grouped_by_state)

    P, q = param_constraints(num_dims, prop, len(pwa_trace.partitions))

    C, d = truncate(C, d)
    P, q = truncate(P, q)
    part_cons_ = np.dot(C, all_vars) - d
    part_cons = (c <= 0 for c in part_cons_)

    param_cons_ = np.dot(P, all_vars) - q
    param_cons = (c <= 0 for c in param_cons_)

    all_cons = it.chain(part_cons, dyn_cons, param_cons)
    #all_cons = (part_cons.tolist() + dyn_cons + param_cons.tolist())

    return all_cons, all_vars


def optsoln2x(x, trace_len):
    '''converts solution of lp: a vector to a concrete trace: numpy
    array'''
    x = np.array(x)
    assert(x.ndim == 1)
    num_vars = len(x)//trace_len
    return np.reshape(x, (trace_len, num_vars))


def feasible(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    cons, Vars = pwatrace2cons(pwa_trace, num_dims, prop)

    #nvars = num_dims.x + num_dims.pi

    #A_ub, b_ub = truncate(A_ub, b_ub)
    obj = 0

    err.warn_severe('faking output of optimizer')
    #res, varval_map = z3opt.polyprog(obj, cons)
    res = True
    varval_map = {v: 0 for v in Vars}

    return optsoln2x([varval_map[v] for v in Vars], len(pwa_trace)) if res else None

    #return lpsoln2x(varval_map, len(pwa_trace)) if res else None


def overapprox_x0(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    raise NotImplementedError
