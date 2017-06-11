from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.linalg as linalg
import sympy as sym
import itertools as it

from globalopts import opts as gopts

import settings
import nonlinprog.nonlinprog as nonlinprog

from IPython import embed

import err
import utils as U
from utils import print
from constraints import IntervalCons


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


def dyn_constraints(models, vars_grouped_by_state):
    """constraints due to dynamics of the pwa model

    Parameters
    ----------
    models : list of pwa models
    vars_grouped_by_state: variables for state vectors grouped
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

    for m, (Vars, next_vars) in zip(models, U.pairwise(vars_grouped_by_state)):
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
    """pwatrace2cons

    Parameters
    ----------
    pwa_trace :
    num_dims :
    prop :

    Returns
    -------
    all_cons: All constraints in the form g(x) <= 0
    all_vars: List of vars, ordered against the trace
    vars_grouped_by_state:

    Notes
    ------
    """

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

    return all_cons, all_vars, vars_grouped_by_state


def optsoln2x(x, trace_len):
    '''converts solution of lp: a vector to a concrete trace: numpy
    array'''
    x = np.array(x)
    assert(x.ndim == 1)
    num_vars = len(x)//trace_len
    return np.reshape(x, (trace_len, num_vars))


def feasible(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    cons, Vars, vars_grouped_by_state = pwatrace2cons(pwa_trace, num_dims, prop)
    cons = list(cons)

    #nvars = num_dims.x + num_dims.pi

    #A_ub, b_ub = truncate(A_ub, b_ub)
    def return_zero(x):
        return 0
    obj = return_zero

    #err.warn_severe('faking output of optimizer')
    #res = True
    #varval_map = {v: 0 for v in Vars}
    # TODO: Make choice of opt engine
    #res, varval_map = z3opt.nlinprog(obj, cons, Vars)

    #res, varval_map = nlpfun(solver)(obj, cons, Vars)
    #ret_val = optsoln2x([varval_map[v] for v in Vars], len(pwa_trace)) if res else None
    res = nlpfun(solver)(obj, cons, Vars)
    ret_val = optsoln2x(res.x, len(pwa_trace)) if res.success else None
    print(res.success)
    #print(cons)
    print(ret_val)
    #embed()
    return ret_val

    #return lpsoln2x(varval_map, len(pwa_trace)) if res else None


#TODO: getting lpsolver every time a linprog is executed!
#@memoize
def nlpfun(solver):
    return nonlinprog.factory(solver)


#TODO: ugly function
def overapprox_x0(num_dims, prop, pwa_trace, solver=gopts.opt_engine):
    cons, Vars, vars_grouped_by_state = pwatrace2cons(pwa_trace, num_dims, prop)
    cons = list(cons)

    num_opt_vars = len(Vars)
    nvars = num_dims.x + num_dims.pi

    #directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    I = np.eye(nvars)
    directions = np.vstack((I, -I))
    left_over_vars = num_opt_vars - nvars
    directions_ext = np.pad(directions, [(0, 0), (0, left_over_vars)], 'constant')

    var_array = np.array(Vars)

    for direction in directions_ext:
        print(np.dot(direction, var_array))
    U.pause()

    lambdafied = tuple(
            sym.lambdify(Vars, np.dot(direction, var_array), str('numpy')) for direction in directions_ext)
    obj_f = tuple(lambda x, l=l: l(*x) for l in lambdafied)

    x_arr = np.array(
            sym.symbols(
                ['x{}'.format(i) for i in range(nvars)]
                ))

    res = solve_mult_opt(nlpfun(solver), obj_f, cons, Vars)

    l = len(res)
    assert(l % 2 == 0)
    min_res, max_res = res[:l//2], res[l//2:]

    ranges = []

    for di, rmin, rmax in zip(directions, min_res, max_res):
        if (rmin.success and rmax.success):
            print('{} \in [{}, {}]'.format(np.dot(di, x_arr), rmin.fun, -rmax.fun))
            ranges.append([rmin.fun, -rmax.fun])
        else:
            if settings.debug:
                print('LP failed')
                print('rmin status:', rmin.status)
                print('rmax status:', rmax.status)
            return None

    r = np.asarray(ranges)
    # due to numerical errors, the interval can be malformed
    try:
        ret_val = IntervalCons(r[:, 0], r[:, 1])
    except ValueError:
        err.warn('linprog fp failure: Malformed Interval! Please fix.')
        return None

    print(ret_val)
    return ret_val


# raise an exception as soon as the first lp fails...better than
# solving all directions when the problem in infeasible
def solve_mult_opt(nlp_fun, directions_ext, cons, Vars):
    res = []
    for obj in directions_ext:
        # A_ub and b_ub do not change in the for loop...can do a warm
        # restart? or some kind of caching?
        ret_val = nlp_fun(obj, cons, Vars)
        if not ret_val.success:
            raise RuntimeError('lp solver failed: {}'.format(ret_val.status))
        else:
            res.append(ret_val)
    return res
