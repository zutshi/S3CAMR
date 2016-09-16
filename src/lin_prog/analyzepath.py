from __future__ import print_function

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as spopt
import sympy as sym


def part_constraints(pwa_trace):
    """constraints due to partitions of the pwa model

    Parameters
    ----------
    pwa_trace : pwa trace form the bmc module

    Returns
    -------
    A, b corresponding to Cx <= d

    Notes
    ------
    """
    C, d = [], []
    # for each sub_model
    for sm in pwa_trace:
        # otherwise the trace is not unique!
        # FIX pwa_trace() in sal_bmc object
        assert(sm.pnext is not None)

        C.append(sm.p.C)
        d.append(sm.p.d)

    # Add the last partition
    C.append(sm.pnext.C)
    d.append(sm.pnext.d)

    A = linalg.block_diag(*C)
    b = np.hstack(d)

    assert(A.shape[0] == b.shape[0])
    return A, b


def dyn_constraints(pwa_trace):
    """constraints due to dynamics of the pwa model

    Parameters
    ----------
    pwa_trace : pwa trace form the bmc module

    Returns
    -------
    A, b corresponding to x' \in Ax + b + [error.l, error.h]

    Notes
    ------
    """
    AA, bb, eeh, eel = [], [], [], []

    # for each sub_model
    for sm in pwa_trace:
        # otherwise the trace is not unique!
        # FIX pwa_trace() in sal_bmc object
        assert(sm.pnext is not None)

        # x' <= Ax + b + e.h
        AA.append(sm.m.A)
        bb.append(sm.m.b)
        eeh.append(sm.m.error.h)
        eel.append(sm.m.error.l)

    # find the dim(x) using any submodel's b vector
    num_dim_x = sm.m.b.size

    # block diagonal of AA
    AA_block = linalg.block_diag(*AA)
    # Add identity matrix to get in x'
    # Of course the blockdiag array is a square
    I = np.eye(AA_block.shape[0])

    # pad the AA_block array
    # add 1 row at the top, and 1 at the right
    padding_scheme = (num_dim_x, 0), (0, num_dim_x)
    AA = np.pad(AA_block, padding_scheme, 'constant')

    # pad the Identity array
    # add 1 row at the top, and 1 at the left [shifted by 1 w.r.t. AA]
    padding_scheme = (num_dim_x, 0), (num_dim_x, 0)
    II = np.pad(I, padding_scheme, 'constant')

    # padding for b
    pad_b = [[0.0] * num_dim_x]

    #print(AA)
    #print(II)
    # Ahx <= bb + eeh
    Ah = -AA + II
    # Alx >= bb + eeh
    Al = -Ah
    bh = np.hstack(pad_b + bb)
    bl = -bh
    eh = np.hstack(pad_b + eeh)
    el = -np.hstack(pad_b + eel)

    A = np.vstack((Ah, Al))
    b = np.hstack((bh, bl)) + np.hstack((eh, el))

    assert(A.shape[0] == b.shape[0])

    return A, b


def prop_constraints(AA, prop, trace_len):
    """constraints due to the initial set, final set and ci/pi

    Parameters
    ----------
    prop :

    Returns
    -------

    Notes
    ------
    """

    iA, ib = prop.init_cons.poly()
    fA, fb = prop.final_cons.poly()

    assert(iA.shape == fA.shape)

    # number of rows and coloumns of iA/fA
    nr, nc = iA.shape
    num_cons = 2 * nr

    # Add 1, because trace captures transitions,
    # and num_states = num_trans + 1
    A = np.zeros((num_cons, AA.num_dims.x * (trace_len + 1)))
    A[0:nr, 0:nc] = iA
    A[-nr:, -nc:] = fA
    b = np.hstack((ib, fb))
    print(A)
    print(b)
    return A, b


def overapprox_x0(AA, prop, opts, pwa_trace):
    C, d = part_constraints(pwa_trace)
    A, b = dyn_constraints(pwa_trace)
    pA, pb = prop_constraints(AA, prop, len(pwa_trace))

    # num vars are the same
    assert(C.shape[1] == A.shape[1])

    A_ub = np.vstack((C, A, pA))
    b_ub = np.hstack((d, b, pb))

    num_vars = A.shape[1]
    dim_x = AA.num_dims.x
    #obj = np.ones(num_vars)
    bounds = [(-np.inf, np.inf)] * num_vars

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x_arr = np.array(sym.symbols(('x1', 'x2')))

    for di in directions:
        obj = np.array(di + (0.0,) * (num_vars - dim_x))
        res = spopt.linprog(
                c=obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='simplex',
                options={'disp': True})
        print(np.dot(di, x_arr))
        print(res)