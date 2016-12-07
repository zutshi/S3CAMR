
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import gurobipy as GB
from IPython import embed

# Tuple contains lp results. It follows the scipy convention
OPTRES = collections.namedtuple('optres', ('fun', 'status', 'success'))


# TODO: make it handle multiple objectives?
# Specifially if just changing an obj offers a warm restart? It will
# also eschew the time required to re-create the LP. WIll need to
# research if such an optimization is legal and does it actually save
# enough time?


def linprog(obj, A, b):
    """ Minimize obj, given Ax <= b

    Parameters
    ----------
    c : obj as a list
    A : as an numpy array
    b : as a numpy array

    Returns
    -------
    s

    Notes
    ------
    """
    model = GB.Model()

    ncons, nvars = A.shape

    # Add variables to model
    #for i in range(nvars):
        #model.addVar(lb=-GB.GRB.INFINITY, ub=GB.GRB.INFINITY, vtype=GB.GRB.CONTINUOUS)
    #model.update()
    #Vars = model.getVars()
    Vars = [model.addVar(lb=-GB.GRB.INFINITY, ub=GB.GRB.INFINITY, vtype=GB.GRB.CONTINUOUS) for i in range(nvars)]
    model.update()

#     for ri, bi in zip(A, b):
#         expr = GB.LinExpr()
#         for cij, vj in zip(ri, Vars):
#             if cij != 0:
#                 expr += cij * vj
#         model.addConstr(expr, GB.GRB.LESS_EQUAL, bi)

    for ri, bi in zip(A, b):
        # XXX: This API usage was found on web, does not have its
        # usage documented.
        expr = GB.LinExpr(((cij, vj) for (cij, vj) in zip(ri, Vars) if cij != 0.0))
        model.addConstr(expr, GB.GRB.LESS_EQUAL, bi)

    # Add objective
#     expr = GB.LinExpr()
#     for ci, vi in zip(obj, Vars):
#         if ci != 0:
#             expr += ci * vi

    expr = GB.LinExpr(((ci, vi) for (ci, vi) in zip(obj, Vars) if ci != 0.0))
    model.setObjective(expr, GB.GRB.MINIMIZE)

    # Why and when exactly shoud this be called?
    #model.update()

    model.setParam('OutputFlag', False)
    model.optimize()

    #model.write('gurobi_out.lp')

    if model.status == GB.GRB.Status.OPTIMAL:
        res = OPTRES(model.objVal, model.status, model.status == GB.GRB.Status.OPTIMAL)
    else:
        dummy_obj = 0
        res = OPTRES(dummy_obj, model.status, model.status == GB.GRB.Status.OPTIMAL)

    return res
