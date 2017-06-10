# Refer example: /home/zutshi/software/pyipopt/examples/hs071.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sympy as sym
import pyipopt

import nonlinprog.spec as spec
import settings

from IPython import embed


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
    nvars = len(Vars)

    x_L = pyipopt.NLP_LOWER_BOUND_INF
    x_U = pyipopt.NLP_UPPER_BOUND_INF
    ncon = len(cons)
    g_L = pyipopt.NLP_LOWER_BOUND_INF
    g_U = 0

    nnzj = 0#?????
    nnzh = 0

    eval_f = obj_f
    eval_grad_f = obj_grad_f

    eval_g = cons_f
    eval_jac_g = jac_cons_f


    nlp = pyipopt.create(nvars, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)

    x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    # import pdb; pdb.set_trace()
    nlp.close()

    def print_variable(variable_name, value):
      for i in xrange(len(value)):
        print variable_name + "["+str(i)+"] =", value[i]

    print
    print "Solution of the primal variables, x"
    print_variable("x", x)
    print
    print "Solution of the bound multipliers, z_L and z_U"
    print_variable("z_L", zl)
    print_variable("z_U", zu)
    print
    print "Solution of the constraint multipliers, lambda"
    print_variable("lambda", constraint_multipliers)
    print
    print "Objective value"
    print "f(x*) =", obj


    return spec.OPTRES(res.fun, res.x, res.status, res.success)
