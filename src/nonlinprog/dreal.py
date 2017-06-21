from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from IPython import embed

import numpy as np
import z3

from sympy2z3.sympy2z3 import sympy2z3

import fileops as fp
import utils as U
from utils import print

from globalopts import opts as gopts

import nonlinprog.spec as spec

DREAL = '/home/zutshi/software/dreal3/build/dReal'
DELTA_SAT_PREC = '0.01'
FNAME = 'tmp.smt2'


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
    # feasibility check
    if obj == 0:
        res = check_feasibility(obj, cons, Vars)
        if res.success and res_is_valid(Vars, cons, res):
            return res
        else:
            return spec.OPTRES(0, None, 'OK', False)
    else:
        import nonlinprog.ipopt as ipopt
        res = check_feasibility(obj, cons, Vars)
        res_nlopt = ipopt.nlinprog(obj, cons, Vars, x0=res.x)
        if not res_nlopt.success:
            return res
        else:
            return res_nlopt


def res_is_valid(Vars, cons, res):
    varval_map = {v: x for v, x in zip(Vars, res.x)}
    for c in cons:
        if c.subs(varval_map) > 0:
            return False
    return True



def check_feasibility(obj, cons, Vars):

    var_str_to_idx = {str(v): idx for idx, v in enumerate(Vars)}

    # Z3
    solver = z3.Solver()
    sym2Z3_varmap, z3_cons = sympy2z3(cons)
    solver.add(z3_cons)

    # dReal3
    LOGIC = '(set-logic QF_NRA)\n'
    smt2_str = LOGIC + solver.to_smt2()
    fp.overwrite(FNAME, smt2_str)
    output = U.strict_call_get_op([DREAL, FNAME, '--model', '--precision', DELTA_SAT_PREC])
    status, res_x = parse_dreal_res(var_str_to_idx, output)
    return spec.OPTRES(0, res_x, 'OK', status)


def parse_dreal_res(var_str_to_idx, output):
    '''
    Parses the below format

    $ ./dReal ./myex/vdp2.smt2 --precision 0.01 --model
    Solution:
    x0 : [ ENTIRE ] = [0.3787226444670079, 0.3865831973010885]
    x1 : [ ENTIRE ] = [-0.1316263925922571, -0.1288160769154083]
    x2 : [ ENTIRE ] = [0.03733948568736267, 0.03979756777764665]
    x3 : [ ENTIRE ] = [-1.98421875, -1.97625]
    x4 : [ ENTIRE ] = [-1, -0.990625]
    x5 : [ ENTIRE ] = [-5.77734375, -5.769375]
    delta-sat with delta = 0.01000000000000000
    '''

    nVars = len(var_str_to_idx)
    UNSAT = 'unsat'
    output = output.strip()

    if output == UNSAT:
        return False, None

    else:
        op = output.splitlines()

        # Make sure the format is as expected
        assert(len(op) == nVars+2)
        assert(op[0] == 'Solution:')
        assert(op[-1][:23] == 'delta-sat with delta = ')

        # we select avg. of the lower bounds and upper bounds for no specific reasons
        x = np.zeros(nVars)
        for a in op[1:-1]:
            v, lb, ub = parse_assignment(a)
            x[var_str_to_idx[v]] = (float(lb)+float(ub))/2
        return True, x


def parse_assignment(a):
    a = a.replace(' ', '')
    lhs, rhs = a.split(':')
    # TODO: not sure about the significance of ENTIRE
    ENTIRE, bounds = rhs.split('=')
    assert(ENTIRE == '[ENTIRE]')
    lb, ub = bounds.strip('[]').split(',')
    return lhs, lb, ub
