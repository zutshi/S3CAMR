from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sympy as sym
import functools as ft
import atexit

from .pymatopt.optimmatlab import init, deinit, fmincon
from utils import print

import nonlinprog.spec as spec
import settings


init()
atexit.register(deinit)


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

    obj_f = lambda x: sym.lambdify(Vars, obj)(*x)

    # The below constraint encoding assumes all cons
    # (constraint exprs) are of the form f(x) <= 0
    lambdafied = []
    for c in cons:
        assert(isinstance(c, sym.LessThan))
        assert(c.args[1] == 0)
        lambdafied.append(sym.lambdify(Vars, c.args[0], str('numpy')))

    def debugf(f, e, x):
        y = f(*x)
        #print(x, ':', e.args[0] <= 0, ':', f(*x))
        #print(e.args[0] <= 0, ':', f(*x))
        return y

    # Must pass l as an arguement, else late binding will make sure
    # that all functions are the same: the last one
    cons_f = tuple(lambda x, l=l: l(*x) for l in lambdafied)
    cons_f = tuple(ft.partial(debugf, l, e) for l, e in zip(lambdafied, cons))

    x0 = np.zeros(len(Vars))

    retcode, res_x, res_f = fmincon(obj_f, x0, A=[], B=[], C=cons_f)
    print('retcode:', retcode)

    return spec.OPTRES(res_f, res_x, 'OK', retcode == 0)


def example():
    import time

    # simple test function
    def banana(x):
        term1 = 100.0 * (x[1] - x[0]**2.0) ** 2.0
        term2 = (1.0 - x[0])**2.0
        return term1 + term2

    # example non linear constraint
    def nonlincon(x):
        return x[0]**2 - x[1]**2

    ti = time.time()
    init()
    print('init time: {}'.format(time.time() - ti))

    ti = time.time()
    x, fval = fmincon(banana, [1.99, 1.55], A=[1.99, 1.55], B=[1.5], C=[nonlincon])
    print('1st solve time: {}'.format(time.time() - ti))

    ti = time.time()
    x, fval = fmincon(banana, [1.99, 1.55], A=[1.99, 1.55], B=[1.5], C=[nonlincon])
    print('2nd solve time: {}'.format(time.time() - ti))
    ti = time.time()
    x, fval = fmincon(banana, [1.99, 1.55], A=[1.99, 1.55], B=[1.5], C=[nonlincon])
    print('3rd solve time: {}'.format(time.time() - ti))
    ti = time.time()
    x, fval = fmincon(banana, [1.99, 1.55], A=[1.99, 1.55], B=[1.5], C=[nonlincon])
    print('4th solve time: {}'.format(time.time() - ti))

    ti = time.time()
    deinit()
    print('de-init time: {}'.format(time.time() - ti))
