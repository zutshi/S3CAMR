from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sympy as sym


def power_array2Poly(power_array, coeff_array):
    '''
    input:
    array([[0, 0],
       [1, 0],
       [0, 1],
       [2, 0],
       [1, 1],
       [0, 2]])
    '''
    pa, ca = power_array, coeff_array
    assert(pa.ndim == 2)
    assert(ca.ndim == 1)

    nvars = power_array.shape[1]
    vars_str = ','.join(('x{}'.format(i) for i in range(nvars)))
    sym_vars = sym.var(vars_str)

    pt = [tuple(i) for i in pa.tolist()]
    d = {p: c for p, c in zip(pt, ca)}
    sym_poly = sym.Poly.from_dict(d, *sym_vars)
    return Poly(sym_poly, sym_vars)


class Poly(object):
    def __init__(self, sym_poly, sym_vars):
        assert(isinstance(sym_poly, sym.Poly))
        self.poly = sym_poly
        self.vars = sym_vars

    def as_dict(self):
        return self.poly.as_dict()

    def subs(self, old2new_map):
        return self.poly.subs(old2new_map)
