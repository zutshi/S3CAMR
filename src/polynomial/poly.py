from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sympy as sym

from utils import print


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

    # TODO: fix this, should not be taking a truncate function. Move
    # the truncate() into a utils file or so
    def truncate_coeffs(self, prec):
        poly = self.poly
        d = poly.as_dict()
        # TODO: string conversions are slow, use proper np.truncate functions
        for powers, coeff in d.iteritems():
            d[powers] = float('{n:0.{p}f}'.format(n=coeff, p=prec))
        #print(poly)
        self.poly = sym.Poly.from_dict(d, poly.free_symbols)
        #print(self.poly)
        return

    def subs_vars(self, old2new_var_map):
        return self.poly.subs(old2new_var_map)
