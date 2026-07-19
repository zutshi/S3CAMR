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
    vars_str = ','.join(f'x{i}' for i in range(nvars))
    sym_vars = sym.var(vars_str)

    pt = [tuple(i) for i in pa.tolist()]
    d = {p: c for p, c in zip(pt, ca)}
    sym_poly = sym.Poly.from_dict(d, *sym_vars)
    return Poly(sym_poly, sym_vars)


class Poly:
    def __init__(self, sym_poly, sym_vars):
        assert(isinstance(sym_poly, sym.Poly))
        self.poly = sym_poly
        self.vars = sym_vars

    def as_dict(self):
        return self.poly.as_dict()

    # TODO: fix this, should not be taking a truncate function. Move
    # the truncate() into a utils file or so
    def truncate_coeffs(self, prec):
        """Return a NEW Poly with coefficients truncated to `prec` digits.

        Non-mutating: the model Poly objects are shared across the PWA graph and
        this is called repeatedly during feasibility/over-approximation, so
        mutating self would be a shared-state side effect. (Truncation is
        idempotent, so the previous in-place version was not a correctness bug,
        but returning a copy is cleaner.)
        """
        poly = self.poly
        d = poly.as_dict()
        # TODO: string conversions are slow, use proper np.truncate functions
        for powers, coeff in d.items():
            d[powers] = float('{n:0.{p}f}'.format(n=coeff, p=prec))
        # sympy expects generators as positional args (ordered); poly.gens is
        # the canonical ordered tuple. (free_symbols is an unordered set and
        # newer sympy rejects passing it directly.)
        return Poly(sym.Poly.from_dict(d, *poly.gens), self.vars)

    def subs_vars(self, old2new_var_map):
        return self.poly.subs(old2new_var_map)
