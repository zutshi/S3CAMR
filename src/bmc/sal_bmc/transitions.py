import collections

from bmc.helpers.expr2str import Expr2Str


class Transition(object):

    def __init__(self, name, g, r):
        self.g = g
        self.r = r
        self.name = name

    def __str__(self):
        s = '{}:\n{} -->\n{}\n'.format(self.name, self.g, self.r)
        return s


class Guard(object):
    def __init__(self, C, d, cell_id=None):
        '''Cx - d <= 0'''
        self.C = C
        self.d = d
        self.cell_id = cell_id

        self.ncons, self.nvars = self.C.shape
        assert(self.ncons == d.size)
        self._vs = None

    @property
    def vs(self):
        if self._vs is None:
            return ['x'+str(j) for j in range(self.nvars)]
        else:
            return self._vs

    @vs.setter
    def vs(self, vs):
        self._vs = vs

    def __str__(self):
        cons = (Expr2Str.linexpr2str(self.vs, self.C[i, :], -self.d[i]) + ' <= 0'
                for i in range(self.ncons))
        cons_str = ' AND '.join(cons)
        cell_str = '' if self.cell_id is None else 'cell = ' + self.cell_id + ' AND '
        return cell_str + cons_str


class Reset(object):
    def __init__(self, A, b, error, next_cell_ids=[]):
        self.A = A
        self.b = b
        self.e = error
        assert(isinstance(next_cell_ids, collections.Iterable))
        self.next_cell_ids = next_cell_ids

        self.nlhs, self.nrhs = self.A.shape
        assert(self.nlhs == b.size)
        self._vs = None
        self._vs_ = None

    @property
    def vs(self):
        if self._vs is None:
            return ['x'+str(j) for j in range(self.nrhs)]
        else:
            return self._vs

    @property
    def vs_(self):
        if self._vs_ is None:
            return ["x{}".format(i) for i in range(self.nlhs)]
        else:
            return self._vs_

    @vs.setter
    def vs(self, vs):
        self._vs = vs

    @vs_.setter
    def vs_(self, vs_):
        self._vs_ = vs_

    def __str__(self):
        s = []

        delta_h = self.b + self.e.h
        delta_l = self.b + self.e.l
        set_def = ("{xi_}' IN {{ r : REAL| "
                   "r >= {Axi_plus_delta_li} AND "
                   "r <= {Axi_plus_delta_hi} }}")
        for vsi_, Ai, dli, dhi in zip(self.vs_, self.A, delta_l, delta_h):
            #if error is not significant, i.e., it is less than the
            # precision used while dumping sal file, ignore it. Else,
            # include it and make the rhs of the assignment a set.
            if Expr2Str.float2str(dli) == Expr2Str.float2str(dhi):
                assignment_stmt = vsi_ + "' = " + Expr2Str.linexpr2str(self.vs, Ai, dli)
            else:
                assignment_stmt = set_def.format(
                        xi_=vsi_,
                        Axi_plus_delta_li=Expr2Str.linexpr2str(self.vs, Ai, dli),
                        Axi_plus_delta_hi=Expr2Str.linexpr2str(self.vs, Ai, dhi)
                        )
            s.append(assignment_stmt)

        new_cell = (";\ncell' IN {" + ', '.join(self.next_cell_ids) + "}"
                    if self.next_cell_ids else '')

        return ';\n'.join(s) + new_cell
