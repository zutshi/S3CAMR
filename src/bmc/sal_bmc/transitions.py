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
    def __init__(self, Cd, Cd_, vs=None, cell_id=None, next_cell_id=None):
        ''' Cx - d <= 0 '''

        if Cd is not None:
            C, d = Cd

            assert(C.shape[0] == d.size)
            assert(vs is not None)
            self.C, self.d = C, d
        else:
            self.C, self.d = [], []

        if Cd_ is not None:
            C_, d_ = Cd_

            assert(C_.shape[0] == d_.size)
            assert(vs is not None)

            self.C_, self.d_ = C_, d_
            self.vs_ = [vsi + "'" for vsi in vs]
        else:
            self.C_, self.d_ = [], []

        self.cell_id = cell_id
        self.next_cell_id = next_cell_id
        self.vs = vs

        if Cd is not None and Cd_ is not None:
            assert(C.shape == C_.shape)
            assert(d.shape == d_.shape)

    def __str__(self):
        pre_cons = (Expr2Str.linexpr2str(self.vs, ci, -di) + ' <= 0'
                    for ci, di in zip(self.C, self.d))

        post_cons = (Expr2Str.linexpr2str(self.vs_, ci, -di) + ' <= 0'
                     for ci, di in zip(self.C_, self.d_))

        cons_str = ' AND '.join(pre_cons) + ' AND '.join(post_cons)
        cell_str = 'TRUE' if self.cell_id is None else 'cell = ' + self.cell_id

#         cell_assignment = (";\ncell' IN {" + ', '.join(self.next_cell_ids) + "}"
#                            if self.next_cell_ids else '')
        cell_assignment = ("cell' = {}".format(self.next_cell_id)
                           if self.next_cell_id else '')

        return cell_str + ' AND ' + cell_assignment + ' AND ' + cons_str


class Reset(object):
    def __init__(self, A, b, error, vs):
        ''' x' = Ax + b +- [error] '''
        self.A = A
        self.b = b
        self.e = error

        self.nlhs, self.nrhs = self.A.shape
        assert(self.nlhs == b.size)
        self.vs = vs
        self.vs_ = vs

    def __str__(self):
        s = []

        # new state assignments
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

        cell_assignment = "cell' IN {c : CELL | TRUE}"
        return ';\n'.join(s) + ';\n' + cell_assignment
