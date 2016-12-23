import itertools as it
import collections

from bmc.helpers.expr2str import Expr2Str
from . import saltrans_dft as slt_dft

PWA_TRANS = collections.namedtuple('TransitionID', ('p pnext m l lnext'))


def var_eq(var, val):
    return "{} = {}".format(var, val)


def next_var_eq(var, val):
    return "{}' = {}".format(var, val)


def bounded_real_set_def(var, lb, ub):
    return "{var}' IN {{ r : REAL| " "r >= {lb} AND " "r <= {ub} }}".format(
            var=var, lb=lb, ub=ub)


def guard(Cd, Cd_, vs=None, cell_id=None, next_cell_id=None):
    ''' Cx - d <= 0 '''

    if Cd is not None:
        C, d = Cd

        assert(C.shape[0] == d.size)
        assert(vs is not None)
        C, d = C, d
    else:
        C, d = [], []

    if Cd_ is not None:
        C_, d_ = Cd_

        assert(C_.shape[0] == d_.size)
        assert(vs is not None)

        C_, d_ = C_, d_
        vs_ = [vsi + "'" for vsi in vs]
    else:
        C_, d_ = [], []

    if Cd is not None and Cd_ is not None:
        assert(C.shape == C_.shape)
        assert(d.shape == d_.shape)

    pre_state_cons = (Expr2Str.linexpr2str(vs, ci, -di) + ' <= 0'
                      for ci, di in zip(C, d))

    post_state_cons = (Expr2Str.linexpr2str(vs_, ci, -di) + ' <= 0'
                       for ci, di in zip(C_, d_))

    pre_cell = '' if cell_id is None else var_eq('cell', cell_id)
    post_cell = '' if next_cell_id is None else next_var_eq('cell', next_cell_id)

    return it.chain((pre_cell, post_cell), pre_state_cons, post_state_cons)


def reset(A, b, error, vs):
    ''' x' = Ax + b +- [error] '''

    nlhs, nrhs = A.shape
    assert(nlhs == b.size)
    vs_ = vs

    # new state assignments
    delta_h, delta_l = b + error.h, b + error.l

    def get_assign(vsi_, Ai, dli, dhi):
        #if error is not significant, i.e., it is less than the
        # precision used while dumping sal file, ignore it. Else,
        # include it and make the rhs of the assignment a set.
        if Expr2Str.float2str(dli) == Expr2Str.float2str(dhi):
            assignment_stmt = next_var_eq(vsi_, Expr2Str.linexpr2str(vs, Ai, dli))
        else:
            Axi_plus_delta_li = Expr2Str.linexpr2str(vs, Ai, dli)
            Axi_plus_delta_hi = Expr2Str.linexpr2str(vs, Ai, dhi)
            assignment_stmt = bounded_real_set_def(
                    vsi_, Axi_plus_delta_li, Axi_plus_delta_hi)
        return assignment_stmt

    assignments = (get_assign(vsi_, Ai, dli, dhi)
                   for vsi_, Ai, dli, dhi in zip(vs_, A, delta_l, delta_h))

    cell_assignment = "cell' IN {c : CELL | TRUE}"
    return it.chain(assignments, (cell_assignment, ))


def convert_transitions(pwa_model, vs):
    conversion_info = collections.OrderedDict()

    #TODO:
    # Store a mapping from a cell id: tuple -> sal loc name: str
    partid2Cid = collections.OrderedDict()
    id_ctr = it.count()

    # TODO: implicitly assumes a self loop on the last state
    # If it is not there, the last p form pnext, would have
    # never been added and get_C will throw an exception.
    def getC(pid):
        """Gets the bmc Cid corresponding to a pwa partition id"""
        #return self.partid2Cid.setdefault(loc, 'C' + str(next(self.id_ctr)))
        if pid not in partid2Cid:
            partid2Cid[pid] = 'C' + str(next(id_ctr))
        return partid2Cid[pid]

    transitions = []
    for idx, sub_model in enumerate(pwa_model):
        p = sub_model.p
        pnext = sub_model.pnexts[0]
        assert(len(sub_model.pnexts) == 1)
        l = getC(p.ID)
        lnext = getC(pnext.ID)

        t = transition(idx, p, pnext, sub_model.m, vs, l, lnext)
        conversion_info[t.name] = PWA_TRANS(p, pnext, sub_model.m, l, lnext)
        transitions.append(t)
    return transitions, conversion_info, partid2Cid


def transition(idx, p, pnext, m, vs, l, lnext):
    g = slt_dft.Guard(guard(None, (pnext.C, pnext.d), vs, l, lnext))
    r = slt_dft.Reset(reset(m.A, m.b, m.error, vs))
    t = slt_dft.Transition('T_{}'.format(idx), g, r)
    return t
