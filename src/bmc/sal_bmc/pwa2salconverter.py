import itertools as it
import collections
import logging

import numpy as np
import scipy.linalg as linalg

from bmc.helpers.expr2str import Expr2Str
from . import saltrans_dft as slt_dft

logger = logging.getLogger(__name__)

PWA_TRANS = collections.namedtuple('TransitionID', ('p pnext m l lnext'))


def var_eq(var, val):
    return "{} = {}".format(var, val)


def next_var_eq(var, val):
    return "{}' = {}".format(var, val)


def next_real_true(var):
    return "{}' IN {{ r : REAL | TRUE }}".format(var)


def bounded_real_set_def(var, lb, ub):
    return "{var}' IN {{ r : REAL| " "r >= {lb} AND " "r <= {ub} }}".format(
            var=var, lb=lb, ub=ub)


def Axb_constraints(A, b, error):
    ''' Matrix conversion from A, b, error -> C, d
        s.t.
        x' = Ax + b + [error_lb, error_ub]
        is converted to
        x' >= Ax + b + error_lb /\ x' <= Ax + b + error_ub
    '''

    nlhs, nrhs = A.shape
    assert(nlhs == b.size)

    # new state assignments
    delta_h, delta_l = b + error.h, b + error.l

# What follows...
#     x_ = Ax + [dl, dh]
#     hence, x_ <= Ax + dh, x_ >= Ax + dl
#     and finally,
#     x_ - Ax <= dh                    (1)
#     -x_ + Ax <= -dl                  (2)


#     [1          a00 a01 a02] [x0_] + [d]
#     [    1      a10 a11 a12] [x1_] + [d]
#     [        1  a20 a21 a22] [x2_] + [d]
#                              [x0 ]
#                              [x1 ]
#                              [x2 ]
    # find the dim(x) using any submodel's b vector
    num_dim_x = nlhs

    I = np.eye(num_dim_x)
    # C_ub is x_ - Ax, refer (1) and (2)
    C_ub = np.hstack((I, -A))
    # C_lb is -x_ + Ax, refer (1) and (2)
    C_lb = np.hstack((-I, A))
    C = np.vstack((C_ub, C_lb))
    # refer to (1) and (2)
    d = np.hstack((delta_h, -delta_l))

    # C [ x'] <= d
    #   [ x ]
    return C, d


# def guard(Cd, Cd_, vs=None, cell_id=None, next_cell_id=None):
#     ''' Cx - d <= 0 '''

#     if Cd is not None:
#         C, d = Cd

#         assert(C.shape[0] == d.size)
#         assert(vs is not None)
#         C, d = C, d
#     else:
#         C, d = [], []

#     if Cd_ is not None:
#         C_, d_ = Cd_

#         assert(C_.shape[0] == d_.size)
#         assert(vs is not None)

#         C_, d_ = C_, d_
#         vs_ = [vsi + "'" for vsi in vs]
#     else:
#         C_, d_ = [], []

#     if Cd is not None and Cd_ is not None:
#         assert(C.shape == C_.shape)
#         assert(d.shape == d_.shape)

#     pre_state_cons = (Expr2Str.linexpr2str(vs, ci, -di) + ' <= 0'
#                       for ci, di in zip(C, d))

#     post_state_cons = (Expr2Str.linexpr2str(vs_, ci, -di) + ' <= 0'
#                        for ci, di in zip(C_, d_))

#     pre_cell = '' if cell_id is None else var_eq('cell', cell_id)
#     post_cell = '' if next_cell_id is None else next_var_eq('cell', next_cell_id)

#     return it.chain((pre_cell, post_cell), pre_state_cons, post_state_cons)


def guard(Cd, vs=None, cell_id=None, next_cell_id=None):
    ''' Cx - d <= 0 '''

    if Cd is not None:
        C, d = Cd

        assert(C.shape[0] == d.size)
        assert(vs is not None)
        C, d = C, d
    else:
        C, d = [], []

    vs_vs = [vsi + "'" for vsi in vs] + vs

    state_cons = (Expr2Str.linexpr2str(vs_vs, ci, -di) + ' <= 0'
                  for ci, di in zip(C, d))

    pre_cell = '' if cell_id is None else var_eq('cell', cell_id)
    post_cell = '' if next_cell_id is None else next_var_eq('cell', next_cell_id)

    return it.chain((pre_cell, post_cell), state_cons)


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


def reset_states_true(vs):
    ''' resets all states to True '''

    state_assignments = (next_real_true(vi) for vi in vs)
    return state_assignments


def reset_cell_true():
    cell_assignments = ("cell' IN {c : CELL | TRUE}", )
    return cell_assignments


class Pwa2Sal(object):

    def __init__(self, module_name, init_cons, final_cons, pwa_graph, vs, init_ps, final_ps):
        self.module_name = module_name
        self.init_cons = init_cons
        self.final_cons = final_cons
        self.pwa_graph = pwa_graph
        self.vs = vs
        self.init_ps = init_ps
        self.final_ps = final_ps

        self.sal2pwa_map = None

    def convert_transitions(self):
        pwa_graph, vs = self.pwa_graph, self.vs
        sal2pwa_map = collections.OrderedDict()

        #TODO:
        # Store a mapping from a cell id: tuple -> sal loc name: str
        partid2Cid = collections.OrderedDict()
        id_ctr = it.count()

        # TODO: implicitly assumes a self loop on the last state
        # If it is not there, the last p form pnext, would have
        # never been added and get_C will throw an exception.
        def getC(pid):
            """Gets the bmc Cid corresponding to a pwa partition id"""
            if pid not in partid2Cid:
                partid2Cid[pid] = 'C' + str(next(id_ctr))
            return partid2Cid[pid]

        transitions = []
#         for idx, sub_model in enumerate(pwa_model):
#             p = sub_model.p
#             pnext = sub_model.pnexts[0]
#             assert(len(sub_model.pnexts) == 1)
#             l = getC(p.ID)
#             lnext = getC(pnext.ID)

#             t = self.sal_transition(idx, p, pnext, sub_model.m, vs, l, lnext)
#             sal2pwa_map[t.name] = PWA_TRANS(p, pnext, sub_model.m, l, lnext)
#             transitions.append(t)

        for idx, relation in enumerate(pwa_graph.relations()):
            p1, p2, m = relation.p1, relation.p2, relation.m
            #assert(len(sub_model.pnexts) == 1)
            l = getC(p1.ID)
            lnext = getC(p2.ID)

            t = self.sal_transition(idx, p1, p2, m, vs, l, lnext)
            sal2pwa_map[t.name] = PWA_TRANS(p1, p2, m, l, lnext)
            transitions.append(t)

        logger.debug('================ bmc - pwa conversion dict ==================')
        logger.debug(sal2pwa_map)
        self.sal2pwa_map = sal2pwa_map
        return transitions, partid2Cid

    def sal_transition(self, idx, p, pnext, m, vs, l, lnext):
        num_states = len(vs)
        assert(num_states == m.b.size)
        assert(num_states == pnext.C.shape[1])
        C1, d1 = Axb_constraints(m.A, m.b, m.error)
        C2 = np.hstack((pnext.C, np.zeros((pnext.C.shape[0], num_states))))
        d2 = pnext.d
        C3 = np.hstack((np.zeros((p.C.shape[0], num_states)), p.C))
        d3 = p.d
        C, d = np.vstack((C1, C2, C3)), np.hstack((d1, d2, d3))
        g = slt_dft.Guard(guard((C, d), vs, l, lnext))
        #r = slt_dft.Reset(reset(m.A, m.b, m.error, vs))
        r = slt_dft.Reset(it.chain(reset_states_true(vs), reset_cell_true()))
        t = slt_dft.Transition('T_{}'.format(idx), g, r)
        return t

    def trans_sys(self):
        sal_transitions, partid2Cid = self.convert_transitions()
        return slt_dft.SALTransSys(
                self.module_name, self.vs,
                self.init_cons, self.final_cons,
                self.init_ps, self.final_ps,
                sal_transitions, partid2Cid)

    def trace(self, tids):

        s2p = self.sal2pwa_map
        models, partitions = [], []
        for tid in tids:
            models.append(s2p[tid].m)
            partitions.append(s2p[tid].p)
        # append the last location/cell/partition id
        partitions.append(s2p[tid].pnext)

        return PWATRACE(partitions, models)


class PWATRACE(object):
    def __init__(self, partitions, models):
        self.partitions = partitions
        self.models = models
        return

    def __len__(self):
        return len(self.partitions)
