from __future__ import print_function

import itertools as it
import abc

import numpy as np

from constraints import IntervalCons
import cellmanager as CM
import err
from utils import print


import matplotlib.pyplot as plt

MIN_TRAIN = 50
MIN_TEST = MIN_TRAIN
MAX_ITER = 25


class Q(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def split(self, *args):
        return

#     @abc.abstractproperty
#     def dim(self):
#         return

#     @abc.abstractmethod
#     def getxy_ignoramous(self, N, sim, t0=0):
#         """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
#         """
#         return

    @abc.abstractmethod
    def modelQ(self, rm):
        return

    @abc.abstractmethod
    def errorQ(self, include_err, X, Y, rm):
        return

    @abc.abstractmethod
    def __hash__(self):
        return hash((self.cell, tuple(self.eps)))

    @abc.abstractmethod
    def __eq__(self, c):
        return self.cell == c.cell and tuple(self.eps) == tuple(c.eps)


class Qxw(Q):
    # wic is the pi_cons of the overall system and has nothing to
    # do with wcell (of course wcell must be \in wic, but that is
    # all.
    wic = None

    @classmethod
    def init(cls, wic):
        cls.wic = wic

    def __init__(self, xcell, wcell):
        """__init__

        Parameters
        ----------
        abs_state : abstract states
        w_cells : cells associated with the abstract state defining
        range of inputs
        """
        if Qxw.wic is None:
            raise NameError('Class was not initialized. wic is None')

        assert(isinstance(xcell, CM.Cell))
        assert(isinstance(wcell, CM.Cell))

        self.xcell = xcell
        self.wcell = wcell
        self.xwcell = CM.Cell.concatenate(xcell, wcell)
        self.sample_UR_x = self.xcell.sample_UR
        self.sample_UR_w = self.wcell.sample_UR
        self.xdim = xcell.dim
        self.wdim = wcell.dim
        self.dim = xcell.dim + wcell.dim
        self.ival_constraints = self.xwcell.ival_constraints
        return

    def split(self, *args):
        qsplits = self.qx.splits(*args)
        return [Qxw(qsplits, self.wcell) for q in qsplits]

    def sim(self, step_sim, xw_array):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci = [np.array([])]*2
        t0 = 0
        Yl = []
        x_array = xw_array[:, 0:self.xdim]
        w_array = xw_array[:, self.xdim:]

        for x, pi in zip(x_array, w_array):
            (t_, x_, s_, d_, pvt_, u_) = step_sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        # return yw_array
        return np.hstack((np.vstack(Yl), w_array))

    def get_rels(self, N, sim):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci = np.array([])
        t0 = 0
        Yl = []

        x_array = self.sample_UR_x(N)
        pi_array = self.sample_UR_w(N)
        for x, pi in zip(x_array, pi_array):
            print(pi)
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        return np.hstack((x_array, pi_array)), np.vstack(Yl)

    def sat(self, XW):
        """returns a sat array, whose elements are false when unsat
        and true when sat"""
        return self.xcell.ival_constraints.sat(XW[:, 0:self.xdim])

    def modelQ(self, rm):

        #print('error%:', rm.error_pc(X, Y))
        # Matrices are extended to include w/pi
        # A = [AB]
        #     [00]
        # AB denotes matrix concatenation.
        # Hence a reset: x' =  Ax + Bw + b
        # can be mimicked as below
        #
        # [x']   = [a00 a01 a02...b00 b01...] * [x] + b + [e0]
        # [w']     [     ...  0 ...         ]   [w]       [e1]
        #
        # This makes x' \in Ax + Bw + b + [el, eh], and
        # makes w' \in [el, eh]
        # We use this to incorporate error and reset w to new values,
        # which in the case of ZOH are just the ranges of w (or pi).

        q = self
        A = np.vstack((rm.A, np.zeros((q.wdim, q.dim))))
        b = np.hstack((rm.b, np.zeros(q.wdim)))
        C, d = q.ival_constraints.poly()
        try:
            assert(A.shape[0] == b.shape[0])    # num lhs (states) is the same
            assert(A.shape[1] == C.shape[1])    # num vars (states + ip) are the same
            assert(C.shape[0] == d.shape[0])    # num constraints are the same
        except AssertionError as e:
            print('\n', A, '\n', b)
            print('\n', C, '\n', d)
            print(A.shape[0], b.shape[0], C.shape[1], d.shape[0])
            raise e
        return A, b

    def errorQ(self, include_err, X, Y, rm):
        q = self
        xic = (rm.error(X, Y) if include_err
               else IntervalCons([0.0]*q.xdim, [0.0]*q.xdim))

        e = IntervalCons.concatenate(xic, Qxw.wic)
        return e

    def __hash__(self):
        return hash(self.xwcell)

    def __eq__(self, qxw):
        return self.xwcell == qxw.xwcell

    def __str__(self):
        return str(self.xwcell)

    def __repr__(self):
        return 'Qxw'+str(self.xwcell)


class Qx(Q):
    def __init__(self, xcell):
        assert(isinstance(xcell, CM.Cell))

        self.xcell = xcell
        self.ival_constraints = xcell.ival_constraints
        self.xdim = xcell.dim
        self.dim = xcell.dim
        return

    def split(self, *args):
        return [Qx(c) for c in self.xcell.split(*args)]

    def sim(self, step_sim, x_array):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci, pi = [np.array([])]*2
        t0 = 0
        Yl = []

        for x in x_array:
            (t_, x_, s_, d_, pvt_, u_) = step_sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        return np.vstack(Yl)

    def get_rels(self, N, sim):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci, pi = [np.array([])]*2
        t0 = 0
        Yl = []

        #plt.figure()

        x_array = self.xcell.sample_UR(N)
        #print(t0)
        #print(x_array)
        for x in x_array:
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)
        #plt.show()
        #exit()
        #raw_input('drawn?')

        return x_array, np.vstack(Yl)

    def sat(self, X):
        """returns a sat array, whose elements are false when unsat
        and true when sat"""
        return self.ival_constraints.sat(X)

    def modelQ(self, rm):
        A, b = rm.A, rm.b
        return A, b

    def errorQ(self, include_err, X, Y, rm):
        e = (rm.error(X, Y) if include_err
             else IntervalCons([0.0]*self.xdim, [0.0]*self.xdim))
        return e

    def __hash__(self):
        return hash(self.xcell)

    def __eq__(self, q):
        return self.xcell == q.xcell

    def __str__(self):
        return str(self.xcell)

    def __repr__(self):
        return 'Qx'+str(self.xcell)


# Unused...should probably delete it if a usecase is not found.
class Qqxw(Q):
    # wic is the pi_cons of the overall system and has nothing to
    # do with wcell (of course wcell must be \in wic, but that is
    # all.
    wic = None

    @classmethod
    def init(cls, wic):
        cls.wic = wic

    def __init__(self, qx, wcell):
        """__init__

        Parameters
        ----------
        abs_state : abstract states
        w_cells : cells associated with the abstract state defining
        range of inputs
        """
        if Qxw.wic is None:
            raise NameError('Class was not initialized. wic is None')

        assert(isinstance(qx, Qx))
        assert(isinstance(wcell, CM.Cell))

        xcell = qx.xcell
        self.xcell = xcell
        self.wcell = wcell
        self.xwcell = CM.Cell.concatenate(xcell, wcell)
        self.sample_UR_x = self.xcell.sample_UR
        self.sample_UR_w = self.wcell.sample_UR
        self.xdim = xcell.dim
        self.wdim = wcell.dim
        self.ival_constraints = self.xwcell.ival_constraints
        return

    def split(self, *args):
        qsplits = self.qx.splits(*args)
        return [Qxw(qsplits, self.wcell) for q in qsplits]

    def get_rels(self, N, sim):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci = np.array([])
        t0 = 0
        Yl = []

        x_array = self.sample_UR_x(N)
        pi_array = self.sample_UR_w(N)
        for x, pi in zip(x_array, pi_array):
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        return np.hstack((x_array, pi_array)), np.vstack(Yl)

    def sat(self, X):
        """returns a sat array, whose elements are false when unsat
        and true when sat"""
        return self.xcell.ival_constraints.sat(X)

    def modelQ(self, rm):

        #print('error%:', rm.error_pc(X, Y))
        # Matrices are extended to include w/pi
        # A = [AB]
        #     [00]
        # AB denotes matrix concatenation.
        # Hence a reset: x' =  Ax + Bw + b
        # can be mimicked as below
        #
        # [x']   = [a00 a01 a02...b00 b01...] * [x] + b + [e0]
        # [w']     [     ...  0 ...         ]   [w]       [e1]
        #
        # This makes x' \in Ax + Bw + b + [el, eh], and
        # makes w' \in [el, eh]
        # We use this to incorporate error and reset w to new values,
        # which in the case of ZOH are just the ranges of w (or pi).

        q = self
        A = np.vstack((rm.A, np.zeros((q.wdim, q.xdim + q.wdim))))
        b = np.hstack((rm.b, np.zeros(q.wdim)))
        C, d = q.ival_constraints.poly()
        try:
            assert(A.shape[0] == b.shape[0])    # num lhs (states) is the same
            assert(A.shape[1] == C.shape[1])    # num vars (states + ip) are the same
            assert(C.shape[0] == d.shape[0])    # num constraints are the same
        except AssertionError as e:
            print('\n', A, '\n', b)
            print('\n', C, '\n', d)
            print(A.shape[0], b.shape[0], C.shape[1], d.shape[0])
            raise e
        return A, b

    def errorQ(self, *args):
        xic = self.qx.errorQ(*args)
        e = IntervalCons.concatenate(xic, Qxw.wic)
        return e

    def __hash__(self):
        return hash(self.xwcell)

    def __eq__(self, qxw):
        return self.xwcell == qxw.xwcell

    def __str__(self):
        return str(self.xwcell)

    def __repr__(self):
        return 'Qxw'+str(self.xwcell)
