from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#import itertools as it
import abc
import six

import numpy as np

import constraints as cons
#import err
from utils import print

import settings
from IPython import embed

from . import cellmanager as CM

if settings.debug and settings.plot:
    import matplotlib.pyplot as plt


@six.add_metaclass(abc.ABCMeta)
class Q(object):
    #__metaclass__ = abc.ABCMeta

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
    def errorQ(self, include_err, rm):
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

    # TODO: a is being as a result of a hack!
    def __init__(self, a, xcell, wcell):
        """__init__

        Parameters
        ----------
        a : abstract state
        xcell: cell over state space
        w_cells : cells associated with the abstract state defining
        range of inputs

        Notes
        -------
        It is not clear if abstract state should be here
        """
        if Qxw.wic is None:
            raise NameError('Class was not initialized. wic is None')

        assert(isinstance(xcell, CM.Cell))
        assert(isinstance(wcell, CM.Cell))

        self.a = a
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

    def get_rels(self, prop, sim, N):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci = np.array([])
        t0 = 0
        Yl = []

        x_array = self.sample_UR_x(N)
        pi_array = self.sample_UR_w(N)
        for x, pi in zip(x_array, pi_array):
            #print(pi)
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        return np.hstack((x_array, pi_array)), np.vstack(Yl)

    def sat(self, XW):
        """returns a sat array, whose elements are false when unsat
        and true when sat"""
        return self.xcell.ival_constraints.sat(XW[:, 0:self.xdim])

    def modelQ(self, rm):

        #print('error%:', rm.max_error_pc(X, Y))
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

    def errorQ(self, include_err, rm):
        xic = (rm.fit_error if include_err
               else cons.zero2ic(self.xdim))

        e = cons.IntervalCons.concatenate(xic, Qxw.wic)
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

        #self.a = a
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

#         if settings.debug and settings.plot:
#             print('plotting grids Qx')
#             ax = plt.gca()

            # plot errors
            # van der pol
#             epsx0 = 0.2
#             epsx1 = 0.2
#             yrng = (-9, 9)
#             xrng = (-3, 3)

#             # bball
#             epsx0 = 50.1
#             epsx1 = 10.1
#             xrng = (-50, 500)
#             yrng = (0, 60)

#             for i in np.arange(0, yrng[1], epsx1):
#                 ax.axhline(i, linestyle='-', color='k')
#             for i in np.arange(0, yrng[0], -epsx1):
#                 ax.axhline(i, linestyle='-', color='k')
#             for i in np.arange(0, xrng[1], epsx0):
#                 ax.axvline(i, linestyle='-', color='k')
#             for i in np.arange(0, xrng[0], -epsx0):
#                 ax.axvline(i, linestyle='-', color='k')

            # For van der pol only
            # plot error states
            #ax.axhline(-5.6, linestyle='-', color='r')
            #ax.axhline(-6.5, linestyle='-', color='r')
            #ax.axvline(-1, linestyle='-', color='r')
            #ax.axvline(-0.7, linestyle='-', color='r')
            #embed()

        return np.vstack(Yl)

    def get_rels(self, prop, sim, N):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci, pi = [np.array([])]*2
        t0 = 0
        Yl = []

        #plt.figure()

        x_array_ = self.xcell.sample_UR(N)
        # remove any sample drawn from the property box
        x_array = x_array_[~prop.final_cons.sat(x_array_), :]
        if x_array.size == 0:
            print(prop.final_cons)
            print(self.xcell.ival_constraints)
        #print(t0)
        #print(x_array)
        for x in x_array:
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)
        if settings.debug_plot:
            # close intermediate plots which can not be switched off;
            # as these are in the system model *.py
            #plt.title('ignore')
            #plt.show()
            #plt.close()
            pass

        # return empty arrays
        if x_array.size == 0:
            return x_array, x_array
        else:
            return x_array, np.vstack(Yl)

    def sat(self, X):
        """returns a sat array, whose elements are false when unsat
        and true when sat"""
        return self.ival_constraints.sat(X)

    def modelQ(self, rm):
        A, b = rm.A, rm.b
        return A, b

    def errorQ(self, include_err, rm):
        e = (rm.fit_error if include_err
             else cons.zero2ic(self.xdim))
        return e

    def poly(self):
        C, d = self.ival_constraints.poly()
        return C, d

    def __hash__(self):
        return hash(self.xcell)

    def __eq__(self, q):
        return self.xcell == q.xcell if isinstance(q, self.__class__) else False

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

    def get_rels(self, prop, sim, N):
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

        #print('error%:', rm.max_error_pc(X, Y))
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
        # The reason for this is to get by without modifying SAL's
        # encoding. Might not be the best way to proceed in general,
        # as we MIGHT be able to reduce the transitions system's size.

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
        # This is a hack to add the constraint wi \in [wl, wh]
        e = cons.IntervalCons.concatenate(xic, Qxw.wic)
        return e

    def __hash__(self):
        return hash(self.xwcell)

    def __eq__(self, qxw):
        return self.xwcell == qxw.xwcell

    def __str__(self):
        return str(self.xwcell)

    def __repr__(self):
        return 'Qxw'+str(self.xwcell)
