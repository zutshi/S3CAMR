#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

"""Uses Assimulo solvers"""

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
#from assimulo.solvers import LSODAR
#from assimulo.solvers import RungeKutta34
import pylab as plt

import utils as U

PLT = False


def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


AA = -1
BB = -2
MAP = np.array([
                [4, 5, 5, 4, 5, 5, 5, 4, 6, 6, 5, 6, 6, 6, 6, 5, 5, 6, 4, 6, 4, 5, 5, 4, 6],
                [4, 5, 4, 5, 3, 4, 4, 4, 4, 5, 6, 6, 4, 6, 5, 6, 6, 6, 4, 5, 6, 6, 4, 5, 4],
                [3, 3, 4, 3, 3, 3, 3, 6, 4, 6, 5, 4, 4, 4, 5, 6, 4, 4, 6, 4, 4, 4, 5, 5, 4],
                [4, 5, 4, 3, 3, 3, 3, 3, 3, 5, 4, 6, 5, 6, 5, 6, 4, 6, 6, 5, 5, 4, 6, 4, 5],
                [3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 4, 5, 5, 6, 4, 6, 5, 6, 5, 5, 6],
                [4, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 6, 5, 6, 5, BB, 6, 5, 5, 5],
                [3, 3, 5, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 5, 6, 4, 6, 5, 6],
                [5, 3, 4, 4, 3, 3, 3, 5, 4, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 4, 5, 4],
                [5, 3, 5, 5, 3, 3, 3, 3, 6, 4, 5, 1, 2, 3, 1, 1, 3, 3, 3, 3, 3, 4, 4, 5, 5],
                [5, 4, 5, 4, 4, 3, 3, 3, 5, 6, 5, 6, 2, 2, 2, 1, 2, 1, 3, 3, 3, 5, 6, 6, 4],
                [4, 5, 4, 5, 5, 3, 3, 3, 4, 4, 5, 6, 6, 6, 1, 1, 3, 3, 3, 4, 5, 5, 4, 5, 5],
                [4, 4, 3, 5, 4, 4, 3, 3, 3, 5, 5, 4, 6, 5, 4, 1, 1, 3, 3, 3, 5, 4, 4, 4, 6],
                [4, 5, 4, 5, 4, 5, 3, 3, 3, 4, 4, 6, 4, 4, 5, 1, 1, 3, 3, 3, 4, 6, 4, 6, 5],
                [5, 4, 4, 4, 3, 4, 3, 3, 3, 5, 5, 5, 5, 4, 4, 2, 2, 3, 3, 3, 4, 5, 4, 5, 4],
                [5, 4, 3, 3, 3, 4, 3, 3, 3, 3, 5, 4, 5, 4, 5, 6, 2, 3, 3, 3, 3, 5, 6, 4, 5],
                [5, 4, 4, 5, 3, 4, 4, 3, 3, 3, 4, 5, 6, 4, 4, 5, 1, 2, 3, 3, 3, 5, 5, 5, 5],
                [3, 5, 3, 3, 5, 4, 4, 3, 3, 3, 6, 5, 6, 6, 6, 6, 2, 3, 3, 3, 3, 5, 4, 4, 5],
                [5, 3, 3, 5, 5, 3, 4, 6, 3, 3, 3, 3, 3, 5, 6, 4, 4, 4, 4, 3, 3, 5, 5, 5, 5],
                [2, 3, 2, 3, 1, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 3],
                [3, 2, 3, 2, 1, 3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
                [2, 1, 3, 2, 3, 1, 1, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, AA, 6, 6, 6],
                [2, 2, 1, 1, 2, 2, 1, 3, 2, 1, 2, 1, 3, 3, 3, 2, 2, 1, 2, 1, 1, 0, 7, 7, 7],
                [2, 1, 2, 2, 2, 1, 2, 3, 1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 2, 1, 1, 0, 7, 7, 6],
                [2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 0, 7, 6, 6],
                [1, 3, 3, 3, 3, 2, 3, 2, 1, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 7, 6, 6]])

MatA = np.array([[-0.5, 0.6], [0.2, -0.7]])
NUM_MODES = MAP.size
#size(MAP,1)*size(MAP,2);
#OUT = np.nan
NAV_SQ_DIM = MAP.shape[0]


@memodict
def modeToij(mode):
    i = int(np.ceil((mode+1)/float(NAV_SQ_DIM)))
    j = ((mode) % NAV_SQ_DIM) + 1
    return i-1, j-1


def AbMap(i, j):
    k = MAP[i, j]
    vx = np.sin(k*np.pi/4)
    vy = np.cos(k*np.pi/4)
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, MatA[0, 0], MatA[0, 1]],
        [0, 0, MatA[1, 0], MatA[1, 1]]])
    b = -np.array([0, 0, MatA[0, 0]*vx+MatA[0, 1]*vy, MatA[1, 0]*vx+MatA[1, 1]*vy])
    return A, b


class Pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)


@memodict
class Bounds():
    def __init__(self, mode):
        i, j = modeToij(mode)
        #print('i,j:', i,j)
        self.l = j if mode % NAV_SQ_DIM != 0 else -np.inf
        self.r = j + 1 if (mode+1) % NAV_SQ_DIM != 0 else np.inf
        mode_a = mode - NAV_SQ_DIM
        mode_b = mode + NAV_SQ_DIM
        self.a = NAV_SQ_DIM - i if mode_a >= 0 else np.inf
        self.b = NAV_SQ_DIM - i - 1 if mode_b <= NUM_MODES-1 else -np.inf

    def __str__(self):
        return 'x=[{}, {}], y=[{}, {}]'.format(self.l, self.r, self.b, self.a)


@memodict
def get_dyn(mode):
    i, j = modeToij(mode)
    if __debug__:
        print(i, j)
    A, b = AbMap(i, j)
    if __debug__:
        print(A)
        print(b)
    return A, b



TOL = 1e-3


def right_ness(pos, bounds):
    return (bounds.r+TOL) - pos.x


def left_ness(pos, bounds):
    return pos.x - (bounds.l-TOL)


def above_ness(pos, bounds):
    #return pos.y - (bounds.a-TOL)
    return (bounds.a+TOL) - pos.y


def below_ness(pos, bounds):
    #return (bounds.b+TOL) - pos.y
    return pos.y - (bounds.b-TOL)


# def is_right(pos, bounds):
#     return pos.x > bounds.r


# def is_left(pos, bounds):
#     return pos.x < bounds.l


# def is_above(pos, bounds):
#     return pos.y < bounds.a


# def is_below(pos, bounds):
#     return pos.y > bounds.b


def get_guard_vals(X, mode):

    bounds = Bounds(mode)
    pos = Pos(X[0], X[1])
    if __debug__:
        print('mode:', mode)
        print(bounds)
        print(pos)
    return [right_ness(pos, bounds),
            left_ness(pos, bounds),
            above_ness(pos, bounds),
            below_ness(pos, bounds)]


@memodict
def right(mode):
#     if (mode+1) % NAV_SQ_DIM == 0:
#         if __debug__:
#             print('returning OUT')
#         return OUT
#     else:
    return mode + 1


@memodict
def left(mode):
#     if mode % NAV_SQ_DIM == 0:
#         if __debug__:
#             print('returning OUT')
#         return OUT
#     else:
    return mode - 1


@memodict
def above(mode):
    mode_ = mode - NAV_SQ_DIM
#     if mode_ < 0:
#         if __debug__:
#             print('returning OUT')
#         return OUT
#     else:
    return mode_


@memodict
def below(mode):
    mode_ = mode + NAV_SQ_DIM
#     if mode_ > NUM_MODES-1:
#         if __debug__:
#             print('returning OUT')
#         return OUT
#     else:
    return mode_


def new_mode(g, mode):
    #print(g)
    ctr = 0
    if g[0] != 0:
        mode_ = right(mode)
        ctr += 1
    if g[1] != 0:
        mode_ = left(mode)
        ctr += 1
    if g[2] != 0:
        mode_ = above(mode)
        ctr += 1
    if g[3] != 0:
        mode_ = below(mode)
        ctr += 1
    # more than 1 guard condition should not be sat: indicates a
    # diagonal transfer!
    #print(ctr)
    assert(ctr <= 1)
    return mode_


# def guard(pos, mode):
#     bound = get_mode_bounds(mode)
#     if is_right(pos, bound):
#         mode_ = right(mode)
#     elif is_left(pos, bound):
#         mode_ = left(mode)
#     elif is_above(pos, bound):
#         mode_ = above(mode)
#     elif is_below(pos, bound):
#         mode_ = below(mode)
#     else:
#         pass

def x2mode(X):
    xh, yh = int(np.floor(X[0])), int(np.floor(X[1]))
    mode = xh + (NAV_SQ_DIM-1 - yh) * NAV_SQ_DIM
    return mode


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    @U.memoize2disk(U.memoize_hash_method)
    def sim(self, TT, X0, D, P, U, I, property_checker):

        m0 = x2mode(X0)

        # If out side the grid, ignore
        if (X0[0] < 0 or X0[0] > 25 or X0[1] < 0 or X0[1] > 25):
            return (TT[0], X0, D, P), False
        #    return (TT[0], X0, D, P), False
        #else:
            #print('ALL OK...mode:', m0)

        #print X0
        #Simulation
        ncp = 200     #Number of communication points
        sw0 = mode2sw(m0)
        self.model.re_init(TT[0], X0, sw0=sw0)
        if __debug__:
            print('re-init:', TT[0], TT[1], X0, x2mode(X0))
        t, y = self.model.simulate(TT[1], ncp) #Simulate
        #Print event information
        #sim.print_event_data()

        # t is a list for some reasons. Maybe there is a better way to
        # fix this.
        t = np.array(t)
        # violating values
        (t_, y_), property_violated = property_checker.first_sat_value_or_end(t, y)

        # if xv and tv are empty, no violation were found

        ret_t, ret_X = t_, y_
        ret_D = D
        ret_P = P
        if PLT:
            plt.plot(y[:, 0], y[:, 1], 'b-', linewidth=2)
        #plt.plot(t, y[:, 1], 'r-', linewidth=2)
        return (ret_t, ret_X, ret_D, ret_P), property_violated


def sw2mode(sw):
    modes = [idx for idx, e in enumerate(sw) if e]
    # Somehow asismulo suppresses this assert! Using prints instead
    #assert(len(modes) == 1)
    if(len(modes) != 1):
        print('WARNING:', modes)
    return modes[0]


@memodict
def mode2sw(mode):
    sw = [False]*NUM_MODES
    sw[mode] = True
    return sw


def create_model():
    def nav(t, X, sw):
        """
        The ODE to be simulated. The parameter sw should be fixed during
        the simulation and only be changed during the event handling.
        """
        A, b = get_dyn(sw2mode(sw))
        return np.dot(A, X) + b

    def state_events(t, X, sw):
        """
        This is our function that keep track of our events, when the sign
        of any of the events has changed, we have an event.
        """
        #TODO: is this the best way?
        mode = sw2mode(sw)
        g = get_guard_vals(X, mode)
        G = [g[0], g[1], g[2], g[3]] # y == 0
        if __debug__:
            print(mode)
            print('G =', G)
        return G

    def handle_event(solver, event_info):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        state_info = event_info[0] #We are only interested in state events info
        if __debug__:
            print('############### EVENT DETECTED')
        g = state_info
        if g[0] <= 0 or g[1] <= 0 or g[2] <= 0 or g[3] <= 0:
            mode = sw2mode(solver.sw)
            mode_ = new_mode(g, mode)
            if __debug__:
                print('############### new_mode =', mode_)
            solver.sw = mode2sw(mode_)

    #Initial values
    y0 = [0., 0., 0., 0.] #Initial states
    t0 = 5.0             #Initial time
    switches0 = [False]*NUM_MODES   #Initial switches
    # Without the below statemment, it hits an error
    switches0[79] = True

    #Create an Assimulo Problem
    mod = Explicit_Problem(nav, y0, t0, sw0=switches0)

    mod.state_events = state_events #Sets the state events to the problem
    mod.handle_event = handle_event #Sets the event handling to the problem
    mod.name = 'nav30'   #Sets the name of the problem

    #Create an Assimulo solver (CVode)
    sim = CVode(mod)
    #sim = LSODAR(mod)
    #sim = RungeKutta34(mod)
    #sim.options['verbosity'] = 20 #LOUD
    #sim.options['verbosity'] = 40 #WHISPER
    sim.verbosity = 40 #WHISPER
    #sim.display_progress = True
    #sim.options['minh'] = 1e-4
    #sim.options['rtol'] = 1e-3

#     #Specifies options
#     sim.discr = 'Adams'     #Sets the discretization method
#     sim.iter = 'FixedPoint' #Sets the iteration method
#     sim.rtol = 1.e-8        #Sets the relative tolerance
#     sim.atol = 1.e-6        #Sets the absolute tolerance

    return sim

