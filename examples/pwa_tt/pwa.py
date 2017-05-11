# -*- coding: utf-8 -*-

"""
PWA System: Time triggered switched linear system w/o inputs

MUST define the global SYS_ID after loading the module
"""

import random

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

import utils as U

PLOT = False


class SIM(object):

    def __init__(self, plt, pvt_init_data):
        """__init__

        Parameters
        ----------
        plt :
        pvt_init_data : pvt_init_data must contain sys_id
        """
        self.dyn = dynamics(int(pvt_init_data))

    @U.memoize2disk(U.memoize_hash_method)
    def sim(self, TT, X0, D, P, U, I, property_checker):
        property_violated_flag = False
        num_dim_x = len(X0)

        if PLOT:
            plt.plot(X0[0], X0[1], 'r*')

        plot_data = ([np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
                     if PLOT else None)

        solver = ode(self.dyn).set_integrator('dopri5')
        solver.set_solout(solout_fun(property_checker, plot_data))  # (2)

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        solver.set_initial_value(X0, t=0.0)
        solver.set_f_params(U)
        X_ = solver.integrate(T)
        # Y = C*x + D*u

        if property_checker.check(Tf, X_):
            property_violated_flag = True

        dummy_D = np.zeros(D.shape)
        dummy_P = np.zeros(P.shape)
        ret_t = Tf
        ret_X = X_
        # ret_Y = Y
        ret_D = dummy_D
        ret_P = dummy_P

        if PLOT:
            plt.plot(plot_data[1][:, 0], plot_data[1][:, 1])

        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


##################################################
# ----------------- Dynamics ---------------------
##################################################


# Bad things happen when you modify the passed in X.
# So, make a copy!
def dyn_1(t, X, u):
    '''decoupled dynamics'''
    X = X.copy()

    #T0 = 0.0
    A0 = np.array([
            [1., 0., 0.],
            [0., -1.5, 0.],
            [0., 0., 0.]
            ])
    T1 = 2.0
    A1 = np.array([
            [-1., 2., 0.],
            [3.4, 0.5, 0.],
            [0., 0., 0.]
            ])
    T2 = 3.0
    A2 = np.array([
            [-5., -3.6, 0.],
            [4., -1., 0.],
            [0., 0., 0.]
            ])
    T3 = 6.0
    A3 = np.array([
            [1., 6., 0.],
            [-5., -2.2, 0.],
            [0., 0., 0.]
            ])
    b = np.array((0., 0., 1.))
    # overwrite the fake time with real time
    t = X[2]
    if t < T1:
        A = A0
    elif t < T2:
        A = A1
    elif t < T3:
        A = A2
    else:
        A = A3

    return np.dot(A, X) + b


def dynamics(dyn_id):
    sys_dyn_map = {
        1: dyn_1,
        }
    return sys_dyn_map[dyn_id]


def solout_fun(property_checker, plot_data):

    def solout(t, Y):
        if PLOT:
            plot_data[0] = np.concatenate((plot_data[0], np.array([t])))
            plot_data[1] = np.concatenate((plot_data[1], np.array([Y])))

        if property_checker.check(t, Y):
            return -1
        else:
            return 0

    return solout
