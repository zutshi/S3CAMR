#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np
from scipy.integrate import ode

import pylab as plt

PLOT = False


class SIM(object):

    def __init__(self, plt, pvt_init_data):
        # atol = 1e-10
        rtol = 1e-6
        nsteps = 1000

        # The orer of these calls is strict... do not change
        # (1):set_integrator -> (2):set_solout -> (3):set_initial_value

        self.solver = ode(dyn).set_integrator(
                'dopri5', rtol=rtol,
                #max_step=max_step,
                #nsteps=nsteps
                )  # (1)

    def sim(self, TT, X0, D, P, U, I, property_checker):
        num_dim_x = len(X0)
        plot_data = ([np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
                     if PLOT else None)

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        solver = self.solver
        solver.set_solout(solout_fun(property_checker, plot_data))  # (2)

        solver.set_initial_value(X0, t=0.0)
        solver.set_f_params(U)
        X_ = self.solver.integrate(T)

        property_violated_flag = property_checker.check(Tf, X_)

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


def solout_fun(property_checker, plot_data):

    def solout(t, Y):
        if PLOT:
            plot_data[0] = np.concatenate((plot_data[0], np.array([t])))
            plot_data[1] = np.concatenate((plot_data[1], np.array([Y])))

        # print Y
        # print t, Y
        if property_checker.check(t, Y):
            #print 'violation found:'#, violating_state[0]
            # return -1 to stop integration
            return -1
        else:
            return 0

    return solout


def dyn(t, X, u):
    # Params
    a = 1.0
    b = 2.5

    Y = X.copy()

    Y[0] = 1 + a * (X[0]**2) * X[1] - (b+1) * X[0]
    Y[1] = b * X[0] - a * (X[0]**2) * X[1]

    return Y
