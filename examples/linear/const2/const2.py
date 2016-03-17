# -*- coding: utf-8 -*-

"""
Defines several constant time dynamical systems for
testing/benchmarking

MUST define the global SYS_ID after loading the module
"""

import random

import numpy as np
from scipy.integrate import ode


class SIM(object):

    def __init__(self, plt, pvt_init_data):
        """__init__

        Parameters
        ----------
        plt :
        pvt_init_data : pvt_init_data must contain sys_id
        """
        #sys_id = int(raw_input('enter system ID: '))
        self.dyn = dynamics(int(pvt_init_data))

    def sim(self, TT, X0, D, P, U, I, property_checker, property_violated_flag):
        # atol = 1e-10
        rtol = 1e-6
        rtol = 1.0

        if property_checker is not None:
            max_step = 1e-2
        else:
            max_step = 0.0
        nsteps = 1000
        num_dim_x = len(X0)
        plot_data = [np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]

        # tt,YY,dummy_D,dummy_P
        # The orer of these calls is strict... do not change
        # (1):set_integrator -> (2):set_solout -> (3):set_initial_value

#         solver = ode(self.dyn).set_integrator('dopri5', rtol=rtol, max_step=max_step,
#                                          nsteps=nsteps)  # (1)
        solver = ode(self.dyn).set_integrator(
                'dopri5',
                rtol=rtol,
                #max_step=max_step,
                #nsteps=nsteps)  # (1)
                )

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        if property_checker:
            violating_state = [()]
            #solver.set_solout(solout_fun(property_checker, violating_state, plot_data))  # (2)

        solver.set_initial_value(X0, t=0.0)
        solver.set_f_params(U)
        X_ = solver.integrate(T)
        # Y = C*x + D*u

        if property_checker is not None:
            if property_checker(Tf, X_):
                property_violated_flag[0] = True

        dummy_D = np.zeros(D.shape)
        dummy_P = np.zeros(P.shape)
        ret_t = Tf
        ret_X = X_
        # ret_Y = Y
        ret_D = dummy_D
        ret_P = dummy_P

        #plt.plot(plot_data[0] + Ti, plot_data[1][:, 0])
        #plt.plot(plot_data[1][:, 0], plot_data[1][:, 1])
        ##plt.plot(plot_data[0] + Ti, np.tile(U, plot_data[0].shape))

        return (ret_t, ret_X, ret_D, ret_P)


def solout_fun(property_checker, violating_state, plot_data):

    def solout(t, Y):

        plot_data[0] = np.concatenate((plot_data[0], np.array([t])))
        plot_data[1] = np.concatenate((plot_data[1], np.array([Y])))

        # print Y
        # print t, Y

#        if property_checker(t, Y):
#            pvf_local[0] = True
#            violating_state[0] = (np.copy(t), np.copy(Y))
#
#            # print 'violation found:', violating_state[0]

        return 0

    return solout


################################################
################ Dynamics ######################
################################################

# Bad things happen when you modify the passed in X.
# So, make a copy!
def dyn_1(t, X, u):
    '''regular dynamics'''
    X = X.copy()
    X[0], X[1] = (1, 2)
    return X


def dyn_2(t, X, u):
    '''slower dynamics'''
    X = X.copy()
    X[0], X[1] = (0.1, 0.2)
    return X


def dyn_3(t, X, u):
    '''switching dynamics: simple, independant
        each dimension is independant of the other'''
    X = X.copy()
    X[0] = np.ceil(X[0]/2)
    X[1] = np.ceil(X[1]/2)
    return X


def dyn_4(t, X, u):
    '''switching dynamics: simple, dependant
        each dimension is dependant on the other'''
    X = X.copy()
    X[0] = 2*np.ceil(X[1]/2)
    X[1] = 2*np.ceil(X[0]/2)
    return X


def dyn_5():
    """PWA dynamics
    Each cell in the grid is assinged dynamics randomly
    These grid dynamics randomly chosen but deterministic accross runs

    - grid size can be changed with grid_eps
    - rngx states the range of derivative"""

    rngx = np.array([[-10, 10],
                     [-10, 10]])
    grid_eps = 0.2
    R = random.Random(0) # seed = 0

    def dyn(t, X, u):
        X = X.copy()
        cell0, cell1 = np.ceil(X[0]/grid_eps), np.ceil(X[1]/grid_eps)
        print cell0, cell1
        R.seed(cell0)
        X[0] = R.random()
        R.seed(cell1)
        X[1] = R.random()
        X_ = rngx[:, 0] + (rngx[:, 1] - rngx[:, 0])*X
        if abs(X_[0]) > 10:
            print X
        #print X_
        return X_

    return dyn


def dynamics(dyn_id):
    sys_dyn_map = {
        1: dyn_1,
        2: dyn_2,
        3: dyn_3,
        4: dyn_4,
        5: dyn_5(),
        }
    return sys_dyn_map[dyn_id]
