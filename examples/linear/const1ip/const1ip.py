# -*- coding: utf-8 -*-

"""
Defines several constant time dynamical systems for
testing/benchmarking

MUST define the global SYS_ID after loading the module
"""

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
        if pvt_init_data is None:
            raise Exception(
                'use --pvt-init-data <test_case_number> to run the '
                'test case')
        self.dyn = dynamics(int(pvt_init_data))
        self.solver = ode(self.dyn).set_integrator(
                    'dopri5',
                    #rtol=rtol,
                    #max_step=max_step,
                    #nsteps=nsteps)  # (1)
                    )

    def sim(self, TT, X0, D, P, U, W, property_checker):

        #num_dim_x = len(X0)
        #plot_data = [np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]

        # tt,YY,dummy_D,dummy_P
        # The orer of these calls is strict... do not change
        # (1):set_integrator -> (2):set_solout -> (3):set_initial_value

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        #solver.set_solout(solout_fun(property_checker, violating_state, plot_data))  # (2)

        self.solver.set_initial_value(X0, t=0.0)
        self.solver.set_f_params(W)
        X_ = self.solver.integrate(T)
        # Y = C*x + D*u

        property_violated_flag = property_checker.check(Tf, X_)

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

        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


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
# ############## Dynamics ######################
################################################

# Bad things happen when you modify the passed in X.
# So, make a copy!
def dyn_1(t, X, w):
    '''regular dynamics'''
    X = X.copy()
    X[0], X[1] = (1+w[0], 2+w[1])
    return X


def dyn_2(t, X, w):
    '''slower dynamics'''
    X = X.copy()
    X[0], X[1] = (0.1+w[0], 1.34+w[1])
    return X


def dyn_3(t, X, w):
    '''switching dynamics: simple, independant
        each dimension is independant of the other'''
    X = X.copy()
    X[0] = np.ceil(X[0]/2) + w[0]
    X[1] = np.ceil(X[1]/2) + w[1]
    return X


def dyn_4(t, X, w):
    '''switching dynamics: simple, dependant
        each dimension is dependant on the other'''
    X = X.copy()
    X[0] = 2*np.ceil(X[1]/2) + w[0]
    X[1] = 2*np.ceil(X[0]/2) + w[1]
    return X


def dynamics(dyn_id):
    sys_dyn_map = {
        1: dyn_1,
        2: dyn_2,
        3: dyn_3,
        4: dyn_4,
        }
    return sys_dyn_map[dyn_id]
