
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as PLT

PLOT = True


class SIM(object):
    def __init__(self, plt, pvt_init_data):
        #print I
        # atol = 1e-10
        rtol = 1e-5
        # tt,YY,dummy_D,dummy_P
        self.solver = ode(dyn).set_integrator('dopri5', rtol=rtol)
        return

    def sim(self, TT, X0, D, P, U, W, property_checker):

        if PLOT:
            num_dim_x = len(X0)
            plot_data = [np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
        else:
            plot_data = None

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        self.solver.set_solout(solout_fun(property_checker, plot_data))  # (2)
        self.solver.set_initial_value(X0, t=0.0)
        self.solver.set_f_params(W)
        X_ = self.solver.integrate(T)

        pvf = property_checker.check(Tf, X_)

        dummy_D = np.zeros(D.shape)
        dummy_P = np.zeros(P.shape)
        ret_t = Tf
        ret_X = X_
        ret_D = dummy_D
        ret_P = dummy_P

        if PLOT:
            PLT.figure(5)
            PLT.plot(plot_data[0], plot_data[1][:, 0])

        return (ret_t, ret_X, ret_D, ret_P), pvf


# State Space Modeling Template
# dx/dt = Ax + Bu
# y = Cx + Du
def dyn(t, X, w):

    if w > 0:
        u = 1.0
    elif w < 0:
        u = -1.0
    else:
        u = 0.0

    x2 = u
    X_ = np.array([x2])
    return X_


def solout_fun(property_checker, plot_data):

    def solout(t, Y):
        if PLOT:
            plot_data[0] = np.concatenate((plot_data[0], np.array([t])))
            plot_data[1] = np.concatenate((plot_data[1], np.array([Y])))

        if property_checker.check(t, Y):
            #violating_state[0] = (np.copy(t), np.copy(Y))
            # print 'violation found:', violating_state[0]
            # return -1 to stop integration
            return -1
        else:
            return 0

        return 0

    return solout
