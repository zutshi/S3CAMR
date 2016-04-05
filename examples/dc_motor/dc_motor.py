# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt

PLOT = False

# TODO: (1) function signatures!
#       (2) all arrays should be matrices!


class SIM(object):
    def __init__(self, plt, pvt_init_data):
        # atol = 1e-10
        rtol = 1e-5
        self.solver = ode(dyn).set_integrator('dopri5', rtol=rtol)
        return

    def plant_sim(self, T, X0, U, pc):
        num_dim_x = len(X0)
        plot_data = [np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
        # tt,YY,dummy_D,dummy_P
        self.solver.set_solout(solout_fun(pc, plot_data))  # (2)
        self.solver.set_initial_value(X0, t=0.0)
        self.solver.set_f_params(U)
        X_ = self.solver.integrate(T)
        pvf = pc.check(T, X_)
        # Y = C*x + D*u

        #plt.plot(plot_data[0] + Ti, plot_data[1][:, 0])
        #plt.plot(plot_data[0] + Ti, plot_data[1][:, 1])
        if PLOT:
            plt.plot(plot_data[1][:, 0], plot_data[1][:, 1])
        ##plt.plot(plot_data[0] + Ti, np.tile(U, plot_data[0].shape))

        return X_, pvf

    def sim(self, TT, X0, D, P, _, W, property_checker):
        print(TT, X0, D, P, W)

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        u, D_ = controller(X0, D, W)
        X_, pvf = self.plant_sim(T, X0, u, property_checker)

        dummy_P = np.zeros(P.shape)
        ret_t = Tf
        ret_X = X_
        # ret_Y = Y
        ret_D = D_
        ret_P = dummy_P
        return (ret_t, ret_X, ret_D, ret_P), pvf

# def dyn(t, x, u):
#     u = np.matrix([u[0], 0.0]).T
#     x = np.matrix(x).T
#     X_ = A*x + B*u
#     return np.array(X_.T)


def controller(X, D, W):
    """controller
    A PI controller for the dc motor.
    Parameters
    ----------
    Y: plant output
    D: controller's discrete state
    W: exogenous noise

    Returns
    -------
    u: control output
    D_: new conrtoller state
    """
    # ##################### PARAMS
    SAT = 20.0
    UPPER_SAT = SAT
    LOWER_SAT = -SAT
    KP = 40.0
    KI = 1.0
    # ##################### PARAMS

    # reference signal
    ref = 1.0

    # TODO: rmeove D
    error_i_prev = D

    y = X[0]

    # error computation is affected by bounded sensor noise
    error = ref - (y + W)
    # to illustrate: ei += e*Ki
    error_i = error * KI + error_i_prev
    error_i_prev = error_i
    pid_op = error * KP + error_i * KI

    if pid_op > UPPER_SAT:
        pid_op = UPPER_SAT
    elif pid_op < LOWER_SAT:
        pid_op = LOWER_SAT
    else:
        pid_op = pid_op

    D_ = error_i_prev
    return pid_op, D_


# State Space Modeling Template
# dx/dt = Ax + Bu
# y = Cx + Du
def dyn(t, X, u):
    # ##################### PARAMS
    J = 0.01
    K = 0.01
    L = 0.5
    R = 1.0
    b = 0.1

    A = np.matrix([[-b/J, K/J], [-K/L, -R/L]])
    B = np.matrix([[0.0, -1/J], [1/L, 0.0]])
    C = np.matrix([1.0, 0.0])
    D = np.matrix([0.0, 0.0])

    # ##################### PARAMS

    Y = X.copy()
    u = np.matrix([u, 0.0]).T
    x = np.matrix(X).T
    Y = A*x + B*u
    return np.array(Y.T)


def solout_fun(property_checker, plot_data):

    def solout(t, Y):

        plot_data[0] = np.concatenate((plot_data[0], np.array([t])))
        plot_data[1] = np.concatenate((plot_data[1], np.array([Y])))

        # print Y
        # print t, Y

        if property_checker.check(t, Y):
            #violating_state[0] = (np.copy(t), np.copy(Y))
            # print 'violation found:', violating_state[0]
            # return -1 to stop integration
            return -1
        else:
            return 0
        return 0

    return solout
