#!/usr/bin/python
# -*- coding: utf-8 -*-

#Bicycle Model with
# - normal force equilibrium for pitching-moments
# - linear tyre model
# state x=[X,Y,psi,vx,vy,omega]
# input u=[delta,omega_f,omega_r]

import numpy as np
from scipy.integrate import ode
import scipy.io as sio

import matplotlib.pyplot as plt

import utils as U
import copy_reg as cr
import types

PLOT = False


def _pickle_method(m):
    """_pickle_method
    Required for successfull pickling of sim()
    """
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

cr.pickle(types.MethodType, _pickle_method)


# As it is created only once, all methods should be static
# methods.
class SIM(object):

    def __init__(self, plt, pvt_init_data):
        self.U_matrix = sio.loadmat('../examples/bicycle_cora/u.mat')['v']
        #self.U_idx = 0
        property_checker = True

        # atol = 1e-10
        rtol = 1e-5
        rtol = 1e-6
        if property_checker is not None:
            max_step = 1e-2
        else:
            max_step = 0.0
        nsteps = 1000

        # tt,YY,dummy_D,dummy_P
        # The orer of these calls is strict... do not change
        # (1):set_integrator -> (2):set_solout -> (3):set_initial_value

        solver_name = 'dopri5'#'dopri5'
        self.solver = ode(dyn).set_integrator(solver_name)  # (1)

    #@U.memoize2disk(U.memoize_hash_method)
    def sim(self, TT, X0, D, P, I, property_checker):
        property_violated_flag = False
        num_dim_x = len(X0)

        plot_data = ([np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
                     if PLOT else None)

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        #if property_checker is not None:
        #violating_state = [()]
        solver = self.solver
        solver.set_solout(solout_fun(property_checker, plot_data))  # (2)

        solver.set_initial_value(X0, t=0.0)
        #solver.set_f_params(self.U_matrix[:, self.U_idx], I)
        solver.set_f_params(self.U_matrix[:, int(D[0])], I)
        #self.U_idx += 1
        X_ = solver.integrate(T)
        # Y = C*x + D*u

        #if property_checker is not None:
        if property_checker.check(Tf, X_):
            property_violated_flag = True

        dummy_P = np.zeros(P.shape)
        ret_D = D + 1
        ret_t = Tf
        ret_X = X_
        # ret_Y = Y
        ret_P = dummy_P

        #plt.plot(plot_data[0] + Ti, plot_data[1][:, 0])

        if PLOT:
            plt.plot(plot_data[1][:, 0], plot_data[1][:, 1], 'b-', linewidth=0.5)

        ##plt.plot(plot_data[0] + Ti, np.tile(U, plot_data[0].shape))

        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


def solout_fun(property_checker, plot_data):

    def solout(t, Y):
        if PLOT:
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

    return solout


#parameters: get parameters from p vector

#body
m = 1750
J = 2500
L = 2.7
l_F = 1.43
l_R = L-l_F
h = 0.5

#street
mu0 = 1
g = 9.81

#tires
#B=p(7),C=p(8) - paceijca parameter in ff_abs = sin(C*atan(B*sf_abs/mu0))
C_F = 10.4*1.3
C_R = 21.4*1.1


#function dx = vmodel_A_bicycle_linear_controlled(t,x,u)
def dyn(t, x, u_, w):
    dx = x.copy()
    u = u_ + w
    #X[0], X[1] = (X[1], 5.0 * (1 - X[0] ** 2) * X[1] - X[0])
    #return X

    #state
    #position
    X = x[0] #ok<NASGU>
    Y = x[1] #ok<NASGU>
    psi = x[2]

    #velocity
    vx = x[3]
    vy = x[4]
    omega = x[5]

    #acceleration
    Fb = x[6]*m
    delta = x[7]

    #control action
    # inputs are values of the state feedback matrix R, the reference state Xn,
    # and the feedforward value W

    #R = [u(1) u(2) u(3) u(4) u(5) u(6) u(7) u(8); u(9) u(10) u(11) u(12) u(13) u(14) u(15) u(16)]
    R = np.array([u[0:8], u[8:16]])
    #Xn = [u(17); u(18); u(19); u(20); u(21); u(22); u(23); u(24)]
    Xn = u[16:24]
    #W = [u(25); u(26)]
    W = u[24:26]
    #print(R.shape, x.shape, Xn.shape, W.shape)
    v = np.dot(-R, (x-Xn)) + W

    #calculate normal forces
    Fzf = (l_R*m*g-h*Fb)/(l_R+l_F)
    Fzr = m*g - Fzf

    #side-slip
    sf = (vy+l_F*omega)/vx-delta
    sr = (vy-l_R*omega)/vx

    #forces
    Fyf = -C_F * Fzf * sf
    Fyr = -C_R * Fzr * sr

    #ACCELERATIONS
    dvx = Fb/m + vy * omega
    dvy = (Fyf+Fyr)/m - vx * omega
    domega = (l_F * Fyf - l_R * Fyr)/J

    #position
    cp = np.cos(psi)
    sp = np.sin(psi)
    dx[0] = cp * vx - sp * vy
    dx[1] = sp * vx + cp * vy
    dx[2] = omega
    #velocity
    dx[3] = dvx
    dx[4] = dvy
    dx[5] = domega
    #accelerationm
    dx[6] = v[0]
    dx[7] = v[1]

    return dx
