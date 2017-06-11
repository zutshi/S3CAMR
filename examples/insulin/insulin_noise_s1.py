#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np
from scipy.integrate import ode

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
        return None

    #@U.memoize2disk(U.memoize_hash_method)
    def sim(self, TT, X0, D, P, I, property_checker):
        property_violated_flag = False
        num_dim_x = len(X0)

        plot_data = ([np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
                     if PLOT else None)

        Ti = TT[0]
        Tf = TT[1]
        #T = Tf - Ti
        T_ = Ti

        while T_ <= Tf:
            T_, X_, D_ = euler_intg_plant(Ti, X0, D[0], I)
            Ti, X0, D = T_, X_, np.array([D_])

        #if property_checker is not None:
        if property_checker.check(Tf, X_):
            property_violated_flag = True

        dummy_P = np.zeros(P.shape)
        ret_t = T_
        ret_X = X_
        # ret_Y = Y
        ret_D = np.array([D_])
        ret_P = dummy_P

        #plt.plot(plot_data[0] + Ti, plot_data[1][:, 0])

        if PLOT:
            plt.plot(plot_data[1][:, 0], plot_data[1][:, 1], 'b-', linewidth=0.5)

        ##plt.plot(plot_data[0] + Ti, np.tile(U, plot_data[0].shape))

        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


def euler_intg_plant(t, state, mode, w):
    # reachability setting
    # stepsize = 0.2;
    # rs.setFixedOrder(3);
    h = 0.2

    X, Isc1, Isc2, Gt, Gp, Il, Ip, I1, Id, Gs, t, uc = state

    # Add noise
    Gs += w

    # mode_0_to_1; G_geq_80
    if mode == 0:
        if (-Gs + 80 <= 0):
            mode = 1
    elif mode == 1:
        # mode_1_to_0; G_leq_75
        if(Gs - 75):
            mode = 0
        # mode_1_to_2: G_geq_120
        if(-Gs + 120):
            mode = 2
    elif mode == 2:
        #mode_2_to_1: G_leq_115
        if(Gs - 115):
            mode = 1
        #mode_2_to_3: G_geq_180
        if(-Gs + 180):
            mode = 3
    elif mode == 3:
        #mode_3_to_2: G_leq_175
        if(Gs - 175):
            mode = 2
        #mode_3_to_4: G_geq_300
        if(-Gs + 300):
            mode = 4
    elif mode == 4:
        #mode_4_to_3: G_leq_295
        if(Gs - 295):
            mode = 3
    else:
        raise RuntimeError

    # System is deterministic and hence invariants are redundant
#     #inv_mode_0: G_leq_80
#     elif(Gs - 80)
#     #inv_mode_1: G_geq_75 /\ G_leq_120
#     elif(-Gs + 75 and Gs - 120)
#     #inv_mode_2: G_geq_115 /\ G_leq_180
#     elif(-Gs + 115 and Gs - 180)
#     #inv_mode_3: G_geq_175/\ G_leq_300
#     elif(-Gs + 175 and Gs - 300):
#     #inv_mode_4: G_geq_295
#     elif(-Gs + 295):

    # meal coefficients
    # t <= 30
    if(t <= 30):
        meal_coeff = (1.140850553428184e-4)*(t**2)*uc + (6.134609247812877e-5)*t*uc
    # 30 <= t <= 80
    elif(t >= 30 and t <= 80):
        meal_coeff = (5.252705445621361e-5)*(t**2)*uc - 0.007468661585336*t*uc + 0.281302431132355*uc
    # 80 <= t <= 360
    elif(t >= 80 and t <= 360):
        meal_coeff = (1.245404770117318e-7)*(t**2)*uc - (9.112517786233733e-5)*t*uc + 0.026475608001390*uc
    # 360 <= t <= 400
    elif(t >= 360 and t <= 400):
        meal_coeff = -(6.307316106980570e-5)*(t**2)*uc + 0.048262107570417*t*uc - 9.190266060911723*uc
    # 400 <= t <= 500
    elif(t >= 400 and t <= 500):
        meal_coeff = (3.552954405332104e-6)*(t**2)*uc - 0.003423642950746*t*uc + 0.823855671531609*uc
    # 500 <= t <= 720
    elif(t >= 500):# and t <= 720):
        meal_coeff = (1.113158675054563e-8)*(t**2)*uc - (1.482047571340105e-5)*t*uc + 0.004900138660598*uc
    else:
        raise RuntimeError

    #insulin rate for all modes (Strategy 1)
    #Polynomial insulin_rate_0("0.97751710655*0.05", stateVars);
    if mode == 0:
        insulin_rate = 0
    elif mode == 1:
        insulin_rate = 0.97751710655*0.1
    elif mode == 2:
        insulin_rate = 0.97751710655*0.2
    elif mode == 3:
        insulin_rate = 0.97751710655*0.5
    elif mode == 4:
        insulin_rate = 0.97751710655*1.4
    else:
        raise RuntimeError

    state_ = state + h*plant_deriv(state, insulin_rate, meal_coeff)
    return t+h, state_, mode


def plant_deriv(state, insulin_rate, meal_coeff):
    X, Isc1, Isc2, Gt, Gp, Il, Ip, I1, Id, Gs, t, uc = state
    # Derivatives
    d_X = -0.0278 * X + 0.0278 * (18.2129 * Ip - 100.25)
    d_Isc1 = -0.0171 * Isc1 + insulin_rate
    d_Isc2 = 0.0152 * Isc1 - 0.0078 * Isc2
    d_Gt = -0.0039*(3.2267+0.0313*X)*Gt * (1 - 0.0026*Gt + (2.5097e-6)*Gt**2) + 0.0581*Gp - 0.0871*Gt
    d_Gp = 4.7314 - 0.0047 * Gp - 0.0121 * Id - 1 - 0.0581*Gp + 0.0871*Gt + meal_coeff
    d_Il = -0.4219 * Il + 0.2250*Ip
    d_Ip = -0.3150 * Ip + 0.1545 * Il + 0.0019 * Isc1 + 0.0078 * Isc2
    d_I1 = -0.0046 * (I1 - 18.2129 * Ip)
    d_Id = -0.0046 * (Id - I1)
    d_Gs = 0.1*(0.5221 * Gp - Gs)
    d_t = 1
    d_uc = 0
    return np.array((d_X, d_Isc1, d_Isc2, d_Gt, d_Gp, d_Il, d_Ip, d_I1, d_Id, d_Gs, d_t, d_uc))


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
