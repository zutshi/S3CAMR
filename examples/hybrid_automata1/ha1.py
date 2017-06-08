#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np

import matplotlib.pyplot as plt
from IPython import embed

import utils as U
import copy_reg as cr
import types

PLOT = True


def _pickle_method(m):
    """_pickle_method
    Required for successfull pickling of sim()
    """
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


cr.pickle(types.MethodType, _pickle_method)

TEMP_UB = 70
TEMP_LB = 50


# As it is created only once, all methods should be static
# methods.
class SIM(object):

    def __init__(self, _, pvt_init_data):
        return None

    def sim(self, TT, X0, D, P, I, property_checker):
        property_violated_flag = False
        num_dim_x = len(X0)

        plot_data = ([np.empty(0, dtype=float), np.empty((0, num_dim_x), dtype=float)]
                     if PLOT else None)

        Ti = TT[0]
        Tf = TT[1]
        T = Tf - Ti

        #u = 'on'
        h = 0.05
        t = Ti
        x = X0

        u = D

        #X_ = [x]
        #embed()
        plot_data_t, plot_data_x = [t], [x]
        while t <= Tf:
            # compute plant's outputs
            x_ = x + h*dyn(x, u, I)
            t_ = t + h

            if x_[0] >= TEMP_UB:
                u = 0
            if x_[0] <= TEMP_LB:
                u = 1
            x = x_
            t = t_

            #X_.append(x_)
            plot_data_t.append(t)
            plot_data_x.append(x)
            #print(t)
            #print(plot_data_t)
            #print(plot_data_x)

        #X_ = np.array(X_)
        plot_data_t, plot_data_x = np.array(plot_data_t), np.array(plot_data_x)
        X_ = x

        #if property_checker is not None:
        if property_checker.check(Tf, X_):
            property_violated_flag = True

        dummy_D = np.array([u])
        dummy_P = np.zeros(P.shape)
        ret_t = Tf
        ret_X = X_
        # ret_Y = Y
        ret_D = dummy_D
        ret_P = dummy_P

        #print('time', plot_data_t.flatten())
        #print('x', plot_data_x.flatten())

        if PLOT:
            plt.plot(plot_data_t.flatten(), plot_data_x.flatten(), 'b-*', linewidth=0.5)

        ##plt.plot(plot_data[0] + Ti, np.tile(U, plot_data[0].shape))

        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


def dyn(x, u, w):
    if u == 1:
        x = 0.1*(100.0+w - x)   # 100
    #elif u == 2:
        #x = 0.1*(70.1 - x)  # 70
    elif u == 0:
        x = 0.50*(20.0+w - x)    # 20
    else:
        print 'error: u =', u
        raise NotImplementedError
    return x
