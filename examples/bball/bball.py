#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

import numpy as np
import PyDSTool as dst
#import matplotlib.pyplot as plt


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    def sim(self, TT, X0, D, P, U, I, property_checker):
        icdict = {'y': X0[0], 'vy': X0[1]}
        self.model.compute(
            'test_traj', tdata=[TT[0], TT[1]], ics=icdict, verboselevel=0, force=True)
        pts = self.model.sample('test_traj', ['y', 'vy'])
        #print pts1
        #exit()
        ret_D = D
        ret_P = P
        ret_t = np.array(pts['t'][-1])
        ret_X = np.hstack((np.array(pts['y'])[-1], np.array(pts['vy'])[-1]))
        ret_t = pts['t'][-1]
        ret_X = np.hstack((pts['y'][-1], pts['vy'][-1]))
        #print ret_X
        #plt.figure(10)
        #plt.plot(pts['t'], pts['y'], 'b', linewidth=2)
        if property_checker is not None:
            property_violated_flag = property_checker.check_array(ret_t, ret_X)
        else:
            property_violated_flag = False
        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


def create_model():
    pars = {'g': 1}
    icdict = {'y': 5, 'vy': 0}

    y_str = 'vy'
    vy_str = '-g'

    event_bounce = dst.makeZeroCrossEvent('y', 0,
                                {'name': 'bounce',
                                 'eventtol': 1e-3,
                                 'term': True,
                                 'active': True},
                        varnames=['y'],
                        parnames=['g'],
                        targetlang='python')  # targetlang is redundant (defaults to python)

    DSargs = dst.args(name='bball')  # struct-like data
    DSargs.events = [event_bounce]
    #DSargs.pars = pars
    #DSargs.tdata = [0, 10]
    #DSargs.algparams = {'max_pts': 3000, 'stiff': False}
    DSargs.algparams = {'stiff': False}
    DSargs.varspecs = {'y': y_str, 'vy': vy_str}
    DSargs.pars = pars
    #DSargs.xdomain = {'y': [0, 100], 'vy': [-100, 100]}

    DSargs.ics = icdict

    DS_fall = dst.embed(dst.Generator.Vode_ODEsystem(DSargs))
    DS_fall_MI = dst.intModelInterface(DS_fall)

    ev_map = dst.EvMapping({'y': 0, 'vy': '-0.75*vy'}, model=DS_fall)

    DS_BBall = dst.makeModelInfoEntry(DS_fall_MI, ['bball'],
                                      [('bounce', ('bball', ev_map))])

    modelInfoDict = dst.makeModelInfo([DS_BBall])
    bball_model = dst.Model.HybridModel({'name': 'Bouncing_Ball', 'modelInfo': modelInfoDict})
    return bball_model
