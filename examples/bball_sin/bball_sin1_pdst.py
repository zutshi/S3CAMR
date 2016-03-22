#!/usr/bin/python
# -*- coding: utf-8 -*-

# This example is from HyCU [Matlab's S3CAM implementation]
# Which was in turn taken from
# Reference:
# The Dynamics of a Bouncing Ball with a SinusoidallyVibrating Table Revisited
# ALBERT C. Jo LUO and RAY P. S. HAN


"""BROKEN: Simulator is really terrible at zero crossing detection
"""

import numpy as np
import PyDSTool as dst
import matplotlib.pyplot as plt


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    def sim(self, TT, X0, D, P, U, I, property_checker, property_violated_flag):

        traj_name = 'tj'

        icdict = {'x': X0[0], 'y': X0[1], 'vx': X0[2], 'vy': X0[3], 'tt': X0[4]}
        self.model.compute(
            traj_name, tdata=[TT[0], TT[1]], ics=icdict,
            verboselevel=0, force=True)
        traj = self.model.trajectories[traj_name]
        pts = traj.sample()
        #pts = self.model.sample('test_traj', ['x', 'y', 'vx', 'vy', 'tt'])
        ret_D = D
        ret_P = P
        ret_t = np.array(pts['t'][-1])
        ret_X = np.hstack((np.array(pts['y'])[-1], np.array(pts['vy'])[-1]))
        ret_t = pts['t'][-1]
        ret_X = np.hstack((pts['x'][-1], pts['y'][-1], pts['vx'][-1], pts['vy'][-1], pts['tt'][-1]))
        plt.figure(10)
        plt.plot(pts['t'], pts['x'], 'b-*', linewidth=2)
        plt.plot(pts['t'], pts['y'], 'r-*', linewidth=2)
        return (ret_t, ret_X, ret_D, ret_P)


def create_model():
    pars = {'g': 9.8}#, 'pi': np.pi}

    #ODE
    ode_def = {
           'x': 'vx',
           'y': 'vy',
           'vx': '-(pi**2)*x',
           'vy': '-g',
           'tt': '1.0',
            }
    event_bounce = dst.makeZeroCrossEvent(
            'x-y', 1,
            {'name': 'bounce',
             'eventtol': 1e-3,
             'term': True,
             'active': True,
             'eventinterval': 1,
             'eventdelay': 1e-2,
             'starttime': 0,
             'precise': True
             },
            varnames=['x', 'y'],
            targetlang='python')  # targetlang is redundant (defaults to python)

    DSargs = dst.args(name='bball_sin')  # struct-like data
    DSargs.events = [event_bounce]
    #DSargs.pars = pars
    #DSargs.tdata = [0, 10]
    #DSargs.algparams = {'max_pts': 3000, 'stiff': False}
    DSargs.algparams = {'stiff': False, 'init_step': 0.01}
    DSargs.varspecs = ode_def
    DSargs.pars = pars
    #DSargs.xdomain = {'y': [0, 100]}

    DS_fall = dst.embed(dst.Generator.Vode_ODEsystem(DSargs))
    DS_fall_MI = dst.intModelInterface(DS_fall)

    # Reset
    ev_map = dst.EvMapping({'y': 'x+0.001', 'vy': '0.9*(vx-vy)'}, model=DS_fall)
    #ev_map = dst.EvMapping({'y': '10', 'x': '20'}, model=DS_fall)

    DS_BBall = dst.makeModelInfoEntry(DS_fall_MI, ['bball_sin'],
                                      [('bounce', ('bball_sin', ev_map))])

    modelInfoDict = dst.makeModelInfo([DS_BBall])
    bball_sin_model = dst.Model.HybridModel(
            {'name': 'Bouncing_Ball_Sinusiodal', 'modelInfo': modelInfoDict})
    return bball_sin_model
