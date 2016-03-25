#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

# REFERENCE:
# Monitoring Bounded LTL Properties Using Interval Analysis
# Daisuke Ishii1 Naoki Yonezaki2
# Tokyo Institute of Technology, Tokyo, Japan
# Alexandre Goldsztejn3
# CNRS, IRCCyN, Nantes, France

"""Uses Assimulo solvers"""

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
from assimulo.solvers import LSODAR
import pylab as plt


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    def sim(self, TT, X0, D, P, U, I, property_checker):
        #print X0
        #Simulation
        ncp = 200     #Number of communication points
        self.model.re_init(TT[0], X0)
        t, y = self.model.simulate(TT[1], ncp) #Simulate
        #Print event information
        #sim.print_event_data()

        property_violated_flag = property_checker.check_array(t, y)

        ret_D = D
        ret_P = P
        ret_t = t[-1]
        ret_X = y[-1]
        #plt.figure(10)
        #plt.plot(t, y[:, 0], 'b-', linewidth=2)
        #plt.plot(t, y[:, 1], 'r-', linewidth=2)
        return (ret_t, ret_X, ret_D, ret_P), property_violated_flag


def create_model():
    def pendulum(t, X, sw):
        """
        The ODE to be simulated. The parameter sw should be fixed during
        the simulation and only be changed during the event handling.
        """

        X = X.copy()
        X[0] = X[1]
        X[1] = -1 + 0.04 * (X[1]**2) * np.sin(X[1])
        X[2] = 1

        return X

    def state_events(t, X, sw):
        """
        This is our function that keep track of our events, when the sign
        of any of the events has changed, we have an event.
        """
        return [X[0] - np.sin(X[2])]

    def handle_event(solver, event_info):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        state_info = event_info[0] #We are only interested in state events info

        if state_info[0] != 0: #Check if the first event function have been triggered

            if solver.sw[0]: #If the switch is True the pendulum bounces
                X = solver.y
                if X[1] - np.cos(X[2]) < 0:
                    X[1] = -0.9*X[1] + 1.9*np.cos(X[2])

            #solver.sw[0] = not solver.sw[0] #Change event function

    #Initial values
    y0 = [0, 0, 0] #Initial states
    t0 = 0.0             #Initial time
    switches0 = [True]   #Initial switches

    #Create an Assimulo Problem
    mod = Explicit_Problem(pendulum, y0, t0, sw0=switches0)

    mod.state_events = state_events #Sets the state events to the problem
    mod.handle_event = handle_event #Sets the event handling to the problem
    mod.name = 'Bouncing Ball on Sonusoidal Platform'   #Sets the name of the problem

    #Create an Assimulo solver (CVode)
    sim = CVode(mod)
    #sim = LSODAR(mod)
    #sim.options['verbosity'] = 20 #LOUD
    sim.options['verbosity'] = 40 #WHISPER
    #sim.options['minh'] = 1e-4
    #sim.options['rtol'] = 1e-3

    #Specifies options
    sim.discr = 'Adams'     #Sets the discretization method
    sim.iter = 'FixedPoint' #Sets the iteration method
    sim.rtol = 1.e-8        #Sets the relative tolerance
    sim.atol = 1.e-6        #Sets the absolute tolerance

    return sim
