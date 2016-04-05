#!/usr/bin/python
# -*- coding: utf-8 -*-
# Must satisfy the signature
# [t,X,D,P] = sim_function(T,X0,D0,P0,I0);

"""Uses Assimulo solvers"""

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
from assimulo.solvers import LSODAR
from assimulo.solvers import RungeKutta34
import pylab as plt

PLT = False


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    def sim(self, TT, X0, D, P, U, I, property_checker):
        tol = 1e-2
        if ((abs(X0[1]) <= tol and abs(X0[3]) <= tol) or X0[1] < 0):
            X0[1] = 0
            X0[3] = 0
            return (TT[0], X0, D, P), False
        #print X0
        #Simulation
        ncp = 200     #Number of communication points
        self.model.re_init(TT[0], X0)
        t, y = self.model.simulate(TT[1], ncp) #Simulate
        #Print event information
        #sim.print_event_data()

        # t is a list for some reasons. Maybe there is a better way to
        # fix this.
        t = np.array(t)
        # violating values
        (t_, y_), property_violated = property_checker.first_sat_value_or_end(t, y)

        # if xv and tv are empty, no violation were found

        ret_t, ret_X = t_, y_
        ret_D = D
        ret_P = P
        if PLT:
            plt.plot(y[:, 0], y[:, 1], 'b-', linewidth=2)
        #plt.plot(t, y[:, 1], 'r-', linewidth=2)
        return (ret_t, ret_X, ret_D, ret_P), property_violated


def create_model():
    def pendulum(t, X, sw):
        """
        The ODE to be simulated. The parameter sw should be fixed during
        the simulation and only be changed during the event handling.
        """
        g = 1

        Y = X.copy()
        Y[0] = X[2]     #x_dot
        Y[1] = X[3]     #y_dot
        Y[2] = 0        #vx_dot
        Y[3] = -g       #vy_dot
        return Y

    def state_events(t, X, sw):
        """
        This is our function that keep track of our events, when the sign
        of any of the events has changed, we have an event.
        """
        return [X[1]] # y == 0

    def handle_event(solver, event_info):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        state_info = event_info[0] #We are only interested in state events info

        if state_info[0] != 0: #Check if the first event function have been triggered
            if solver.sw[0]:
                X = solver.y
                if X[3] < 0: # if the ball is falling (vy < 0)
                    # bounce!
                    X[1] = 1e-5
                    X[3] = -0.75*X[3]

            #solver.sw[0] = not solver.sw[0] #Change event function

    #Initial values
    y0 = [0., 0., 0., 0.] #Initial states
    t0 = 0.0             #Initial time
    switches0 = [True]   #Initial switches

    #Create an Assimulo Problem
    mod = Explicit_Problem(pendulum, y0, t0, sw0=switches0)

    mod.state_events = state_events #Sets the state events to the problem
    mod.handle_event = handle_event #Sets the event handling to the problem
    mod.name = 'Bouncing Ball in X-Y'   #Sets the name of the problem

    #Create an Assimulo solver (CVode)
    sim = CVode(mod)
    #sim = LSODAR(mod)
    #sim = RungeKutta34(mod)
    #sim.options['verbosity'] = 20 #LOUD
    sim.verbosity = 40 #WHISPER
    #sim.display_progress = True
    #sim.options['minh'] = 1e-4
    #sim.options['rtol'] = 1e-3

#     #Specifies options
#     sim.discr = 'Adams'     #Sets the discretization method
#     sim.iter = 'FixedPoint' #Sets the iteration method
#     sim.rtol = 1.e-8        #Sets the relative tolerance
#     sim.atol = 1.e-6        #Sets the absolute tolerance

    return sim
