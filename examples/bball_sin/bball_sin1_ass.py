
#!/usr/bin/python
# -*- coding: utf-8 -*-

# This example is from HyCU [Matlab's S3CAM implementation]
# Which was in turn taken from
# Reference:
# The Dynamics of a Bouncing Ball with a SinusoidallyVibrating Table Revisited
# ALBERT C. Jo LUO and RAY P. S. HAN

"""Uses Assimulo solvers"""

import numpy as np
import pylab as plt
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
from assimulo.solvers import LSODAR


class SIM(object):

    def __init__(self, _, pvt_init_data):
        self.model = create_model()

    def sim(self, TT, X0, D, P, U, I, property_checker, property_violated_flag):
        #Simulation
        ncp = 200     #Number of communication points
        self.model.re_init(TT[0], X0)
        t, y = self.model.simulate(TT[1], ncp) #Simulate
        #Print event information
        #sim.print_event_data()

        ret_D = D
        ret_P = P
        ret_t = t[-1]
        ret_X = y[-1]
#         plt.figure(10)
#         plt.plot(t, y[:, 0], 'b-*', linewidth=2)
#         plt.plot(t, y[:, 1], 'r-*', linewidth=2)
        return (ret_t, ret_X, ret_D, ret_P)

def create_model():
    def pendulum(t, X, sw):
        """
        The ODE to be simulated. The parameter sw should be fixed during
        the simulation and only be changed during the event handling.
        """

        #X = X.copy()
        g = 9.81
        x = X[0]
        vx = X[2]
        vy = X[3]

        return np.array([vx, vy, -(np.pi**2)*x, -g, 1.0])

    def state_events(t, X, sw):
        """
        This is our function that keep track of our events, when the sign
        of any of the events has changed, we have an event.
        """
        x = X[0]
        y = X[1]

        return [x - y]
        #return np.array([x - y])

    def handle_event(solver, event_info):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        state_info = event_info[0] #We are only interested in state events info

        if state_info[0] != 0: #Check if the first event function have been triggered

            if solver.sw[0]: #If the switch is True the pendulum bounces
                X = solver.y
                X[1] = X[0] + 1e-3
                X[3] = 0.9*(X[2] - X[3])

            #solver.sw[0] = not solver.sw[0] #Change event function

    #Initial values
    phi = 1.0241592; Y0 = -13.0666666; A = 2.0003417; w = np.pi
    y0 = [A*np.sin(phi), 1+A*np.sin(phi), A*w*np.cos(phi), Y0 - A*w*np.cos(phi)-1, 0] #Initial states
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
    sim.options['verbosity'] = 40

    #Specifies options
    sim.discr = 'Adams'     #Sets the discretization method
    sim.iter = 'FixedPoint' #Sets the iteration method
    sim.rtol = 1.e-8        #Sets the relative tolerance
    sim.atol = 1.e-6        #Sets the absolute tolerance

    return sim
