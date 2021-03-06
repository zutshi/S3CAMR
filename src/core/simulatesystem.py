#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from . import state as st
from . import traces
from globalopts import opts as gopts

import err
import utils as U
#import progBar as pb
#import utils

import logging
import numpy as np

# import matplotlib.pyplot as plt


# from guppy import hpy

# hp=hpy()

logger = logging.getLogger(__name__)


def step_sim(psim, delta_t, t0, x0, d0, pvt0, pi):
    concrete_states = get_concrete_state_obj(t0, x0, d0, pvt0, pi)

    # ignore pvf
    concrete_states_, _ = psim.simulate(concrete_states, delta_t)

    (t, x, d, pvt, _) = get_individual_states(concrete_states_)

    return (t[0], x[0], d[0], pvt[0])


# exactly the same as simulate, but does not use closures
def simulate_system(sys, T, concrete_state):
    (t, x, d, pvt, pi_array) = get_individual_states(concrete_state)
    t0 = t
    tf = t + T

    T = tf - t0
    num_segments = int(np.ceil(T / sys.delta_t))
    # num_points = num_segments + 1
    trace = traces.Trace(sys.num_dims, num_segments + 1)
    t = t0

    for i in xrange(num_segments):
        pi = pi_array[i]
        (t_, x_, d_, pvt_) = step_sim(sys.plant_sim, sys.delta_t, t, x, d, pvt, pi)
        trace.append(t=t, x=x, d=d, pi=pi)
        (t, x, d, pvt) = (t_, x_, d_, pvt_)

    trace.append(t=t, x=x, d=d, pi=pi)
    return trace


# Helper function for simulating a system from 0 to T
# TODO: Does not handle error states of the controller or the plant simulator..
# Must include provision for states which can not be simulated for some
# reasons...
def simulate(system_sim, T, concrete_state):
    (t, x, d, pvt, pi_array) = get_individual_states(concrete_state)
    t0 = t
    tf = t + T
    trace = system_sim(x, d, pvt, t0, tf, pi_array)
    return trace


# arguements must be numpy arrays
# Uses the dimension info to correctly create the state array
def get_concrete_state_obj(t0, x0, d0, pvt0, pi):
    if x0.ndim == 1:
        concrete_states = st.StateArray(
            t=np.array([t0]),
            x=np.array([x0]),
            d=np.array([d0]),
            pvt=np.array([pvt0]),
            pi=np.array([pi]),
            )
    elif x0.ndim == 2:

        concrete_states = st.StateArray(
            t=t0,
            x=x0,
            d=d0,
            pvt=pvt0,
            pi=pi,
            )
    else:
        raise err.Fatal('dimension must be 1 or 2...: {}!'.format(x0.ndim))
    return concrete_states


def get_individual_states(concrete_states):
    t = concrete_states.t
    x = concrete_states.cont_states
    d = concrete_states.discrete_states
    pvt = concrete_states.pvt_states
    pi = concrete_states.plant_extraneous_inputs

    return (t, x, d, pvt, pi)


def get_system_simulator(sys):
    step_sim = get_step_simulator(sys.controller_sim,
                                  sys.plant_sim,
                                  sys.delta_t)

    def system_simulator(
            x,
            d, pvt,
            t0, tf,
            pi_array
            ):

        T = tf - t0
        # use the ceiling function from utils to detect and fix
        # floating point issues
        num_segments = int(U.ceil(T / sys.delta_t, np.ceil))
        # num_points = num_segments + 1
        trace = traces.Trace(sys.num_dims, num_segments + 1)
        t = t0

        #print('num_segments:', num_segments)
        # need the below try catch shenanigans to print the error message,
        # because matlab does not.
        #try:
        #    raise e
        for i in range(num_segments):
            pi = pi_array[i]
            (t_, x_, d_, pvt_) = step_sim(t, x, d, pvt, pi)
            trace.append(t=t, x=x, d=d, pi=pi)
            (t, x, d, pvt) = (t_, x_, d_, pvt_)

        trace.append(t=t, x=x, d=d, pi=pi)
        return trace
    return system_simulator


def get_step_simulator(csim, psim, delta_t):

    def simulate_basic(t0, x0, d0, pvt0, pi):
        concrete_states = get_concrete_state_obj(t0, x0, d0, pvt0, pi)

        # ignore pvf
        concrete_states_, _ = psim.simulate(concrete_states, delta_t)

        (t, x, d, pvt, _) = get_individual_states(concrete_states_)
        return (t[0], x[0], d[0], pvt[0])
    return simulate_basic
