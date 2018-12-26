#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


import logging

logger = logging.getLogger(__name__)


# TODO: all states flying around....split it into controller and plant state array!

class StateArray(object):

    # t: time
    # x: continous plant state [change to x]
    # d: discrete plant state
    # pi: plant disturbance
    # pvt: plant private state [plant simulator can associate pvt states?]

    def __init__(
            self,
            t,
            x,
            d,
            pvt,
            pi=None):

        # self.sanity_chec()

        # ##!!##logger.debug('sanity_check() disabled!')

        # TODO: why have the time array? whats the use?
        # time array

        self.t = t

        # numpy arr

        self.cont_states = x

        # numpy arr

        self.discrete_states = d
        self.pvt_states = pvt
        self.plant_extraneous_inputs = pi

    @property
    def n(self):

        # number of states, find from the number of continous states
        # This is OK, as the sanity check has verified that all are equal

        return len(self.cont_states)

    def __len__(self):
        return self.n

    def iterable(self):
        #i = 0
        #while i < self.n:
        for t, x, d, p, pi in zip(self.t,
                                            self.cont_states,
                                            self.discrete_states,
                                            self.pvt_states,
                                            self.plant_extraneous_inputs,
                                            ):
                yield State(t, x, d, p, pi)
        return

    def __getitem__(self, key):
        i = key

        return StateArray(
            t=self.t[i],
            x=self.cont_states[i],
            d=self.discrete_states[i],
            pvt=self.pvt_states[i],
            pi=self.plant_extraneous_inputs[i],
            )

    # not a good idea to return another class object! Creates issues with list
    # pseudoi indexing functionality

#        return State(self.t[i],
#                self.cont_states[i],
#                self.discrete_states[i],
#                self.pvt_states[i],
#                None,
#                self.controller_states[i],
#                None,
#                self.controller_outputs[i])

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __deleteitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        s = ''
        s += '''t_array: {} '''.format(self.t)
        s += '''plant_state_array {} '''.format(self.cont_states)
        s += '''discrete_state_array {} '''.format(self.discrete_states)
        s += '''pvt_state_array {} '''.format(self.pvt_states)
        s += '''plant_extraneous_inputs_array {} '''.format(self.plant_extraneous_inputs)
        return s


class State(object):

    def __init__(
            self,
            t,
            x,
            d,
            pvt,
            pi,
            ):

        self.t = t  # time
        self.x = x  # continuous states
        self.d = d  # plant discrete states
        self.pvt = pvt  # plant pvt_states
        self.pi = pi

    def __repr__(self):
        l = []
        l.append('t=' + str(self.t))
        l.append('x=' + str(self.x))
        l.append('d=' + str(self.d))
        l.append('pvt=' + str(self.pvt))
        l.append('pi=' + str(self.pi))
        return '(' + ','.join(l) + ')'
