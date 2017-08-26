#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


# X: Plant states

# import time

import logging
import numpy as np

# import matplotlib
# matplotlib.use('GTK3Agg')
# import matplotlib.pyplot as plt

import utils as U
from utils import print
import err

from graphs import graph as g
from . import state as st
from . import cellmanager as CM

import settings

logger = logging.getLogger(__name__)


def abstraction_factory(*args, **kwargs):
    return TopLevelAbs(*args, **kwargs)


# mantains a graph representing the abstract relation
# Each state is a tuple (abstract_plant_state, abstract_controller_state)
# Each relation is A1 -> A2, and is annotated by the concrete states
# abstraction states are a tuple (plant_state, controller_state)

class TopLevelAbs:

    @staticmethod
    def get_abs_state(plant_state):
        return AbstractState(plant_state)

    # takes in the plant abstraction object which is
    # instatiation of their respective parameterized abstract classes

    #TODO: split this ugly init function into smaller ones
    def __init__(
        self,
        config_dict,
        ROI,
        T,
        num_dims,
        min_smt_sample_dist,
        plant_abstraction_type,
        graph_lib,
        plant_abs=None,
        controller_abs=None,
        prog_bar=False,
        ):

        # super(Abstraction, self).__init__()

        self.ROI = ROI
        self.graph_lib = graph_lib
        self.num_dims = num_dims
        self.G = g.factory(graph_lib)
        self.T = T
        self.N = None
        self.state = None
        self.scale = None
        self.min_smt_sample_dist = min_smt_sample_dist
        self.plant_abstraction_type = plant_abstraction_type

        # The list of init_cons is interpreted as [ic0 \/ ic1 \/ ... \/ icn]
#        self.init_cons_list = init_cons_list
#        self.final_cons_list = final_cons_list

        self.eps = None

#        self.refinement_factor = 0.5

        self.delta_t = None
        self.num_samples = None

        # TODO: replace this type checking by passing dictionaries and parsing
        # outside the class. This will also avoid parsing code duplication.
        # Keep a single configuration format.

        self.parse_config(config_dict)

        # TAG:Z3_IND - default init
        smt_solver = None

        if plant_abstraction_type == 'cell':
            #from PACell import *
            from . import PACell as PA
        else:
            raise NotImplementedError
        # Overriding the passed in plant and conctroller abstractions
        plant_abs = PA.PlantAbstraction(
            self.T,
            self.N,
            self.num_dims,
            self.delta_t,
            self.eps,
            self.refinement_factor,
            self.num_samples,
            smt_solver, #TAG:Z3_IND - Add solver param
            )

        print(U.decorate('new abstraction created'))
        print('eps:', self.eps)
        print('num_samples:', self.num_samples)
        print('refine:', self.refinement_factor)
        print('deltaT:', self.delta_t)
        print('TH:', self.T)
        print('num segments:', self.N)
        print('=' * 50)

        logger.debug('new abstraction created')
        logger.debug('eps:{}'.format(self.eps))
        logger.debug('num_samples:{}'.format(self.num_samples))
        logger.debug('refine:{}'.format(self.refinement_factor))
        logger.debug('deltaT:{}'.format(self.delta_t))
        logger.debug('TH:{}'.format(self.T))
        logger.debug('num traces:{}'.format(self.N))
        logger.debug('=' * 50)

        # ##!!##logger.debug('==========abstraction parameters==========')
        # ##!!##logger.debug('eps: {}, refinement_factor: {}, num_samples: {},delta_t: {}'.format(str(self.eps), self.refinement_factor, self.num_samples, self.delta_t))

        self.final_augmented_state_set = set()

        # self.sanity_check()

        self.plant_abs = plant_abs

    def parse_config(self, config_dict):

        # ##!!##logger.debug('parsing abstraction parameters')

        if config_dict['type'] == 'string':
            try:
                grid_eps_str = config_dict['grid_eps']
                # remove braces
                grid_eps_str = grid_eps_str[1:-1]
                self.eps = np.array([float(eps) for eps in grid_eps_str.split(',')])

                pi_grid_eps_str = config_dict['pi_grid_eps']
                # remove braces
                pi_grid_eps_str = pi_grid_eps_str[1:-1]
                #self.pi_eps = np.array([float(pi_eps) for pi_eps in pi_grid_eps_str.split(',')])

                self.refinement_factor = float(config_dict['refinement_factor'])
                self.num_samples = int(config_dict['num_samples'])
                self.delta_t = float(config_dict['delta_t'])
                self.N = int(np.ceil(self.T / self.delta_t))

                # Make the accessed data as None, so presence of spurious data can be detected in a
                # sanity check

                config_dict['grid_eps'] = None
                config_dict['pi_grid_eps'] = None
                config_dict['refinement_factor'] = None
                config_dict['num_samples'] = None
                config_dict['delta_t'] = None
            except KeyError, key:
                raise err.Fatal('expected abstraction parameter undefined: {}'.format(key))
        else:
            for attr in config_dict:
                setattr(self, attr, config_dict[attr])
            self.N = int(np.ceil(self.T / self.delta_t))
            self.refinement_factor = 2.0

        return

    def in_ROI(self, abs_state):
        """ is the abstract state in ROI?
        Only checks plant's abstract state for now. Not sure if the
        controller's state should matter.

        Returns
        -------
        True/False
        """
        return self.plant_abs.in_ROI(abs_state.ps, self.ROI) #and self.controller_abs.in_ROI()

    # TODO: remove this eventually...

    def is_terminal(self, abs_state):
        return self.plant_abs.is_terminal(abs_state.plant_state)

    # Add the relation(abs_state_src, rchd_abs_state)
    # and update the abstraction function

    def add_relation(
            self,
            abs_state_src,
            rchd_abs_state,
            pi
            ):

        # get new distance/position from the initial state
        # THINK:
        # n can be calculated in two ways
        #   - only in the abstraction world: [current implementation]
        #       Completely independant of the simulated times
        #       i.e. if A1->A2, then A2.n = A1.n + 1
        #   - get it from simulation trace:
        #       n = int(np.floor(t/self.delta_t))

        self.G.add_edge(abs_state_src, rchd_abs_state, pi=pi)
        return

    def get_reachable_states(self, abs_state, system_params):
        abs2rchd_abs_state_set = set()
        #print(abs_state.ps.cell_id)
        # TODO: RECTIFY the below GIANT MESS
        # Sending in self and the total abstract_state to plant and controller
        # abstraction!!

        # ##!!##logger.debug('getting reachable states for: {}'.format(abs_state))

        intermediate_state = sample_abs_state(abs_state, self, system_params)
        abs2rchd_abs_state_pi_list = self.plant_abs.get_reachable_abs_states(intermediate_state, self, system_params)

        for (rchd_abs_state, pi_cell) in abs2rchd_abs_state_pi_list:
            self.add_relation(abs_state, rchd_abs_state, pi_cell)
            abs2rchd_abs_state_set.add(rchd_abs_state)

        # ##!!##logger.debug('found reachable abs_states: {}'.format(abs2rchd_abs_state_set))
        return abs2rchd_abs_state_set

#     def states_along_paths(self, paths):
#         MAX_ERROR_PATHS = 2
#         bounded_paths = U.bounded_iter(paths, MAX_ERROR_PATHS)

#         ret_list = []
#         for path in bounded_paths:
#             ret_list.append(path)
#         return ret_list


    def compute_error_paths(self, initial_state_set, final_state_set, MAX_ERROR_PATHS):
        # length of path is num nodes, whereas N = num segments
        max_len = self.N + 1
        return self.G.get_path_generator(initial_state_set, final_state_set, max_len, MAX_ERROR_PATHS)

    # memoized because the same function is called twice for ci and pi
    # FIXME: Need to fix it
    #@U.memodict
    def get_seq_of_pi(self, path):
        attr_map = self.G.get_path_attr_list(path, ['pi'])
        #print('attr_map:', attr_map)
        return attr_map['pi']


    def get_error_paths_not_normalized(self, initial_state_set,
                        final_state_set, pi_ref,
                        pi_cons,
                        max_paths):
        '''
        @type pi_cons: constraints.IntervalCons
        '''

        MAX_ERROR_PATHS = max_paths
        pi_dim = self.num_dims.pi
        path_list = []
        pi_seq_list = []

        error_paths = self.compute_error_paths(initial_state_set, final_state_set, MAX_ERROR_PATHS)
        bounded_error_paths = error_paths

        def get_pi_seq(path):
            return self.get_seq_of_pi(path)[1]

        def get_empty(_):
            return []

        get_pi = get_pi_seq if pi_dim != 0 else get_empty

        unique_paths = set()
        for path in bounded_error_paths:
            pi_seq_cells = get_pi(path)
            pi_ref.update_from_path(path, pi_seq_cells)
            pi_seq = [CM.ival_constraints(pi_cell, pi_ref.eps) for pi_cell in pi_seq_cells]

            #FIXME: Why are uniqe paths found only for the case when dim(ci) != 0?
            plant_states_along_path = tuple(state.plant_state for state in path)
            if plant_states_along_path not in unique_paths:
                unique_paths.add(plant_states_along_path)
                pi_seq_list.append(pi_seq)
                path_list.append(path)

        return (path_list, pi_seq_list)

    def get_initial_states_from_error_paths(self, initial_state_set, final_state_set, pi_ref, _, pi_cons, __, max_paths):
        '''extracts the initial state from the error paths'''
        path_list, pi_seq_list = self.get_error_paths_not_normalized(initial_state_set, final_state_set, pi_ref, pi_cons, max_paths)
        init_list = [path[0] for path in path_list]
        return init_list, pi_seq_list, pi_seq_list

    def get_abs_state_from_concrete_state(self, concrete_state):

        # ##!!##logger.debug(U.decorate('get_abs_state_from_concrete_state'))

        abs_plant_state = \
            self.plant_abs.get_abs_state_from_concrete_state(concrete_state)

        #TODO: why do we have the below code?
        if abs_plant_state is None:
            return None
        else:
            abs_state = TopLevelAbs.get_abs_state(abs_plant_state)

        # ##!!##logger.debug('concrete state = {}'.format(concrete_state))
        # ##!!##logger.debug('abstract state = {}'.format(abs_state))
        # ##!!##logger.debug(U.decorate('get_abs_state_from_concrete_state done'))

        return abs_state

#    def get_ival_cons_from_abs_state(self, abstract_state):
#        return (PlantAbstraction.get_concrete_state_constraints(abstract_state.plant_state, ))

    def __repr__(self):
        return '<abstraction_object>'


def sample_abs_state(abs_state,
                     A,
                     system_params):

    samples = system_params.sampler.sample(abs_state, A, system_params, A.num_samples)

    total_num_samples = samples.n

    x_array = samples.x_array

    # print s_array

    t_array = samples.t_array
    pi_array = samples.pi_array

    d = np.array([abs_state.plant_state.d])
    pvt = np.array([abs_state.plant_state.pvt])

    d_array = np.repeat(d, samples.n, axis=0)
    pvt_array = np.repeat(pvt, samples.n, axis=0)

    # sanity check
    assert(len(d_array) == total_num_samples)
    assert(len(pvt_array) == total_num_samples)
    assert(len(x_array) == total_num_samples)
    assert(len(t_array) == total_num_samples)

    state = st.StateArray(
        t=t_array,
        x=x_array,
        d=d_array,
        pvt=pvt_array,
        pi=pi_array,
        )

    return state


global abs_store
abs_store = {}


def AbstractState(ps):
    global abs_store

    new_as = AbstractState_(ps)
    if new_as not in abs_store:
        abs_store[new_as] = new_as
    return abs_store[new_as]


class AbstractState_(object):
    if settings.MEMLEAK_TEST:
        instance_ctr = 0

    def __init__(self, plant_state):
        self.plant_state = plant_state
        if settings.MEMLEAK_TEST:
            self.__class__.instance_ctr += 1
        return

    # rename/shorten name hack

    @property
    def ps(self):
        return self.plant_state

    def __eq__(self, x):
        if isinstance(x, self.__class__):
            return self.ps == x.ps
        else:
            return False

    def __hash__(self):

        # print('abstraction_hash_invoked')

        return hash(self.ps)

    def __repr__(self):
        return 'p={' + self.plant_state.__repr__() + '}'
