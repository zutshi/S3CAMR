#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import Queue
import numpy as np

from . import abstraction as AA
from . import sample as SaMpLe

from utils import print
#from utils import print

logger = logging.getLogger(__name__)

np.set_printoptions(suppress=True)


# TODO: plant state centric....attaches other states to plant states. make it
# neutral, gets touples of ((cons, d), s)

def init(
        A,
        init_cons_list_plant,
        final_cons,
        init_d,
        controller_init_state,
        ):

    PA, CA = A.plant_abs, A.controller_abs
    d, pvt, n = init_d, (0, ), 0

    plant_initial_state_set = set()

    def f(init_cons_): return PA.get_abs_state_set_from_ival_constraints(init_cons_, n, d, pvt)

    for init_cons in init_cons_list_plant:
        init_abs_states = f(init_cons)

        # filters away zero measure sets
        def fnzm(as_):
            intsec = PA.get_ival_cons_abs_state(as_) & init_cons
            return (intsec is not None) and not intsec.zero_measure

        filt_init_abs_states = filter(fnzm, init_abs_states)

        plant_initial_state_set.update(filt_init_abs_states)


# Old code to compute plant_initial_state_set
#     plant_initial_state_list = []
#     for init_cons in init_cons_list_plant:
#         plant_initial_state_list += \
#             PA.get_abs_state_set_from_ival_constraints(init_cons, n, d, pvt)
#     plant_initial_state_set = set(plant_initial_state_list)


    # The below can be very very expensive in time and memory for large final
    # sets!
    # plant_final_state_set = \
    #    set(PA.get_abs_state_set_from_ival_constraints(final_cons, 0, 0, 0))

    # ##!!##logger.debug('{0}initial plant states{0}\n{1}'.format('=' * 10, plant_initial_state_set))

    # set control states for initial states
    # TODO: ideally the initial control states should be supplied by the
    # user and the below initialization should be agnostic to the type

    controller_init_abs_state = \
        CA.get_abs_state_from_concrete_state(controller_init_state)

    initial_state_list = []
    for plant_init_state in plant_initial_state_set:
        initial_state_list.append(AA.TopLevelAbs.get_abs_state(plant_state=plant_init_state,
                                  controller_state=controller_init_abs_state))

    #final_state_list = []

#    for plant_final_state in plant_final_state_set:
#        final_state_list.append(AA.TopLevelAbs.get_abs_state(
#            plant_state=plant_final_state,
#            controller_state=controller_init_abs_state))

    # print('='*80)
    # print('all final states')
    # for ps in plant_final_state_set:
    #    print(ps)
    # print('='*80)

    def is_final(_A, abs_state):

        #print('----------isfinal-----------')
        #print(abs_state.plant_state.cell_id == ps.cell_id)
        #print(abs_state.plant_state in plant_final_state_set)
        #print(hash(abs_state.plant_state), hash(ps))
        #print(abs_state.plant_state == ps)
        #if abs_state.plant_state.cell_id == ps.cell_id:
            #exit()

        # return abs_state.plant_state in plant_final_state_set

        #print('---------------------------------------------')
        #print(_A.plant_abs.get_ival_constraints(abs_state.ps))
        #print('---------------------------------------------')
        pabs_ic = _A.plant_abs.get_ival_cons_abs_state(abs_state.plant_state)
        intersection = pabs_ic & final_cons
        return not(intersection is None or intersection.zero_measure)

    # ##!!##logger.debug('{0}initial{0}\n{1}'.format('=' * 10, plant_initial_state_set))

    return (set(initial_state_list), is_final)


# TODO: move set_n, get_n here, because no other exploration process might want to use it?
# Calls the simulator for each abstract state

def discover(A, system_params, initial_state_set, budget=None):
    final_state_set = set()
    Q = Queue.Queue(maxsize=0)
    examined_state_set = set()

    # initialize the Q with initial states

    # ##!!##logger.debug('Adding initial states to Q')

    for init_state in initial_state_set:
        Q.put(init_state)

    while not Q.empty():
        abs_state = Q.get(False)

        # ##!!##logger.debug('{} = Q.get()'.format(abs_state))

        if not (A.is_terminal(abs_state)
                or abs_state in examined_state_set
                or not A.in_ROI(abs_state)):

            # ##!!##logger.debug('decided to process abs_state')

            # Mark it as examined

            examined_state_set.add(abs_state)

            # Find all reachable abstract states using simulations

            abs2rch_abs_state_dict = get_reachable_abs_states(A, system_params, [abs_state])

            # add the new reached states only if they have not been
            # processed before

            # ##!!##logger.debug('abs2rch_abs_state_dict.values()\n{}'.format(abs2rch_abs_state_dict.values()))

            rchd_abs_state_set = abs2rch_abs_state_dict[abs_state]

            # TODO: abstract away the graph maybe??
            # Hide the call behind add_relation(A1, A2)

            for rchd_abs_state in rchd_abs_state_set:

                # ##!!##logger.debug('reached abstract state {}'.format(rchd_abs_state))

                # query if the reached state is a final state or not?
                # If yes, tag it so

                if system_params.is_final(A, rchd_abs_state):
                    final_state_set.add(rchd_abs_state)
                else:

                    # print('found a final state')
                    # exit()

                    Q.put(rchd_abs_state, False)

                # moving below to the abstraction itself
                # A.add_relation(abs_state, rchd_abs_state)

                # A.G.add_edge(abs_state, rchd_abs_state)
#                    n = self.get_n_for(abs_state) + 1
#                    self.set_n_for(rchd_abs_state, n)

    # end while loop

    # ##!!##logger.debug('Abstraction discovery done')
    # ##!!##logger.debug('Printing Abstraction\n {}'.format(str(A)))
    return final_state_set

# Same as discover_old, but accumulates all abstract states from the Q before
# calling get_reachable_abs_states() on the entire group
# This makes it potentially faster, because fewer simulator calls are
# needed.


def discover_batch(A, budget=None):
    Q = Queue.Queue(maxsize=0)
    examined_state_set = set()

    abs_state_list_to_examine = []

    # initialize the Q with initial states

    # ##!!##logger.debug('Adding initial states to Q')

    for init_state in A.initial_state_set:
        Q.put(init_state)

    while not Q.empty():
        abs_state_list_to_examine = []

        # Empty the Q

        while not Q.empty():
            abs_state = Q.get(False)

            # ##!!##logger.debug('{} = Q.get()'.format(abs_state))

            if not (A.is_terminal(abs_state) or abs_state
                    in examined_state_set):

                # ##!!##logger.debug('decided to process abs_state')

                # Mark it as examined

                examined_state_set.add(abs_state)

                # Collect all abstract states which need to be examined

                abs_state_list_to_examine.append(abs_state)
            else:

                # ##!!##logger.debug('NOT going to process abstract state')

                pass

        # end inner while

        # ##!!##logger.debug('get_reachable_abs_states() for...\n{}'.format(abs_state_list_to_examine))

        if abs_state_list_to_examine:
            abs2rch_abs_state_dict = get_reachable_abs_states(A, abs_state_list_to_examine)

            # print(abs2rch_abs_state_dict)

            # ##!!##logger.debug('abs2rch_abs_state_dict.values()\n{}'.format(abs2rch_abs_state_dict.values()))

            rchd_abs_state_set = set.union(*abs2rch_abs_state_dict.values())

            # add the new reached states only if they have not been
            # processed before

            for rchd_abs_state in rchd_abs_state_set:

                # ##!!##logger.debug('reached abstract state {}'.format(rchd_abs_state))

                Q.put(rchd_abs_state, False)

            # Add the relation even if state has been seen before,
            # because the edge might
            # not have had been...can check for edge's presence too,
            # but will need to change
            # later when we will add weights...where we will probably
            # update weights in
            # adfition to adding edges.
            # encode the new relations in the graph, no weights for
            # now!
            # TODO: abstract away the graph maybe??
            # Hide the call behind add_relation(A1, A2)

            for (abs_state, rchd_abs_state_set) in abs2rch_abs_state_dict.items():
                for rchd_abs_state in rchd_abs_state_set:
                    A.add_relation(abs_state, rchd_abs_state)

                    # A.G.add_edge(abs_state, rchd_abs_state)
#                        n = self.get_n_for(abs_state) + 1
#                        self.set_n_for(rchd_abs_state, n)

    # end while loop

    # ##!!##logger.debug('Abstraction discovery done')
    # ##!!##logger.debug('Printing Abstraction\n {}'.format(str(A)))


def refine_state(A, RA, abs_state):
    ps = abs_state.ps
    cs = abs_state.cs
    ps_ival = A.plant_abs.get_ival_cons_abs_state(ps)
    refined_ps_set = RA.plant_abs.get_abs_state_set_from_ival_constraints(ps_ival, ps.n, ps.d, ps.pvt)
    abs_state_list = []
    if A.controller_abs.is_symbolic:
        for rps in refined_ps_set:
            x_smt2 = RA.plant_abs.get_smt2_constraints(rps, cs.x)
            cs.C = RA.controller_abs.solver.And(cs.C, x_smt2)
            AA.AbstractState(rps, cs)
            abs_state_list.append(abs_state)
    else:
        for rps in refined_ps_set:
            AA.AbstractState(rps, cs)
            abs_state_list.append(abs_state)
    return abs_state_list


def refine_param_dict(A):
    new_eps = A.eps / A.refinement_factor
    #new_pi_eps = A.pi_eps / A.refinement_factor
    param_dict = {
        'eps': new_eps,
        #'pi_eps': new_pi_eps,
        'refinement_factor': A.refinement_factor,
        'num_samples': A.num_samples,
        'delta_t': A.delta_t,
        'N': A.N,
        'type': 'value',
        }
    return param_dict


def refine_trace_based(A, error_paths, system_params):

    # ##!!##logger.debug('executing trace based refinement')

    traversed_state_set = set()
    sap = A.states_along_paths(error_paths)
    for path in sap:
        traversed_state_set.update(path)

    param_dict = refine_param_dict(A)

    RA = AA.abstraction_factory(
        param_dict,
        A.T,
        A.num_dims,
        A.controller_sym_path_obj,
        A.min_smt_sample_dist,
        A.plant_abstraction_type,
        A.controller_abstraction_type
        )

    # split the traversed states

    # construct a list of sets and then flatten
    #refined_ts_list = U.flatten( [refine_state(A, RA, ts) for ts in traversed_state_set])

    #abs2rch_abs_state_dict = get_reachable_abs_states(RA, system_params, refined_ts_list)

    return RA


# refine using init states
def refine_init_based(A, promising_initial_abs_states,
                      original_plant_cons_list):#, pi_ref, ci_ref):

    # ##!!##logger.debug('executing init based refinement')

    # checks if the given constraint has a non empty intersection with the
    # given plant initial states. These are the actual initial plant sets
    # specified in the tst file.

    def in_origianl_initial_plant_cons(ic):
        for oic in original_plant_cons_list:
            if oic & ic:
                return True
        assert('Should never happen. Should be caught by SS.filter_invalid_abs_states')
        return False

    init_cons_list = []

    # ignore cells which have no overlap with the initial state

    for init_state in promising_initial_abs_states:
        ic = A.plant_abs.get_ival_cons_abs_state(init_state.plant_state)
        if in_origianl_initial_plant_cons(ic):
            init_cons_list.append(ic)

    param_dict = refine_param_dict(A)

#    AA.AbstractState.clear()

    refined_abs = AA.abstraction_factory(
        param_dict,
        A.ROI,
        A.T,
        A.num_dims,
        A.controller_sym_path_obj,
        #A.ci_grid_eps/2,
        A.min_smt_sample_dist,
        A.plant_abstraction_type,
        A.controller_abstraction_type,
        A.graph_lib
        )

    # TODO: what a hack!
    #pi_ref = A.plant_abs.pi_ref
#     pi_ref.refine()
#     ci_ref.refine()
#     refined_abs.plant_abs.pi_ref = pi_ref
#     refined_abs.controller_abs.ci_ref = ci_ref

#    refined_abs = AA.GridBasedAbstraction(param_dict,
#                                          A.plant_sim,
#                                          A.T,
#                                          A.sample,
#                                          init_cons_list,
#                                          A.final_cons,
#                                          A.controller_sim,
#                                          A.num_dims,
#                                          prog_bar=True)

    return (refined_abs, init_cons_list)


# samples a list of abstract states and collates the resulting samples

def sample_abs_state_list(A, system_params, abs_state_list):

    # ##!!##logger.debug(U.decorate('sampling begins'))

    # List to store reachble concrete states:
    # stores number of samples in the same order as abstract states in
    # abs_state_list and of course
    # len(abs_state_list) =  abs_state2samples_list

    abs_state2samples_list = []

    consolidated_samples = SaMpLe.Samples()

    for abs_state in abs_state_list:

        # scatter the continuous states

        # ##!!##logger.debug('sampling({})'.format(abs_state))

        samples = system_params.sampler.sample(abs_state, A, system_params, A.num_samples)

        # ##!!##logger.debug('{}'.format(samples))

        abs_state2samples_list.append(samples.n)
        consolidated_samples.append(samples)

#    # ##!!##logger.debug('{}'.format(consolidated_samples))

    #total_num_samples = consolidated_samples.n

    # ##!!##logger.debug('num_samples = {}'.format(total_num_samples))
    # ##!!##logger.debug('samples = \n{}'.format(samples))
    # ##!!##logger.debug(U.decorate('sampling done'))

    return (abs_state2samples_list, consolidated_samples)


# Returns abs2rchd_abs_state_map: a mapping
# abstract_state |-> set(reached abstract states)
# It is debatable if this is the best return format! ;)

def get_reachable_abs_states(A, system_params, abs_state_list):

    # Dictionary mapping: abstract state |-> set(reached abstract state)

    abs2rchd_abs_state_map = {}

    for abs_state in abs_state_list:
        abs2rchd_abs_state_map[abs_state] = A.get_reachable_states(abs_state, system_params)

    return abs2rchd_abs_state_map


# TODO: ugly...should it be another function?
# Only reason its been pulled out from random_test is to ease the detection of
# the case when no valid abstract state is left!
# VERY INEFFICIENT
# Repeats some work done in random_test...
def filter_invalid_abs_states(state_list, pi_seq_list, ci_seq_list, A, init_cons):
    valid_idx_list = []

    for idx, abs_state in enumerate(state_list):
        ival_cons = A.plant_abs.get_ival_cons_abs_state(abs_state.plant_state)

        # ##!!##logger.debug('ival_cons: {}'.format(ival_cons))

        # find the intersection b/w the cell and the initial cons
        # print('init_cons', init_cons)

        ic = ival_cons & init_cons
        if (ic is not None) and (not ic.zero_measure):
            valid_idx_list.append(idx)
            #valid_state_list.append(abs_state)

    # TODO: this should be logged and not printed
    if valid_idx_list == []:
        for abs_state in state_list:
            ival_cons = A.plant_abs.get_ival_cons_abs_state(abs_state.plant_state)
            print(ival_cons)

    valid_state_list = []
    respective_pi_seq_list = []
    respective_ci_seq_list = []
    for i in valid_idx_list:
        valid_state_list.append(state_list[i])
        respective_pi_seq_list.append(pi_seq_list[i])
        respective_ci_seq_list.append(ci_seq_list[i])
    return valid_state_list, respective_pi_seq_list, respective_ci_seq_list
