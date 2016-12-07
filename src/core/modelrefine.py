from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools as it

#from collections import defaultdict
import collections

import numpy as np

from . import simulatesystem as simsys
from pwa import pwa
from pwa import simulator as pwa_sim
from pwa import relational as rel
from . import random_testing as rt
from bmc import bmc as BMC
from bmc.bmc_spec import InvarStatus
import modeling.affinemodel as AFM
from .cellmodels import Qxw, Qx
from . import cellmanager as CM
from graphs.graph import factory as graph_factory
from graphs.graph import class_factory as graph_class
from pwa import analyzepath as azp
from . import state

import settings

import utils as U
from utils import print
import err
from constraints import IntervalCons, top2ic, zero2ic

from IPython import embed

from globalopts import opts as gopts

TESTCODE = True

#np.set_printoptions(suppress=True, precision=2)

# multiply num samples with the
MORE_FACTOR = 100
#TEST_FACTOR = 10

MAX_TRAIN = 500

#MAX_TEST = 200

KMAX = 1
KMIN = 1


KMAX_EXCEEDED = 0
SUCCESS = 1
TERMINAL = 2

PWASYSPROP = collections.namedtuple('pwa_sys_prop', 'pwa_model init_partitions final_partitions')


def abs_state2cell(abs_state, AA):
    return CM.Cell(abs_state.plant_state.cell_id, AA.plant_abs.eps)


def ic2multicell(ic, eps):
    cells = CM.ic2cell(ic, eps)
    return cells


def ic2cell(ic, eps):
    cells = CM.ic2cell(ic, eps)
    # Due to path normalization the entire pi range gets added as an
    # edge by S3CAM wherever paths are shorter than the max path
    # length. Please fix this and switch on the assertion.
    # FIXED
    assert(len(cells) == 1)
    return cells[0]
    #return CM.Cell(CM.cell_from_concrete(pi, eps), eps)


def simulate_pwa(pwa_model, x_samples, N):
    return [pwa_sim.simulate(pwa_model, x0, N) for x0 in x_samples]


def simulate(AA, sp, pwa_model, max_path_len, S0):
    NUM_SIMS = 100
    # sample only initial abstract state
    x0_samples = (sp.sampler.sample_multiple(S0, AA, sp, NUM_SIMS)).x_array
    #print(x0_samples)
    # sample the entire given initial set
    #X0 = sp.init_cons
    #x0_samples = sample.sample_ival_constraints(X0, n=1000)

    print('path length: {}'.format(max_path_len))
    traces = [i for i in simulate_pwa(pwa_model, x0_samples, N=max_path_len)]
    return traces


def sim_n_plot(sp, AA, prop, error_paths, depth, pwa_model):
    # intial abs state set
    S0 = {path[0] for path in error_paths}
    s = (state for path in error_paths for state in path)
    print('simulating using depth = {} ...'.format(depth))
    pwa_traces = simulate(AA, sp, pwa_model, depth, S0)
    print('done')
    gopts.plotting.figure()
    print('plotting...')
    gopts.plotting.plot_abs_states(AA, prop, s)
    gopts.plotting.plot_rect(sp.final_cons.rect())
    gopts.plotting.plot_pwa_traces(pwa_traces)
    #fig = BP.figure(title='S3CAMR')
    #fig = plt.figure()
    gopts.plotting.show()


# def get_qgraph(sp, AA, G, error_paths, pi_seqs):

#     for path, pi_seq in zip(error_paths, pi_seqs):
#         for (a1, a2), pi_ic in zip(U.pairwise(path), pi_seq):
#             x1cell, x2cell = abs_state2cell(a1, AA), abs_state2cell(a2, AA)
#             if AA.num_dims.pi:
#                 wcell = ic2cell(pi_ic, sp.pi_ref.eps)
#             else:
#                 wcell = None
#             q1, q2 = Qx(x1cell), Qx(x2cell)
#             G.add_edge(q1, q2, pi=wcell)
#     return G


class QGraph(graph_class(gopts.graph_lib)):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.init = set()
        self.final = set()

#TODO: URGENT: The error path and pi_seq generated from S3CAM is not
# clear. Are we capturing multiple pi values between same states?
# Please review the code where the pi values are added to the edges
# and error paths and pi_seqs extracted.

# Subsumes get_qgraph
def get_qgraph_xw(sp, AA, error_paths, pi_seqs):
#     for i, pi_seq in enumerate(pi_seqs):
#         for j, pi in enumerate(pi_seq):
#             try:
#                 ic2cell(pi, sp.pi_ref.eps)
#             except AssertionError:
#                 print(i,j)
#                 print(pi)
#      exit()
    #G = graph_factory(gopts.graph_lib)
    G = QGraph()

    for path, pi_seq in zip(error_paths, pi_seqs):
        # Normalize the (x, w) list
        # e.g., for a 3 state length path, we have 3 abstract states
        # and 2 pi cells (ic)
        # a0 --pi01--> a1 --pi12--> a2
        # Which means that the new flattened graph will have nodes
        # like (a0, p01), (a1, p12), (a2, ?)
        # a2 can not have None as its pi due to self loops being
        # modled. As a pi value is not known for self loop dynamics,
        # we assume all possible pi values, i.e., pi \in pi_cons and
        # append it at the end of the pi_seq.
        # Note, instead of directly appending it, we append None, and
        # detect it below and replace accordingly. This is done in
        # order to keep ic2cell() and ic2multicell() separate. This
        # helps in catching bugs! Can be removed later to improve
        # performance.
        pi_seq.append(None)

#         ###############################
#         # Add final and initial states
#         #TODO: Only works for AA.num_dims.pi = 0
#         a0, af = path[0], path[-1]
#         x0cell, xfcell = abs_state2cell(a0, AA), abs_state2cell(af, AA)
#         #q0, qf = Qx(a0, x0cell), Qx(af, xfcell)
#         q0, qf = Qx(x0cell), Qx(xfcell)
#         G.init.add(q0)
#         G.final.add(qf)
#         ###############################

        # TODO: Merge the two branches?
        if AA.num_dims.pi:
            # Fix this crap: have wcell on edge?
            raise NotImplementedError
            for (a1, a2), (pi1_ic, pi2_ic) in it.izip_longest(U.pairwise(path), U.pairwise(pi_seq)):
                x1cell, x2cell = abs_state2cell(a1, AA), abs_state2cell(a2, AA)
                w1cell = ic2cell(pi1_ic, sp.pi_ref.eps)

                q1 = Qxw(a1, x1cell, w1cell)

                if pi2_ic is None:
                    w2cells = ic2multicell(sp.pi_ref.i_cons, sp.pi_ref.eps)
                    q2s = [Qxw(a2, x2cell, w2cell) for w2cell in w2cells]
                    for q2 in q2s:
                        G.add_edge(q1, q2)
                else:
                    w2cell = ic2cell(pi2_ic, sp.pi_ref.eps)
                    q2 = Qxw(a2, x2cell, w2cell)
                    G.add_edge(q1, q2)
        else:
            for (a1, a2) in U.pairwise(path):
                x1cell, x2cell = abs_state2cell(a1, AA), abs_state2cell(a2, AA)
                #q1, q2 = Qx(a1, x1cell), Qx(a2, x2cell)
                q1, q2 = Qx(x1cell), Qx(x2cell)
                G.add_edge(q1, q2)
    return G


def error_graph2qgraph_xw(sp, AA, initial_state_set, final_state_set, error_graph):
    G = QGraph()

#     # Add all initial and final q
#     #TODO: works only for AA.num_dims.pi = 0
#     ################################
#     for a in error_graph.nodes():
#         if a in initial_state_set:
#             xcell = abs_state2cell(a, AA)
#             #q = Qx(a, xcell)
#             q = Qx(xcell)
#             G.init.add(q)
#         elif a in final_state_set:
#             xcell = abs_state2cell(a, AA)
#             #q = Qx(a, xcell)
#             q = Qx(xcell)
#             G.final.add(q)
#         else:
#             pass
#     ##############################

    for edge in error_graph.all_edges():

        a1, a2 = edge

        x1cell, x2cell = abs_state2cell(a1, AA), abs_state2cell(a2, AA)

        if AA.num_dims.pi:
            pi1_ic = error_graph.edge_attrs(edge)['pi']
            assert(pi1_ic is not None)
            w1cell = ic2cell(pi1_ic, sp.pi_ref.eps)
            # This is a mess. Instead of being an edge, pi is actuall
            # in the node....fix it somehow
            assert(False)
            w2cell = ic2cell(pi2_ic, sp.pi_ref.eps)

            q1, q2 = Qxw(x1cell, w1cell), Qxw(x2cell, w2cell)
        else:
            #q1, q2 = Qx(a1, x1cell), Qx(a2, x2cell)
            q1, q2 = Qx(x1cell), Qx(x2cell)

        G.add_edge(q1, q2)
    return G


def get_pwa_system(sys, prop, sp, qgraph):
    #pwa_sys_prop.pwa_model

    pwa_sys_prop = build_pwa_model(sys, prop, qgraph, sp, 'dft')


#     if settings.debug:
#         qg.draw_graphviz()
#         qg.draw_mplib()

    draw_model(gopts.construct_path(sys.sys_name), pwa_sys_prop.pwa_model)

    #max_path_len = max([len(path) for path in error_paths])
    #print('max_path_len:', max_path_len)

    # depth = max_path_len - 1, because path_len = num_states along
    # path. This is 1 less than SAL depth and simulation length
    #depth = max(int(np.ceil(AA.T/AA.delta_t)), max_path_len - 1)

    depth = int(np.ceil(prop.T/sys.delta_t))

    assert(settings.CE) # adding a depth to accomodate the last
    # transition which takes a location to the CE
    depth += 1
    print('depth :',  depth)

#     if settings.debug_plot:
#         # Need to define a new function to simulate inputs
#         sim_n_plot(sp, AA, prop, error_paths, depth, pwa_model)


    # TODO: The reason we need this is to encode the transitions which
    # land to error states. This is required to discard bmc paths
    # which do the below:
    #
    # p0 -m01-> p1 -m12-> p2
    # Where p1->p2 is valid, but m12(x) \not-in p2(x)
    # cell = p1 and p1(x) -->
    # cell = p2 and x' = m12(x)
    #
    # Clearly, this does not enforce x' \in p2(x)
    # Hence, even if x' \in prop, it did not reallty reach there by
    # following the rules of evolution of the pwa system.
    # Specifically, ignored: x' \in p2(x). And hence, it used m12 to
    # go outside p2(x), but the map m12 was meant specifically to go
    # to p2
    #
    # The real reason for this is, that the current BMC encoding is
    # faulty. It encodes partitions, but instead, we should be
    # encoding transitions. That is, each location of the sal
    # transition system is actually a transitions pi --mij-->  pj
    # Hence, we need to know the error locations, from where we add
    # one more partition.

    # older calculation of prop_cells
    #prop_cells = {abs_state2cell(path[-1], AA) for path in error_paths}

    #prop_cells = {abs_state2cell(s, AA) for s in (final_state_set)}

#     if gopts.max_paths > 0:
#         prop_cells = {abs_state2cell(path[-1], AA) for path in error_paths}
#         init_cells = {abs_state2cell(path[0], AA) for path in error_paths}
#     else:
#         prop_cells = {abs_state2cell(s, AA) for s in (final_state_set)}
#         init_cells = {abs_state2cell(s, AA) for s in (initial_state_set)}

    assert(pwa_sys_prop.init_partitions)
    assert(pwa_sys_prop.final_partitions)

#     # partitions do not have a hash function. Hence, using a work
#     # around to avoid duplications
#     # Create a mapping from Q -> partitions
#     Q_p_map = {sm.p.ID.xcell: sm.p for sm in pwa_model if sm.p.ID.xcell in prop_cells}
#     # Q have a hash function
#     prop_partitions = [sm.p for sm in pwa_model if sm.p.ID.xcell in prop_cells]

#     init_partitions = {sm.p.ID.xcell: sm.p for sm in pwa_model if sm.p.ID.xcell in init_cells}.values()
    return pwa_sys_prop, depth


def refine_dft_model_based(AA, errors, initial_state_set, final_state_set, sp, sys_sim, sys, prop):

    # initialize Qxw class
    if AA.num_dims.pi:
        Qxw.init(sp.pi_ref.i_cons)

    if gopts.max_paths > 0:
        error_paths, pi_seqs = errors
        qgraph = get_qgraph_xw(sp, AA, error_paths, pi_seqs)
    else:
        error_graph = errors
        qgraph = error_graph2qgraph_xw(sp, AA, initial_state_set,
                                       final_state_set, error_graph)

    # make sure init and final are not empty
    #qgraph = get_qgraph_xw(sp, AA, error_paths, pi_seqs)
    #qgraph = error_graph2qgraph_xw(sp, AA, error_graph)

    # Add init and final states to qgraph
    qgraph.init = {q for q in qgraph if q.xcell.ival_constraints & prop.init_cons}
    qgraph.final = {q for q in qgraph if q.xcell.ival_constraints & prop.final_cons}
#     assert(qgraph.init == init)
#     assert(qgraph.final == final)

    assert(qgraph.init)
    assert(qgraph.final)

    pwa_sys_prop, depth = get_pwa_system(sys, prop, sp, qgraph)

    # flush all plots: must block
    gopts.plotting.show(block=True)

    if TESTCODE:


        bmc = lala(pwa_sys_prop.pwa_model, depth,
                   pwa_sys_prop.init_partitions,
                   pwa_sys_prop.final_partitions,
                   sys.sys_name, 'dft', AA, sys, prop, sp)
        list(bmc.print_all_CE(0))

#         qgraph_ref_gen = bmc.print_all_CE(1)
#         qgraphs = list(qgraph_ref_gen)

#         for qgraph_ref in qgraphs:
#             U.pause('qgraph refined, checking it')
#             pwa_sys_prop, depth = get_pwa_system(sys, prop, sp, qgraph_ref)
#             bmc =\
#                 lala(pwa_sys_prop.pwa_model, depth,
#                      pwa_sys_prop.init_partitions,
#                      pwa_sys_prop.final_partitions,
#                      sys.sys_name, 'dft', AA, sys, prop, sp)
#             qqgraph_ref_gen = bmc.print_all_CE(2)
#             qqgraphs = list(qqgraph_ref_gen)
#             if not qqgraphs:
#                 U.pause('refinement fails: NO CE!')


# #         bmc = lala(pwa_sys_prop.pwa_model, depth,
# #                    pwa_sys_prop.init_partitions,
# #                    pwa_sys_prop.final_partitions,
# #                    sys.sys_name, 'dft', AA, sys, prop, sp)
# #         list(bmc.print_all_CE(0))
        exit()

    check4CE(pwa_sys_prop.pwa_model, depth,
             pwa_sys_prop.init_partitions,
             pwa_sys_prop.final_partitions,
             sys.sys_name, 'dft', AA, sys, prop, sp)


def lala(pwa_model, depth, init_partitions, prop_partitions, sys_name, model_type, AA, sys, prop, sp):

    # Remove prop_partitions
    assert(settings.CE)
    # Extend both init set and final set to include inputs if any
    dummy_cons = top2ic(AA.num_dims.pi) # T <=> [-inf, inf]
    safety_prop = IntervalCons.concatenate(sp.final_cons, dummy_cons)
    init_cons = (sp.init_cons if AA.num_dims.pi == 0
                 else IntervalCons.concatenate(
                     sp.init_cons,
                     sp.pi_ref.i_cons))

    xs = ['x'+str(i) for i in range(AA.num_dims.x)]
    ws = ['w'+str(i) for i in range(AA.num_dims.pi)]
    # Order is important
    vs = xs + ws

    bmc = BMC.factory(
            gopts.bmc_engine,
            sys,
            prop,
            vs,
            pwa_model, init_cons, safety_prop,
            init_partitions,
            prop_partitions,
            gopts.construct_path,
            '{}_{}'.format(sys_name, model_type),
            model_type)

    return bmc

def check4CE(pwa_model, depth, init_partitions, prop_partitions, sys_name, model_type, AA, sys, prop, sp):


    # Remove prop_partitions
    assert(settings.CE)
    # Extend both init set and final set to include inputs if any
    dummy_cons = top2ic(AA.num_dims.pi) # T <=> [-inf, inf]
    safety_prop = IntervalCons.concatenate(sp.final_cons, dummy_cons)
    init_cons = (sp.init_cons if AA.num_dims.pi == 0
                 else IntervalCons.concatenate(
                     sp.init_cons,
                     sp.pi_ref.i_cons))

    xs = ['x'+str(i) for i in range(AA.num_dims.x)]
    ws = ['w'+str(i) for i in range(AA.num_dims.pi)]
    # Order is important
    vs = xs + ws

    bmc = BMC.factory(
            gopts.bmc_engine,
            sys,
            prop,
            vs,
            pwa_model, init_cons, safety_prop,
            init_partitions,
            prop_partitions,
            gopts.construct_path,
            '{}_{}'.format(sys_name, model_type),
            model_type)

    status = bmc.check(depth)
    if status == InvarStatus.Safe:
        print('Safe')
        exit()
    elif status == InvarStatus.Unsafe:
        bmc_trace = bmc.get_last_trace()
        print(bmc_trace)
        print(bmc_trace.to_array())
        while bmc_trace is not None:
            pwa_trace = bmc.get_last_pwa_trace()
            if settings.debug:
                print(pwa_trace)
                print(bmc_trace)
            print('Unsafe...trying to concretize...')
            #verify_bmc_trace(AA, sys, prop, sp, bmc.trace, xs, ws)
            verify_bmc_trace(AA, sys, prop, sp, bmc_trace, pwa_trace)

            bmc.get_new_disc_trace()
            bmc_trace = bmc.get_last_trace()
            #U.pause()

    elif status == InvarStatus.Unknown:
        print('Unknown...exiting')
        exit()
    else:
        raise err.Fatal('Internal')


def verify_bmc_trace(AA, sys, prop, sp, bmc_trace, pwa_trace):
    """Get multiple traces and send them for random testing
    """

    xw_array = bmc_trace.to_array()
    x_array, w_array = np.split(xw_array, [AA.num_dims.x], axis=1)
    pi_seq = w_array
    #init_assignments = trace[0].assignments
    #x0_array = np.array([init_assignments[x] for x in xs])
    # Trace consists of transitions, but we want to interpret it as
    # locations (abs_states + wi). Hence, subtract 1 from trace.
    num_trace_states = len(bmc_trace)-1

    # TODO: fix inputs!!
    #pi_seq = [[step.assignments[w] for w in ws] for step in bmc_trace[:-1]]
    gopts.plotting.new_session()
    res = rt.concretize_bmc_trace(sys, prop, AA, sp, num_trace_states, x_array, pi_seq)

    gopts.plotting.new_session()
    init_cons_subset = azp.overapprox_x0(AA.num_dims, prop, pwa_trace)
    rt.concretize_init_cons_subset(sys, prop, AA, sp, num_trace_states, x_array, pi_seq, init_cons_subset)
    return


def get_abstract_path(AA, x_array):
    # old function, not sure about its current health: marking it non
    # implemented
    raise NotImplementedError

    abs_path = []
    t, d, pvt, ci, s, pi, u = [0]*7
    for x in x_array:
        concrete_state = state.State(t, x, d, pvt, ci, s, pi, u)
        abs_state = AA.get_abs_state_from_concrete_state(concrete_state)
        abs_path.append(abs_state)
    return abs_path


def refine_dmt_model_based(AA, error_paths, pi_seq_list, sp, sys_sim, bmc_engine):
    """refine using discrete time models

    Parameters
    ----------
    A :
    error_paths :
    pi_seq_list :
    sp :
    sys_sim :

    Returns
    -------

    Notes
    ------
    does not handle pi_seq_list yet
    """
    # Unmantained old code. Update before using
    raise NotImplementedError
    # traversed_abs_state_set
    tas = {state for path in error_paths for state in path}

    pwa_models = build_pwa_dt_model(AA, tas, sp, sys_sim)

    bmc = BMC.factory(bmc_engine)
    prop = sp.final_cons

    bmc.init(AA.num_dims.x, pwa_models, sp.init_cons, prop, 'vdp_dmt', 'dmt')
    bmc.check()


def build_pwa_model(sys, prop, qgraph, sp, model_type):
    """build_pwa_model
    Builds both dft and rel models

    Parameters
    ----------
    abs_objs : Either abs_states (for dft models) or relations
              [tuple(abs_state_src, abs_state_target)] for rel model
    sp : system params
    tol : modeling error tolerance
    include_err : include error in determining next state
                  x' = x +- errror
    """
    tol = gopts.max_model_error
    include_err = gopts.model_err

    # number of training samples
    #TODO : should be min and not max!
    ntrain = min(gopts.regression_sim_samples * MORE_FACTOR, MAX_TRAIN)
    # number of test samples
    #ntest = min(ntrain * TEST_FACTOR, MAX_TEST)

    dt = sys.delta_t
    step_sim = simsys.get_step_simulator(sp.controller_sim, sp.plant_sim, dt)

    #abs_state_models = {}

    pwa_model = rel.PWARelational()
    init_partitions = set()
    final_partitions = set()

    # for ever vertex (abs_state) in the graph
    for q in qgraph:
        if settings.debug:
            print('modeling: {}'.format(q))
        for sub_model in q_affine_models(prop, ntrain, step_sim, tol, include_err, qgraph, q):
            assert(sub_model is not None)
            # sub_model.pnexts[0] = sub_model.p.ID to enforce self loops
            print(U.colorize('{} -> {}, e%:{}, status: {}, e: {}, A:{}, b:{}'.format(
                sub_model.p.ID,
                [p.ID for p in sub_model.pnexts],
                np.trunc(sub_model.max_error_pc),
                sub_model_status(sub_model),
                sub_model.m.error,
                str(sub_model.m.A).replace('\n', ''),
                sub_model.m.b)))
            pwa_model.add(sub_model)
            #abs_state_models[abs_state] = sub_model
        # Even if a state gets split, its recorded
        if q in qgraph.init:
            init_partitions.add(pwa.Partition(*q.poly(), part_id=q))
        # TODO: If we split a final cell to increase precision for
        # the transition to concrete error_states and break the
        # terminal self-loop, both will get recorded and get
        # weird.
        if q in qgraph.final:
            final_partitions.add(pwa.Partition(*q.poly(), part_id=q))

    pwa_sys_prop = PWASYSPROP(pwa_model, init_partitions, final_partitions)
    return pwa_sys_prop


# TODO: fix this mess and move it to pwa models
def sub_model_status(s, status2str={KMAX_EXCEEDED: 'kamx exceeded', SUCCESS: 'success', TERMINAL: 'terminal'}):
    return status2str[s.status]


def draw_model(sys_name, pwa_model):
    """Adds a 'label' attribute to the edges. This is useful for
    graphs rendered using graphviz, as it annotates the edges with the
    value of those attributes. Currently, the label's value is the
    error between a learned relation between two nodes."""

#     # update edge attribute by adding it again
#     def label_edge_with_error(e, error):
#         e_attrs = qgraph.G[e[0]][e[1]]
#         e_attrs['label'] = np.round(error, 2)
#         qgraph.G.add_edge(*e, attr_dict=e_attrs)

#     # keep on looping through the submodels till the right one is
#     # found
#     for e in qgraph.G.edges():
#         q1 = e[0]
#         q2 = e[1]
#         for sub_model in pwa_model:
#             # TODO: builds a list every time!
#             if sub_model.p.ID == q1 and any(q2 == pn.ID for pn in sub_model.pnexts):
#                 label_edge_with_error(e, sub_model.max_error_pc)
#                 break

    G = graph_factory(gopts.graph_lib)

    for sub_model in pwa_model:
        for p_ in sub_model.pnexts:
            #e_attr = {'label': np.round(sub_model.max_error_pc, 2)}
            error = np.trunc(sub_model.max_error_pc)
            color = 'red' if sub_model.status == KMAX_EXCEEDED else 'black'
            G.add_edge(sub_model.p.ID, p_.ID, label=error, color=color)
        G.node_attrs(sub_model.p.ID)['label'] = sub_model.p.ID
        G.node_attrs(sub_model.p.ID)['tooltip'] = sub_model.p.ID.ival_constraints

    G.draw_graphviz(sys_name)
    #G.draw_mplib(sys_name)


# TODO: it is a superset of build_pwa_dft_model
# and should replace it
def build_pwa_dt_model(AA, abs_states, sp, sys_sim):
    """build_pwa_dt_model

    Parameters
    ----------
    AA :
        AA is
    abs_states :
        abs_states is
    sp :
        sp is
    sys_sim :
        sys_sim is

    Returns
    -------

    Notes
    ------
    Builds a model with time as a discrete variable.
    i.e., models the behaviors resulting from several chosen time
    steps and not only the one specified in .tst as delta_t.
    """

    dt_steps = [0.01, 0.1, AA.plant_abs.delta_t]
    err.warn('using time steps: {}'.format(dt_steps))
    step_sims = [simsys.get_step_simulator(sp.controller_sim, sp.plant_sim, dt)
                 for dt in dt_steps]

    pwa_models = {}
    for dt, step_sim in zip(dt_steps, step_sims):
        pwa_model = pwa.PWA()
        for abs_state in abs_states:
            sub_model = affine_model(abs_state, AA, sp, step_sim)
            pwa_model.add(sub_model)

        pwa_models[dt] = pwa_model
    return pwa_models


def model(tol, X, Y):
    try:
        rm = AFM.OLS(X, Y)
    except AFM.UdetError:
        return []
    e_pc = rm.max_error_pc(X, Y)
    if settings.debug:
        err.imp('error%: {}'.format(e_pc))
    error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]
    error_exceeds_tol = len(error_dims) > 0
    refine = error_exceeds_tol
    #TODO
    status = SUCCESS if not refine else KMAX_EXCEEDED
    return [(rm, e_pc, status)]


def mdl_1relational(prop, tol, step_sim, qgraph, q, X, Y):
    assert(X.shape[1] == q.dim)
    assert(Y.shape[1] == q.dim)
    assert(X.shape[0] == Y.shape[0])

    ms = []
    # prevent repetition in case a self loop exists
    neighbors_including_self = set(qgraph.neighbors(q))
    neighbors_including_self.add(q)

    for qi in neighbors_including_self:
        if settings.debug:
            print('checking qi: ', qi)
        sat = qi.sat(Y)

        if any(sat):
            rm_qseq = model(tol, X[sat], Y[sat])
            l = [(rm_, (q, qi), e_pc_, status_) for rm_, e_pc_, status_ in rm_qseq]
            ms.extend(l)
        else:
            if(qi == q):
                # no self loop observed
                if settings.debug:
                    print('no self loop found')
            else:
                err.warn('out of samples...Giving up on the edge!')

    # TODO: this will happen when the last location fails? confirm
    if not ms:
        # This means, all samples which landed are in a cell which
        # was never initially explored by S3CAM. Can happen, but
        # possibility is very very low.
        err.Fatal('Very low prob. of happening. Check code')
#     else:
#         status = SUCCESS
#         if settings.debug:
#             print('error is under control...')
#         ms = [(rm, [], e_pc, status)]

    return ms


def mdl(AA, prop, tol, step_sim, qgraph, q, XY, Y_, k, kmin, kmax):
    X, Y = XY
    assert(X.shape[1] == q.dim)
    assert(Y_.shape[1] == q.dim)
    assert(X.shape[0] == Y.shape[0] == Y_.shape[0])

    #print(U.colorize('# samples = {}'.format(X.shape[0])))

    if k >= kmin:
        rm = AFM.OLS(X, Y)
        e_pc = rm.max_error_pc(X, Y)
        if settings.debug:
            err.imp('error%: {}'.format(e_pc))
        error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]
        error_exceeds_tol = len(error_dims) > 0
        refine = error_exceeds_tol
        #err.warn('e%:{}, |e%|:{}'.format(e_pc, np.linalg.norm(e_pc, 2)))
        if settings.debug_plot:
            rm.plot(X, Y, tol, 'q:{}, err:{}'.format(q, e_pc))
        gopts.plotting.show()
    else:
        refine = True

    if refine:
        ms = []
        if settings.debug:
            print('error exceeds...')
        if k >= kmax:
            assert(k == kmax)
            status = KMAX_EXCEEDED
            if settings.debug:
                err.warn('max depth exceeded but the error > tol. Giving up!')
            ms = [(rm, [], e_pc, status)]
        else:
            # first check for existance of a self loop
            #if any(q.sat(Y)):
                #err.imp('self loop exists')

            for qi in it.chain([q], qgraph.neighbors(q)):
                if settings.debug:
                    print('checking qi: ', qi)

                Y__ = qi.sim(step_sim, Y_)
                sat = qi.sat(Y__)

                # TODO: If we are out of samples, we can't do much. Need to
                # handle this situation better? Not sure? Request for more
                # samples? Give up?
                from matplotlib import pyplot as plt
                if settings.debug and settings.plot:
                    #ax = plt.gca()
                    #ax.set_color_cycle(['b'])
                    #ax.set_ylim([-10, 10])
                    #ax.set_xlim([-2, 2])
                    gopts.plotting.acquire_global_fig()
                    gopts.plotting.single_color('b')
                    gopts.plotting.set_range((-2, 2), (-7, 7))
                    # TODO: pull out abstract state from concrete
                    # state Y_. q.a is no longer being mantained
                    # It can be mantained, but it would then be an
                    # abstract state w/o a valid abstraction
                    gopts.plotting.plot_abs_states(AA, prop, [q.a])
                    if any(sat):
                        gopts.plotting.plot(Y__[sat, 0], Y__[sat, 1], '*')
                        gopts.plotting.plot(Y_[sat, 0], Y_[sat, 1], '.')
                    gopts.plotting.plot_abs_states(AA, prop, [qi.a])

                if any(sat):
                    rm_qseq = mdl(AA, prop, tol, step_sim, qgraph, qi, (X[sat], Y[sat]), Y__[sat], k+1, kmin, kmax)
                    l = [(rm_, [qi]+qseq_, e_pc_, status_)
                         for rm_, qseq_, e_pc_, status_ in rm_qseq]
                    ms.extend(l)
                else:
                    if(qi == q):
                        pass # no self loop observed
                    else:
                        err.warn('out of samples...Giving up on the edge!')

#############################################
#############################################
            # Delete unused traj seg: assumes 1-relational modeling
            if settings.paper_plot:
                ax = plt.gca()
                remove = set()
                for l in ax.lines:
                    xy = l.get_xydata()
                    if np.all(q.sat(xy[0:1, :])):
                        if not any([np.all(qi.sat(xy[-1:, :])) for qi in it.chain([q], qgraph.neighbors(q))]):
                            remove.add(l)
                for l in remove:
                    ax.lines.remove(l)

##############################################
##############################################

        # TODO: this will happen when the last location fails? confirm
        if not ms:
            # This means, all samples which landed are in a cell which
            # was never initially explored by S3CAM. Can happen, but
            # possibility is very very low.
            err.Fatal('Very low prob. of happening. Check code')
    else:
        status = SUCCESS
        if settings.debug:
            print('error is under control...')
        ms = [(rm, [], e_pc, status)]

    return ms


def dummy_sub_model(q):
    # stationary dynamics
    A, b = np.eye(q.dim), np.zeros(q.dim)
    # no error
    e = zero2ic(q.dim)
    dmap = rel.DiscreteAffineMap(A, b, e)

    C, d = q.ival_constraints.poly()
    p = pwa.Partition(C, d, q)

    future_partitions = []
    # self loop
    pnexts = [p]

    sub_model = rel.KPath(dmap, p, pnexts, future_partitions)
    sub_model.max_error_pc = np.zeros(q.dim)
    sub_model.status = TERMINAL
    return sub_model


def q_affine_models_old(AA, prop, ntrain, step_sim, tol, include_err, qgraph, q):
    """Find affine models for a given Q

    Parameters
    ----------
    cell : cell
    step_sim : 1 time step (delta_t) simulator
    tol : each abs state is split further into num_splits cells
    in order to meet: modeling error < tol (module ntests samples)

    Returns
    -------
    pwa.SubModel()

    Notes
    ------
    """
    # Is non relational modeling being done? No, by default
    pwa_non_relational = False
    sub_models = []

    try_again = True
    ntries = 1
    MAX_TRIES = 2
    while try_again:
        last_node = not qgraph.edges(q)
        X, Y = q.get_rels(prop, step_sim, ntrain)
        assert(not X.size == 0 or Y.size == 0)
        assert(not Y.size == 0 or X.size == 0)
        if X.size == 0:
            # make sure it is the last node: had no edges
            assert(last_node)
            # The cell is completely inside the property
            # If not, it means that the volume of Cell - prop is very
            # small and a sample wasnt found in there.
            assert(prop.final_cons.contains(q.ival_constraints))
            return [dummy_sub_model(q)]

        try:
            regression_models = mdl(AA, prop, tol, step_sim, qgraph, q, (X, Y), X, k=0, kmin=KMIN, kmax=KMAX)
            # we are done!
            if regression_models:
                try_again = False
            # else try again
            else:
                err.warn('no model found')
                if ntries > MAX_TRIES:
                    if last_node:
                        err.warn('giving up')
                        try_again = False
                    else:
                        err.warn('can happen rarely...')
        except AFM.UdetError:
            pass
        print('trying again')
        # double the number of samples and try again
        ntrain *= 2
        # repeat!
        ntries += 1

    # try again on failure, and settle with non relational models
    if not regression_models:
        err.warn('No model found for q: {}'.format(q))
        regression_models = mdl(AA, prop, np.inf, step_sim, qgraph, q, (X, Y), X, k=0, kmin=0, kmax=1)
        assert(regression_models)
        # No model found, get a non-relational model as the worst case
        pwa_non_relational = True
        #import IPython
        #IPython.embed()

#     # TODO: fix this messy handling...?
#     if not regression_models:
#         # no model was found...node must be a sink node, otherwise
#         # such a condition is not possible!
#         # It must be due to missing neighbors of th sink node.
#         assert(qgraph.out_degree(q) == 0)
#         # Now request for the model once more but given an infinite
#         # tolerance so that we always get one. K=1 for sanity's sake,
#         # as a depth > 1 should never be reached with tol = Inf.
#         regression_models = mdl(np.inf, step_sim, qgraph, q, (X, Y), X, 1)
#         # Due to the tolerance being Inf, we should get back a single
#         # model
#         assert(len(regression_models) == 1)

    for rm, q_seq, e_pc, status in regression_models:
        A, b = q.modelQ(rm)
        e = q.errorQ(include_err, rm)
        dmap = rel.DiscreteAffineMap(A, b, e)

        C, d = q.ival_constraints.poly()
        p = pwa.Partition(C, d, q)

        future_partitions = []

        pnexts = []

        #if len(q_seq) == 1:
        # if q_seq is empty, all its neighbours are reachable
        if not q_seq:
            # Relational modeling is currently forced as KMIN = 1
            #assert(False)
            assert(pwa_non_relational)
            # No relational modeling was done. Use the relations from
            # the graph. Add transitions to cell only seen in the
            # subgraph.
            # Force self loops just in case. The other option is to
            # examin in the mdl() function if a self loop is possible
            #err.warn('forcing self loops for every location!')
            for qi in it.chain([q], qgraph.neighbors(q)):
                C, d = qi.ival_constraints.poly()
                #pnexts.append(pwa.Partition(C, d, qi))
                pnexts = [pwa.Partition(C, d, qi)]

                sub_model = rel.KPath(dmap, p, pnexts, future_partitions)
                sub_model.max_error_pc = e_pc
                sub_model.status = status
                sub_models.append(sub_model)

        # Relational modeling is available. Add the edge which was
        # used to model this transition.
        else:
            # Add the immediate next reachable state
            qnext = q_seq[0]
            C, d = qnext.ival_constraints.poly()
            pnexts.append(pwa.Partition(C, d, qnext))
            # Add the states reachable in future
            for qi in q_seq[1:]:
                C, d = qi.ival_constraints.poly()
                future_partitions.append(pwa.Partition(C, d, qi))

            sub_model = rel.KPath(dmap, p, pnexts, future_partitions)
            sub_model.max_error_pc = e_pc
            sub_model.status = status
            sub_models.append(sub_model)
    return sub_models


# models can be split
def q_affine_models(prop, ntrain, step_sim, tol, include_err, qgraph, q):
    """Find affine models for a given Q

    Parameters
    ----------
    cell : cell
    step_sim : 1 time step (delta_t) simulator
    tol : each abs state is split further into num_splits cells
    in order to meet: modeling error < tol (module ntests samples)

    Returns
    -------
    pwa.SubModel()

    Notes
    ------
    """

    try_again = True
    ntries = 1
    #MAX_TRIES = 2
    MAX_TRIES = 0
    while try_again:
        last_node = not qgraph.edges(q)
        X, Y = q.get_rels(prop, step_sim, ntrain)
        assert(not X.size == 0 or Y.size == 0)
        assert(not Y.size == 0 or X.size == 0)
        if X.size == 0:
            # make sure it is the last node: had no edges
            assert(last_node)
            # The cell is completely inside the property
            # If not, it means that the volume of Cell - prop is very
            # small and a sample wasnt found in there.
            assert(prop.final_cons.contains(q.ival_constraints))
            return [dummy_sub_model(q)]

        try:
            regression_models = mdl_1relational(prop, tol, step_sim, qgraph, q, X, Y)
            # we are done!
            if regression_models:
                try_again = False
            # else try again
            else:
                err.warn('no model found')
                if ntries > MAX_TRIES:
                    if last_node:
                        err.warn('giving up on last node')
                    else:
                        err.warn('can happen rarely...')
                    try_again = False
        except AFM.UdetError:
            pass
        if try_again:
            print('trying again')
        # double the number of samples and try again
        ntrain *= 2
        # repeat!
        ntries += 1

    # try again on failure, and settle with non relational models
    #assert(regression_models)

    sub_models = []
    for rm, (qi, qj), e_pc, status in regression_models:
        A, b = qi.modelQ(rm)
        e = qi.errorQ(include_err, rm)
        dmap = rel.DiscreteAffineMap(A, b, e)

        C, d = qi.ival_constraints.poly()
        p = pwa.Partition(C, d, qi)

        future_partitions = []

        pnexts = []

        # Relational modeling is available. Add the edge which was
        # used to model this transition.
        # Add the immediate next reachable state
        C, d = qj.ival_constraints.poly()
        pnexts.append(pwa.Partition(C, d, qj))

        sub_model = rel.KPath(dmap, p, pnexts, future_partitions)
        sub_model.max_error_pc = e_pc
        sub_model.status = status
        sub_models.append(sub_model)
    return sub_models

def build_pwa_ct_model(AA, abs_states, sp, sys_sim):
    """Build a time continuous pwa model

    Parameters
    ----------
    AA :
        AA is
    abs_states :
        abs_states is
    sp :
        sp is
    sys_sim :
        sys_sim is

    Returns
    -------

    Notes
    ------
    Builds a model with time as a bounded continuous variable.
    i.e., models the behaviors resulting from several time steps and
    not only the one chosen one.
    """
    raise NotImplementedError


################################################
# ############# CEMETERY #######################
################################################


# #AA.plant_abs.get_abs_state_cell(abs_state.plant_state),
# def cell_affine_models(q, step_sim, ntrain, ntest, tol, include_err):
#     """cell_affine_models

#     Parameters
#     ----------
#     cell : cell
#     step_sim : 1 time step (delta_t) simulator
#     tol : each abs state is split further into num_splits cells
#     in order to meet: modeling error < tol (module ntests samples)

#     Returns
#     -------
#     pwa.SubModel()

#     Notes
#     ------
#     """
#     # XXX: Generate different samples for each time step or reuse?
#     # Not clear!
#     sub_models = []

#     X, Y = q.getxy_ignoramous(ntrain, step_sim)
#     rm = RegressionModel(X, Y)
#     X, Y = q.getxy_ignoramous(ntest, step_sim)
#     e_pc = rm.max_error_pc(X, Y) # error %
#     if settings.debug:
#         print('error%:', e_pc)
#     #error = np.linalg.norm(e_pc, 2)
#     # indices where error exceeds tol
#     error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]

#     if len(error_dims) > 0:
#         err.warn('splitting on e%:{}, |e%|:{}'.format(
#             e_pc, np.linalg.norm(e_pc, 2)))
#         for split_cell in q.split(axes=error_dims):
#             sub_models_ = cell_affine_models(
#                     split_cell, step_sim, ntrain, ntest, tol, include_err)
#             sub_models.extend(sub_models_)
#         return sub_models
#     else:
#         #print('error%:', rm.max_error_pc(X, Y))
#         A, b, C, d = q.modelQ(rm)
#         e = q.error(include_err, X, Y, rm)
#         dmap = pwa.DiscreteAffineMap(A, b, e)
#         part = pwa.Partition(C, d, q)
#         sub_model = pwa.SubModel(part, dmap)
#         if __debug__:
#             print('----------------Finalized------------------')
#     return [sub_model]



# def getxy_rel_ignoramous_force_min_samples(cell1, cell2, force, N, sim, t0=0):
#     """getxy_rel_ignoramous

#     """
#     xl = []
#     yl = []
#     sat_count = 0
#     if __debug__:
#         obs_cells = set()
#     while True:
#         x_array, y_array = getxy_ignoramous(cell1, N, sim, t0=0)
#         if __debug__:
#             for i in y_array:
#                 obs_cells.add(CM.cell_from_concrete(i, cell1.eps))
#             print('reachable cells:', obs_cells)

#         # satisfying indexes
#         sat_array = cell2.ival_constraints.sat(y_array)
#         sat_count += np.sum(sat_array)
#         xl.append(x_array[sat_array])
#         yl.append(y_array[sat_array])
#         # If no sample is found and force is True, must keep sampling till
#         # satisfying samples are found
#         if (sat_count >= MIN_TRAIN) or (not force):
#             break
#         if __debug__:
#             print('re-sampling, count:', sat_count)

#     print('found samples: ', sat_count)
#     return np.vstack(xl), np.vstack(yl)


# #AA.plant_abs.get_abs_state_cell(abs_state.plant_state),
# def cell_rel_affine_models(cell1, cell2, force, step_sim, ntrain, ntest, tol, include_err):
#     """cell_affine_models

#     Parameters
#     ----------
#     cell1 : source cell
#     cell2 : target cell
#     step_sim : 1 time step (delta_t) simulator
#     tol : each abs state is split further into num_splits cells
#     in order to meet: modeling error < tol (module ntests samples)

#     Returns
#     -------
#     pwa.SubModel()

#     Notes
#     ------
#     """
#     # XXX: Generate different samples for each time step or reuse?
#     # Not clear!
#     sub_models = []

#     X, Y = getxy_rel_ignoramous(cell1, cell2, force, ntrain, step_sim)
#     # No samples found => no model
#     training_samples_found = len(X) != 0
#     if not training_samples_found:
#         return [None]
#     rm = RegressionModel(X, Y)
#     X, Y = getxy_rel_ignoramous(cell1, cell2, True, ntest, step_sim)
#     testing_samples_found = len(X) != 0
#     # If valid samples are found, compute e_pc (error %) and dims
#     # where error % >= given tol
#     if testing_samples_found:
#         e_pc = rm.max_error_pc(X, Y)
#         error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]
#     # Otherwise, forget it!
#     else:
#         e_pc = None
#         error_dims = []

#     if __debug__:
#         print('error%:', e_pc)

#     if len(error_dims) > 0:
#         err.warn('splitting on e%:{}, |e%|:{}'.format(
#             e_pc, np.linalg.norm(e_pc, 2)))
#         for split_cell1 in cell1.split(axes=error_dims):
#             sub_models_ = cell_rel_affine_models(
#                     split_cell1, cell2, False, step_sim, ntrain, ntest, tol, include_err)
#             sub_models.extend(sub_models_)
#         return sub_models
#     else:
#         A, b = rm.A, rm.b
#         C1, d1 = cell1.ival_constraints.poly()
#         C2, d2 = cell2.ival_constraints.poly()

#         e = rm.error(X, Y) if (include_err and testing_samples_found) else None

#         dmap = rel.DiscreteAffineMap(A, b, e)
#         part1 = rel.Partition(C1, d1, cell1)
#         part2 = rel.Partition(C2, d2, cell2)
#         sub_model = rel.Relation(part1, part2, dmap)
#         if __debug__:
#             print('----------------Finalized------------------')
#     return [sub_model]


# def refine_rel_model_based(
#         AA, error_paths, pi_seq_list, sp, sys_sim, sys, prop):
#     '''does not handle pi_seq_list yet'''

#     # abs_state relations: maps an abs_state to other abs_states
#     # reachable in one time step
#     abs_relations = defaultdict(set)
#     for path in error_paths:
#         # abs_state_1 -> abs_state_2
#         for a1, a2 in U.pairwise(path):
#             abs_relations[a1].add(a2)

#     flat_relations = []
#     for abs_state, rch_states in abs_relations.iteritems():
#         flat_relation = list(itertools.product([abs_state], rch_states))
#         flat_relations.extend(flat_relation)

#     pwa_model = build_pwa_model(
#             AA, flat_relations, sp, gopts.max_model_error,
#             gopts.model_err, 'rel')

#     if __debug__:
#         sim_n_plot(error_paths, pwa_model, AA, sp)
#     check4CE(pwa_model, error_paths, sys.sys_name, 'rel', AA, sys, prop, sp, gopts.bmc_engine)



# def q_affine_models(ntrain, ntest, step_sim, tol, include_err, qgraph, q):
#     """cell_affine_models

#     Parameters
#     ----------
#     cell : cell
#     step_sim : 1 time step (delta_t) simulator
#     tol : each abs state is split further into num_splits cells
#     in order to meet: modeling error < tol (module ntests samples)

#     Returns
#     -------
#     pwa.SubModel()

#     Notes
#     ------
#     """
#     # XXX: Generate different samples for each time step or reuse?
#     # Not clear!
#     sub_models = []

#     X, Y = q.getxy_ignoramous(ntrain, step_sim, qgraph)
#     rm = RegressionModel(X, Y)
#     X, Y = q.getxy_ignoramous(ntest, step_sim)
#     e_pc = rm.max_error_pc(X, Y) # error %
#     if __debug__:
#         print('error%:', e_pc)
#     #error = np.linalg.norm(e_pc, 2)
#     # error exceeds tol in error_dims
#     error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]

#     if len(error_dims) > 0:
#         err.warn('splitting on e%:{}, |e%|:{}'.format(
#             e_pc, np.linalg.norm(e_pc, 2)))
#         for split_q in q.split(axes=error_dims):
#             sub_models_ = q_affine_models(
#                     ntrain, ntest,
#                     split_q, step_sim, tol, include_err)
#             sub_models.extend(sub_models_)
#         return sub_models
#     else:
#         A, b, C, d = q.modelQ(rm)
#         e = q.errorQ(include_err, X, Y, rm)
#         dmap = pwa.DiscreteAffineMap(A, b, e)
#         part = pwa.Partition(C, d, q)
#         sub_model = pwa.SubModel(part, dmap)
#         if __debug__:
#             print('----------------Finalized------------------')
#     return [sub_model]

# def get_qs_from_error_paths(sp, AA, error_paths, pi_seqs):
#     if AA.num_dims.pi == 0:
#         # traversed_abs_state_set
#         tas = {state for path in error_paths for state in path}
#         qs = [Qx(abs_state2cell(a, AA)) for a in tas]
#     else:
#         pi_eps = sp.pi_ref.eps
#         # collect all pi which were encountered with the abs_state
#         abs_state_pi = defaultdict(set)
#         for path, pi_seq in zip(error_paths, pi_seqs):
#             for abs_state, pi in zip(path[:-1], pi_seq):
#                 abs_state_pi[abs_state].add(pi)

#         qs = []
#         for abs_state, pi_ic_list in abs_state_pi.iteritems():
#             xcell = abs_state2cell(abs_state, AA)
#             for pi_ic in pi_ic_list:
#                 wcell = ic2cell(pi_ic, pi_eps)
#                 qs.append(Qxw(xcell, wcell, sp.pi_ref.i_cons))
#     return qs
#
# def q_graph_models(ntrain, ntest, step_sim, tol, include_err, qgraph):
#     qmodels = {}
#     # Make a model for every q in the graph
#     for q in qgraph:
#         X, Y = q.get_rels(ntrain, step_sim)
#         models = mdl(tol, qgraph, q, (X, Y), X, K)
#         qmodels[q] = models
#     return qmodels


# def mdl_old(tol, step_sim, qgraph, q, (X, Y), Y_, k):
#     assert(X.shape[1] == q.dim)
#     assert(Y_.shape[1] == q.dim)
#     assert(X.shape[0] == Y.shape[0] == Y_.shape[0])

#     if k == -1:
#         err.warn('max depth exceeded but the error > tol. Giving up!')
#         return []

#     rm = RegressionModel(X, Y)
#     e_pc = rm.max_error_pc(X, Y)
#     err.imp('error%: {}'.format(e_pc))
#     error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]
#     error_exceeds_tol = len(error_dims) > 0
#     #err.warn('e%:{}, |e%|:{}'.format(e_pc, np.linalg.norm(e_pc, 2)))
#     if error_exceeds_tol:
#         ms = []
#         for qi in qgraph.neighbors(q):
#             Y__ = qi.sim(step_sim, Y_)
#             sat = qi.sat(Y__)
#             # TODO: If we are out of samples, we can't do much. Need to
#             # handle this situation better? Not sure? Request for more
#             # samples? Give up?
#             if any(sat):
#                 rm_qseq = mdl(tol, step_sim, qgraph, qi, (X[sat], Y[sat]), Y__[sat], k-1)
#                 l = [(rm_, [q]+qseq_, e_pc_) for rm_, qseq_, e_pc_ in rm_qseq]
#                 ms.extend(l)
#             else:
#                 err.warn('out of samples...Giving up!')

#         # The loop never ran due to q not having any neighbors,
#         # Or, no samples were left. We do the best with what we have
#         # then.
#         if not ms:
#             ms = [(rm, [q], e_pc)]
#         return ms
#     else:
#         return [(rm, [q], e_pc)]

