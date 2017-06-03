#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


#import matplotlib
# Force GTK3 backend. By default GTK2 gets loaded and conflicts with
# graph-tool
#matplotlib.use('GTK3Agg')
#global plt
#import matplotlib.pyplot as plt
import logging
import numpy as np
#from optparse import OptionParser
import argparse
#import argcomplete
import time
import sys as SYS

import dill

from core import abstraction
from core import sample
from core import simulatesystem as simsys
from core import scattersim as SS
from core import loadsystem
#import traces
from core import wmanager
from plot import plotting
from core import random_testing as RT
from core import properties


import globalopts
#from globalopts import opts as gopts

import err
import fileops as fp
import utils as U
from utils import print

import settings

from IPython import embed

#from guppy import hpy
#from pkgcore.config import load_config

#from pympler import tracker
#from pympler import refbrowser



#matplotlib.use('GTK3Agg')

#precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=True, nanstr=None, infstr=None, formatter=Nonu)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

###############################
# terminal color printing compatibility for windows
# https://pypi.python.org/pypi/colorama/0.2.4

# from colorama import init
# init()

# use this when we add windows portability
###############################


# start logger
def setup_logger():
    FORMAT = '[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s'
    FORMAT2 = '%(levelname) -10s %(asctime)s %(module)s:\
               %(lineno)s %(funcName)s() %(message)s'

    logging.basicConfig(filename='{}_secam.log'.format(TIME_STR), filemode='w', format=FORMAT2,
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger


# Remove this re-structuring!! Use existing structures instead.
class SystemParams:

    def __init__(
            self,
            #initial_state_set,
            #final_state_set,
            is_final,
            plant_sim,
            controller_sim,
            ci,
            pi,
            sampler,
            init_cons,
            final_cons,
            pi_ref,
            ci_ref
            ):

        #self.initial_state_set = initial_state_set
        #self.final_state_set = final_state_set
        self.is_final = is_final
        self.plant_sim = plant_sim
        self.controller_sim = controller_sim
        self.ci = ci
        self.pi = pi
        self.sampler = sampler
        self.init_cons = init_cons
        self.final_cons = final_cons
        self.pi_ref = pi_ref
        self.ci_ref = ci_ref
        return


def sanity_check_input(sys, prop):
    return


#side effect: creates the dir
def setup_dir(sys, dirname):
    # If no output folder is specified, generate one
    if dirname is None:
        dirname = '{}_{}'.format(sys.sys_name, TIME_STR)
    if fp.file_exists(dirname):
        raise err.Fatal('output dir exists! Please provide a new dir name to prevent override.')

    def construct_path(fname): return fp.construct_path(fname, dirname)
    # create otuput directory
    fp.make_dir(dirname)
    return construct_path


def create_abstraction(sys, prop):
    num_dims = sys.num_dims
    plant_config_dict = sys.plant_config_dict
    controller_path_dir_path = sys.controller_path_dir_path
    controller_object_str = sys.controller_object_str

    T = prop.T

    METHOD = globalopts.opts.METHOD

    plant_abstraction_type = 'cell'
    if METHOD == 'concolic':
        controller_abstraction_type = 'concolic'

        # Initialize concolic engine

        var_type = {}

        # state_arr is not symbolic in concolic exec,
        # with concrete controller states

        var_type['state_arr'] = 'int_arr'
        var_type['x_arr'] = 'int_arr'
        var_type['input_arr'] = 'int_arr'
        concolic_engine = CE.concolic_engine_factory(
            var_type,
            num_dims,
            controller_object_str)

        # sampler = sample.Concolic(concolic_engine)

        sampler = sample.IntervalConcolic(concolic_engine)
    elif METHOD == 'concrete':
        sampler = sample.IntervalSampler()
        controller_abstraction_type = 'concrete'
        controller_sym_path_obj = None

    elif METHOD == 'concrete_no_controller':
        sampler = sample.IntervalSampler()
        controller_abstraction_type = 'concrete_no_controller'
        controller_sym_path_obj = None

        # TODO: manual contruction of paths!!!!
        # use OS independant APIs from fileOps
    elif METHOD == 'symbolic':
        sampler = None
        if globalopts.opts.symbolic_analyzer == 'klee':
            controller_abstraction_type = 'symbolic_klee'
            if globalopts.opts.cntrl_rep == 'smt2':
                controller_path_dir_path += '/klee/'
            else:
                raise err.Fatal('KLEE supports only smt2 files!')
        elif globalopts.opts.symbolic_analyzer == 'pathcrawler':
            controller_abstraction_type = 'symbolic_pathcrawler'
            if globalopts.opts.cntrl_rep == 'smt2':
                controller_path_dir_path += '/pathcrawler/'
            elif globalopts.opts.cntrl_rep == 'trace':
                controller_path_dir_path += '/controller'
            else:
                raise err.Fatal('argparse should have caught this!')

            # TAG:PCH_IND
            # Parse PC Trace
            import CSymLoader as CSL
            controller_sym_path_obj = CSL.load_sym_obj((globalopts.opts.cntrl_rep, globalopts.opts.trace_struct), controller_path_dir_path)
        else:
            raise err.Fatal('unknown symbolic analyzer requested:{}'.format(globalopts.opts.symbolic_analyzer))

    else:
        raise NotImplementedError

    # TODO: parameters like controller_sym_path_obj are absraction dependant
    # and should not be passed directly to abstraction_factory. Instead a
    # flexible structure should be created which can be filled by the
    # CAsymbolic abstraction module and supplied as a substructure. I guess the
    # idea is that an abstraction module should be 'pluggable'.
    current_abs = abstraction.abstraction_factory(
        plant_config_dict,
        prop.ROI,
        T,
        num_dims,
        sys.min_smt_sample_dist,
        plant_abstraction_type,
        globalopts.opts.graph_lib,
        )
    return current_abs, sampler


#def falsify(sut, init_cons, final_cons, plant_sim, controller_sim, init_cons_list, ci, pi, current_abs, sampler):
def falsify(sys, prop, current_abs, sampler):
    # sys
    controller_sim = sys.controller_sim
    plant_sim = sys.plant_sim

    # prop
    init_cons_list = prop.init_cons_list
    init_cons = prop.init_cons
    final_cons = prop.final_cons
    ci = prop.ci
    pi = prop.pi
    initial_discrete_state = prop.initial_discrete_state
    initial_controller_state = prop.initial_controller_state
    MAX_ITER = prop.MAX_ITER

    #TODO: hack to make random_test sample ci_cells when doing
    # ss-concrete. It is false if ss-symex (and anything else) is
    # asked for, because then ci_seq consists if concrete values. Can
    # also be activated for symex as an option, but to be done later.
    sample_ci = globalopts.opts.METHOD == 'concrete' or globalopts.opts.METHOD == 'concrete_no_controller'

    # options
    #plot = globalopts.opts.plot

    initial_discrete_state = tuple(initial_discrete_state)
    initial_controller_state = np.array(initial_controller_state)

    # make a copy of the original initial plant constraints

    original_plant_cons_list = init_cons_list

    pi_ref = wmanager.WMap(pi, sys.pi_grid_eps)
    ci_ref = wmanager.WMap(ci, sys.ci_grid_eps) if sample_ci else None

#            f1 = plt.figure()
##
##            plt.grid(True)
##
##            ax = f1.gca()
##           eps = current_abs.plant_abs.eps
##            #ax.set_xticks(np.arange(0, 2, eps[0]))
##            #ax.set_yticks(np.arange(0, 20, eps[1]))
##
##            f1.suptitle('abstraction')

    args = (current_abs,
            init_cons_list,
            final_cons,
            initial_discrete_state,
            initial_controller_state,
            plant_sim,
            controller_sim,
            ci,
            pi,
            sampler,
            #plot,
            init_cons,
            original_plant_cons_list,
            MAX_ITER,
            sample_ci,
            pi_ref,
            ci_ref,
            )

    if globalopts.opts.refine == 'init':
        refine_init(*args)
    elif globalopts.opts.refine == 'trace':
        refine_trace(*args)
    elif (globalopts.opts.refine == 'model-dft'
         or globalopts.opts.refine == 'model-dmt'
         or globalopts.opts.refine == 'model-dct'):
        sys_sim = simsys.get_system_simulator(sys)
        falsify_using_model(*args, sys_sim=sys_sim, sys=sys, prop=prop)
    else:
        raise err.Fatal('internal')


def refine_trace(
        current_abs,
        init_cons_list,
        final_cons,
        initial_discrete_state,
        initial_controller_state,
        plant_sim,
        controller_sim,
        ci,
        pi,
        sampler,
        #plot,
        init_cons,
        original_plant_cons_list):

    (initial_state_set, final_state_set, is_final) = \
        SS.init(current_abs, init_cons_list, final_cons,
                initial_discrete_state, initial_controller_state)

    system_params = SystemParams(
        initial_state_set,
        final_state_set,
        is_final,
        plant_sim,
        controller_sim,
        ci,
        pi,
        sampler,
        init_cons,
        final_cons,
        )

    SS.discover(current_abs, system_params)

    #POFF
#     if plot:
#         plt.autoscale()
#         plt.show()

    while True:

        if not system_params.final_state_set:
            print('did not find any abstract counter example!', file=SYS.stderr)
            return True
        else:
            print('analyzing graph...')
        (promising_initial_states, ci_seq_list) = \
            current_abs.get_initial_states_from_error_paths(initial_state_set, final_state_set)

        # ##!!##logger.debug('promising initial states: {}'.format(promising_initial_states))

        print('begin random testing!')
        #POFF
#         if plot:
#             f2 = plt.figure()
#             f2.suptitle('random testing')


        # TODO: ugly...should it be another function?
        # Read the TODO above the function definition for more details
        valid_promising_initial_state_list = SS.filter_invalid_abs_states(
                promising_initial_states,
                current_abs,
                init_cons)
        if valid_promising_initial_state_list == []:
            print('no valid sample found during random testing. STOP', file=SYS.stderr)
            return True

        done = RT.random_test(
            current_abs,
            system_params,
            valid_promising_initial_state_list,
            ci_seq_list,
            initial_discrete_state,
            initial_controller_state,
            )
        #POFF
#         if plot:
#             plt.show()
        if done:
            print('Concretized', file=SYS.stderr)
            return True
        current_abs = SS.refine_trace_based(
                current_abs,
                current_abs.compute_error_paths(initial_state_set, final_state_set),
                system_params)
        #init_cons_list = [current_abs.plant_abs.get_ival_constraints(i) for i in valid_promising_initial_state_list]


def ERROR_PATHS(A):
    from core import state
    l = [[0.01961446,    3.47045684,    9.98620188,    4.98716176],
         [49.95062386,   15.90625493,    9.98620188,   -0.01283824],
         [99.88163325,    3.3420208,    9.98620188,   -5.01283824],
         [149.81264265,    8.93715771,    9.98620188,   -0.14237774],
         [199.74365204,    2.47800605,    9.98620188,    2.26046069],
         [249.67466144,    1.28030005,    9.98620188,   -2.73953931],
         [299.60567083,    0.43858743,    9.98620188,   -2.18741369],
         [330.06358656,    1.01642601,    9.98620188,   -1.07331661],
         [330.06358656,    1.01642601,    9.98620188,   -1.07331661]]

    x = np.array(l).reshape(9, 4)
    s = [state.State(0, xi, 0, 0, [], [], [], []) for xi in x]
    abs_trace = [A.get_abs_state_from_concrete_state(si) for si in s]
    return [abs_trace]


# returns a True when its done
def falsify_using_model(
        current_abs,
        init_cons_list,
        final_cons,
        initial_discrete_state,
        initial_controller_state,
        plant_sim,
        controller_sim,
        ci,
        pi,
        sampler,
        #plot,
        init_cons,
        original_plant_cons_list,
        MAX_ITER,
        sample_ci,
        pi_ref,
        ci_ref,
        sys_sim,
        sys,
        prop):

    from core import modelrefine as MR
    # TODO: temp function ss.init()

    (initial_state_set, is_final) = \
        SS.init(current_abs, init_cons_list, final_cons,
                initial_discrete_state, initial_controller_state)

    system_params = SystemParams(
        #initial_state_set,
        #final_state_set,
        is_final,
        plant_sim,
        controller_sim,
        ci,
        pi,
        sampler,
        init_cons,
        final_cons,
        pi_ref,
        ci_ref
        )

    # Modifies system_params
    final_state_set = SS.discover(current_abs, system_params, initial_state_set)

    # Flush the first plot

    plt = globalopts.opts.plotting
    if settings.paper_plot:
        plt.acquire_global_fig()
        plt.plot_abs_states(current_abs, prop, (i for i in current_abs.G.nodes_iter()))
        #plt.plot_rect(prop.init_cons.rect(), 'g')
        #plt.plot_rect(prop.final_cons.rect(), 'r')
        plt.set_range((-2.5, 2.5), (-8, 8))
    plt.show(block=True)

# switch of abstraction dump: memory leak?
#     print('dumping abstraction')
#     fp.write_data(globalopts.opts.construct_path('{}_graph.dump'.format(sys.sys_name)),
#                   dill.dumps(current_abs))

    if settings.MEMLEAK_TEST:
        import objgraph
        from core.PACell import PlantAbstractState as PS
#         objgraph.show_backrefs(
#                 PS.instance_store[0], max_depth=10, too_many=10,
#                 highlight=None, filename='objgraph2.png',
#                 extra_info=None, refcounts=False, shortnames=False)
#                 #extra_ignore=[id(locals())])
        objgraph.show_growth(shortnames=False)
        U.pause()

    if not final_state_set:
        print('did not find any abstract counter example!', file=SYS.stderr)
        return False

    #ep = ERROR_PATHS(current_abs)[0]
    #assert(ep in error_paths)

    if globalopts.opts.max_paths > 0:
        print('analyzing graph...')
        # creates a new pi_ref, ci_ref
        (error_paths,
         pi_seq_list) = current_abs.get_error_paths_not_normalized(
                 initial_state_set,
                 final_state_set,
                 pi_ref,
                 pi,
                 globalopts.opts.max_paths)
        errors = (error_paths, pi_seq_list)
    else:
        errors = current_abs.G.subgraph_source2target(initial_state_set, final_state_set)

    #error_paths = ERROR_PATHS(current_abs)
#     if ep in error_paths:
#         U.pause('yay, found!')
#     else:
#         U.pause('not found')

    print('Refining...')
    if globalopts.opts.refine == 'model-dft':
        MR.refine_dft_model_based(current_abs,
                                  #error_paths,
                                  #error_graph,
                                  errors,
                                  initial_state_set,
                                  final_state_set,
                                  system_params,
                                  sys_sim,
                                  sys,
                                  prop)
    elif globalopts.opts.refine == 'model-dmt':
        MR.refine_dmt_model_based(current_abs,
                                  error_paths,
                                  pi_seq_list,
                                  system_params,
                                  sys_sim,
                                  sys,
                                  prop)
    elif globalopts.opts.refine == 'model-dct':
        raise NotImplementedError
    else:
        assert(False)

    return


# returns a True when its done
def refine_init(
        current_abs,
        init_cons_list,
        final_cons,
        initial_discrete_state,
        initial_controller_state,
        plant_sim,
        controller_sim,
        ci,
        pi,
        sampler,
        #plot,
        init_cons,
        original_plant_cons_list,
        MAX_ITER,
        sample_ci,
        pi_ref,
        ci_ref,
        ):

    i = 1
    while i <= MAX_ITER:
        print('iteration:', i)
        # TODO: temp function ss.init()

        (initial_state_set, is_final) = \
            SS.init(current_abs, init_cons_list, final_cons,
                    initial_discrete_state, initial_controller_state)

        logger.debug('initial state set:\n{}'.format('\n'.join([str(current_abs.plant_abs.get_ival_cons_abs_state(s0.ps)) for s0 in initial_state_set])))

        system_params = SystemParams(
            #initial_state_set,
            #final_state_set,
            is_final,
            plant_sim,
            controller_sim,
            ci,
            pi,
            sampler,
            init_cons,
            final_cons,
            pi_ref,
            ci_ref
            )
        final_state_set = SS.discover(current_abs, system_params, initial_state_set)

        globalopts.opts.plotting.show()

        if not final_state_set:
            print('did not find any abstract counter example!', file=SYS.stderr)
            return False

        print('analyzing graph...')
        pi_ref.cleanup()
        if ci_ref is not None:
            ci_ref.cleanup()
        # creates a new pi_ref, ci_ref
        (promising_initial_states,
            ci_seq_list,
            pi_seq_list) = current_abs.get_initial_states_from_error_paths(initial_state_set,
                                                                           final_state_set,
                                                                           pi_ref,
                                                                           ci_ref,
                                                                           pi,
                                                                           ci,
                                                                           globalopts.opts.max_paths)

        # ##!!##logger.debug('promising initial states: {}'.format(promising_initial_states))

        print('begin random testing!')
        #POFF
#         if plot:
#             f2 = plt.figure()
#             f2.suptitle('random testing')

        print(len(promising_initial_states), len(ci_seq_list), len(pi_seq_list))
        #U.pause()
        # TODO: ugly...should it be another function?
        # Read the TODO above the function definition for more details
        (valid_promising_initial_state_list,
            pi_seq_list, ci_seq_list) = SS.filter_invalid_abs_states(
                        promising_initial_states,
                        pi_seq_list,
                        ci_seq_list,
                        current_abs,
                        init_cons)
        print(len(valid_promising_initial_state_list), len(ci_seq_list), len(pi_seq_list))
        #U.pause()
        if valid_promising_initial_state_list == []:
            print('no valid sample found during random testing. STOP', file=SYS.stderr)
            return False
        res = RT.random_test(
            current_abs,
            system_params,
            valid_promising_initial_state_list,
            ci_seq_list,
            pi_seq_list,
            initial_discrete_state,
            initial_controller_state,
            sample_ci,
            return_vio_only=True
            )

        globalopts.opts.plotting.show()

        if res:
            op_fname = globalopts.opts.construct_path('violations.log')
            print('Concretized', file=SYS.stderr)
            #fp.append_data(globalopts.opts.op_fname,
            fp.append_data(op_fname,
                           '{0} Concrete Traces({2}) for: {1} {0}\n'.
                           format('='*20, globalopts.opts.sys_path, len(res)))
            traces_string = '\n'.join(str(trace) for trace in res)
            fp.append_data(op_fname, traces_string)
            return True

        (current_abs, init_cons_list) = SS.refine_init_based(
                current_abs,
                promising_initial_states, # should it not be valid_promising_initial_state_list?
                original_plant_cons_list)#, pi_ref, ci_ref)
        pi_ref.refine()
        if ci_ref is not None:
            ci_ref.refine()
        i += 1
    print('Failed: MAX iterations {} exceeded'.format(MAX_ITER), file=SYS.stderr)
    # raise an exception maybe?


def dump_trace(trace_list):
    print('dumping trace[0]')
    trace_list[0].dump_matlab()

def simulate(sys, prop):
    plt = globalopts.opts.plotting
    #plot = globalopts.opts.plot
    if not isinstance(
            globalopts.opts.property_checker,
            properties.PropertyChecker):
        raise err.Fatal('property checker must be enabled when '
                        'random testing!')
    trace_list = RT.simulate(sys, prop)
    #print(len(list(trace_list)))
#         for trace in trace_list:
#             print(trace)
    #for trace in trace_list:
    #    fp.append_data('trace_log', str(trace))

    if globalopts.opts.dump_trace:
        dump_trace(trace_list)

    if settings.paper_plot:
        # because the plot is craeted inside the simulator, get
        # the global handle
        plt.acquire_global_fig()
        plt.plot_rect(prop.init_cons.rect(), 'g')
        plt.plot_rect(prop.final_cons.rect(), 'r')
        plt.set_range((-2.5, 2.5), (-8, 8))

    plt.plot_trace_list(trace_list)

#         if settings.paper_plot:
#             plt.plot_rect(prop.init_cons.rect(), 'g')
#             plt.plot_rect(prop.final_cons.rect(), 'r')
#             plt.set_range((-2, 2), (-7, 7))

    plt.show()

def run_secam(sys, prop):
    MODE = globalopts.opts.MODE

    start_time = time.time()

    if MODE == 'simulate':
        simulate(sys, prop)
    elif MODE == 'falsify':
        # ignore time taken to create_abstraction: mainly to ignore parsing
        # time
        current_abs, sampler = create_abstraction(sys, prop)
        start_time = time.time()
        falsify(sys, prop, current_abs, sampler)
    else:
        raise err.Fatal('bad MODE supplied: {}'.format(MODE))

    stop_time = time.time()
    print('*'*20)
    print('time spent(s) = {}'.format(stop_time - start_time), file=SYS.stderr)
    return


def main():
    #U.pause()

    #c = load_config()
#     hp = hpy()
#     hp.setrelheap()
#     h_before = hp.heap()
    # critical section here

    #tr = tracker.SummaryTracker()
    #ib = refbrowser.InteractiveBrowser(root)

    logger.info('execution begins')
    LIST_OF_SYEMX_ENGINES = ['klee', 'pathcrawler']
    LIST_OF_CONTROLLER_REPRS = ['smt2', 'trace']
    LIST_OF_TRACE_STRUCTS = ['list', 'tree']
    LIST_OF_REFINEMENTS = ['init', 'trace', 'model-dft', 'model-dmt', 'model-dct']
    LIST_OF_GRAPH_LIBS = ['nxlm', 'nx', 'gt', 'g']
    LIST_OF_PLOT_LIBS = ['mp', 'pg']
    LIST_OF_BMC = ['sal', 's3camsmt', 'pwa', 'pretty-printer']
    LIST_OF_LP = ['scipy', 'glpk', 'gurobi']
    LIST_OF_CLUSTERING = ['cell', 'box', 'hull']
    LIST_OF_MODELS = ['affine', 'poly']

    DEF_BMC_PREC = 4
    DEF_BMC = 'sal'
    DEF_LP = 'glpk'
    DEF_MAX_PATHS = 100
    DEF_GRAPH_LIB = 'nx'
    DEF_VIO_LOG = 'vio.log'
    DEF_CLUSTERING = 'cell'
    DEF_MODEL = 'affine'

    parser = argparse.ArgumentParser(
            description='S3CAM',
            usage='%(prog)s <filename>',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename', default=None, metavar='file_path.tst')

    #parser.add_argument('--run-benchmarks', action="store_true", default=False,
    #                    help='run pacakged benchmarks')

    parser.add_argument('-s', '--simulate', type=int, metavar='num-sims',
                        help='simulate')
    parser.add_argument('-c', '--ss-concrete', action="store_true",
                        help='scatter & simulate')

    parser.add_argument('-cn', '--ss-concrete-no-controller', action="store_true",
                        help='scatter & simulate')

    parser.add_argument('--ss-concolic', action="store_true",
                        help='scatter & simulate with concolic execution using KLEE')
    parser.add_argument('-x', '--ss-symex', type=str, choices=LIST_OF_SYEMX_ENGINES,
                        help='SS + SymEx with static paths')

    parser.add_argument('-r', '--cntrl-rep', type=str, choices=LIST_OF_CONTROLLER_REPRS,
                        help='Controller Representation')

#     parser.add_argument('-p', '--plot', action='store_true',
#                         help='enable plotting')

    parser.add_argument('--dump', action='store_true',
                        help='dump trace in mat file')

    parser.add_argument('--seed', type=int, metavar='integer_seed_value',
                        help='seed for the random generator')

    # TAG:MSH
    parser.add_argument('--meng', type=str, metavar='engine_name',
                        help='Shared Matlab Engine name')

    parser.add_argument('-t', '--trace-struct', type=str, default='tree',
                        choices=LIST_OF_TRACE_STRUCTS, help='structure for cntrl-rep')

    parser.add_argument('--refine', type=str, default=None,
                        choices=LIST_OF_REFINEMENTS, help='Refinement method')

    parser.add_argument('--incl-error', action='store_true',
                        help='Include errors in model for bmc')

#     parser.add_argument('-o', '--output', type=str,
#                         default=DEF_VIO_LOG,
#                         help='violation log')

    parser.add_argument('-g', '--graph-lib', type=str,
                        default=DEF_GRAPH_LIB,
                        choices=LIST_OF_GRAPH_LIBS,
                        help='graph library')

    parser.add_argument('-p', '--plot', type=str, nargs='?',
                        default=None, const='mp',
                        choices=LIST_OF_PLOT_LIBS, help='plot library')

    parser.add_argument('--plots', type=plotting.plot_opts_parse, default='',
                        nargs='+', help='plots x vs y: t-x1 x0-x1')

    parser.add_argument('--max-paths', type=int, default=DEF_MAX_PATHS,
                        help='max number of paths to use for refinement')

    # TODO: remove this before thigns get too fancy.
    # Add a simple package interface so benchmarks/tests can be
    # written as python files and can call s3cam using an API
    parser.add_argument('--pvt-init-data', type=str, default=None,
                        help='will set pvt_init_data in the supplied .tst file')

    # TODO: This error can be computed against the cell sizes?
    parser.add_argument('--max-model-error', type=float, default=float('inf'),
                        help='split cells till model error (over a single step) <= max-error')

    parser.add_argument('--plot-opts', type=str, default=(),
                        nargs='+', help='additional lib specific plotting opts')

    parser.add_argument('--prop-check', action='store_true',
                        help='Check violations by analyze the entire '
                        'trace, instead of relying only on x(t_end).')

    parser.add_argument('--bmc-engine', type=str,
                        choices=LIST_OF_BMC,
                        default=DEF_BMC,
                        help='Choose the bmc engine')

    parser.add_argument('--lp-engine', type=str,
                        choices=LIST_OF_LP,
                        default=DEF_LP,
                        help='Choose the LP engine')

    # TODO: fix this hack
    parser.add_argument('--enable-regression-plots', action='store_true',
                        help='Disables showing/rendering of regression plots')

    # TODO: fix this hack
    parser.add_argument('--debug', action='store_true',
                        help='Enables debug flag')

    parser.add_argument('--bmc-prec', type=int, default=DEF_BMC_PREC,
                        help='number of decimals to retain when\
                        translating the pwa system to the transitions\
                        system for BMC.')

    parser.add_argument('--par', action='store_true',
                        help='parallel simulation')

    parser.add_argument('-o', '--output', type=str,
                        default=None,
                        help='output directory')

    parser.add_argument('--clustering', type=str,
                        default=DEF_CLUSTERING,
                        choices=LIST_OF_CLUSTERING,
                        help='clustering method')

    parser.add_argument('--model-type', type=str,
                        default=DEF_MODEL,
                        choices=LIST_OF_MODELS,
                        help='type of model')

#    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    #print(args)

    if args.filename is None:
        print('No file to test. Please use --help')
        exit()
    else:
        filepath = args.filename

    if args.seed is not None:
        np.random.seed(args.seed)

    # TODO put plot hack as a switch in settings
    settings.debug_plot = args.enable_regression_plots
    settings.debug = args.debug
    #TODO:
    # Another hack to make sure that we know plot is on at some random
    # places which is not usoing plotlib but overriding plotting with
    # matplotlib.
    settings.plot = args.plot is not None

    opts = globalopts.Options()

    if args.simulate is not None:
        opts.MODE = 'simulate'
        opts.num_sim_samples = args.simulate
    elif args.ss_concrete:
        opts.MODE = 'falsify'
        opts.METHOD = 'concrete'
    elif args.ss_concrete_no_controller:
        opts.MODE = 'falsify'
        opts.METHOD = 'concrete_no_controller'
    elif args.ss_concolic:
        opts.MODE = 'falsify'
        opts.METHOD = 'concolic'
        print('removed concolic (KLEE)')
        exit(0)
    elif args.ss_symex is not None:
        opts.MODE = 'falsify'
        opts.METHOD = 'symbolic'
        opts.symbolic_analyzer = args.ss_symex
        #if opts.symbolic_analyzer not in LIST_OF_SYEMX_ENGINES:
        #    raise err.Fatal('unknown symbolic analyses engine requested.')
        if args.cntrl_rep is None:
            raise err.Fatal('controller representation must be provided')
        else:
            opts.cntrl_rep = args.cntrl_rep
            opts.trace_struct = args.trace_struct
    else:
        raise err.Fatal('no options passed. Check usage.')

    if args.refine is None and opts.MODE == 'falsify':
        print('No refinement strategy is specified!  Please use --help')
        exit()

    # TODO: fixed arguement, remove it from opts
    opts.vio_fname = DEF_VIO_LOG

    #opts.plot = args.plot
    opts.dump_trace = args.dump
    opts.refine = args.refine

    #opts.op_fname = args.output
    #opts.op_path = args.output

    opts.sys_path = filepath
    opts.graph_lib = args.graph_lib
    opts.max_paths = args.max_paths
    opts.max_model_error = args.max_model_error
    opts.plotting = plotting.factory(args.plot, args.plots, *args.plot_opts)
    #opts.plots = args.plots
    opts.model_err = args.incl_error
    opts.bmc_engine = args.bmc_engine
    opts.lp_engine = args.lp_engine
    # Default bmc prec
    opts.bmc_prec = args.bmc_prec
    opts.par = args.par
    opts.clustering = args.clustering
    opts.model_type = args.model_type

    sys, prop = loadsystem.parse(filepath, args.pvt_init_data)
    if args.prop_check:
        opts.property_checker = properties.PropertyChecker(prop.final_cons)
    else:
        opts.property_checker = properties.PropertyCheckerNeverDetects()

    opts.construct_path = setup_dir(sys, args.output)
    print('\nOutput dir: {}\n'.format(opts.construct_path('')))

    # What a hack!
    opts.regression_sim_samples = sys.plant_config_dict['num_samples']

    # TAG:MSH
    matlab_engine = args.meng
    sys.init_sims(opts.plotting, opts.property_checker, psim_args=matlab_engine)

    sanity_check_input(sys, prop)

    globalopts.opts = opts

    run_secam(sys, prop)

    #embed()
    # ##!!##logger.debug('execution ends')
    #tr.print_diff()
    #ib.main()
#     h_after = hp.heap()
#     leftover = h_after - h_before
#     print(leftover)
#     embed()
    #import pdb; pdb.set_trace()

#######################################################################
# GLobals: is there a better way to do this?
TIME_STR = fp.time_string()
logger = setup_logger()

if __name__ == '__main__':
    main()
