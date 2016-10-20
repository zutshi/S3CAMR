#!/usr/bin/python
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

from . import abstraction
from . import sample
from . import simulatesystem as simsys
from . import scattersim as SS
from . import loadsystem
#import traces
from . import wmanager
from .plot import plotting
from . import random_testing as RT
from . import properties

import err
import fileops as fp
#import utils as U
from utils import print

import settings

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

FORMAT = '[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s'
FORMAT2 = '%(levelname) -10s %(asctime)s %(module)s:\
           %(lineno)s %(funcName)s() %(message)s'

logging.basicConfig(filename='log.secam', filemode='w', format=FORMAT2,
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Remove this re-structuring!! Use existing structures instead.
class SystemParams:

    def __init__(
            self,
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
            pi_ref,
            ci_ref
            ):

        self.initial_state_set = initial_state_set
        self.final_state_set = final_state_set
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


def sanity_check_input(sys, prop, opts):
    return


def create_abstraction(sys, prop, opts):
    num_dims = sys.num_dims
    plant_config_dict = sys.plant_config_dict
    controller_path_dir_path = sys.controller_path_dir_path
    controller_object_str = sys.controller_object_str

    T = prop.T

    METHOD = opts.METHOD

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
        if opts.symbolic_analyzer == 'klee':
            controller_abstraction_type = 'symbolic_klee'
            if opts.cntrl_rep == 'smt2':
                controller_path_dir_path += '/klee/'
            else:
                raise err.Fatal('KLEE supports only smt2 files!')
        elif opts.symbolic_analyzer == 'pathcrawler':
            controller_abstraction_type = 'symbolic_pathcrawler'
            if opts.cntrl_rep == 'smt2':
                controller_path_dir_path += '/pathcrawler/'
            elif opts.cntrl_rep == 'trace':
                controller_path_dir_path += '/controller'
            else:
                raise err.Fatal('argparse should have caught this!')

            # TAG:PCH_IND
            # Parse PC Trace
            import CSymLoader as CSL
            controller_sym_path_obj = CSL.load_sym_obj((opts.cntrl_rep, opts.trace_struct), controller_path_dir_path)
        else:
            raise err.Fatal('unknown symbolic analyzer requested:{}'.format(opts.symbolic_analyzer))

    else:
        raise NotImplementedError

    # TODO: parameters like controller_sym_path_obj are absraction dependant
    # and should not be passed directly to abstraction_factory. Instead a
    # flexible structure should be created which can be filled by the
    # CAsymbolic abstraction module and supplied as a substructure. I guess the
    # idea is that an abstraction module should be 'pluggable'.
    current_abs = abstraction.abstraction_factory(
        plant_config_dict,
        T,
        num_dims,
        controller_sym_path_obj,
        sys.min_smt_sample_dist,
        plant_abstraction_type,
        controller_abstraction_type,
        opts.graph_lib,
        )
    return current_abs, sampler


#def falsify(sut, init_cons, final_cons, plant_sim, controller_sim, init_cons_list, ci, pi, current_abs, sampler):
def falsify(sys, prop, opts, current_abs, sampler):
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
    sample_ci = opts.METHOD == 'concrete' or opts.METHOD == 'concrete_no_controller'

    # options
    #plot = opts.plot

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
            opts)

    if opts.refine == 'init':
        refine_init(*args)
    elif opts.refine == 'trace':
        refine_trace(*args)
    elif (opts.refine == 'model-dft'
         or opts.refine == 'model-dmt'
         or opts.refine == 'model-dct'):
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
        opts,
        sys_sim,
        sys,
        prop):

    from . import modelrefine as MR
    # TODO: temp function ss.init()

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
        pi_ref,
        ci_ref
        )

    SS.discover(current_abs, system_params)

    opts.plotting.show()
    print('dumping abstraction')
    fp.write_data('{}_graph.dump'.format(sys.sys_name), dill.dumps(current_abs))

    if not system_params.final_state_set:
        print('did not find any abstract counter example!', file=SYS.stderr)
        return False

    print('analyzing graph...')
    # creates a new pi_ref, ci_ref
    (error_paths,
     ci_seq_list,
     pi_seq_list) = current_abs.get_error_paths_not_normalized(initial_state_set,
                                                final_state_set,
                                                pi_ref,
                                                ci_ref,
                                                pi,
                                                ci,
                                                opts.max_paths)
    print('Refining...')
    if opts.refine == 'model-dft':
        MR.refine_dft_model_based(current_abs,
                                  error_paths,
                                  pi_seq_list,
                                  system_params,
                                  sys_sim,
                                  opts,
                                  sys,
                                  prop)
    elif opts.refine == 'model-dmt':
        MR.refine_dmt_model_based(current_abs,
                                  error_paths,
                                  pi_seq_list,
                                  system_params,
                                  sys_sim,
                                  opts,
                                  sys,
                                  prop)
    elif opts.refine == 'model-dct':
        raise NotImplementedError
    else:
        assert(False)

    exit()


    print('begin random testing!')

    print(len(promising_initial_states), len(ci_seq_list), len(pi_seq_list))

    (valid_promising_initial_state_list,
     pi_seq_list, ci_seq_list) = SS.filter_invalid_abs_states(promising_initial_states,
                                                              pi_seq_list,
                                                              ci_seq_list,
                                                              current_abs,
                                                              init_cons)

    print(len(valid_promising_initial_state_list), len(ci_seq_list), len(pi_seq_list))

    if valid_promising_initial_state_list == []:
        print('no valid sample found for random testing. STOP', file=SYS.stderr)
        return False

    res = RT.random_test(
        current_abs,
        system_params,
        valid_promising_initial_state_list,
        ci_seq_list,
        pi_seq_list,
        initial_discrete_state,
        initial_controller_state,
        sample_ci
        )

    if res:
        print('Concretized', file=SYS.stderr)
        fp.append_data(opts.op_fname,
                       '{0} Concrete Traces({2}) for: {1} {0}\n'.
                       format('='*20, opts.sys_path, len(res)))
        fp.append_data(opts.op_fname, '{}\n'.format(res))
        return True

    #print('Failed: MAX iterations {} exceeded'.format(MAX_ITER), file=SYS.stderr)


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
        opts):

    i = 1
    while i <= MAX_ITER:
        print('iteration:', i)
        # TODO: temp function ss.init()

        (initial_state_set, final_state_set, is_final) = \
            SS.init(current_abs, init_cons_list, final_cons,
                    initial_discrete_state, initial_controller_state)

        logger.debug('initial state set:\n{}'.format('\n'.join([str(current_abs.plant_abs.get_ival_cons_abs_state(s0.ps)) for s0 in initial_state_set])))

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
            pi_ref,
            ci_ref
            )
        SS.discover(current_abs, system_params)

        opts.plotting.show()

        if not system_params.final_state_set:
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
                                                                           opts.max_paths)

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
            sample_ci
            )

        opts.plotting.show()

        if res:
            print('Concretized', file=SYS.stderr)
            fp.append_data(opts.op_fname,
                           '{0} Concrete Traces({2}) for: {1} {0}\n'.
                           format('='*20, opts.sys_path, len(res)))
            fp.append_data(opts.op_fname, '{}\n'.format(res))
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


def run_secam(sys, prop, opts):
    MODE = opts.MODE
    #plot = opts.plot

    if MODE == 'simulate':
        if not isinstance(
                opts.property_checker,
                properties.PropertyChecker):
            raise err.Fatal('property checker must be enabled when '
                            'random testing!')
        start_time = time.time()
        trace_list = RT.simulate(sys, prop, opts)
        print(len(list(trace_list)))
#         for trace in trace_list:
#             print(trace)
        if opts.dump_trace:
            dump_trace(trace_list)
        opts.plotting.plot_trace_list(trace_list, opts.plots)
        opts.plotting.show()
    elif MODE == 'falsify':
        # ignore time taken to create_abstraction: mainly to ignore parsing
        # time
        current_abs, sampler = create_abstraction(sys, prop, opts)
        start_time = time.time()
        falsify(sys, prop, opts, current_abs, sampler)
    else:
        raise err.Fatal('bad MODE supplied: {}'.format(MODE))

    stop_time = time.time()
    print('*'*20)
    print('time spent(s) = {}'.format(stop_time - start_time), file=SYS.stderr)
    return


def main():
    logger.info('execution begins')
    LIST_OF_SYEMX_ENGINES = ['klee', 'pathcrawler']
    LIST_OF_CONTROLLER_REPRS = ['smt2', 'trace']
    LIST_OF_TRACE_STRUCTS = ['list', 'tree']
    LIST_OF_REFINEMENTS = ['init', 'trace', 'model-dft', 'model-dmt', 'model-dct']
    LIST_OF_GRAPH_LIBS = ['nx', 'gt', 'g']
    LIST_OF_PLOT_LIBS = ['mp', 'pg']
    LIST_OF_BMC = ['sal', 's3camsmt']

    DEF_BMC_PREC = 4
    DEF_BMC = 'sal'
    DEF_MAX_PATHS = 100
    DEF_GRAPH_LIB = 'nx'
    DEF_VIO_LOG = 'vio.log'

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

    parser.add_argument('-o', '--output', type=str,
                        default=DEF_VIO_LOG,
                        help='violation log')

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

    # TODO:
    # dynamicall generate an opt class to mimic the same defined in
    # loadsystem.py
    # switch this to a reg class later.
    Options = type(str('Options'), (), {})
    opts = Options()
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

    #opts.plot = args.plot
    opts.dump_trace = args.dump
    opts.refine = args.refine
    opts.op_fname = args.output
    opts.sys_path = filepath
    opts.graph_lib = args.graph_lib
    opts.max_paths = args.max_paths
    opts.max_model_error = args.max_model_error
    opts.plotting = plotting.factory(args.plot, *args.plot_opts)
    opts.plots = args.plots
    opts.model_err = args.incl_error
    opts.bmc_engine = args.bmc_engine
    # Default bmc prec
    opts.bmc_prec = args.bmc_prec
    opts.par = args.par

    sys, prop = loadsystem.parse(filepath, args.pvt_init_data)
    if args.prop_check:
        opts.property_checker = properties.PropertyChecker(prop.final_cons)
    else:
        opts.property_checker = properties.PropertyCheckerNeverDetects()

    # TAG:MSH
    matlab_engine = args.meng
    sys.init_sims(opts.plotting, opts.property_checker, psim_args=matlab_engine)

    sanity_check_input(sys, prop, opts)
    run_secam(sys, prop, opts)
    # ##!!##logger.debug('execution ends')

if __name__ == '__main__':
    main()
