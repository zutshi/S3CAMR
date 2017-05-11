from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging

import functools as ft
#import cPickle as cP

import numpy as np
import tqdm
from blessed import Terminal

import err
import utils as U
#import fileops as fops
from utils import print
from streampickle import PickleStreamReader, PickleStreamWriter

from . import sample
from . import traces
from . import state as st
from . import simulatesystem as simsys
from .properties import PropertyChecker

import multiprocessing as mp

import globalopts

logger = logging.getLogger(__name__)
term = Terminal()

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)


# TODO: make a module of its own once we add more general property using
# monitors...
def check_prop_violation(prop, trace):
    """check_prop_violation

    Parameters
    ----------
    trace :
    prop :

    Returns
    -------

    Notes
    ------
    """
    # check using random sims
    idx = prop.final_cons.sat(trace.x_array+0.0001)
    sat_x, sat_t = trace.x_array[idx], trace.t_array[idx]
    if sat_x.size != 0:
        print('x0={} -> x={}, t={}'.format(
            trace.x_array[0, :],
            sat_x[0, :],    # the first violating state
            sat_t[0],       # corresponding time instant
            ))
        return True
    else:
        return False


# def pickle_res(f, arg):
#     return cP.dumps(f(arg), protocol=cP.HIGHEST_PROTOCOL)


def simulate(sys, prop):
    if globalopts.opts.par:
        return simulate_par(sys, prop)()
    else:
        return simulate_single(sys, prop)


def f(sys, prop, fd, _):
    cs = sample.sample_init_UR(sys, prop, 1)
    trace = simsys.simulate_system(sys, prop.T, cs[0])
    #fd.write(trace)
    if check_prop_violation(prop, trace):
        #num_violations += 1
        pass


def mp_imap(self, sim, concrete_states):
    CHNK = 6250
    num_violations = 0
    #TODO: concrete_states should be an iterator/generator
    #f = ft.partial(pickle_res, sim)
    #writer.write(pool.imap_unordered(sim, concrete_states, chunksize=CHNK))
    #with fops.StreamWrite(self.fname, mode='wb') as sw:
#
    pool = mp.Pool(self.nworkers)
    with PickleStreamWriter(self.fname) as writer:
        for trace in pool.imap_unordered(sim, concrete_states, chunksize=CHNK):
            writer.write(trace)
            if check_prop_violation(self.prop, trace):
                num_violations += 1
    pool.close()
    pool.join()


import time
def worker(prop, sim, writer, concrete_states):
    print('burden: {}'.format(len(concrete_states)))
    ti = time.time()
    num_violations = 0
    with writer:
        for cs in concrete_states:
            trace = sim(cs)
            writer.write(trace)
            if check_prop_violation(prop, trace):
                num_violations += 1
    tf = time.time()
    print('time taken = {}'.format(tf-ti))
    return num_violations


def mp_custom(self, sim, concrete_states):
    jobs = []
    nworkers = self.nworkers
    work = len(concrete_states)
    work_load = int(work/nworkers)
    left_over_jobs = work % nworkers
    assert(work_load * nworkers + left_over_jobs == work)

    for i in range(nworkers):
        fname = self.fname + str(i)
        writer = PickleStreamWriter(fname)
        cs_slice = concrete_states[i*work_load:(i+1)*work_load]
        # if its the last worker, send off all the remaining jobs
        if i == nworkers - 1:
            cs_slice = concrete_states[i*work_load:]
        # else, slice evenly
        else:
            cs_slice = concrete_states[i*work_load:(i+1)*work_load]
        p = mp.Process(target=worker, args=(self.prop, sim, writer, cs_slice))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()


def numap(self, sim, concrete_states):
    num_violations = 0
    from numap import NuMap
    with PickleStreamWriter(self.fname) as writer:
        for trace in NuMap(func=sim, iterable=concrete_states,
                           ordered=False, stride=1, buffer=1000):
            writer.write(trace)
            if check_prop_violation(self.prop, trace):
                num_violations += 1


def mp_shared_mem(self, num_samples):
    CHNK = 6250
    pool = mp.Pool(self.nworkers)
    fd = open('delme.dump', 'w')
    ff = ft.partial(f, self.sys, self.prop, fd)
    for trace in pool.imap_unordered(ff, xrange(num_samples), chunksize=CHNK):
        pass
    fd.close()
    pool.close()
    pool.join()


def jb_parallel(self, sim, concrete_states):
    num_violations = 0
    import tempfile
    import os
    import joblib as jb
    from joblib import load, dump

    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib_test.mmap')
    if os.path.exists(filename):
        os.unlink(filename)
    dump(concrete_states, filename)
    large_memmap = load(filename, mmap_mode='r+')

    with PickleStreamWriter(self.fname) as writer:
        for trace in jb.Parallel(n_jobs=self.nworkers, verbose=0, batch_size='auto')(jb.delayed(sim, check_pickle=False)(i) for i in large_memmap):
            writer.write(trace)
            if check_prop_violation(self.prop, trace):
                num_violations += 1


class simulate_par(object):
    def __init__(self, sys, prop):

        self.par_option = 'mp_custom'

        self.sys = sys
        self.prop = prop
        fname = '{}.simdump'.format(sys.sys_name)
        self.fname = globalopts.opts.construct_path(fname)

        #self.nworkers = mp.cpu_count()
        self.nworkers = int(raw_input('Enter number of workers'))
        print('Num workers: {}'.format(self.nworkers))
        return

    def trace_gen(self):
        # This case is different as multiple dumps are produced
        if self.par_option == 'mp_custom':
            # combine all files
            readers = (PickleStreamReader(self.fname+str(i)) for i in range(self.nworkers))
            for reader in readers:
                for trace in reader.read():
                    yield trace

        else:
            reader = PickleStreamReader(self.fname)
            for trace in reader.read():
                yield trace

    def __call__(self):
        par_option = self.par_option

        num_samples = globalopts.opts.num_sim_samples
        concrete_states = sample.sample_init_UR(self.sys, self.prop, num_samples)
        sim = ft.partial(simsys.simulate_system, self.sys, self.prop.T)

        if par_option == 'mp':
            mp_imap(self, sim, concrete_states)
        elif par_option == 'mp_custom':
            mp_custom(self, sim, concrete_states)
        elif par_option == 'mp_shared_mem':
            raise NotImplementedError
            #mp_shared_mem(self, num_samples)
        elif par_option == 'joblib':
            jb_parallel()
        else:
            raise NotImplementedError
            #single threaded

        #print('number of violations: {}'.format(num_violations))
        return self.trace_gen()


def simulate_single(sys, prop):
    num_samples = globalopts.opts.num_sim_samples
    num_violations = 0

    concrete_states = sample.sample_init_UR(sys, prop, num_samples)
    trace_list = []

    sys_sim = simsys.get_system_simulator(sys)
    for i in tqdm.trange(num_samples):
        trace = simsys.simulate(sys_sim, prop.T, concrete_states[i])
        trace_list.append(trace)
        if check_prop_violation(prop, trace):
            num_violations += 1
            print('violation counter: {}'.format(num_violations))

    print('number of violations: {}'.format(num_violations))
    return trace_list


def random_test(
        A,
        system_params,
        initial_state_list,
        ci_seq_list,
        pi_seq_list,
        init_d,
        initial_controller_state,
        sample_ci,
        return_vio_only=True
        ):

    # ##!!##logger.debug('random testing...')
    logger.debug('initial states :\n{}'.format('\n'.join([str(A.plant_abs.get_ival_cons_abs_state(s0.ps)) for s0 in initial_state_list])))
    init_cons = system_params.init_cons
    A.prog_bar = False

    res = []
    # initial_state_set = set(initial_state_list)

    if A.num_dims.ci != 0:
        if sample_ci:
            ci_seq_array = np.array([np.array(ci_seq_list).T]).T
        else:
            ci_seq_array = np.array(ci_seq_list)

        # print('ci_seq_array', ci_seq_array)
        # print('ci_seq_array.shape', ci_seq_array.shape)

    if A.num_dims.pi != 0:
        pi_seq_array = np.array([np.array(pi_seq_list).T]).T

    #print(ci_seq_array.shape)
    #print(pi_seq_array.shape)
    x_array = np.empty((0, A.num_dims.x), dtype=float)

    print('checking initial states')

    # for abs_state in initial_state_set:

    for abs_state in initial_state_list:
        ival_cons = A.plant_abs.get_ival_cons_abs_state(abs_state.plant_state)

        # ##!!##logger.debug('ival_cons: {}'.format(ival_cons))

        # find the intersection b/w the cell and the initial cons
        # print('init_cons', init_cons)

        ic = ival_cons & init_cons
        if (ic is not None) and (not ic.zero_measure):

            # scatter the continuous states
            x_samples = ic.sample_UR(A.num_samples)

            # ##!!##logger.debug('ic: {}'.format(ic))
            # ##!!##logger.debug('samples: {}'.format(x_samples))

            x_array = np.concatenate((x_array, x_samples))
        else:
            raise err.Fatal('Can not happen! Invalid states have already been filtered out by filter_invalid_abs_states()')

            # # ##!!##logger.debug('{}'.format(samples.x_array))

            # ##!!##logger.debug('ignoring abs states: {}'.format(ival_cons))

            # ignore the state as it is completely outside the initial
            # constraints

    #x_array[-1, :] = np.array([0.4, -0.4])
    # print(x_array)
    print(x_array.shape)
    num_samples = len(x_array)
    if num_samples == 0:
        print(initial_state_list)
        print('no valid sample found during random testing. STOP')
        return False
    else:

        # ##!!##logger.debug('num_samples = 0')

        print('simulating {} samples'.format(num_samples))

    trace_list = [traces.Trace(A.num_dims, A.N+1) for i in range(num_samples)]

    s_array = np.tile(initial_controller_state, (num_samples, 1))

#     if system_params.pi is not None:
#         pi_array = SaMpLe.sample_ival_constraints(system_params.pi, num_samples)
#         print(pi_array)
#         exit()
#     else:
#         pi_array = None

    t_array = np.tile(0.0, (num_samples, 1))

    d_array = np.tile(init_d, (num_samples, 1))

    # TODO: initializing pvt states to 0

    p_array = np.zeros((num_samples, 1))

    # save x_array to print x0 in case an error is found
    # TODO: need to do something similar for u,ci,pi

    x0_array = x_array
    d0_array = d_array

    for i, trace in enumerate(trace_list):
        trace.append(x_array[i], 0, t_array[i], d_array[i])

    # sanity check

    if len(x_array) != len(s_array):
        raise err.Fatal('internal: how is len(x_array) != len(s_array)?')

    # while(simTime < A.T):
    sim_num = 0
    simTime = 0.0
    i = 0
    # records the actual pis used. These are printed in case a
    # violation is found for reproducibility
    pi_seqs_used = []
    while sim_num < A.N:
        if A.num_dims.ci == 0:
            ci_array = np.zeros((num_samples, 0))
        else:
            if sample_ci:
                ci_cons_list = list(ci_seq_array[:, i, :])
                ci_cons_list = [ci_cons.tolist()[0] for ci_cons in ci_cons_list]

                ci_lb_list = [np.tile(ci_cons.l, (A.num_samples, 1)) for ci_cons in ci_cons_list]
                ci_ub_list = [np.tile(ci_cons.h, (A.num_samples, 1)) for ci_cons in ci_cons_list]

                ci_cons_lb = ft.reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), ci_lb_list)
                ci_cons_ub = ft.reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), ci_ub_list)

                random_arr = np.random.rand(num_samples, A.num_dims.ci)

                ci_array = ci_cons_lb + random_arr * (ci_cons_ub - ci_cons_lb)
            else:
                ci_array = ci_seq_array[:, i, :]
                ci_array = np.repeat(ci_array, A.num_samples, axis=0)

        if A.num_dims.pi == 0:
            pi_array = np.zeros((num_samples, 0))
        else:
            pi_cons_list = list(pi_seq_array[:, i, :])
            pi_cons_list = [pi_cons.tolist()[0] for pi_cons in pi_cons_list]
            #print(pi_cons_list)
            #pi_cons_list = map(A.plant_abs.get_ival_cons_pi_cell, pi_cells)

            pi_lb_list = [np.tile(pi_cons.l, (A.num_samples, 1)) for pi_cons in pi_cons_list]
            pi_ub_list = [np.tile(pi_cons.h, (A.num_samples, 1)) for pi_cons in pi_cons_list]

            pi_cons_lb = ft.reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), pi_lb_list)
            pi_cons_ub = ft.reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), pi_ub_list)

            #print(pi_cons_lb)
            #print(pi_cons_ub)
            #U.pause()

            random_arr = np.random.rand(num_samples, A.num_dims.pi)

#             print('pi_cons_lb.shape:', pi_cons_lb.shape)
#             print('pi_cons_ub.shape:', pi_cons_ub.shape)
#             print('num_samples', num_samples)
            pi_array = pi_cons_lb + random_arr * (pi_cons_ub - pi_cons_lb)
            pi_seqs_used.append(pi_array)

        (s_array_, u_array) = compute_concrete_controller_output(
            A,
            system_params.controller_sim,
            ci_array,
            x_array,
            s_array,
            num_samples,
            )
        concrete_states = st.StateArray(  # t
                                        # cont_state_array
                                        # abs_state.discrete_state
                                        # abs_state.pvt_stat
                                        t_array,
                                        x_array,
                                        d_array,
                                        p_array,
                                        pi_array,
                                        )

        # print(concrete_states)

        # enforce property checking even if it has not been requested
        # by the user
        pc = PropertyChecker(system_params.final_cons)
        rchd_concrete_state_array, property_violated_flag = (
            system_params.plant_sim.simulate_with_property_checker(
                concrete_states,
                A.delta_t,
                pc
                ))

        for kdx, rchd_state in enumerate(rchd_concrete_state_array.iterable()):
            trace = trace_list[kdx]
            trace.append(rchd_state.x, rchd_state.pi, rchd_state.t, rchd_state.d)

        if property_violated_flag:
            print(U.decorate('concretized!'))
            for (idx, xf) in enumerate(rchd_concrete_state_array.iterable()):
                if xf.x in system_params.final_cons:
                    res.append(idx)
                    print(x0_array[idx, :], d0_array[idx, :], '->', '\t', xf.x, xf.d)
                    tmp_pi = [pi[idx, :] for pi in pi_seqs_used]
                    print('pi_seq:', tmp_pi)
                    #if A.num_dims.ci != 0:
                    #    print('ci:', ci_array[idx])
                    #if A.num_dims.pi != 0:
                    #    print('pi:', pi_array[idx])
            break
        i += 1
        sim_num += 1

        # increment simulation time

        simTime += A.delta_t
        t_array += A.delta_t
        concrete_states = rchd_concrete_state_array
        x_array = concrete_states.cont_states
        d_array = concrete_states.discrete_states
        p_array = concrete_states.pvt_states

        # u_array =

    if return_vio_only:
        return list(map(trace_list.__getitem__, res))
    else:
        #for trace in trace_list:
            #print(trace.x_array[0, :])
        #exit()
        return trace_list, bool(res)


def compute_concrete_controller_output(*args):
    return U.inf_list(0), U.inf_list(0)
