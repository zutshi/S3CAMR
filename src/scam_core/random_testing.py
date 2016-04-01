from __future__ import print_function

import numpy as np
import tqdm

import err
import utils as U
from utils import print

import sample
import traces
import state as st
from properties import PropertyChecker
import simulatesystem as simsys
import state
import cellmanager as CM


def concretize_bmc_trace(sys, prop, AA, sp, x0, pi_seq):
    """
    Tries to concretize a BMC trace

    Parameters
    ----------
    AA :
    sys :
    prop :
    x0 : list of concrete states
    pi_seq : associated list of concrete pi sequences
    Notes
    ------
    Uses 4 different ways to check
    """
    sys_sim = simsys.get_system_simulator(sys)
    # 1)
    # Check exact returned trace.
    print('Checking trace returned by BMC...')
    trace = trace_violates(sys_sim, sys, prop, x0, pi_seq)
    if trace:
        print('violation found')
        print(trace)
        exit()
    else:
        print('nothing found')
    # 2)
    # Check using random sims. Find the abstract state of the trace's
    # X0, and send it to random_test() along with pi_seqs
    print('concretizing using sampling X0...[fixed X0+pi_seq]')
    if abstract_trace_violates(sys, sp, prop, AA, x0, pi_seq):
        print('our job is done...')
        exit()
    else:
        print('nothing found')


    # 3)
    # Check using random sims. Poll the BMC solver to provide more
    # samples for the same discrete sequence.
    print('concretizing using SMT sampling...[fixed discrete seq.]')
    raise NotImplementedError

    # 4)
    # Try different discrete sequences
    print('concretizing using different discrete sequences...[paths]')
    raise NotImplementedError

    print('bad luck!')
    exit()


def trace_violates(sys_sim, sys, prop, x, pi_seq):
    num_segments = int(np.ceil(prop.T / sys.delta_t))
    #num_segments = len(pi)
    z = np.array(np.empty((1, 0)))
    pvt = z
    ci_array = np.empty((1, num_segments, 0))
    s = z
    u = z
    t = np.array([[0]])
    x = np.array([x])
    d = np.array([prop.initial_discrete_state])
    pi_array = np.array([pi_seq])
    concrete_states = state.StateArray(
            t, x, d,
            pvt, s, u, pi_array, ci_array)

#     print(concrete_states.n)
#     print(concrete_states.t.shape)
#     print(concrete_states.cont_states.shape)
#     print(concrete_states.controller_extraneous_inputs.shape)
#     print(concrete_states.controller_outputs.shape)
#     print(concrete_states.discrete_states.shape)
#     print(concrete_states.controller_states.shape)
#     print(concrete_states.pvt_states.shape)
#     print(concrete_states.plant_extraneous_inputs.shape)

    trace = simsys.simulate(sys_sim, concrete_states[0], prop.T)
    if check_prop_violation(trace, prop):
        return trace
    else:
        return None


def abstract_trace_violates(sys, sp, prop, AA, x, pi_seq):
    if AA.num_dims.pi != 0:
        pi_eps = sp.pi_ref.pi_eps
        pi_seq = [
                CM.ival_constraints(
                    CM.cell_from_concrete(pi, pi_eps)
                    ) for pi in pi_seq
                    ]

    z = np.array(np.empty(1))
    pvt = z
    ci_array = np.empty(1)
    pi_array = np.empty(1)
    s = z
    u = z
    t = np.array(0)
    x = np.array(x)
    d = np.array(prop.initial_discrete_state)

    concrete_state = state.State(
            t=t, x=x, d=d,
            pvt=pvt, ci=ci_array, s=s, pi=pi_array, u=u)

    initial_state_list = [AA.get_abs_state_from_concrete_state(concrete_state)]

    print(
            CM.Cell(
                AA.get_abs_state_from_concrete_state(concrete_state).ps.cell_id,
                AA.plant_abs.eps).ival_constraints
            )

    return random_test(
        AA,
        sp,
        initial_state_list,
        [],
        [pi_seq],
        prop.initial_discrete_state,
        initial_controller_state=None,
        sample_ci=False
        )


# TODO: make a module of its own once we add more general property using
# monitors...
def check_prop_violation(trace, prop):
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
    idx = prop.final_cons.contains(trace.x_array)
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


def simulate(sys, prop, opts):
    num_samples = opts.num_sim_samples
    num_violations = 0

    concrete_states = sample.sample_init_UR(sys, prop, num_samples)
    trace_list = []

    sys_sim = simsys.get_system_simulator(sys)
    for i in tqdm.trange(num_samples):
        trace = simsys.simulate(sys_sim, concrete_states[i], prop.T)
        trace_list.append(trace)
        if check_prop_violation(trace, prop):
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
        sample_ci
        ):

    # ##!!##logger.debug('random testing...')
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
    x_array = np.empty((0.0, A.num_dims.x), dtype=float)

    print('checking initial states')

    # for abs_state in initial_state_set:

    for abs_state in initial_state_list:
        ival_cons = A.plant_abs.get_ival_cons_abs_state(abs_state.plant_state)

        # ##!!##logger.debug('ival_cons: {}'.format(ival_cons))

        # find the intersection b/w the cell and the initial cons
        # print('init_cons', init_cons)

        ic = ival_cons & init_cons
        if ic is not None:

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

    # print(x_array)
    # print(x_array.shape)

    print(x_array.shape)
    num_samples = len(x_array)
    if num_samples == 0:
        print(initial_state_list)
        print('no valid sample found during random testing. STOP')
        return False
    else:

        # ##!!##logger.debug('num_samples = 0')

        print('simulating {} samples'.format(num_samples))

    trace_list = [traces.Trace(A.num_dims, A.N) for i in range(num_samples)]

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

    # sanity check

    if len(x_array) != len(s_array):
        raise err.Fatal('internal: how is len(x_array) != len(s_array)?')

    # while(simTime < A.T):
    sim_num = 0
    simTime = 0.0
    i = 0
    while sim_num < A.N:
        if A.num_dims.ci == 0:
            ci_array = np.zeros((num_samples, 0))
        else:
            if sample_ci:
                ci_cons_list = list(ci_seq_array[:, i, :])
                ci_cons_list = [ci_cons.tolist()[0] for ci_cons in ci_cons_list]

                ci_lb_list = [np.tile(ci_cons.l, (A.num_samples, 1)) for ci_cons in ci_cons_list]
                ci_ub_list = [np.tile(ci_cons.h, (A.num_samples, 1)) for ci_cons in ci_cons_list]

                ci_cons_lb = reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), ci_lb_list)
                ci_cons_ub = reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), ci_ub_list)

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

            pi_cons_lb = reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), pi_lb_list)
            pi_cons_ub = reduce(lambda acc_arr, arr: np.concatenate((acc_arr, arr)), pi_ub_list)

            random_arr = np.random.rand(num_samples, A.num_dims.pi)

#             print('pi_cons_lb.shape:', pi_cons_lb.shape)
#             print('pi_cons_ub.shape:', pi_cons_ub.shape)
#             print('num_samples', num_samples)
            pi_array = pi_cons_lb + random_arr * (pi_cons_ub - pi_cons_lb)

        (s_array_, u_array) = compute_concrete_controller_output(
            A,
            system_params.controller_sim,
            ci_array,
            x_array,
            s_array,
            num_samples,
            )
        print(x_array)
        concrete_states = st.StateArray(  # t
                                        # cont_state_array
                                        # abs_state.discrete_state
                                        # abs_state.pvt_stat
                                        t_array,
                                        x_array,
                                        d_array,
                                        p_array,
                                        s_array_,
                                        u_array,
                                        pi_array,
                                        ci_array,
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
            trace.append(rchd_state.s, rchd_state.u, rchd_state.x, rchd_state.ci, rchd_state.pi, rchd_state.t, rchd_state.d)

        if property_violated_flag:
            print(U.decorate('concretized!'))
            for (idx, xf) in enumerate(rchd_concrete_state_array.iterable()):
                if xf.x in system_params.final_cons:
                    res.append(idx)
                    print(x0_array[idx, :], d0_array[idx, :], '->', '\t', xf.x, xf.d)
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
        s_array = concrete_states.controller_states
        d_array = concrete_states.discrete_states
        p_array = concrete_states.pvt_states

        # u_array =

    return map(trace_list.__getitem__, res)


def compute_concrete_controller_output(*args):
    return U.inf_list(0), U.inf_list(0)
