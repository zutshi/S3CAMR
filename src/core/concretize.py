from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys as SYS
import logging

import numpy as np
from blessed import Terminal

from . import simulatesystem as simsys
from . import state
from . import cellmanager as CM
import globalopts
from core.random_testing import check_prop_violation, random_test

#from IPython import embed

logger = logging.getLogger(__name__)
term = Terminal()


def concretize_bmc_trace(sys, prop, AA, sp, x_array, pi_seq):
    """
    Tries to concretize a BMC trace

    Parameters
    ----------
    AA :
    sys :
    prop :
    x_array : discrete trace represented by a numpy array of x: concrete states
    pi_seq : associated list of concrete pi sequences
    Notes
    ------
    Uses 4 different ways to check
    """
    sys_sim = simsys.get_system_simulator(sys)
    # 1)
    # Check exact returned trace.
    print('Checking trace returned by BMC...')
    trace = trace_violates(sys_sim, sys, prop, x_array, pi_seq)
    if trace:
        print(term.green('violation found'))
        print('violation found', file=SYS.stderr)
        res = True
    else:
        print('nothing found')
        print('nothing found', file=SYS.stderr)
        print(term.red('nothing found'))
        res = False

    return res
#######################################
#######################################
    # 2)
    # Check using random sims. Find the abstract state of the trace's
    # X0, and send it to random_test() along with pi_seqs
    print('concretizing using sampling X0...[fixed X0+pi_seq]')
    if abstract_trace_violates(sys, sp, prop, AA, x_array, pi_seq):
        print('our job is done...')
        print('our job is done...', file=SYS.stderr)
        exit()
    else:
        print('nothing found')
        print('nothing found', file=SYS.stderr)
        exit()

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


def trace_violates(sys_sim, sys, prop, x_array, pi_seq):
    x0_samples = [x_array[0, :]]
    x_array_bmc = x_array
    return traces_violate(sys_sim, sys, prop, x0_samples, pi_seq, x_array_bmc)


def concretize_init_cons_subset(sys, prop, AA, sp, x_array, pi_seq, init_cons_subset):
    x0_samples = init_cons_subset.sample_UR(100)
    sys_sim = simsys.get_system_simulator(sys)
    return traces_violate(sys_sim, sys, prop, x0_samples, pi_seq, x_array)


def traces_violate(sys_sim, sys, prop, x0_samples, pi_seq, x_array_bmc=None):
    x_array = x_array_bmc
    # The property can be violated at t <= Time Horizon. In that case
    # simulate only as much as the trace length allows.
    #num_segments = trace_len
    num_segments = len(x_array_bmc)-1

    z = np.array(np.empty((1, 0)))
    pvt = z
    t = np.array([[0]])
    d = np.array([prop.initial_discrete_state])
    pi_array = np.array([pi_seq])

    traces = []
    for x0 in x0_samples:
        concrete_states = state.StateArray(
                t, np.array([x0]), d,
                pvt, pi_array)

        traces.append(simsys.simulate(sys_sim, sys.delta_t*num_segments, concrete_states[0]))

    #from matplotlib import pyplot as plt
    #print(trace)

    # t_array is the same
    t_array = np.arange(0., prop.T, sys.delta_t)

    globalopts.opts.plotting.plot_trace_list(traces)#, x_vs_y=globalopts.opts.plots)
    if x_array is not None:
        globalopts.opts.plotting.plot_trace_array(t_array, x_array, 'r*-', linewidth=2)
    globalopts.opts.plotting.show()

    vio_traces = [trace for trace in traces if check_prop_violation(prop, trace)]
    return vio_traces


def abstract_trace_violates(sys, sp, prop, AA, x_array, pi_seq):
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
    x0 = x_array[0, :]
    x = np.array(x0)
    d = np.array(prop.initial_discrete_state)

    concrete_state = state.State(
            t=t, x=x, d=d,
            pvt=pvt, ci=ci_array, s=s, pi=pi_array, u=u)

    initial_state = AA.get_abs_state_from_concrete_state(concrete_state)

    print(
            CM.Cell(
                AA.get_abs_state_from_concrete_state(concrete_state).ps.cell_id,
                AA.plant_abs.eps).ival_constraints
            )

    # TODO: AA.samples is too low?
    # TODO: Also, the samples themselves seem fishy
    AA.num_samples = 100
    trace_list, vio_found = random_test(
        AA,
        sp,
        [initial_state],
        [],
        [pi_seq],
        prop.initial_discrete_state,
        initial_controller_state=None,
        sample_ci=False,
        return_vio_only=False
        )

    #globalopts.opts.plotting.figure()
    globalopts.opts.plotting.plot_trace_list(trace_list, x_vs_y=globalopts.opts.plots)
    globalopts.opts.plotting.plot_abs_states(AA, {'init': [initial_state]})
    globalopts.opts.plotting.show()

    return vio_found
