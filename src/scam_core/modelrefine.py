import numpy as np
#import bokeh.plotting as BP
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import simulatesystem as simsys
from modeling.pwa import pwa
from modeling.pwa import simulator as pwa_sim
#import utils as U
#import constraints as C
import err
from bmc import bmc
from modeling.affinemodel import AffineModel
import sample

#np.set_printoptions(suppress=True, precision=2)

# multiply num samples with the
MORE_FACTOR = 10
TEST_FACTOR = 10

INF = float('inf')


def simulate_pwa(pwa_model, x_samples, N):
    #ps = pwa_sim.PWA_SIM(pwa_model)
    return [pwa_sim.simulate(pwa_model, x0, N) for x0 in x_samples]


def plot_rect(r, fig):
    ax = fig.gca()
    ax.add_patch(
        patches.Rectangle(r[0], *r[1], fill=False)
        #patches.Rectangle((0.1, 0.1), 0.5, 0.5, fill=False)
    )


def plot_abs_states(AA, abs_states, fig):
    for abs_state in abs_states:
        #r = AA.plant_abs.rect(abs_state.plant_state)
        r = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).rect()
        #if c.dim != 2:
        #    raise StandardError('dim should be 2 for plotting 2D!')
        plot_rect(r, fig)


def plot_trace(tx, fig):
    #print tx
    #t = np.array(tx[0])
    x = np.vstack(tx[1])
    ax = fig.gca()
    ax.plot(x[:, 0], x[:, 1], '-*')
#     fig.line(x[:, 0], x[:, 1])
#     print x
#     BP.output_server('hover')
#     BP.show(fig)


def refine_dft_model_based(AA, error_paths, pi_seq_list, sp, sys_sim):
    '''does not handle pi_seq_list yet'''

    # traversed_abs_state_set
    tas = {state for path in error_paths for state in path}
    # intial abs state set
    S0 = {path[0] for path in error_paths}
    max_path_len = max([len(path) for path in error_paths])

    #fig = BP.figure(title='S3CAMR')
    fig = plt.figure()
    plot_abs_states(AA, tas, fig)
    plot_rect(sp.final_cons.rect(), fig)

    pwa_model = build_pwa_dft_model(AA, tas, sp, sys_sim)

    # sample only initial abstract state
    x0_samples = (sp.sampler.sample_multiple(S0, AA, sp, 500)).x_array
    #print x0_samples
    # sample the entire given initial set
    #X0 = sp.init_cons
    #x0_samples = sample.sample_ival_constraints(X0, n=1000)
    print 'simulating and plotting...'
    for i in simulate_pwa(pwa_model, x0_samples, N=max_path_len):
        plot_trace(i, fig)
    plt.show()

    sal_bmc = bmc.factory('sal')
    prop = sp.final_cons

#     err.warn_severe('overwriting safety property!')
#     prop = C.IntervalCons(
#             np.array([-INF, -INF]),
#             np.array([-1, INF]))
#     prop = C.IntervalCons(
#             np.array([1, -INF]),
#             np.array([INF, INF]))

    sal_bmc.init(AA.num_dims.x, pwa_model, sp.init_cons, prop, 'vdp_dft', 'dft')
    sal_bmc.check(max_path_len)
    #bmc.check(depth=2)
    #if bmc.sat:
    #    return bmc.soln
    #else:
    #    return None


def refine_dmt_model_based(AA, error_paths, pi_seq_list, sp, sys_sim):
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

    # traversed_abs_state_set
    tas = {state for path in error_paths for state in path}

    pwa_models = build_pwa_dt_model(AA, tas, sp, sys_sim)

    sal_bmc = bmc.factory('sal')
    prop = sp.final_cons

    sal_bmc.init(AA.num_dims.x, pwa_models, sp.init_cons, prop, 'vdp_dmt', 'dmt')
    sal_bmc.check()


def getxy(abs_state, state_samples, sim, t0=0):
    d = abs_state.plant_state.d
    pvt = abs_state.plant_state.pvt
    Yl = []

    for s, x, ci, pi, t in state_samples.iterable():
        (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
        Yl.append(x_)
        #trace = sys_sim(x, s, d, 0, t0, t0 + dt, ci, pi)

    Y = np.vstack(Yl)
    X = state_samples.x_array
    return X, Y


def build_pwa_dft_model(AA, abs_states, sp, sys_sim):
    dt = AA.plant_abs.delta_t
    step_sim = simsys.get_step_simulator(sp.controller_sim, sp.plant_sim, dt)

    abs_state_models = {}
    M = pwa.PWA()

    for abs_state in abs_states:
        sub_model = model(abs_state, AA, sp, step_sim)
        M.add(sub_model)
        abs_state_models[abs_state] = sub_model

    return M


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
            sub_model = model(abs_state, AA, sp, step_sim)
            pwa_model.add(sub_model)

        pwa_models[dt] = pwa_model
    return pwa_models


def model(abs_state, AA, sp, step_sim):
    # XXX: Generate different samples for each time step or reuse?
    # Not clear!
    state_samples = sp.sampler.sample(abs_state, AA, sp, AA.num_samples*MORE_FACTOR)
    test_samples = sp.sampler.sample(abs_state, AA, sp, AA.num_samples*MORE_FACTOR*TEST_FACTOR)

    X, Y = getxy(abs_state, state_samples, step_sim)

    am = AffineModel(X, Y)
    A, b = am.Ab
    C, d = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).poly()
    sub_model = pwa.PWA.sub_model(A, b, C, d)

    if __debug__:
        X, Y = getxy(abs_state, test_samples, step_sim)
        e = am.model_error(X, Y)
        print 'error% in pwa model', e
        print '='*20
        #U.pause()
    return sub_model


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
