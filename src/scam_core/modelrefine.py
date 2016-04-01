from __future__ import print_function
from collections import defaultdict
import itertools

import numpy as np

import simulatesystem as simsys
from modeling.pwa import pwa
from modeling.pwa import simulator as pwa_sim
from modeling.pwa import relational as rel

import utils as U
from utils import print
import err
from constraints import IntervalCons, top2ic

import random_testing as rt

from bmc import bmc
from bmc.bmc_spec import InvarStatus
from modeling.affinemodel import RegressionModel
import cellmanager as CM

import functools

#np.set_printoptions(suppress=True, precision=2)

# multiply num samples with the
MORE_FACTOR = 10
TEST_FACTOR = 10

MAX_TRAIN = 200
MAX_TEST = 200

MIN_TRAIN = 50
MIN_TEST = MIN_TRAIN
MAX_ITER = 25

INF = float('inf')


def abs_state2cell(abs_state, eps):
    return CM.Cell(abs_state.plant_state.cell_id, eps)


def ic2cell(ic, eps):
    cells = CM.ic2cell(ic, eps)
    assert(len(cells) == 1)
    return cells[0]
    #return CM.Cell(CM.cell_from_concrete(pi, eps), eps)


class Q(object):
    def __init__(self, xcell, wcell):
        """__init__

        Parameters
        ----------
        abs_state : abstract states
        w_cells : cells associated with the abstract state defining
        range of inputs
        """

        assert(isinstance(xcell, CM.Cell))
        assert(isinstance(wcell, CM.Cell))

        self.xcell = xcell
        self.wcell = wcell
        self.xwcell = CM.Cell.concatenate(xcell, wcell)
        self.sample_UR_x = self.xcell.sample_UR
        self.sample_UR_w = self.wcell.sample_UR
        self._ival_constraints = None
        return

    def split(self, *args):
        xwsplits = self.xwcell.split(*args)
        l = []
        for xwcell in xwsplits:
            for xcell, wcell in xwcell.un_concatenate(self.xcell.dim):
                l.append(Q(xcell, wcell))
        return l

    @property
    def xdim(self):
        return self.xcell.dim

    @property
    def wdim(self):
        return self.wcell.dim

    @property
    def ival_constraints(self):
        return self.xwcell.ival_constraints

    def getxy_ignoramous(self, N, sim, t0=0):
        """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
        """
        d, pvt, s = [np.array([])]*3
        ci = np.array([])
        t0 = 0
        Yl = []

        x_array = self.sample_UR_x(N)
        pi_array = self.sample_UR_w(N)
        for x, pi in zip(x_array, pi_array):
            (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
            Yl.append(x_)

        return np.hstack((x_array, pi_array)), np.vstack(Yl)


def simulate_pwa(pwa_model, x_samples, N):
    #ps = pwa_sim.PWA_SIM(pwa_model)
    return [pwa_sim.simulate(pwa_model, x0, N) for x0 in x_samples]


def simulate(AA, s, sp, pwa_model, max_path_len, S0):
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


def sim_n_plot(error_paths, pwa_model, AA, sp, opts):
    max_path_len = max([len(path) for path in error_paths])
    # intial abs state set
    S0 = {path[0] for path in error_paths}
    s = {
        'init': {path[0] for path in error_paths},
        'final': {path[-1] for path in error_paths},
        'regular': {state for path in error_paths for state in path[1:-1]}
        }
    print('simulating...')
    pwa_traces = simulate(AA, s, sp, pwa_model, max_path_len, S0)
    print('done')
    opts.plotting.figure()
    print('plotting...')
    opts.plotting.plot_abs_states(AA, s)
    opts.plotting.plot_rect(sp.final_cons.rect())
    opts.plotting.plot_pwa_traces(pwa_traces)
    #fig = BP.figure(title='S3CAMR')
    #fig = plt.figure()
    opts.plotting.show()


# TODO: Merge with refine_dft
def refine_dftX_model_based(
        AA, error_paths, pi_seqs, sp, sys_sim, opts, sys, prop):

    #abs_state_pi_map = {}
    # traversed_abs_state_set
    #for path in error_paths:
    #for state in path
    pi_eps = sp.pi_ref.eps
    # collect all pi which were encountered with the abs_state
    abs_state_pi = defaultdict(set)
    for path, pi_seq in zip(error_paths, pi_seqs):
        for abs_state, pi in zip(path[:-1], pi_seq):
            abs_state_pi[abs_state].add(pi)

    Qs = []
    for abs_state, pi_ic_list in abs_state_pi.iteritems():
        xcell = abs_state2cell(abs_state, AA.plant_abs.eps)
        for pi_ic in pi_ic_list:
            wcell = ic2cell(pi_ic, pi_eps)
            Qs.append(Q(xcell, wcell))

    pwa_model = build_pwa_model(
            AA, Qs, sp, opts.max_model_error,
            opts.model_err, 'dftX')
    if __debug__:
        pass
        # Need to define a new function to simulate inputs
        #sim_n_plot(error_paths, pwa_model, AA, sp, opts)
    check4CE(pwa_model, error_paths, sys.sys_name, 'dftX', AA, sys, prop, sp, opts.bmc_engine)


def refine_dft_model_based(
        AA, error_paths, pi_seqs, sp, sys_sim, opts, sys, prop):

    # traversed_abs_state_set
    tas = {state for path in error_paths for state in path}

    pwa_model = build_pwa_model(
            AA, tas, sp, opts.max_model_error,
            opts.model_err, 'dft')
    if __debug__:
        sim_n_plot(error_paths, pwa_model, AA, sp, opts)
    check4CE(pwa_model, error_paths, sys.sys_name, 'dft', AA, sys, prop, sp, opts.bmc_engine)


def check4CE(pwa_model, error_paths, sys_name, model_type, AA, sys, prop, sp, bmc_engine):
    max_path_len = max([len(path) for path in error_paths])
    print('max_path_len:', max_path_len)
    # depth = max_path_len - 1, because path_len = num_states along
    # path. This is 1 less than SAL depth and simulation length
    depth = max(int(np.ceil(AA.T/AA.delta_t)), max_path_len - 1)
    print('depth :',  depth)

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

    sal_bmc = bmc.factory(
            bmc_engine,
            vs,
            pwa_model, init_cons, safety_prop,
            '{}_{}'.format(sys_name, model_type),
            model_type)

    status = sal_bmc.check(depth)
    #if status == InvarStatus.Safe:
    #    print('Safe')
    if status == InvarStatus.Unsafe:
        print('Unsafe...trying to concretize...')
        verify_bmc_trace(AA, sys, prop, sp, sal_bmc.trace, xs, ws)
    elif status == InvarStatus.Unknown:
        print('Unknown...exiting')
        exit()
    else:
        raise err.Fatal('Internal')


def verify_bmc_trace(AA, sys, prop, sp, trace, xs, ws):
    """Get multiple traces and send them for random testing
    """
    print(trace[0])
    init_assignments = trace[0].assignments
    x_array = np.array([init_assignments[x] for x in xs])
    pi_seq = [[step.assignments[w] for w in ws] for step in trace[:-1]]
    print(x_array)
    print(pi_seq)
    rt.concretize_bmc_trace(sys, prop, AA, sp, x_array, pi_seq)
    return


def refine_rel_model_based(
        AA, error_paths, pi_seq_list, sp, sys_sim, opts, sys, prop):
    '''does not handle pi_seq_list yet'''

    # abs_state relations: maps an abs_state to other abs_states
    # reachable in one time step
    abs_relations = defaultdict(set)
    for path in error_paths:
        # abs_state_1 -> abs_state_2
        for a1, a2 in U.pairwise(path):
            abs_relations[a1].add(a2)

    flat_relations = []
    for abs_state, rch_states in abs_relations.iteritems():
        flat_relation = list(itertools.product([abs_state], rch_states))
        flat_relations.extend(flat_relation)

    pwa_model = build_pwa_model(
            AA, flat_relations, sp, opts.max_model_error,
            opts.model_err, 'rel')

    if __debug__:
        sim_n_plot(error_paths, pwa_model, AA, sp, opts)
    check4CE(pwa_model, error_paths, sys.sys_name, 'rel', AA, sys, prop, sp, opts.bmc_engine)


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

    sal_bmc = bmc.factory(bmc_engine)
    prop = sp.final_cons

    sal_bmc.init(AA.num_dims.x, pwa_models, sp.init_cons, prop, 'vdp_dmt', 'dmt')
    sal_bmc.check()


def getxy_ignoramous(cell, N, sim, t0=0):
    """TODO: EXPLICITLY ignores t, d, pvt, ci, pi
    """
    d, pvt, s = [np.array([])]*3
    ci, pi = [np.array([])]*2
    t0 = 0
    Yl = []

    x_array = cell.sample_UR(N)
    for x in x_array:
        (t_, x_, s_, d_, pvt_, u_) = sim(t0, x, s, d, pvt, ci, pi)
        Yl.append(x_)

    return x_array, np.vstack(Yl)


def getxy_rel_ignoramous_force_min_samples(cell1, cell2, force, N, sim, t0=0):
    """getxy_rel_ignoramous

    Parameters
    ----------
    force : force to return non-zero samples. Will loop for infinitiy
            if none exists.
    """
    xl = []
    yl = []
    sat_count = 0
    if __debug__:
        obs_cells = set()
    while True:
        x_array, y_array = getxy_ignoramous(cell1, N, sim, t0=0)
        if __debug__:
            for i in y_array:
                obs_cells.add(CM.cell_from_concrete(i, cell1.eps))
            print('reachable cells:', obs_cells)

        # satisfying indexes
        sat_array = cell2.ival_constraints.sat(y_array)
        sat_count += np.sum(sat_array)
        xl.append(x_array[sat_array])
        yl.append(y_array[sat_array])
        # If no sample is found and force is True, must keep sampling till
        # satisfying samples are found
        if (sat_count >= MIN_TRAIN) or (not force):
            break
        if __debug__:
            print('re-sampling, count:', sat_count)

    print('found samples: ', sat_count)
    return np.vstack(xl), np.vstack(yl)


def getxy_rel_ignoramous(cell1, cell2, force, N, sim, t0=0):
    """getxy_rel_ignoramous

    Parameters
    ----------
    force : force to return non-zero samples. Will loop for infinitiy
            if none exists.
    """
    xl = []
    yl = []
    sat_count = 0
    iter_count = itertools.count()
    print(cell1.ival_constraints)
    print(cell2.ival_constraints)
    if __debug__:
        obs_cells = set()
    while next(iter_count) <= MAX_ITER:
        x_array, y_array = getxy_ignoramous(cell1, N, sim, t0=0)
        if __debug__:
            for i in y_array:
                obs_cells.add(CM.cell_from_concrete(i, cell1.eps))
            print('reachable cells:', obs_cells)
        # satisfying indexes
        sat_array = cell2.ival_constraints.sat(y_array)
        sat_count += np.sum(sat_array)
        xl.append(x_array[sat_array])
        yl.append(y_array[sat_array])
        if sat_count >= MIN_TRAIN:
            break
        # If no sample is found and force is True, must keep sampling till
        # satisfying samples are found
    if __debug__:
        if sat_count < MIN_TRAIN:
            err.warn('Fewer than MIN_TRAIN samples found!')
    print('found samples: ', sat_count)
    return np.vstack(xl), np.vstack(yl)


def build_pwa_model(AA, abs_objs, sp, tol, include_err, model_type):
    """build_pwa_model
    Builds both dft and rel models

    Parameters
    ----------
    AA : AA
    abs_objs : Either abs_states (for dft models) or relations
              [tuple(abs_state_src, abs_state_target)] for rel model
    sp : system params
    tol : modeling error tolerance
    include_err : include error in determining next state
                  x' = x +- errror
    """
    # number of training samples
    ntrain = min(AA.num_samples * MORE_FACTOR, MAX_TRAIN)
    # number of test samples
    ntest = min(ntrain * TEST_FACTOR, MAX_TEST)

    dt = AA.plant_abs.delta_t
    step_sim = simsys.get_step_simulator(sp.controller_sim, sp.plant_sim, dt)

    #abs_state_models = {}
    modelers = {
            'dft': (pwa.PWA, abs_state_affine_models),
            'rel': (rel.PWARelational, abs_rel_affine_models),
            'dftX': (pwa.PWA, functools.partial(q_affine_models, ntrain, ntest, sp.pi_ref.i_cons))
            }
    M, model = modelers[model_type]
    pwa_model = M()

    for a in abs_objs:
        print('modeling: {}'.format(a))
        for sub_model in model(a, AA, step_sim, tol, sp, include_err):
            if sub_model is not None:
                pwa_model.add(sub_model)
            #abs_state_models[abs_state] = sub_model
    return pwa_model


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


def abs_rel_affine_models(abs_state_rel, AA, step_sim, tol, sp, include_err):
    cell1 = CM.Cell(abs_state_rel[0].plant_state.cell_id, AA.plant_abs.eps)
    cell2 = CM.Cell(abs_state_rel[1].plant_state.cell_id, AA.plant_abs.eps)
    # number of training samples
    ntrain = min(AA.num_samples * MORE_FACTOR, MAX_TRAIN)
    # number of test samples
    ntest = min(ntrain * TEST_FACTOR, MAX_TEST)

    return cell_rel_affine_models(
            cell1, cell2, True, step_sim, ntrain, ntest, tol, include_err)


#AA.plant_abs.get_abs_state_cell(abs_state.plant_state),
def cell_rel_affine_models(cell1, cell2, force, step_sim, ntrain, ntest, tol, include_err):
    """cell_affine_models

    Parameters
    ----------
    cell1 : source cell
    cell2 : target cell
    step_sim : 1 time step (delta_t) simulator
    tol : each abs state is split further into num_splits cells
    in order to meet: modeling error < tol (module ntests samples)

    Returns
    -------
    pwa.SubModel()

    Notes
    ------
    """
    # XXX: Generate different samples for each time step or reuse?
    # Not clear!
    sub_models = []

    X, Y = getxy_rel_ignoramous(cell1, cell2, force, ntrain, step_sim)
    # No samples found => no model
    training_samples_found = len(X) != 0
    if not training_samples_found:
        return [None]
    rm = RegressionModel(X, Y)
    X, Y = getxy_rel_ignoramous(cell1, cell2, True, ntest, step_sim)
    testing_samples_found = len(X) != 0
    # If valid samples are found, compute e_pc (error %) and dims
    # where error % >= given tol
    if testing_samples_found:
        e_pc = rm.error_pc(X, Y)
        error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]
    # Otherwise, forget it!
    else:
        e_pc = None
        error_dims = []

    if __debug__:
        print('error%:', e_pc)

    if len(error_dims) > 0:
        err.warn('splitting on e%:{}, |e%|:{}'.format(
            e_pc, np.linalg.norm(e_pc, 2)))
        for split_cell1 in cell1.split(axes=error_dims):
            sub_models_ = cell_rel_affine_models(
                    split_cell1, cell2, False, step_sim, ntrain, ntest, tol, include_err)
            sub_models.extend(sub_models_)
        return sub_models
    else:
        A, b = rm.A, rm.b
        C1, d1 = cell1.ival_constraints.poly()
        C2, d2 = cell2.ival_constraints.poly()

        e = rm.error(X, Y) if (include_err and testing_samples_found) else None

        dmap = rel.DiscreteAffineMap(A, b, e)
        part1 = rel.Partition(C1, d1, cell1)
        part2 = rel.Partition(C2, d2, cell2)
        sub_model = rel.Relation(part1, part2, dmap)
        if __debug__:
            print('----------------Finalized------------------')
    return [sub_model]


# TODO: Fix the excess arguement issue
def q_affine_models(ntrain, ntest, wic, q, dummy1, step_sim, tol, dummy2, include_err):
    """cell_affine_models

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
    # XXX: Generate different samples for each time step or reuse?
    # Not clear!
    sub_models = []

    X, Y = q.getxy_ignoramous(ntrain, step_sim)
    rm = RegressionModel(X, Y)
    X, Y = q.getxy_ignoramous(ntest, step_sim)
    e_pc = rm.error_pc(X, Y) # error %
    if __debug__:
        print('error%:', e_pc)
    #error = np.linalg.norm(e_pc, 2)
    # error exceeds tol in error_dims
    error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]

    if len(error_dims) > 0:
        err.warn('splitting on e%:{}, |e%|:{}'.format(
            e_pc, np.linalg.norm(e_pc, 2)))
        for split_q in q.split(axes=error_dims):
            sub_models_ = q_affine_models(
                    ntrain, ntest, wic,
                    split_q, dummy1, step_sim, tol, dummy2, include_err)
            sub_models.extend(sub_models_)
        return sub_models
    else:
        #print('error%:', rm.error_pc(X, Y))
        # Matrices are extended to include w/pi
        # A = [AB]
        #     [00]
        # AB denotes matrix concatenation.
        # Hence a reset: x' =  Ax + Bw + b
        # can be mimicked as below
        #
        # [x']   = [a00 a01 a02...b00 b01...] * [x] + b + [e0]
        # [w']     [     ...  0 ...         ]   [w]       [e1]
        #
        # This makes x' \in Ax + Bw + b + [el, eh], and
        # makes w' \in [el, eh]
        # We use this to incorporate error and reset w to new values,
        # which in the case of ZOH are just the ranges of w (or pi).

        A = np.vstack((rm.A, np.zeros((q.wdim, q.xdim + q.wdim))))
        b = np.hstack((rm.b, np.zeros(q.wdim)))
        C, d = q.ival_constraints.poly()
        try:
            assert(A.shape[0] == b.shape[0])    # num lhs (states) is the same
            assert(A.shape[1] == C.shape[1])    # num vars (states + ip) are the same
            assert(C.shape[0] == d.shape[0])    # num constraints are the same
        except AssertionError as e:
            print('\n', A, '\n', b)
            print('\n', C, '\n', d)
            print(A.shape[0], b.shape[0], C.shape[1], d.shape[0])
            raise e

        xic = (rm.error(X, Y) if include_err
               else IntervalCons([0.0]*q.xdim, [0.0]*q.xdim))

        e = IntervalCons.concatenate(xic, wic)
        dmap = pwa.DiscreteAffineMap(A, b, e)
        part = pwa.Partition(C, d, q)
        sub_model = pwa.SubModel(part, dmap)
        if __debug__:
            print('----------------Finalized------------------')
    return [sub_model]


def abs_state_affine_models(abs_state, AA, step_sim, tol, sp, include_err):
    cell = CM.Cell(abs_state.plant_state.cell_id, AA.plant_abs.eps)
    # number of training samples
    ntrain = min(AA.num_samples * MORE_FACTOR, MAX_TRAIN)
    # number of test samples
    ntest = min(ntrain * TEST_FACTOR, MAX_TEST)

    return cell_affine_models(
            cell, step_sim, ntrain, ntest, tol, include_err)


#AA.plant_abs.get_abs_state_cell(abs_state.plant_state),
def cell_affine_models(cell, step_sim, ntrain, ntest, tol, include_err):
    """cell_affine_models

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
    # XXX: Generate different samples for each time step or reuse?
    # Not clear!
    sub_models = []

    X, Y = getxy_ignoramous(cell, ntrain, step_sim)
    rm = RegressionModel(X, Y)
    X, Y = getxy_ignoramous(cell, ntest, step_sim)
    e_pc = rm.error_pc(X, Y) # error %
    if __debug__:
        print('error%:', e_pc)
    #error = np.linalg.norm(e_pc, 2)
    # indices where error exceeds tol
    error_dims = np.arange(len(e_pc))[np.where(e_pc >= tol)]

    if len(error_dims) > 0:
        err.warn('splitting on e%:{}, |e%|:{}'.format(
            e_pc, np.linalg.norm(e_pc, 2)))
        for split_cell in cell.split(axes=error_dims):
            sub_models_ = cell_affine_models(
                    split_cell, step_sim, ntrain, ntest, tol, include_err)
            sub_models.extend(sub_models_)
        return sub_models
    else:
        #print('error%:', rm.error_pc(X, Y))
        A, b = rm.A, rm.b
        C, d = cell.ival_constraints.poly()
        # TODO: guess the dimensions. Fix it
        ndim = X.shape[1]
        e = (rm.error(X, Y) if include_err
             else IntervalCons([0.0]*ndim, [0.0]*ndim))
        dmap = pwa.DiscreteAffineMap(A, b, e)
        part = pwa.Partition(C, d, cell)
        sub_model = pwa.SubModel(part, dmap)
        if __debug__:
            print('----------------Finalized------------------')
    return [sub_model]


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

























##############################################
################## CEMETEREY #################
##############################################

# TODO: move it to PACell/ or use an existing function
def split_abs_state(AA, abs_state):
    import cellmanager as cm
    import PACell as P
    import abstraction as A
    child_cells = cm.get_children(abs_state.ps.cell_id)
    child_states = [
        A.AbstractState(
            P.PlantAbstractState(
                          cell,
                          abs_state.ps.n,
                          abs_state.ps.d,
                          abs_state.ps.pvt),
            abs_state.cs)
        for cell in child_cells
        ]
    return child_states

# DO NOT USE
# TODO: DELETE
def test_model_(abs_state, AA, sp, am, step_sim):
    test_samples = sp.sampler.sample(abs_state, AA, sp, AA.num_samples*MORE_FACTOR*TEST_FACTOR)
    X, Y = getxy(abs_state, test_samples, step_sim)
    e = am.model_error(X, Y)
    if __debug__:
        print(e)
    return e

def getxy_generic(abs_state, state_samples, sim, t0=0):
    """TODO: uses t, d, pvt, ci, pi
    but its getting ignored while modeling!
    """
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
