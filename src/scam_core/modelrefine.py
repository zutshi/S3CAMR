import pwa
import simulatesystem as simsys
from sklearn import linear_model as skl_lm
import numpy as np

#import utils as U

np.set_printoptions(suppress=True, precision=2)

# multiply num samples with the
MORE_FACTOR = 10


def refine_model_based(A, error_paths, pi_seq_list, sp, sys_sim):
    '''does not handle pi_seq_list yet'''

    # traversed_abs_state_set
    tas = {state for path in error_paths for state in path}

    pwa_model = build_pwa_model(A, tas, sp, sys_sim)

    bmc = BMC_factory('sal')
    bmc.init(A.num_dims.x, pwa_model, sp.init_cons, sp.final_cons, 'vdp')
    bmc.check()
    #bmc.check(depth=2)
    #if bmc.sat:
    #    return bmc.soln
    #else:
    #    return None


def BMC_factory(bmc_engine='sal'):
    if bmc_engine == 'sal':
        import salbmc as sbmc
        return SALBMC(sbmc)
    else:
        raise NotImplementedError


class SALBMC(object):
    def __init__(self, sbmc):
        self.sbmc = sbmc

    def init(self, nd, pwa_model, init_state, safety_prop, sys_name):
        self.sbmc = self.sbmc.BMC(nd, pwa_model, init_state, safety_prop, sys_name)
        return

    def check(self):
        self.sbmc.dump()
        self.sbmc.check(3)
        return




def build_pwa_model(AA, abs_states, sp, sys_sim):
    dt = AA.plant_abs.delta_t
    step_sim = simsys.get_step_simulator(sp.controller_sim, sp.plant_sim, dt)

    abs_state_models = {}
    M = pwa.PWA()

    for abs_state in abs_states:
        #ic = AA.plant_abs.get_ival_cons_abs_state(abs_state)
        #sampled_states = sample.sample_ival_constraints(ic, AA.num_samples)
        state_samples = sp.sampler.sample(abs_state, AA, sp, AA.num_samples*MORE_FACTOR)
        d = abs_state.plant_state.d
        pvt = abs_state.plant_state.pvt

        Yl = []

        for s, x, ci, pi, t in state_samples.iterable():
            (t_, x_, s_, d_, pvt_, u_) = step_sim(dt, x, s, d, pvt, ci, pi)
            Yl.append(x_)
            #trace = sys_sim(x, s, d, 0, t0, t0 + dt, ci, pi)

        Y = np.vstack(Yl)
        X = state_samples.x_array

        A, b = linear_model(X, Y)
        C, d = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).poly()

        sub_model = pwa.PWA.sub_model(A, b, C, d)
        M.add(sub_model)
        abs_state_models[abs_state] = sub_model

    return M


# based on scipy.linalg.lstsq
# TODO: directly use lstsq
def linear_model(x, y):
    clf = skl_lm.LinearRegression(normalize=True)
    clf.fit(x, y)
    #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    #print clf.coef_, clf.intercept_
    return clf.coef_, clf.intercept_

