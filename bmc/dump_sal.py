import numpy as np
import pickle

import saltrans as st
import fileOps as fops

PICKLE_FNAME = '../python_proto/pickled_model'
np.set_printoptions(suppress=True, precision=2)

def get_pwa_model():
    pickled_pwa = fops.get_data(PICKLE_FNAME)
    return pickle.loads(pickled_pwa)


def pwa2sal(pwa_rep):
    sal_trans_sys = st.SALTransSys()
    for sub_model in pwa_rep:
        g = st.Guard(sub_model.p.C, sub_model.p.d)
        r = st.Reset(sub_model.m.A, sub_model.m.b)
        t = st.Transition(g, r)
        sal_trans_sys.add_transition(t)
    return sal_trans_sys


def main():
    prop = 'G(x0>=1)'
    pwa_rep = get_pwa_model()
    sal_trans_sys = pwa2sal(pwa_rep)
    sal_trans_sys.add_prop(prop)
    print sal_trans_sys


if __name__ == '__main__':
    main()
