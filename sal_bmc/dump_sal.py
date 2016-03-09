import numpy as np
import pickle
import argparse

import saltrans as st
import fileops as fops

PICKLE_FNAME = '../python_proto/pickled_model'
np.set_printoptions(suppress=True, precision=2)


def get_pwa_model(pickle_file):
    pickled_pwa = fops.get_data(pickle_file)
    return pickle.loads(pickled_pwa)


def pwa2sal(pwa_rep):
    sal_trans_sys = st.SALTransSys()
    for sub_model in pwa_rep:
        g = st.Guard(sub_model.p.C, sub_model.p.d)
        r = st.Reset(sub_model.m.A, sub_model.m.b)
        t = st.Transition(g, r)
        sal_trans_sys.add_transition(t)
    return sal_trans_sys


def main(pickle_file, sal_file):
    prop = 'G(x0>=1)'
    pwa_rep = get_pwa_model(pickle_file)
    sal_trans_sys = pwa2sal(pwa_rep)
    sal_trans_sys.add_prop(prop)
    fops.write_data(sal_file, str(sal_trans_sys))


if __name__ == '__main__':
    usage = '%(prog)s <filename>'
    parser = argparse.ArgumentParser(description='demo bmc', usage=usage)
    parser.add_argument('pickle_file', default=None, type=str)
    parser.add_argument('-o', default=None, metavar='sal_file', type=str, required=True)
    args = parser.parse_args()
    main(args.pickle_file, args.o)
