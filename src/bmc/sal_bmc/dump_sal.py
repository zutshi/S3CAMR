from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pickle
import argparse

import saltrans as st
import fileops as fops

import constraints as C

PICKLE_FNAME = '../python_proto/pickled_model'


def get_pwa_model(pickle_file):
    pickled_pwa = fops.get_data(pickle_file)
    return pickle.loads(pickled_pwa)


# vsp example
def vdp_pwa2sal(pwa_rep, module_name):
    init_set = C.IntervalCons(np.array([-0.4, -0.4]), np.array([0.4, 0.4]))
    prop = C.IntervalCons(np.array([-1, -6.5]), np.array([-0.7, -5.6]))
    sal_trans_sys = st.SALTransSys(module_name, 2, init_set, prop)

    for sub_model in pwa_rep:
        g = st.Guard(sub_model.p.C, sub_model.p.d)
        r = st.Reset(sub_model.m.A, sub_model.m.b)
        t = st.Transition(g, r)
        sal_trans_sys.add_transition(t)
    return sal_trans_sys


def main(pickle_file, sal_module_name):
    pwa_rep = get_pwa_model(pickle_file)
    sal_trans_sys = vdp_pwa2sal(pwa_rep, sal_module_name)
    sal_file = sal_module_name + '.sal'
    fops.write_data(sal_file, str(sal_trans_sys))


if __name__ == '__main__':
    usage = '%(prog)s <input> -o <output>'
    parser = argparse.ArgumentParser(description='demo bmc', usage=usage)
    parser.add_argument('pickle_file', default=None, type=str, help='pwa pickle input')
    parser.add_argument('-o', default=None, metavar='sal_file', type=str, required=True, help='name of sal module, no ext!')
    args = parser.parse_args()
    main(args.pickle_file, args.o)
