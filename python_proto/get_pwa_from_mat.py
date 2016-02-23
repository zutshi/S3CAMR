import argparse
import scipy.io as sio

import scipy as sp
import pickle

import pwa
import fileOps as fops

#sp.set_printoptions(suppress=True, precision=2)

#MODEL_MAT = '../examples/vdp/vdp_model_data_x_1e6.mat'
#PICKLE_FNAME = '../examples/vdp/pickled_model'


def populate(model_mat):
    model_data = sio.loadmat(model_mat)
    # flattened model is stored as [(P0,M0),...(Pi, Mi),...(Pn,Mn)]
    fm = model_data['fm']
    pwa_rep = pwa.PWA()

    # fm should be a row vector
    # for everey sub_model in the system model
    for i in range(fm.shape[0]):
        partition = fm[i, 0]['P'][0, 0]
        model = fm[i, 0]['M'][0, 0]
        p = pwa.Partition(partition['A'][0, 0], partition['b'][0, 0])
        m = pwa.Model(model['A'][0, 0], model['b'][0, 0])
        pwa_rep.add_sub_model(p, m)
        #print fm[pi, 0]['P'][0, 0]['b'][0, 0]

    return pwa_rep


def test_pickling(pwa_rep, pickle_fname):
    pickled_pwa_w = pickle.dumps(pwa_rep, pickle.HIGHEST_PROTOCOL)
    fops.write_data(pickle_fname, pickled_pwa_w)
    pickled_pwa_r = fops.get_data(pickle_fname)
    return pickle.loads(pickled_pwa_r)


def main(model_mat, pickle_fname):
    pwa_rep = populate(model_mat)
    #print pwa_rep
    pwa_rep = test_pickling(pwa_rep, pickle_fname)
    print pwa_rep

if __name__ == '__main__':
    usage = '%(prog)s <filename>'
    parser = argparse.ArgumentParser(description='demo pwa', usage=usage)
    parser.add_argument('mat_file', default=None, type=str)
    parser.add_argument('-o', default=None, metavar='pickle', type=str, required=True)
    args = parser.parse_args()
    main(args.mat_file, args.o)
