import scipy.io as sio
import pickle

import pwa
import fileOps as fops

MODEL_MAT = '../matlab_proto/vdp_model_data_x_1e6.mat'
PICKLE_FNAME = 'pickled_model'


def populate():
    model_data = sio.loadmat(MODEL_MAT)
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


def test_pickling(pwa_rep):
    pickled_pwa_w = pickle.dumps(pwa_rep, pickle.HIGHEST_PROTOCOL)
    fops.write_data(PICKLE_FNAME, pickled_pwa_w)
    pickled_pwa_r = fops.get_data(PICKLE_FNAME)
    return pickle.loads(pickled_pwa_r)


def main():
    pwa_rep = populate()
    #print pwa_rep
    pwa_rep = test_pickling(pwa_rep)
    print pwa_rep

if __name__ == '__main__':
    main()
