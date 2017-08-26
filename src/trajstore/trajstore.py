# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#external deps
from IPython import embed
import numpy as np

#core deps
#import logging
import collections
import itertools as it

# pyutils
from utils import print_function
import utils as U

# internal deps
from globalopts import opts as gopts
from core import cellmanager as CM


class Data(object):
    def __init__(self):
        self.X = []
        self.X_ = []


def wrap_sim(sim_fn, ts):
    def sim_wrapper(concrete_states, *args):
        cs = concrete_states
        cs_, pvf = sim_fn(cs, *args)
        for x, x_ in zip(cs.cont_states, cs_.cont_states):
            ts.add_traj(x, x_)
        return cs_, pvf
    return sim_wrapper



class TrajStore(object):
    def __init__(self, eps, sys):
        self.num_dims = sys.num_dims
        # doesn't handle inputs for now
        if sys.num_dims.pi != 0:
            raise NotImplementedError
        self.eps = eps
        self.rel2traj = collections.defaultdict(Data)

    def add_traj(self, x, x_):
        """ 
        cs is concrete state object of Class core.state.State, 
        such that:
        cs_ = sim(cs, dt)
        """
        
        c = CM.cell_from_concrete(x, self.eps)
        c_ = CM.cell_from_concrete(x_, self.eps)
        
        # for now, just use source cell as it is easier to integrate
        # with existing code in cellmodels.py
        #relation = tuple(it.chain(c, c_))
        relation = c

        self.rel2traj[relation].X.append(x)
        self.rel2traj[relation].X_.append(x_)

    def get_traj(self, c):#, c_):
        # for now, just use source cell as it is easier to integrate
        # with existing code in cellmodels.py
        #relation = tuple(it.zip(c, c_))

        relation = c

        data = self.rel2traj[relation]
        if data.X:
            return np.stack(data.X), np.stack(data.X_)
        else:
            return np.empty((0, self.num_dims.x)), np.empty((0, self.num_dims.x))
