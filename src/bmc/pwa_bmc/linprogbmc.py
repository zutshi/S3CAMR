from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from bmc.bmc_spec import BMCSpec, InvarStatus, TraceSimple
from bmc.sal_bmc.pwa2salconverter import PWATRACE
import pwa.analyzepath as azp
from pwa import pwagraph
from graphs.graph import class_factory as graph_class
from globalopts import opts as gopts

#import fileops as fops
import utils as U
import err

from blessed import Terminal
term = Terminal()

logger = logging.getLogger(__name__)


class BMC(BMCSpec):
    def __init__(self, sys, prop, vs, pwa_model, init_state, safety_prop,
                 init_partitions, prop_partitions, fname_constructor,
                 module_name, model_type):

        assert(isinstance(pwa_model, pwagraph.PWAGraph))
        assert(model_type == 'dft')

        self.trace = None
        self.vs = vs
        self.sources = {p.ID for p in init_partitions}
        self.targets = {p.ID for p in prop_partitions}
        self.pwa_model = pwa_model
        self.num_dims = sys.num_dims
        self.prop = prop

        self.CE_gen = None
        #self.last_trace = None
        self.last_pwa_trace = None
        self.last_X0 = None

        return None

    def check(self, *args):
        if self.CE_gen is None:
            self.CE_gen = self.get_CE_gen()
            # check
            self.last_pwa_trace, self.last_X0 = next(self.CE_gen)
            return InvarStatus.Safe if self.last_pwa_trace is None else InvarStatus.Unsafe
        else:
            raise err.Fatal('check should be called only once!')

    def get_trace(self):
        raise NotImplementedError
        return None
        # can fake a bmc trace by sapling and simulation if required
        #return self.last_trace

    def get_new_disc_trace(self):
        raise NotImplementedError
        # check
        self.last_pwa_trace, self.last_X0 = next(self.CE_gen)
        return None

    def get_pwa_trace(self):
        return Trace(self.pwa_trace)

    #def get_last_X0(self):
    #    return self.last_X0

    def get_CE_gen(self):
        path_gen = self.pwa_model.get_all_path_generator(self.sources, self.targets)
        for path in path_gen:
            ptrace = [self.pwa_model.node_p(qi) for qi in path]
            mtrace = [self.pwa_model.edge_m((qi, qj)) for qi, qj in U.pairwise(path)]
            pwa_trace = PWATRACE(partitions=ptrace, models=mtrace)
            #TODO: replace with feasible()
            ret_val = azp.overapprox_x0(self.num_dims, self.prop, pwa_trace)
            if ret_val is not None:
                print('Model Found')
                yield pwa_trace, ret_val
        return

    def print_all_CE(self, d):
        path_gen = self.pwa_model.get_all_path_generator(self.sources, self.targets)

        print('')
        print(term.move_up + term.move_up)
        print('checking models: ', end='')

        for ctr, path in enumerate(path_gen):

            with term.location():
                print(ctr)

            ptrace = [self.pwa_model.node_p(qi) for qi in path]
            mtrace = [self.pwa_model.edge_m((qi, qj)) for qi, qj in U.pairwise(path)]
            pwa_trace = PWATRACE(partitions=ptrace, models=mtrace)
            #TODO: replace with feasible()
            ret_val = azp.overapprox_x0(self.num_dims, self.prop, pwa_trace)
            if ret_val is not None:
                print('Model Found: {}'.format(d))
                U.pause()
                qgraph = self.refine_CE(pwa_trace)
                if qgraph is None:
                    print('refinement failed to build a graph, moving to new CE')
                    continue
                else:
                    yield qgraph
        print('total paths checked:{}'.format(ctr))
        return

    def refine_edge(self, qi, qj):
        pass

    #TODO: Why make an explicit graph !!
    def refine_CE(self, pwa_trace):
        qgraph = QGraph()
        for pi, pj in U.pairwise(pwa_trace.partitions):
            qi, qj = pi.ID, pj.ID
            qi_splits = qi.split()
            qj_splits = qj.split()

            # update init and final
            if qi in self.sources:
                qgraph.init.update(qi_splits)
            if qj in self.targets:
                qgraph.final.update(qj_splits)

            # add edges
            for qik in qi_splits:
                for qjk in qj_splits:
                    assert(not qgraph.has_edge(qik, qjk))
                    qgraph.add_edge(qik, qjk)

        return qgraph


class QGraph(graph_class(gopts.graph_lib)):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.init = set()
        self.final = set()


class Trace(TraceSimple):
    """Simple Trace: provides minimal functionality"""

    def __init__(self, pwa_trace):
        self.pwa_trace = pwa_trace
        partitions models

    def __getitem__(self, idx):
        raise NotImplementedError
        return

    def __iter__(self):
        raise NotImplementedError
        return

    def to_array(self):
        return None

    def __len__(self):
        raise NotImplementedError
        return

    def __str__(self):
        raise NotImplementedError
        return
