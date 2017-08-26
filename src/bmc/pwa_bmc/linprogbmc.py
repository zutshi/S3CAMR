from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from bmc.bmc_spec import BMCSpec, InvarStatus, TraceSimple
from bmc.sal_bmc.pwa2salconverter import PWATRACE
from pwa import pwagraph
from graphs.graph import class_factory as graph_class
from globalopts import opts as gopts

#import fileops as fops
import utils as U
import err

from blessed import Terminal
term = Terminal()

logger = logging.getLogger(__name__)


if gopts.model_type == 'affine':
    import pwa.analyzepath as azp
elif gopts.model_type == 'poly':
    import pwa.analyzepathnl as azp
else:
    raise NotImplementedError


class BMC(BMCSpec):
    def __init__(self, sys, prop, vs, pwa_model, init_state, safety_prop,
                 init_partitions, prop_partitions, fname_constructor,
                 module_name, model_type, *args):

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
        self.pwa_trace = None

        return None

    def trace_generator(self, depth):
        # TODO: make the graph search depth bounded?
        err.warn('ignoring depth')
        path_gen = self.pwa_model.get_all_path_generator(self.sources, self.targets)
        path_ctr = 0
        for path in path_gen:
            path_ctr += 1
            ptrace = [self.pwa_model.node_p(qi) for qi in path]
            mtrace = [self.pwa_model.edge_m((qi, qj)) for qi, qj in U.pairwise(path)]
            pwa_trace = PWATRACE(partitions=ptrace, models=mtrace)
            x_array = azp.feasible(self.num_dims, self.prop, pwa_trace)
            if x_array is not None:
                concrete_trace = ConcreteTrace(x_array, pwa_trace)
                print('Model Found')
                yield concrete_trace, pwa_trace
        print('Total paths checked: {}'.format(path_ctr))
        return

    def compute_next_trace(self):
        try:
            next(self.CE_gen)
        except StopIteration:
            pass
        return None

    def check(self, *args):
        if self.CE_gen is None:
            self.CE_gen = self.get_CE_gen()
            self.compute_next_trace()
            return InvarStatus.Safe if self.pwa_trace is None else InvarStatus.Unsafe
        else:
            raise err.Fatal('check should be called only once!')

    def get_trace(self):
        return self.trace
        # can fake a bmc trace by sapling and simulation if required
        #return self.last_trace

    def gen_new_disc_trace(self):
        self.compute_next_trace()

    def get_pwa_trace(self):
        return self.pwa_trace

    #def get_last_X0(self):
    #    return self.last_X0

    def get_CE_gen(self):
        path_gen = self.pwa_model.get_all_path_generator(self.sources, self.targets)
        for path in path_gen:
            ptrace = [self.pwa_model.node_p(qi) for qi in path]
            mtrace = [self.pwa_model.edge_m((qi, qj)) for qi, qj in U.pairwise(path)]
            pwa_trace = PWATRACE(partitions=ptrace, models=mtrace)
            x_array = azp.feasible(self.num_dims, self.prop, pwa_trace)
            if x_array is not None:
                self.trace = ConcreteTrace(x_array, pwa_trace)
                print('Model Found')
                #ret_val = azp.overapprox_x0(self.num_dims, self.prop, pwa_trace)
                #self.pwa_trace, self.X0 = pwa_trace, ret_val
                self.pwa_trace = pwa_trace
                yield
            else:
                #self.tace, self.pwa_trace, self.X0 = None, None, None
                self.trace, self.pwa_trace = None, None
        return

    def print_all_CE(self, d):
        # XXX:Works OK...but remove in the future
        raise NotImplementedError

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
            x_array = azp.feasible(self.num_dims, self.prop, pwa_trace)
            #ret_val = azp.overapprox_x0(self.num_dims, self.prop, pwa_trace)
            #if ret_val is not None:
            if x_array is not None:
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


class ConcreteTrace(TraceSimple):
    """Simple Trace: provides minimal functionality"""

    def __init__(self, x_array, pwa_trace):
        self.x_array = x_array
        self.pwa_trace = pwa_trace

    def to_array(self):
        return self.x_array

    def __getitem__(self, idx):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
