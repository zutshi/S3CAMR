from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import abc
import logging
from IPython import embed

from . import modelspec
from . import relational as R

from scam_core.graphs.graph import class_factory as graph_class
#from .graphs.graph import factory as graph_factory
from globalopts import opts as gopts


# TODO: ideal solution: derive the class PWAGraph form the appropriate
# gopts.graph_lib. Better than storing an object G
# It can be done by storing the class in gopts.graph_lib, instead of
# the class string and using a factory method to return the class and
# not an object of the class.
class PWAGraph(graph_class(gopts.graph_lib)):
    def __init__(self):
        super(self.__class__, self).__init__()

    def add_relation(self, p1, p2, m):
        #assert(not self.has_node(p1.ID))
        #assert(not self.has_node(p2.ID))
        if not self.G.has_node(p1.ID):
            self.add_node(p1.ID, p=p1)
        if not self.has_node(p2.ID):
            self.add_node(p2.ID, p=p2)
        # sm.p -> pi
        try:
            assert(not self.has_edge(p1.ID, p2.ID))
        except AssertionError:
            print('assert(not self.has_edge(p1.ID, p2.ID)) is false')
            embed()

        self.add_edge(p1.ID, p2.ID, m=m)

    def node_p(self, n):
        return self.node_attrs(n)['p']

    def edge_m(self, e):
        return self.edge_attrs(e)['m']


class SubModel(modelspec.PartitionedDiscreteAffineModel):
    pass


class Partition(modelspec.Partition):
    pass # calls super


class DiscreteAffineMap(modelspec.DiscreteAffineMap):
    pass # calls super

from modeling.pwa import pwa
def recurse_add(q_split_map, pwa_graph, pi, pj, m):

    pwa_graph.add_relation(pi, pj, m)

    qj = pj.ID
    for qjk in q_split_map[qj]:
        recurse_add(q_split_map, pwa_graph,
                    pi,
                    pwa.Partition(*qjk.ival_constraints.poly(), part_id=qjk),
                    m)

# ignores the constraints on pnexts: p_future
def convert_pwarel2pwagraph(pwa_rel_model):
    assert(isinstance(pwa_rel_model, R.PWARelational))
    from scam_core.modelrefine import q_split_map

    pwa_graph = PWAGraph()

    for sm in pwa_rel_model:
        assert(isinstance(sm, R.KPath))
#         for pi in sm.pnexts:
#             pwa_graph.add_relation(sm.p, pi, sm.m)
        # there should be only one pnext due to 1-relational modeling
        assert(len(sm.pnexts) == 1)
        pnext = sm.pnexts[0]
        #pwa_graph.add_relation(sm.p, pnext, sm.m)

        qi = pnext.ID
        #pi = pwa.Partition(*qjk.ival_constraints.poly(), part_id=qjk)
        recurse_add(q_split_map, pwa_graph, sm.p, pnext, sm.m)

    return pwa_graph

# ignores the constraints on pnexts: p_future
def convert_pwarel2pwagraph_old(pwa_rel_model):
    assert(isinstance(pwa_rel_model, R.PWARelational))

    pwa_graph = PWAGraph()

    for sm in pwa_rel_model:
        assert(isinstance(sm, R.KPath))
        assert(len(sm.pnexts) >= 1)
        for pi in sm.pnexts:
            pwa_graph.add_relation(sm.p, pi, sm.m)

    return pwa_graph
