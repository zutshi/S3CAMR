# Low memory graph NX
# Mantains an internal dict and follows an igraph/graph-tool
# methodology to create a graph of integers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#global nx

import ast
import subprocess
import logging

import networkx as nx
from heapq import heappush, heappop
from itertools import count
from blessed import Terminal

from .graphNX import GraphNX

import utils as U
import err

#from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot

import settings

logger = logging.getLogger(__name__)

GVIZ_GRAPH_PATH = '{}_graph.dot'
GRAPH_SVG_PATH = '{}_graph.svg'
MPLIB_GRAPH_PATH = '{}_graph.png'

term = Terminal()


class GraphNXLM(object):
    def __init__(self):

        self.G = GraphNX()
        self.ID2n = {}
        self.n2ID = {}
        self.ID_ctr = 0

    def genID(self):
        self.ID_ctr += 1
        return self.ID_ctr

    def num_nodes(self):
        return self.G.num_nodes()

    def num_edges(self):
        return self.G.num_edges()

    def nodes(self):
        return [self.ID2n[nID] for nID in self.G.nodes()]

    def node_attrs(self, node):
        nID = self.n2ID[node]
        return self.G.node_attrs(nID)

    def edge_attrs(self, edge):
        n1, n2 = edge
        n1ID, n2ID = self.n2ID[n1], self.n2ID[n2]
        return self.G.edge_attrs((n1ID, n2ID))

    def edges(self, node):
        """Get edges of a node """
        nID = self.n2ID[node]
        return [(self.ID2n[n1ID], self.ID2n[n2ID]) for (n1ID, n2ID) in self.G.edges(nID)]

    def all_edges(self):
        """Get all edges """
        return [(self.ID2n[n1ID], self.ID2n[n2ID]) for (n1ID, n2ID) in self.G.all_edges()]

    def nodes_iter(self):
        return (self.ID2n[nID] for nID in self.G.nodes_iter())

    def new_node(self, n):
        nID = self.genID()
        self.n2ID[n] = nID
        self.ID2n[nID] = n
        return nID

    # TODO: An edge b/w two states stores a unique value of ci and pi.
    # In other words, if x -> x', only the last discovered ci/pi is
    # stored.
    def add_edge(self, n1, n2, **attr_dict_arg):#ci=None, pi=None, weight=1):
#         nID1 = self.n2ID[n1]
#         nID2 = self.n2ID[n2]
        
        n1ID = self.n2ID[n1] if n1 in self.n2ID else self.new_node(n1)
        n2ID = self.n2ID[n2] if n2 in self.n2ID else self.new_node(n2)

#         nID1 = self.n2ID.setdefault(n1, self.genID())
#         nID2 = self.n2ID.setdefault(n2, self.genID())

        if self.G.has_edge(n1ID, n2ID):
            return

        return self.G.add_edge(n1ID, n2ID, **attr_dict_arg)

    def add_edges_from(self, edge_list, **attr_dict_arg):#ci=None, pi=None, weight=1):
        for edge in edge_list:
            self.add_edge(*edge, **attr_dict_arg)

    def add_node(self, node, **attrs):
        #nID = self.n2ID.setdefault(node, self.genID())
        nID = self.n2ID[node] if node in self.n2ID else self.new_node(node)
        return self.G.add_node(nID, **attrs)

    def has_edge(self, n1, n2):
        has_edge = False
        if self.has_node(n1) and self.has_node(n2):
            n1ID, n2ID = self.n2ID[n1], self.n2ID[n2]
            has_edge = self.G.has_edge(n1ID, n2ID)
        return has_edge

    def has_node(self, node):
        return node in self.n2ID

    def get_path_attr_list(self, path, attrs):
        raise NotImplementedError

    def dump(self, *args):
        return self.G.dump(*args)

    def subgraph_source2target(self, sources, targets):
        sources_ = [self.n2ID[n] for n in sources]
        targets_ = [self.n2ID[n] for n in targets]

        nxG = self.G.subgraph_source2target(sources_, targets_)
        
        G_ = self.__class__()
        G_.G = nxG
        G_.ID2n = self.ID2n
        G_.n2ID = self.n2ID
        G_.ID_ctr = self.ID_ctr
        return G_

    def k_shortest_paths( self, G, source, target, k=1, weight='weight',):
        raise NotImplementedError

    def get_all_path_generator(self, sources, sinks):
        sources_ = [self.n2ID[n] for n in sources]
        sinks_ = [self.n2ID[n] for n in sinks]
        path_gen = self.G.get_all_path_generator(sources_, sinks_)
        return ([self.ID2n[nID] for nID in p] for p in path_gen)

    def get_path_generator(self, sources, sinks, max_depth, max_paths):
        sources_ = [self.n2ID[n] for n in sources]
        sinks_ = [self.n2ID[n] for n in sinks]
        path_gen = self.G.get_path_generator(sources_, sinks_)
        #return (self.ID2n[nID] for p in path_gen for nID in p)
        return ([self.ID2n[nID] for nID in p] for p in path_gen)


    # TODO: Add this method to the abstract graph class
    def __iter__(self):
        return (self.ID2n[nID] for nID in self.G.__iter__())

    def neighbors(self, node):
        nID = self.n2ID[node]
        return [self.ID2n[nID_] for nID_ in self.G.neighbors(nID)]

    def out_degree(self, node):
        nID = self.n2ID[node]
        return self.G.out_degree(nID)

    def iteredges(self):
        return ((self.ID2n[n1ID], self.ID2n[n2ID]) for (n1ID,n2ID) in self.G.edges_iter())

    def draw(self, *args):
        return self.G.draw(*args)

    def draw_graphviz(self, *args):
        return self.G.draw_graphviz(*args)

    def draw_mplib(self, *args):
        return self.G.draw_graphviz(*args)

    def __contains__(self, node):
        return self.n2ID[node] in self.G

    def __repr__(self):
        return self.G.__repr__()
