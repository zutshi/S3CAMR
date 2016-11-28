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
from collections import defaultdict
from blessed import Terminal


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


class GraphNX(object):

    @staticmethod
    def compare(G1, G2):
        G1 = G1.G
        G2 = G2.G

        G1_nodes_set = set(G1.nodes())
        G2_nodes_set = set(G2.nodes())

        G1_edges_set = set(G1.edges())
        G2_edges_set = set(G2.edges())

        G1_in_G2_nodes = G1_nodes_set.issubset(G2_nodes_set)
        G2_in_G1_nodes = G2_nodes_set.issubset(G1_nodes_set)

        G1_in_G2_edges = G1_edges_set.issubset(G2_edges_set)
        G2_in_G1_edges = G2_edges_set.issubset(G1_edges_set)

        G1_in_G2 = G1_in_G2_nodes and G1_in_G2_edges
        G2_in_G1 = G2_in_G1_nodes and G2_in_G1_edges

        print('G1_in_G2_nodes: {}, G1_in_G2_edges: {}'.format(G1_in_G2_nodes, G1_in_G2_edges))
        print('G2_in_G1_nodes: {}, G2_in_G1_edges: {}'.format(G2_in_G1_nodes, G2_in_G1_edges))

        print('G1_nodes_set - G2_nodes_set: {}').format(G1_nodes_set - G2_nodes_set)

        G1_and_G2_are_equal = G1_in_G2 and G2_in_G1

        print('G1_in_G2: {}, G2_in_G1: {}\n'.format(G1_in_G2, G2_in_G1))

        return G1_and_G2_are_equal

    @staticmethod
    def compare_nodes(n1, n2):
        raise NotImplementedError

    @staticmethod
    def compare_edges():
        raise NotImplementedError

    def __init__(self, G=None, Type=None):

        # unused maxVert

        self.maxVertices = 0

        self.ctr = 0

        # create a Di-graph if not created already

        if G is None:

            self.G = nx.DiGraph()
            self.Type = 'test_no'
        else:
            self.G = G
            self.Type = Type

    def num_nodes(self):
        return self.G.number_of_nodes()

    def num_edges(self):
        return self.G.number_of_edges()

    def nodes(self):
        return self.G.nodes()

    def node_attrs(self, node):
        return self.G.node[node]

    def edge_attrs(self, edge):
        return self.G.get_edge_data(*edge)

    def edges(self, node):
        """Get edges of a node
        """
        return self.G.edges(node)

    def all_edges(self):
        """Get edges of a node
        """
        return self.G.edges()

    def nodes_iter(self):
        return self.G.nodes_iter()

    # TODO: An edge b/w two states stores a unique value of ci and pi.
    # In other words, if x -> x', only the last discovered ci/pi is
    # stored.
    def add_edge(self, v1, v2, **attr_dict_arg):#ci=None, pi=None, weight=1):
        if self.G.has_edge(v1, v2):
            return
        #TODO: remove default attrs
        attrs = {'ci': None, 'pi': None, 'weight': 1}
        #print('nx:', v1, v2)
        attrs.update(attr_dict_arg)

#         if __debug__:
#             if self.G.has_edge(v1, v2):
#                 err.warn('overwriting an edge')

        self.G.add_edge(v1, v2, attrs)

        self.ctr += 1

        if self.ctr % 1000 == 0:
            with term.location(x=100, y=term.height-10):
                print(term.green('nodes={}, edges={}'
                                 .format(
                                    self.G.number_of_nodes(),
                                    self.G.number_of_edges())))

    def add_edges_from(self, edge_list, **attr_dict_arg):#ci=None, pi=None, weight=1):
        attrs = {'ci': None, 'pi': None, 'weight': 1}
        attrs.update(attr_dict_arg)
        self.G.add_edges_from(edge_list, attrs)

    def add_node(self, v, **attrs):
        self.G.add_node(v, attrs)

    def has_edge(self, u, v):
        return self.G.has_edge(u, v)

    def has_node(self, v):
        return self.G.has_node(v)

    def get_path_attr_list(self, path, attrs):
        attr_map = defaultdict(list)
        for (v1, v2) in U.pairwise(path):
            for attr in attrs:
                attr_map[attr].append(self.G[v1][v2][attr])
        return attr_map

    def dump(self, fname):
        print('saving graph')
        nx.write_gpickle(self.G, fname)
        return

    def multiple2single_st(self, sources, targets):
        # Create a shallow copy of the graph
        H = nx.DiGraph(self.G)

        # All modifications are now done on this shallow copy H

        # Define super source and sink nodes
        # A Super source node has a directed edge to each source node in the
        # source_list
        # Similarily, a Super sink node has a directed edge from each sink node
        # in the sink_list

        dummy_super_source_node = 'source'
        dummy_super_target_node = 'sink'

        for s in sources:
            H.add_edge(dummy_super_source_node, s)
        for t in targets:
            H.add_edge(t, dummy_super_target_node)

        return H, dummy_super_source_node, dummy_super_target_node

    def subgraph_source2target(self, sources, targets):
        H, S, T = self.multiple2single_st(sources, targets)
        G = H.subgraph(nx.descendants(H, S) & nx.ancestors(H, T))
        return self.__class__(G)

    # ###################### KSP 1 ##################################################
    # https://gist.github.com/guilhermemm/d4623c574d4bccb6bf0c
    # __author__ = 'Guilherme Maia <guilhermemm@gmail.com>'
    # __all__ = ['k_shortest_paths']

    def k_shortest_paths(
        self,
        G,
        source,
        target,
        k=1,
        weight='weight',
        ):
        """Returns the k-shortest paths from source to target in a weighted graph G.

        Parameters
        ----------
        G : NetworkX graph

        source : node
           Starting node

        target : node
           Ending node

        k : integer, optional (default=1)
            The number of shortest paths to find

        weight: string, optional (default='weight')
           Edge data key corresponding to the edge weight

        Returns
        -------
        lengths, paths : lists
           Returns a tuple with two lists.
           The first list stores the length of each k-shortest path.
           The second list stores each k-shortest path.

        Raises
        ------
        NetworkXNoPath
           If no path exists between source and target.

        Examples
        --------
        >>> G=nx.complete_graph(5)
        >>> print(k_shortest_paths(G, 0, 4, 4))
        ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])

        Notes
        ------
        Edge weight attributes must be numerical and non-negative.
        Distances are calculated as sums of weighted edges traversed.

        """

        if source == target:
            return ([0], [[source]])

        (length, path) = nx.single_source_dijkstra(G, source, target,
                weight=weight)
        if target not in length:
            raise nx.NetworkXNoPath('node %s not reachable from %s' % (source,
                                    target))

        lengths = [length[target]]
        paths = [path[target]]
        c = count()
        B = []

        # Is deep copy really required?
        #   Fails due to embedded Ctype objects which can not be pickled
        # # G_original = G.copy()
        # Swapping with shallow copy...will it work?

        G_original = G
        if nx.is_directed(G_original):
            G = nx.DiGraph(G_original)
        else:
            G = nx.Graph(G_original)

        ######################################
        #TODO: wrap this up somehow
        print('')
        print(term.move_up + term.move_up)
        ######################################
        print('getting K:{} paths...'.format(k), end='')
        for i in range(1, k):
            with term.location():
                print(i)
            for j in range(len(paths[-1]) - 1):
                spur_node = paths[-1][j]
                root_path = (paths[-1])[:j + 1]

                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G.has_edge(u, v):
                            edge_attr = G.edge[u][v]
                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))

                for n in range(len(root_path) - 1):
                    node = root_path[n]

                    # out-edges

                    for (u, v, edge_attr) in G.edges_iter(node, data=True):

                        # print('lala1: {} -> {}'.format(u,v))

                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

                    if G.is_directed():

                        # in-edges

                        for (u, v, edge_attr) in G.in_edges_iter(node,
                                data=True):

                            # print('lala2: {} -> {}'.format(u,v))

                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))

                (spur_path_length, spur_path) = nx.single_source_dijkstra(G,
                        spur_node, target, weight=weight)
                if target in spur_path and spur_path[target]:
                    total_path = root_path[:-1] + spur_path[target]
                    total_path_length = self.get_path_length(G_original,
                            root_path, weight) + spur_path_length[target]
                    heappush(B, (total_path_length, next(c), total_path))

                for e in edges_removed:
                    (u, v, edge_attr) = e
                    G.add_edge(u, v, edge_attr)

            if B:
                (l, _, p) = heappop(B)
                lengths.append(l)
                paths.append(p)
            else:
                break

        return (lengths, paths)

    def get_path_length(
        self,
        G,
        path,
        weight='weight',
        ):
        length = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]

                length += G.edge[u][v].get(weight, 1)

        return length

    # ################################### KSP2 ###########################
    # ######################## GREG BERNSTEIN #######################
    # https://groups.google.com/forum/#!topic/networkx-discuss/0niVmZZXxGA

    def ksp_gregBern(
        self,
        G,
        source,
        target,
        k=1,
        weight='weight',
        ):
        YKSP = YenKShortestPaths(G, weight)
        path_list = []

        wp0 = YKSP.findFirstShortestPath(source, target)
        p0 = wp0.nodeList
        path_list.append(p0)

        print('getting K:{} paths...'.format(k))
        for i in range(k):
            wpi = YKSP.getNextShortestPath()
            if wpi is None:
                break
            pi = wpi.nodeList
            path_list.append(pi)
            print(i, ' ')

        # print(path_list)

        return path_list

    # ################################### KSP END ###########################

    def get_all_path_generator(self, source_list, sink_list):

        # Create a shallow copy of the graph
        H = nx.DiGraph(self.G)

        # All modifications are now done on this shallow copy H

        # Define super source and sink nodes
        # A Super source node has a directed edge to each source node in the
        # source_list
        # Similarily, a Super sink node has a directed edge from each sink node
        # in the sink_list

        dummy_super_source_node = 'source'
        dummy_super_sink_node = 'sink'
        num_source_nodes = len(source_list)
        num_sink_nodes = len(sink_list)

        # Add edges:
        #   \forall source \in source_list. super source node -> source

        edge_list = zip([dummy_super_source_node] * num_source_nodes,
                        source_list)

        H.add_edges_from(edge_list, weight=1)

        logger.debug('source -> list')
        for e in edge_list:
            logger.debug(e)

        # Add edges:
        #   \forall sink \in sink_list. sink -> super sink node

        edge_list = zip(sink_list, [dummy_super_sink_node] * num_sink_nodes)
        H.add_edges_from(edge_list, weight=1)

        logger.debug('sink -> list')
        for e in edge_list:
            logger.debug(e)

        def path_gen():
            # Remove the first (super source)
            # and the last element (super sink)
            logger.debug('======== Paths =========')
            for p in nx.all_simple_paths(H, dummy_super_source_node, dummy_super_sink_node):
                logger.debug(p[1:-1])
                yield p[1:-1]

        # return lambda: [yield i[1:-1] for i in nx.all_simple_paths(H,
        # dummy_super_source_node, dummy_super_sink_node)]

        return path_gen()

    def get_path_generator(
            self,
            source_list,
            sink_list,
            max_depth,
            max_paths
            ):

        # Create a shallow copy of the graph

        H = nx.DiGraph(self.G)

        # All modifications are now done on this shallow copy H

        # Define super source and sink nodes
        # A Super source node has a directed edge to each source node in the
        # source_list
        # Similarily, a Super sink node has a directed edge from each sink node
        # in the sink_list

        dummy_super_source_node = 'source'
        dummy_super_sink_node = 'sink'
        num_source_nodes = len(source_list)
        num_sink_nodes = len(sink_list)

        # increment max_depth by 2 to accommodate edges from 'super source' and
        # to 'super sink'
        max_depth += 2

        # Add edges:
        #   \forall source \in source_list. super source node -> source

        edge_list = zip([dummy_super_source_node] * num_source_nodes,
                        source_list)

        H.add_edges_from(edge_list, weight=1)

        logger.debug('source -> list')
        for e in edge_list:
            logger.debug(e)

        # Add edges:
        #   \forall sink \in sink_list. sink -> super sink node

        edge_list = zip(sink_list, [dummy_super_sink_node] * num_sink_nodes)
        H.add_edges_from(edge_list, weight=1)

        logger.debug('sink -> list')
        for e in edge_list:
            logger.debug(e)

#        print('='*80)
        # TODO: WHY?
        # Switching this on with def path_gen(), results in empty path and no further results!!
        # #xplanation required!
#        for path in nx.all_simple_paths(H, dummy_super_source_node, dummy_super_sink_node):
#            print(path)
#        print('='*80)

        # TODO: how to do this with lambda?
        # Also, is this indeed correct?

#         if settings.debug:
#             # delete below lines till exit
#             K = max_paths
#             (len_list, path_list) = self.k_shortest_paths(H,
#                                                           dummy_super_source_node,
#                                                           dummy_super_sink_node,
#                                                           k=K)
#             for p in path_list:
#                 print(p)

        def path_gen():

            # all_shortest_paths
            # all_simple_paths
            #

            #K = 100
            K = max_paths
            (len_list, path_list) = self.k_shortest_paths(H,
                                                          dummy_super_source_node,
                                                          dummy_super_sink_node,
                                                          k=K)

            # path_list = self.ksp_gregBern(H, dummy_super_source_node,
            #                                              dummy_super_sink_node,
            #                                              k=K)

            # using simple paths
            # for i in nx.all_simple_paths(H, dummy_super_source_node,
            #                             dummy_super_sink_node,
            #                             cutoff=max_depth):

            # using all sohrtest paths
            # for i in nx.all_shortest_paths(H, dummy_super_source_node, dummy_super_sink_node):
                # Remove the first (super source)
                # and the last element (super sink)

            logger.debug('======== Paths =========')
            for p in path_list:
                l = len(p)
                #print(l, max_depth)
                if l <= max_depth:
                    logger.debug(p[1:-1])
                    yield p[1:-1]

        # return lambda: [yield i[1:-1] for i in nx.all_simple_paths(H,
        # dummy_super_source_node, dummy_super_sink_node)]

        return path_gen()

    # TODO: Add this method to the abstract graph class
    def __iter__(self):
        return self.G.__iter__()

    def neighbors(self, node):
        return self.G.neighbors(node)

    def out_degree(self, node):
        return self.G.out_degree(node)

    def iteredges(self):
        return self.G.edges_iter()

    def draw(self, pos_dict=None):
        nx.draw_networkx(self.G, pos=pos_dict, labels=pos_dict,
                         with_labels=True)

    def draw_graphviz(self, fname):
        #pos = graphviz_layout(self.G)
        #nx.draw_graphviz(self.G, pos)
        #nx.draw(self.G, pos)
        gdot_fname = GVIZ_GRAPH_PATH.format(fname)
        gsvg_fname = GRAPH_SVG_PATH.format(fname)
        print('generating graphs:', gdot_fname, gsvg_fname)
        write_dot(self.G, gdot_fname)
        opts = ['-Efontsize=10', '-Nfontsize=10', '-Gnodesep=1', '-Granksep=3', '-Tsvg']
        subprocess.call(['dot'] + opts + [gdot_fname, '-o', gsvg_fname])

    def draw_mplib(self, fname):
        from matplotlib import pyplot as plt
        plt.figure()
        #pos = nx.spring_layout(self.G)

        node_labels = {n: str(n) for n in self.G.nodes()}
        # literal_eval evals strings: '(x1,x2)' -> (x1,x2)
        pos_dict = {n: ast.literal_eval(str(n)) for n in self.G.nodes()}
        # assert 2-dim
        if not (len(ast.literal_eval(str(self.G.nodes()[0]))) == 2):
            return
        nx.draw(self.G, pos=pos_dict, labels=node_labels)

        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(self.G, pos=pos_dict, edge_labels=edge_labels)

        plt.savefig(MPLIB_GRAPH_PATH.format(fname))
        #plt.show()

    def __contains__(self, key):
        return key in self.G

    def __repr__(self):
        s = ''
        s += '''==== Nodes ====\n {} \n'''.format(self.G.nodes())
        s += '''==== Edges ====\n {} \n'''.format(self.G.edges())
        return s
