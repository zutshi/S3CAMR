from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import time
from heapq import heappush, heappop
from itertools import count
#import sys

import graph_tool.all as gt
# can not import sub modules for some reasons!!
# e.e. gt.topology gives an error
#import graph_tool as gt
from blessed import Terminal

import utils as U
import err

import settings

term = Terminal()


class GraphGTError(Exception):
    pass


class GraphGT(object):

    def __init__(self, G=None, Type=None):

        # unused maxVert

        #self.maxVertices = 0

        self.G = (gt.Graph(directed=True) if G is None
                  else gt.Graph(G))
        # vertex_dict
        self.vd = {}#defaultdict(self.G.add_vertex)
        self.v_attr = self.G.new_vertex_property('object')
        # edge_dict
        self.edge_attr = self.G.new_edge_property('object')

    # TODO: Does't work!
    def shallow_copy(self):
        '''Something is wrong in this shallow copy.
        Gets errors when adding new edges even though add_missing=True'''
        H = GraphGT()
        H.G = gt.Graph(self.G)
        H.vd = self.vd
        H.edge_attr = self.edge_attr
        return H

    def add_edge(self, n1, n2, **attr_dict_arg):#ci=None, pi=None
        attrs = {'ci': None, 'pi': None}
        #U.eprint('gt:', n1, n2)

        #v1 = self.vd[n1]
        #v2 = self.vd[n2]

        try:
            v1 = self.vd[n1]
        except KeyError:
            v1 = self.G.add_vertex()
            self.vd[n1] = v1
        try:
            v2 = self.vd[n2]
        except KeyError:
            v2 = self.G.add_vertex()
            self.vd[n2] = v2

        #v2 = self.vd[n2]

        self.v_attr[v1] = n1
        self.v_attr[v2] = n2

        #print(v1, v2)
        e = self.G.edge(v1, v2, add_missing=True)
        #TODO: The attributes of an existing edge will be updated.
        # Should this be the intended behavior?
        #self.edge_attr[e] = {'ci': ci, 'pi': pi}
        attrs.update(attr_dict_arg)
        self.edge_attr[e] = attrs

    def add_edges_from(self, iterable):
        for e in iterable:
            self.add_edge(*e)

    def get_path_attr_list(self, path, attrs):
        attr_map = defaultdict(list)
        for (n1, n2) in U.pairwise(path):
            v1 = self.vd[n1]
            v2 = self.vd[n2]
            e = self.G.edge(v1, v2, add_missing=False)
            for attr in attrs:
                attr_map[attr].append(self.edge_attr[e][attr])
        return attr_map

    def get_path_generator(self, *args):
        return GraphGT.get_all_path_generator(self, *args)

    @staticmethod
    def get_all_path_generator(
            G,
            source_list,
            sink_list,
            max_depth,
            max_paths
            ):
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

        edge_list = list(zip([dummy_super_source_node] * num_source_nodes,
                        source_list))
        G.add_edges_from(edge_list)

        if settings.debug:
            U.eprint('source -> list')
            for e in edge_list:
                U.eprint(e)

        # Add edges:
        #   \forall sink \in sink_list. sink -> super sink node

        edge_list = list(zip(sink_list, [dummy_super_sink_node] *
                             num_sink_nodes))
        G.add_edges_from(edge_list)

        if settings.debug:
            U.eprint('sink -> list')
            for e in edge_list:
                U.eprint(e)

        if settings.debug:
            #print(the graph first)
            U.eprint('Printing graph...')
            for e in G.G.edges():
                s, t = e
                U.eprint('{}, {}'.format(G.v_attr[s], G.v_attr[t]))
            U.eprint('Printing graph...done')

        path_it = G.all_shortest_paths(dummy_super_source_node,
                                       dummy_super_sink_node,
                                       )

        if settings.debug:
            U.eprint('path list')
            paths = list(path_it)
            for path in paths[0:max_paths]:
                p = [G.v_attr[v] for v in path]
                U.eprint(p)
            path_it = (i for i in paths)

#############################################################
        # TODO: Remove the extra step to count paths
        # It is there just as a debug print
        paths = list(U.bounded_iter(path_it, max_paths))
        num_paths = len(paths)
        err.warn('counting paths...found: {}'.format(num_paths))

        def path_gen():
            for path in paths:
                p = [G.v_attr[v] for v in path]
                yield p[1:-1]

        # END: CODE TO BE REMOVED
        # CORRECT CODE BELOW
#         def path_gen():
#             for path in U.bounded_iter(path_it, max_paths):
#                 p = [G.v_attr[v] for v in path]
#                 yield p[1:-1]
#############################################################
        return path_gen()

    def all_shortest_paths(self, source, target):
        v_source = self.vd[source]
        v_target = self.vd[target]
        path_iterator = gt.all_shortest_paths(self.G, v_source, v_target)
        return path_iterator

    def get_ksp_generator(
            self,
            source_list,
            sink_list,
            max_depth,
            max_paths
            ):

        # Create a shallow copy of the graph

        #H = gt.Graph(self.G)
        H = self.copy()

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

        edge_list = list(zip([dummy_super_source_node] * num_source_nodes,
                         source_list))
        H.add_edges_from(edge_list)

#        print(edge_list)

        # Add edges:
        #   \forall sink \in sink_list. sink -> super sink node

        edge_list = list(zip(sink_list, [dummy_super_sink_node] *
                             num_sink_nodes))
        H.add_edges_from(edge_list)

#        print(edge_list)

#        print('='*80)
        # TODO: WHY?
        # Switching this on with def path_gen(), results in empty path and no further results!!
        # #xplanation required!
#        for path in nx.all_simple_paths(H, dummy_super_source_node, dummy_super_sink_node):
#            print(path)
#        print('='*80)

        # TODO: how to do this with lambda?
        # Also, is this indeed correct?

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

            for p in path_list:
                l = len(p)
                #print(l, max_depth)
                if l <= max_depth:
                    yield p[1:-1]

        # return lambda: [yield i[1:-1] for i in nx.all_simple_paths(H,
        # dummy_super_source_node, dummy_super_sink_node)]

        return path_gen()


    # ###################### KSP 1 ##################################################
    # https://gist.github.com/guilhermemm/d4623c574d4bccb6bf0c
    # __author__ = 'Guilherme Maia <guilhermemm@gmail.com>'
    # __all__ = ['k_shortest_paths']

    @staticmethod
    def k_shortest_paths(
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

        #(length, path) = nx.single_source_dijkstra(G, source, target, weight=weight)
        (vlist, elist) = gt.topology.shortest_path(G.G, source, target)

        if not vlist:
            raise GraphGTError('node %s not reachable from %s' % (source, target))

        lengths = [length[target]]
        paths = [path[target]]
        c = count()
        B = []

        # Is deep copy really required?
        #   Fails due to embedded Ctype objects which can not be pickled
        # # G_original = G.copy()
        # Swapping with shallow copy...will it work?

        G_original = G
        G = GraphGT(G_original)

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

                        for (u, v, edge_attr) in G.in_edges_iter(node, data=True):

                            # print('lala2: {} -> {}'.format(u,v))

                            G.remove_edge(u, v)
                            edges_removed.append((u, v, edge_attr))

                (spur_path_length, spur_path) = nx.single_source_dijkstra(
                        G, spur_node, target, weight=weight)
                if target in spur_path and spur_path[target]:
                    total_path = root_path[:-1] + spur_path[target]
                    total_path_length = self.get_path_length(
                            G_original, root_path, weight
                            ) + spur_path_length[target]
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

############################## UNFINISHED FROM HERE

    # Actually draws the graph!! Need to rewrite get_path_generator() from
    # scratch for gt. Also, destroys the passed in graph (oops) :D
    # Hence, use this function only for debugging!!
    # # TODO: Fix it, of course?

    def plot(
            self,
            source_list,
            sink_list,
            max_depth,
            max_paths
            ):

        print('WARNING: This is actually a plotting function!!!')

        num_source_nodes = len(source_list)
        num_sink_nodes = len(sink_list)

        # super_source_vertex = g.add_vertex()
        # super_sink_vertex = g.add_vertex()

        super_source_vertex = 'super_source_vertex'
        super_sink_vertex = 'super_sink_vertex'

        edge_list = list(zip([super_source_vertex] * num_source_nodes,
                             source_list))
        for e in edge_list:
            self.add_edge(*e)

        edge_list = list(zip(sink_list, [super_sink_vertex] *
                             num_sink_nodes))
        for e in edge_list:
            self.add_edge(*e)

        g = self.G

        pos = gt.arf_layout(g, max_iter=0)
        gt.graph_draw(g, pos=pos, vertex_text=self.G.vertex_index)
        time.sleep(1000)
        exit()

        gt.graph_draw(self.G, vertex_text=self.G.vertex_index)
        time.sleep(1000)

#        print(edge_list)

        # Add edges:
        #   \forall sink \in sink_list. sink -> super sink node

        edge_list = list(zip(sink_list, [dummy_super_sink_node] *
                             num_sink_nodes))
        H.add_edges_from(edge_list)

#        print(edge_list)

#        print('='*80)
        # TODO: WHY?
        # Switching this on with def path_gen(), results in empty path and no further results!!
        # #xplanation required!
#        for path in nx.all_simple_paths(H, dummy_super_source_node, dummy_super_sink_node):
#            print(path)
#        print('='*80)

        # TODO: how to do this with lambda?
        # Also, is this indeed correct?

        def path_gen():
            for i in nx.all_simple_paths(H, dummy_super_source_node,
                    dummy_super_sink_node):

                # Remove the first (super source)
                # and the last element (super sink)

                yield i[1:-1]

        # return lambda: [yield i[1:-1] for i in nx.all_simple_paths(H,
        # dummy_super_source_node, dummy_super_sink_node)]

        return path_gen()

    def neighbors(self, node):
        v = self.vd[node]
        return v.out_neighbours

    def __contains__(self, key):
        return key in self.vd

    def __repr__(self):
        raise NotImplementedError
        s = ''
        s += '''==== Nodes ==== {} '''.format(self.G.nodes())
        s += '''==== Edges ==== {} '''.format(self.G.edges())
        return s
