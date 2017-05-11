from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import graphGT
import graphNX

#grid_eps = [0.50, 0.04]
#delta_t = 0.4
#num_samples = 2

#time python -O ./scamr.py -f ../examples/vdp/vanDerPol.tst -cn --refine model_dft -g g --seed 1 >nx_log 2>gt_log


class Graph(object):

    def __init__(self, G=None, Type=None):
        self.Ggt = graphGT.GraphGT()
        self.Gnx = graphNX.GraphNX()

    def add_edge(self, n1, n2, ci=None, pi=None):
        #n1 = n1.plant_state.cell_id
        #n2 = n2.plant_state.cell_id
        #print('{}, {}'.format(n1, n2))

        self.Ggt.add_edge(n1, n2)
        self.Gnx.add_edge(n1, n2)

    def get_path_attr_list(self, path, attrs):
        self.Ggt.get_path_attr_list(path, attrs)
        self.Gnx.get_path_attr_list(path, attrs)

    def get_path_generator(
            self,
            source_list,
            sink_list,
            max_depth,
            max_paths
            ):

        #source_list = [i.plant_state.cell_id for i in source_list]
        #sink_list = [i.plant_state.cell_id for i in sink_list]

        self.Ggt.get_path_generator(
                source_list,
                sink_list,
                max_depth,
                max_paths)

        return self.Gnx.get_path_generator(
                source_list,
                sink_list,
                max_depth,
                max_paths)
        raise NotImplementedError

    def neighbors(self, node):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
