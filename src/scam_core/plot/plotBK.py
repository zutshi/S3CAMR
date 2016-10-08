from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import bokeh.plotting as BP

from plotting_abstract import PlottingBase


class Plotting(PlottingBase):
    def plot_trace_list(self, trace_list):
        return

    def plot_vecs(self, tx):
        x = np.vstack(tx[1])
        fig.line(x[:, 0], x[:, 1])
        BP.output_server('hover')
        BP.show(fig)
