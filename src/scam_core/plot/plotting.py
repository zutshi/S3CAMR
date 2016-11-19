from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import err

from .plotting_abstract import PlottingBase


def factory(lib_name, plots, *args):
    # Matplotlib
    if lib_name == 'mp':
        from . import plotMP
        plotting = plotMP.Plotting(plots, *args)
    # pyQtGraph
    elif lib_name == 'pg':
        from . import plotPG
        plotting = plotPG.Plotting(plots, *args)
    # Bokeh
    elif lib_name == 'bk':
        from . import plotBK
        raise NotImplementedError
    # Plotting disabled
    elif lib_name is None:
        plotting = PlottingDisabled()
    else:
        raise err.Fatal('unknown plotting library requested: {}'.format(lib_name))
    
    if lib_name is not None:
        assert(isinstance(plotting, PlottingBase))
    return plotting


#class PlottingDisabled(PlottingBase):
class PlottingDisabled(object):
    def __getattribute__(*args):
        return lambda *args, **kwargs: None
#     def figure(self):
#         return

#     def show(self):
#         return

#     def plot_rect(self, r, edgecolor='k'):
#         return

#     def plot_abs_states(self, AA, s):
#         return

#     def plot_trace_list(self, trace_list, x_vs_y=None):
#         return

#     def plot_pwa_traces(self, txs):
#         return

#     def plot(self, *args):
#         return

#     def title(self, *args):
#         return


def plot_opts_parse(popts):
    if popts:
        xVsy = popts.split('-')
        return xVsy
    else:
        return []
