from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six


# SPEC: Any plotting lib must implement these functions
@six.add_metaclass(abc.ABCMeta)
class PlottingBase():
    #__metaclass__ = abc.ABCMeta

    ######################################
    # #### Debatable methods
#     @abc.abstractmethod
#     def figure(self):
#         """draws a figure like Matplotlib"""
#         return

    @abc.abstractmethod
    def show(self):
        """show the plot like Matplotlib"""
        return
    # #### Debatable methods
    ######################################

    @abc.abstractmethod
    def plot_rect(self, r, edgecolor='k'):
        """plot 2-d rectangles"""
        return

    @abc.abstractmethod
    def plot_abs_states(self, AA, s):
        """plot abstract state as 2-d rectangles"""
        return

    @abc.abstractmethod
    def plot_trace_list(self, trace_list, x_vs_y=None):
        """plot all the traces in the trace list"""
        return

    @abc.abstractmethod
    def plot_pwa_traces(self, txs):
        """plot pwa traces in the trace list"""
        return

    #TODO: exists in matplotlib..but does it exists in other libs?
    @abc.abstractmethod
    def plot(self, *args):
        """generic library plot calls"""
        return
