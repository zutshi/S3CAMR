import abc


# SPEC: Any plotting lib must implement these functions
class PlottingBase():
    __metaclass__ = abc.ABCMeta

    ######################################
    # #### Debatable methods
    @abc.abstractmethod
    def figure(self):
        """draws a figure like Matplotlib"""
        return

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
    def plot_trace_list(self, trace_list, x_vs_y=[]):
        """plot all the traces in the trace list"""
        return

    @abc.abstractmethod
    def plot_pwa_traces(self, txs):
        """plot pwa traces in the trace list"""
        return
