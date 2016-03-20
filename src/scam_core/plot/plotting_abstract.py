import abc


# SPEC: Any plotting lib must implement these functions
class PlottingBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def plot_trace_list(self, trace_list):
        """plot all the traces in the trace list"""
        return
