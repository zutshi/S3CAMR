from plotting_abstract import PlottingBase

import err


def factory(lib_name):
    # Matplotlib
    if lib_name == 'mp':
        import plotMP
        plotting = plotMP.Plotting()
    # pyQtGraph
    elif lib_name == 'pg':
        import plotPG
        plotting = plotPG.Plotting()
    # Bokeh
    elif lib_name == 'bk':
        import plotBK
        raise NotImplementedError
    # Plotting disabled
    elif lib_name is None:
        plotting = PlottingDisabled()
    else:
        raise err.Fatal('unknown plotting library requested: {}'.format(lib_name))

    assert(isinstance(plotting, PlottingBase))
    return plotting


class PlottingDisabled(PlottingBase):
    def plot_trace_list(self, trace_list):
        return
