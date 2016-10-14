# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from plotting_abstract import PlottingBase


class Plotting(PlottingBase):
    def __init__(self):
        return

    def plot_trace_list(self, trace_list):

        app = pg.mkQApp()
        app.processEvents()
        ## Putting this at the beginning or end does not have much effect


        win = pg.GraphicsWindow(title="Basic plottting examples")
        win.resize(1000, 600)

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        p = win.addPlot(title="x0-x1")

        xy = []
        for trace in trace_list:
            x_array = trace.x_array
            #p.plot(x_array[:, 0:2])
            xy.append(x_array)

        xy_array = np.vstack(xy)
        path = pg.arrayToQPath(xy_array[:, 0], xy_array[:, 1], connect='all')

        item = QtGui.QGraphicsPathItem(path)
        item.setPen(pg.mkPen('w'))
        p.addItem(item)

        #p.show()
        QtGui.QApplication.instance().exec_()



# REFERENCE:
# https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/Plotting.py

# import sys

# #QtGui.QApplication.setGraphicsSystem('raster')
# app = QtGui.QApplication([])
# #mw = QtGui.QMainWindow()
# #mw.resize(800,800)

# win = pg.GraphicsWindow(title="Basic plotting examples")
# win.resize(1000, 600)
# win.setWindowTitle('pyqtgraph example: Plotting')

# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)

# p1 = win.addPlot(title="Basic array plotting", y=np.random.normal(size=100))

# p2 = win.addPlot(title="Multiple curves")
# p2.plot(np.random.normal(size=100), pen=(255, 0, 0), name="Red curve")
# p2.plot(np.random.normal(size=110)+5, pen=(0, 255, 0), name="Blue curve")
# p2.plot(np.random.normal(size=120)+10, pen=(0, 0, 255), name="Green curve")

# p3 = win.addPlot(title="Drawing with points")
# p3.plot(np.random.normal(size=100), pen=(200, 200, 200), symbolBrush=(255, 0, 0), symbolPen='w')
