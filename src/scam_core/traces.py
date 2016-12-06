#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy import io

import settings

import fileops as fops
from streampickle import PickleStreamReader

## commented off because matplotlib is not installed for python installation
## used with matlab
#import matplotlib
## Force GTK3 backend. By default GTK2 gets loaded and conflicts with
## graph-tool
#matplotlib.use('GTK3Agg')
#global plt
#import matplotlib.pyplot as plt

plot_figure_for_paper = False


class Trace(object):

    def __init__(self, num_dims, num_points):
        self.idx = 0
        self.t_array = np.empty(num_points)
        self.x_array = np.empty((num_points, num_dims.x))
        self.s_array = np.empty((num_points, num_dims.s))
        self.u_array = np.empty((num_points, num_dims.u))
        self.d_array = np.empty((num_points, num_dims.d))
        self.ci = np.empty((num_points, num_dims.ci))
        self.pi = np.empty((num_points, num_dims.pi))

    def append(
            self,
            s=None,
            u=None,
            x=None,
            ci=None,
            pi=None,
            t=None,
            d=None,
            ):
        #if s is None or u is None or r is None or x is None or ci is None \
        #    or pi is None or t is None or d is None:
        #    raise err.Fatal('one of the supplied arguement is None')

        i = self.idx
        self.t_array[i] = t
        self.x_array[i, :] = x
        self.s_array[i, :] = s
        self.u_array[i, :] = u
        self.ci[i, :] = ci
        self.pi[i, :] = pi
        self.idx += 1

#########################################
# replacement for plot_trace_list()
#########################################
# TODO: unfinished function...
# Need to take care of matplotlib format and test the function!!
    def plot(self, plot_cmd):
        raise NotImplementedError
        parsed_plot_cmd = None
        while(parsed_plot_cmd is None):
            plot_cmd = get_plot_cmd_from_stdin()
            parsed_plot_cmd = parse_plot_cmd(plot_cmd)

        x, y = parsed_plot_cmd
        plt.plot(x, y)

#         ###### used to generate heater plots for the rtss paper##############
#         plt.rc('text', usetex=True)
#         plt.rc('font', family='serif')
#         plt.title(r'Room-Heater-Thermostat: Random Simulations',fontsize=30)
#         plt.xlabel(r'Time (s)',fontsize=28)
#         plt.ylabel(r'Room Temp. ($^\circ$F)',fontsize=28)
#         plt.plot([0, 10], [76, 76], 'r-', lw=2)
#         plt.plot([0, 10], [52, 52], 'r-', lw=2)
#         #####################################################################

    # plt.figure()
    # AC = plt.gca()
    # plt.title('ci')

        #   AX_list[i+1].plot(x_array[:, 0], x_array[:, 1])

        # plt_x1 = AX1.plot(t_array, x_array[:, 1], label='x1')
        # plt_x2 = AX2.plot(t_array, x_array[:, 2], label='x2')
        # plt_x0x1 = AX0X1.plot(x_array[:, 0], x_array[:, 1], label='x0x1')

        # #plt_s0 = plt.plot(t_array, trace.s_array[:, 0], label='err')
        # plt_s1 = plt.plot(t_array, trace.s_array[:, 1], label='ref')
        # plt_u = AU.plot(t_array, trace.u_array[:, 0], label='u')
        # plt_ci = AC.plot(t_array, trace.ci[:, 0], label='ci')
        # print(trace.s_array)
        # plt.legend()
        # plt.legend([plt_x0, plt_x1, plt_s0, plt_s1], ['x0', 'x1', 's0', 's1'])
    # plt.plot(t_array, ref_signal, drawstyle='steps-post')
    # plt.autoscale()
        settings.plt_show()

    def dump_matlab(self):
        data = {'T': self.t_array,
                'X': self.x_array,
                'S': self.s_array,
                'U': self.u_array,
                'CI': self.ci,
                'PI': self.pi}
        io.savemat('mat_file.mat', data, appendmat=False, format='5',
                   do_compression=False, oned_as='column')

    def __str__(self):
        s = '''t:{},\nx:{},\ns:{},\nu:{},\nci:{},\npi:{}'''.format(
            self.t_array,
            self.x_array,
            self.s_array,
            self.u_array,
            self.ci,
            self.pi,
            )
        return s
    def serialize(self):
        s = '''{}\n{}\n{}\n{}\n{}\n{}'''.format(
            self.t_array.tobytes(),
            self.x_array.tobytes(),
            self.s_array.tobytes(),
            self.u_array.tobytes(),
            self.ci.tobytes(),
            self.pi.tobytes(),
            )
        return s

def get_simdump_gen(dirpath):
    files = fops.get_file_list_matching('*.simdump*', dirpath)

    for f in files:
        reader = PickleStreamReader(f)
        for trace in reader.read():
            yield trace
