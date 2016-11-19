from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import matplotlib
# Force GTK3 backend. By default GTK2 gets loaded and conflicts with
# graph-tool
matplotlib.use('GTK3Agg')
#global plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import matplotlib.cm as cm
import numpy as np

from .plotting_abstract import PlottingBase
import err
import settings

plot_figure_for_paper = False


class Plotting(PlottingBase):

    def __init__(self, plots, *args):
        self._ax = None
        self.fast_plot = True if 'fast-plot' in args else False
        self.block = False if 'no-block' in args else True
        self.plots = plots
        self.x_vs_y = plots
        self.new_session()
        return

#     def figure(self):
#         fig = plt.figure()
#         self._ax = fig.gca()

    def force_ax(self, ax):
        self._ax = ax

    def single_color(self, c):
        self.ax().set_color_cycle([c])

    def acquire_global_fig(self):
        ax = plt.gca()
        self._ax = ax

    def new_session(self):
        self.xy2fig_map = collections.defaultdict(plt.figure)

    def ax(self, x_vs_y=None):
        # creates throwable figures
        if x_vs_y is None:
            if self._ax is None:
                fig = plt.figure()
                self._ax = fig.gca()
        else:
            ax = self.xy2fig_map[x_vs_y].gca()
            ax.set_title(x_vs_y)
            if settings.paper_plot:
                ax.set_title('')
            self._ax = ax

        return self._ax

    def set_range(self, x_range, y_range):
        self.ax().set_xlim(x_range)
        self.ax().set_ylim(y_range)

    def show(self, block=None):
        if block is None:
            block = self.block

        plt.show(block)

    def title(self, *args, **kwargs):
        self.ax().set_title(*args, **kwargs)

    # discouraged!
    def plot(self, *args, **kwargs):
        self.ax().plot(*args, **kwargs)

    def gen_x_vs_y(self, nd):
        if self.x_vs_y is None:
            x_vs_y = [('t', 'x{}'.format(i)) for i in range(nd)]
        else:
            x_vs_y = self.x_vs_y
        return x_vs_y

    def tx_array_selectors(self, x_vs_y):
        for a, o in x_vs_y:

            a_str = a[0]
            o_str = o[0]

            a_idx = int(a[1:]) if a[1:] else 0
            o_idx = int(o[1:]) if o[1:] else 0

            title = '{} - {}'.format(a, o)

            yield a_str, o_str, a_idx, o_idx, title

    def plot_trace_array(self, t_array, x_array, *args, **kwargs):

        x_vs_y = self.gen_x_vs_y(x_array.shape[1])
        ao = self.tx_array_selectors(x_vs_y)

        for a_str, o_str, a_idx, o_idx, title in ao:
            x = t_array if a_str == 't' else x_array[:, a_idx]
            y = t_array if o_str == 't' else x_array[:, o_idx]
            self.ax(title).plot(x, y, *args, **kwargs)

    def plot_abs_states(self, AA, prop, abs_states):
        def get_color(s):
            if AA.plant_abs.get_ival_cons_abs_state(s.ps) & prop.init_cons:
                color = 'g'
            elif AA.plant_abs.get_ival_cons_abs_state(s.ps) & prop.final_cons:
                color = 'r'
            else:
                color = 'k'
            return color

#         if not isinstance(states, dict):
#             s = {'regular': states}
#         for atype, abs_states in s.iteritems():
#             for abs_state in abs_states:
#                 #r = AA.plant_abs.rect(abs_state.plant_state)
#                 #if c.dim != 2:
#                 #    raise StandardError('dim should be 2 for plotting 2D!')
#                 r = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).rect()
#                 self.plot_rect(r, color_map[atype])
        for abs_state in abs_states:
            r = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).rect()
            self.plot_rect(r, get_color(abs_state))

    def plot_rect(self, r, edgecolor='k'):
        if len(r[0]) > 2:
            err.warn('dim>2, projecting before plottting the rectangle on the first 2 dims.')
            c = r[0][0:2]
            rng = r[1][0:2]
        else:
            c = r[0]
            rng = r[1]
        self.ax().add_patch(
            patches.Rectangle(c, *rng, fill=False, edgecolor=edgecolor, linewidth=2)
            #patches.Rectangle((0.1, 0.1), 0.5, 0.5, fill=False)
        )

    def plot_pwa_traces(self, txs):
        N = np.array([None])
        x = []
        y = []
        for tx in txs:
            #t = np.array(tx[0])
            xy = np.vstack(tx[1])
            x.append(xy[:, 0])
            x.append(N)
            y.append(xy[:, 1])
            y.append(N)
            #self.ax.plot(x[:, 0], x[:, 1], '-*')
        self.ax().plot(np.hstack(x), np.hstack(y), '-*')

    def parse_plot_cmd(self, plot_cmd, trace_obj):
        if len(plot_cmd) != 4:
            print('plot command NOT of length 4: {}'.format(plot_cmd))
            return None
        x_axis_str, x_idx, y_axis_str, y_idx = plot_cmd
        try:
            x_axis = getattr(trace_obj, x_axis_str)
            y_axis = getattr(trace_obj, y_axis_str)
        except AttributeError, e:
            print('unexpected plot command received: {}'.format(plot_cmd))
            print(e)
            return None
        x_idx = int(x_idx)
        y_idx = int(y_idx)
        try:
            if x_axis_str != 't':
                x = x_axis[x_idx, :]
        except:
            print('unexpected indices for the first var: {}'.format(plot_cmd))
            return None
        try:
            if y_axis_str != 't':
                y = y_axis[y_idx, :]
        except:
            print('unexpected indices for the second var: {}'.format(plot_cmd))
            return None

        return x, y

    def get_plot_cmd_from_stdin(self):
        plot_cmd_format = \
            '''########### plot command format ########## \n
               [t,x,s,u,ci,pi][0-n][t,x,s,u,ci,pi][0-n]   \n
               e.g. (a) phase plot,    x[1] vs x[0]: x0x1 \n
                    (b) state vs time, t    vs x[0]: t0x0 \n
               ########################################## \n'''
        print(plot_cmd_format)

        # For python2/3 compatibility
        try:
            input = raw_input
        except NameError:
            pass
        corrected_plot_cmd = input('please type the correct command:')
        return corrected_plot_cmd

    def plot_trace_list_slow(self, trace_list):
        '''
        @type plt: matplotlib.pyplot
        '''

        # plot for each plant state
        # NUM_PLOTS = num_dims.x+1

        # plot all continuous plant states against time
        NUM_PLOTS = trace_list[0].x_array.shape[1]
    #     AX_list = []
        #plt.figure()
        #AX0X1 = plt.gca()

    #     for i in range(NUM_PLOTS):
    #         plt.figure()
    #         ax = plt.gca()
    #         AX_list.append(ax)
    #         plt.title('x{}'.format(i))

    #     for trace in trace_list:
    #         x_array = trace.x_array
    #         t_array = trace.t_array

    #         # plt_x0 = AX0.plot(t_array, x_array[:, 10], label='x10')
    #         for i in range(NUM_PLOTS):
    #             AX_list[i].plot(t_array, x_array[:, i])

    #         #plt_x0x1 = AX0X1.plot(x_array[:, 0], x_array[:, 1], label='x0x1')

    #         # plt.legend([plt_x0, plt_x1, plt_s0, plt_s1], ['x0', 'x1', 's0', 's1'])

    #     # plt.plot(t_array, ref_signal, drawstyle='steps-post')
    #     # plt.autoscale()
    #     plt.show()

        if plot_figure_for_paper:
            # count num of plotted sims
            ctr_total = 0
            import plothelper as ph
            line_list = []
            for i in range(NUM_PLOTS):
                plt.figure()
                ax = plt.gca()
                plt.title('x{}'.format(i))
                for trace in trace_list:
                    x_array = trace.x_array
                    t_array = trace.t_array
                    if not (x_array[0, i] <= 70.0 and x_array[0, i] >= 69.9):
                        # select to plot with probability 20%
                        if np.random.random() >= 0.05:
                            continue
                    line, = ax.plot(t_array, x_array[:, i])
                    line_list.append(line)
                    ctr_total += 1
                print('plotted {} sims'.format(ctr_total))
                ph.figure_for_paper(ax, line_list)
        else:
    #         for i in range(NUM_PLOTS):
    #             plt.figure()
    #             ax = plt.gca()
    #             plt.title('x{}'.format(i))
    #             for trace in trace_list:
    #                 x_array = trace.x_array
    #                 t_array = trace.t_array
    #                 ax.plot(t_array, x_array[:, i])
    #             #plt.show()

            plt.figure()
            ax = plt.gca()
            plt.title('x0-x1')
            for trace in trace_list:
                x_array = trace.x_array
                ax.plot(x_array[:, 0], x_array[:, 1])
            #plt.show()

    # Move this function to Traces class!
    def plot_trace_list(self, trace_list):
        """plot all state vars """

        nd = trace_list[0].x_array.shape[1]
        x_vs_y = self.gen_x_vs_y(nd)

        if self.fast_plot:
            self.plot_trace_list_xvsy_sc(trace_list, x_vs_y)
        else:
            self.plot_trace_list_xvsy_dc(trace_list, x_vs_y)


#     def plot_trace_list(self, trace_list):
#         """plot all state vars """
#         if not self.x_vs_y2ax_map:
#             nd = trace_list[0].x_array.shape[1]
#             self.x_vs_y2ax_map = {('t', 'x{}'.format(i)): plt.figure().gca() for i in range(nd)}
#         if self.fast_plot:
#             self.plot_trace_list_xvsy_sc(trace_list)
#         else:
#             self.plot_trace_list_xvsy_dc(trace_list)

    def plot_trace_list_xvsy_sc(self, trace_list, x_vs_y):
        """plot_trace_list: Same Color but faster?

        Parameters
        ----------
        trace_list :
        x_vs_y : [(x,y)]: x Vs y
               e.g. [(t, x0), (t, x0),(t, x1),(x0, x1)]
        """

        # Would be good to handle errors in the x Vs y string
        #sanity_check_xVsy()
        print('plotting...')

        def prep_t_trace(trace_list):
            N = np.array([None])
            t_ = []
            for trace in trace_list:
                t_.append(trace.t_array)
                t_.append(N)
            return t_

        def prep_x_trace(trace_list, idx):
            N = np.array([None])
            x_ = []
            for trace in trace_list:
                x_.append(trace.x_array[:, idx])
                x_.append(N)
            return x_

        ao = self.tx_array_selectors(x_vs_y)

        # ordinate, abcissa
        for a_str, o_str, a_idx, o_idx, title in ao:

            # collect t_array
            x = (prep_t_trace(trace_list) if a_str == 't'
                 else prep_x_trace(trace_list, a_idx))
            # collect t_array
            y = (prep_t_trace(trace_list) if o_str == 't'
                 else prep_x_trace(trace_list, o_idx))

            self.ax(title).plot(np.hstack(x), np.hstack(y), '.-', linewidth=0.5)

    def plot_trace_list_xvsy_dc(self, trace_list, x_vs_y):
        """plot_trace_list: Different Color but slower?

        Parameters
        ----------
        trace_list :
        x_vs_y : [(x,y)]: x Vs y
               e.g. [(t, x0), (t, x0),(t, x1),(x0, x1)]
        """

        # Would be good to handle errors in the x Vs y string
        #sanity_check_xVsy()

        print('plotting...')

        def prep_t_trace(trace_list):
            return [trace.t_array for trace in trace_list]

        def prep_x_trace(trace_list, idx):
            return [trace.x_array[:, idx] for trace in trace_list]

        ao = self.tx_array_selectors(x_vs_y)

        # ordinate, abcissa
        for a_str, o_str, a_idx, o_idx, title in ao:

            # collect t_array
            x = (prep_t_trace(trace_list) if a_str == 't'
                 else prep_x_trace(trace_list, a_idx))
            # collect t_array
            y = (prep_t_trace(trace_list) if o_str == 't'
                 else prep_x_trace(trace_list, o_idx))

            for i, j in zip(x, y):
                self.ax(title).plot(i, j, '-*')

    def plot_trace_list_sat(self, trace_list):
        '''
        @type plt: matplotlib.pyplot
        '''
        NUM_PLOTS = 1
        for i in range(NUM_PLOTS):
            plt.figure()
            #ax = plt.gca(projection='3d')
            ax = plt.gca()
            #for trace in trace_list:
            #    x_array = trace.x_array
            #    t_array = trace.t_array
            #    ax.plot(t_array, x_array[:, i])
            plt.title('x-y'.format(i))
            for trace in trace_list:
                x_array = trace.x_array
                #ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2])
                ax.plot(x_array[:, 0], x_array[:, 1])
            plt.figure()
            ax = plt.gca()
            plt.title('y-z'.format(i))
            for trace in trace_list:
                x_array = trace.x_array
                #ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2])
                ax.plot(x_array[:, 1], x_array[:, 2])
            plt.figure()
            ax = plt.gca()
            plt.title('x-z'.format(i))
            for trace in trace_list:
                x_array = trace.x_array
                #ax.plot(x_array[:, 0], x_array[:, 1], x_array[:, 2])
                ax.plot(x_array[:, 0], x_array[:, 2])
            plt.show()
