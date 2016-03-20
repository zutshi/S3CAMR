import matplotlib
# Force GTK3 backend. By default GTK2 gets loaded and conflicts with
# graph-tool
matplotlib.use('GTK3Agg')
#global plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from plotting_abstract import PlottingBase

plot_figure_for_paper = False


class Plotting(PlottingBase):

    def __init__(self):
        self.fig = None
        return

    def figure(self):
        fig = plt.figure()
        self.ax = fig.gca()

    def show(self):
        plt.show()

    def plot_abs_states(self, AA, s):
        color_map = {
                    'init': 'g',
                    'final': 'r',
                    'regular': 'k'
                    }
        for atype, abs_states in s.iteritems():
            for abs_state in abs_states:
                #r = AA.plant_abs.rect(abs_state.plant_state)
                #if c.dim != 2:
                #    raise StandardError('dim should be 2 for plotting 2D!')
                r = AA.plant_abs.get_ival_cons_abs_state(abs_state.plant_state).rect()
                self.plot_rect(r, color_map[atype])

    def plot_rect(self, r, edgecolor='k'):
        self.ax.add_patch(
            patches.Rectangle(r[0], *r[1], fill=False, edgecolor=edgecolor)
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
        self.ax.plot(np.hstack(x), np.hstack(y), '-*')

    def parse_plot_cmd(self, plot_cmd, trace_obj):
        if len(plot_cmd) != 4:
            print 'plot command NOT of length 4: {}'.format(plot_cmd)
            return None
        x_axis_str, x_idx, y_axis_str, y_idx = plot_cmd
        try:
            x_axis = getattr(trace_obj, x_axis_str)
            y_axis = getattr(trace_obj, y_axis_str)
        except AttributeError, e:
            print 'unexpected plot command received: {}'.format(plot_cmd)
            print e
            return None
        x_idx = int(x_idx)
        y_idx = int(y_idx)
        try:
            if x_axis_str != 't':
                x = x_axis[x_idx, :]
        except:
            print 'unexpected indices for the first var: {}'.format(plot_cmd)
            return None
        try:
            if y_axis_str != 't':
                y = y_axis[y_idx, :]
        except:
            print 'unexpected indices for the second var: {}'.format(plot_cmd)
            return None

        return x, y

    def get_plot_cmd_from_stdin(self):
        plot_cmd_format = \
            '''########### plot command format ########## \n
               [t,x,s,u,ci,pi][0-n][t,x,s,u,ci,pi][0-n]   \n
               e.g. (a) phase plot,    x[1] vs x[0]: x0x1 \n
                    (b) state vs time, t    vs x[0]: t0x0 \n
               ########################################## \n'''
        print plot_cmd_format
        corrected_plot_cmd = raw_input('please type the correct command:')
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

    def plot_trace_list(self, trace_list):
        '''
        @type plt: matplotlib.pyplot
        '''

        NUM_PLOTS = trace_list[0].x_array.shape[1]
    #         for i in range(NUM_PLOTS):
    #             plt.figure()
    #             ax = plt.gca()
    #             plt.title('x{}'.format(i))
    #             for trace in trace_list:
    #                 x_array = trace.x_array
    #                 t_array = trace.t_array
    #                 ax.plot(t_array, x_array[:, i])
    #             #plt.show()

        #plt.figure()
        #ax = plt.gca()
        #plt.title('x0-x1')
        ax.title('x0-x1')
        x = []
        y = []
        N = np.array([None])
        for trace in trace_list:
            x_array = trace.x_array
            x.append(x_array[:, 0])
            x.append(N)
            y.append(x_array[:, 1])
            y.append(N)
            #ax.plot(x_array[:, 0], x_array[:, 1])
        self.ax.plot(np.hstack(x), np.hstack(y))
        #plt.show()

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
