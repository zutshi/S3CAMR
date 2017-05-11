###############################################################################
# File name: test_exifc.py
# Author: Aditya
# Python Version: 2.7
#
#                       #### Description ####
# Test system loading and simulation using external_interface.py
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import external_interface as exifc
import traces

import numpy as np
import time

import matplotlib
# Force GTK3 backend. By default GTK2 gets loaded and conflicts with
# graph-tool
matplotlib.use('GTK3Agg')
#global plt
import matplotlib.pyplot as plt


# TODO: test all examples using the examplel isting
def main():
    systems = [
        './examples/fuzzy_invp/fuzzy_invp.tst',
        './examples/heater/heater.tst',
        './examples/dc_motor/dci.tst',
        './examples/toy_model_10u/toy_model_10u.tst',
        './examples/heat/heat.tst',
        './examples/spi/spi.tst',
            ]

    for s in systems:
        one_shot_sim, prop = exifc.load_system(s)
        NUM_SIMS = 1
        x0 = prop.init_cons.sample_UR(NUM_SIMS)

        w0 = prop.ci.sample_UR(np.ceil(prop.T/prop.delta_t))

        trace_list = []
        tic = time.time()
        for i in range(NUM_SIMS):
            trace = one_shot_sim(x0[i], 0.0, prop.T, w0)
            trace_list.append(trace)
        toc = time.time()
        print('time taken for simulations: {}s'.format(toc-tic))
        traces.plot_trace_list(trace_list, plt)

if __name__ == '__main__':
    main()
