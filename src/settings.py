from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

debug = False
debug_plot = False
plot = False


# CE hack is ON
CE = True


def plt_show():
    from matplotlib import pyplot as plt
    if debug_plot:
        plt.show()
    else:
        plt.close()
