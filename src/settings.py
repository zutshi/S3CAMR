debug = False
debug_plot = False


# CE hack is ON
CE = True


def plt_show():
    from matplotlib import pyplot as plt
    if debug_plot:
        plt.show()
    else:
        plt.close()
