PLOT = True


def plt_show():
    from matplotlib import pyplot as plt
    if PLOT:
        plt.show()
    else:
        plt.close()
