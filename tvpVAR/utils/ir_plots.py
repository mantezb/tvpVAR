import numpy as np
import matplotlib.pyplot as plt


def ir_plots(ir_sort, bands, scale_adj, s):
    """
    Generate impulse response plots
    :param ir_sort: impulse response results
    :param bands: quantiles to be used for plotting
    :param scale_adj:
    :param s: no of time points
    :return: 4 plots
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.plot((ir_sort[0, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
             linestyle='dashed')
    ax1.plot((ir_sort[0, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
    ax1.set_title('Impulse Response of Variable x to y')
    ax1.set_ylabel('Impulse Response')
    ax1.set_xlabel('Time (in quarters)')


    ax2.plot((ir_sort[1, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
             linestyle='dashed')
    ax2.plot((ir_sort[1, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
    ax2.set_title('Impulse Response of variable y to y')
    ax2.set_ylabel('Impulse Response')
    ax2.set_xlabel('Time (in quarters)')


    ax3.plot((ir_sort[2, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
             linestyle='dashed')
    ax3.plot((ir_sort[2, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
    ax3.set_title('Impulse Response of variable z to y')
    ax3.set_ylabel('Impulse Response')
    ax3.set_xlabel('Time (in quarters)')


    ax4.plot((ir_sort[0, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
             linestyle='dashed')
    ax4.plot((ir_sort[1, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
    ax4.set_title('Impulse Response of variable y to y, different confidence bands')
    ax4.set_ylabel('Impulse Response')
    ax4.set_xlabel('Time (in quarters)')

    return fig
