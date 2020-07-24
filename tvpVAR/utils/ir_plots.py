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
    fig, ax = plt.subplots()

    plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2, linestyle='dashed')
    plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
    plt.title('Impulse Response of variable x to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')

    plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2, linestyle='dashed')
    plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
    plt.title('Impulse Response of variable y to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')

    plt.plot(np.squeeze(np.arange(0, s), ir_sort[2, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2, linestyle='dashed')
    plt.plot(np.squeeze(np.arange(0, s), ir_sort[2, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
    plt.title('Impulse Response of variable z to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')

    plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2, linestyle='dashed')
    plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
    plt.title('Impulse Response of variable y to y, different confidence bands')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')

    return fig
