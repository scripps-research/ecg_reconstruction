import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_2d_distribution(x, y, x_label, y_label, output_file: str, xticks=None, xbins=None, ybins=None, gamma=0.1, xlim=None, rotate=False):

    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots()

    if xbins is not None and ybins is not None:

        bins = (xbins, ybins)
    
    elif xbins is not None:

        bins = (xbins, xbins)

    elif ybins is not None:

        bins = (ybins, ybins)

    else:

        bins= (100, 100)

    h = ax.hist2d(x, y, bins=bins, density=True, norm=mcolors.PowerNorm(gamma))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True)

    if xticks is not None:

        ax.set_xticks([0] + ( (np.arange(1, len(xticks)+1) - .5) * (len(xticks)-1) / len(xticks) ).tolist() + [len(xticks)-1])

        ax.set_xticklabels([''] + xticks + [''])

    if xlim is not None:

        ax.set_xlim(xlim)
        
    if rotate:
        
        plt.xticks(rotation = 90)

    fig.colorbar(h[3], ax=ax, label='Probability Density Function')

    plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()

def plot_multi_2d_distribution(x_list, y_list, x_label, y_label, labels, colors, output_file: str, xticks=None, alpha=0.5):

    _, ax = plt.subplots()

    for x, y, label, color in zip(x_list, y_list, labels, colors):

        x = np.asarray(x)
        y = np.asarray(y)
    
        ax.scatter(x, y, alpha=alpha, label=label, color=color)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.grid(True)

        if xticks is not None:

            ax.set_xticks(np.arange(len(xticks)))

            ax.set_xticklabels(xticks)
            
    if len(labels) % 3 == 0:
        ncol=3
    else:
        ncol=2

    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=ncol)

    plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()
