import numpy as np
from plot_functions.plot_utils import get_colors
import matplotlib.pyplot as plt


def plot_violin_distribution(data, xlabels, ylabel: str, output_file: str, yticks=None, ylims=None, plot_format='png'):
    
    colors = get_colors(len(data))

    _, axs = plt.subplots(1, 1)

    quantiles = [[0.05, 0.25, 0.75, 0.95]] * len(data)

    distributions = axs.violinplot(data, showmeans=False, showextrema=True, showmedians=False, points=500, quantiles=quantiles)

    medians = [np.median(x) for x in data]
    means = [np.mean(x) for x in data]

    axs.scatter(range(1, len(data)+1), medians, label='Median', marker='o', color='white', edgecolors='royalblue', s=30, zorder=3)
    axs.scatter(range(1, len(data)+1), means, label='Mean', marker='d', color='white', edgecolors='royalblue', s=30, zorder=3)

    axs.set_xticks(np.arange(1, len(xlabels) + 1))

    axs.set_xticklabels(xlabels, rotation='vertical')

    axs.set_ylabel(ylabel)
    
    if colors is not None:
        
        for dist, color in zip(distributions['bodies'], colors):
            dist.set_facecolor(color)

    if yticks is not None:

        axs.set_yticks(yticks)
        axs.set_ylim(yticks[0], yticks[-1])
        
    elif ylims is not None:
        
        axs.set_ylim(ylims[0], ylims[-1])

    axs.grid(which='both')
    axs.legend()

    plt.savefig(output_file + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.clf()
    plt.close()
    

def plot_box_distribution(data, xlabels, ylabel: str, output_file: str, yticks=None, ylims=None, plot_format='png'):
    
    colors = get_colors(len(data))

    _, axs = plt.subplots(1, 1)
    
    distributions = axs.boxplot(data,
                                vert=True,
                                patch_artist=True,
                                notch=True,
                                labels=xlabels,
                                showfliers=False,
                                whis=0.95)
    
    for patch, color in zip(distributions['boxes'], colors):
        patch.set_facecolor(color)
    
    for median in distributions['medians']:
        median.set_color('snow')
    

    axs.set_xticks(np.arange(1, len(xlabels) + 1))

    axs.set_xticklabels(xlabels, rotation='vertical')

    axs.set_ylabel(ylabel)

    if yticks is not None:

        axs.set_yticks(yticks)
        axs.set_ylim(yticks[0], yticks[-1])
        
    elif ylims is not None:
        
        axs.set_ylim(ylims[0], ylims[-1])

    axs.grid(which='both')

    plt.savefig(output_file + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.clf()
    plt.close()
    