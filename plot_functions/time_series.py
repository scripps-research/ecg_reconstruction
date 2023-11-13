import numpy as np
from plot_functions.plot_utils import get_colors
import matplotlib.pyplot as plt


def plot_time_series(data: np.ndarray, step_num: int, yticks, point_num: int, xlabel: str, ylabel: str, output_file: str):

    _, axs = plt.subplots(1, 1)

    data_num = len(data)

    if point_num > data_num:
        point_num = data_num

    point_len = int(data_num / point_num)

    data = np.resize(data[:point_len * point_num], (point_num, point_len))

    data_mean = np.mean(data, axis=1)
    data_median = np.median(data, axis=1)
    data_percent_75 = np.percentile(data, 75, axis=1)
    data_percent_25 = np.percentile(data, 25, axis=1)

    steps = np.linspace(0, step_num, num=point_num)

    axs.fill_between(steps, data_percent_25, data_percent_75, color='lightsalmon')

    axs.plot(steps, data_mean, linestyle='--', label='Mean', color='orangered')
    axs.plot(steps, data_median, linestyle='-.', label='Median', color='orangered')
    axs.plot(steps, data_percent_25, linestyle='-', label='25th percentile', color='orangered')
    axs.plot(steps, data_percent_75, linestyle='-', label='75th percentile', color='orangered')

    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    axs.set_xlim(0, step_num)

    if yticks is not None:
        axs.set_yticks(yticks)
        axs.set_ylim(yticks[0], yticks[-1])
    else:
        axs.set_ylim([np.min(data_percent_25), np.max(data_percent_75[int(point_num/20):])])

    axs.grid()
    axs.legend()

    plt.savefig(output_file, bbox_inches='tight')
    
    plt.yscale("log")

    plt.clf()
    plt.close()


def plot_multi_time_series(multi_data, multi_labels, multi_step, yticks, max_point_num: int, xlabel: str, ylabel: str, output_file: str, plot_format='png'):
    
    multi_colors = get_colors(len(multi_data))

    _, axs = plt.subplots(1, 1)

    max_index = np.argmax(multi_step)
    max_step = multi_step[max_index]

    for data, step, label, color in zip(multi_data, multi_step, multi_labels, multi_colors):

        data_num = len(data)

        point_num = int(max_point_num * step / max_step)
        
        point_len = int(data_num / point_num)

        data = np.resize(data[-point_len * point_num:], (point_num, point_len))

        data_mean = np.mean(data, axis=1)

        data_75_percent = np.percentile(data, 75, axis=1)
        data_25_percent = np.percentile(data, 25, axis=1)

        steps = np.linspace(0, step, num=point_num)

        axs.plot(steps, data_mean, label=label + ' (mean)', color=color, linestyle='-.')

        axs.plot(steps, data_75_percent, label=label + ' (25-75th percentile)', color=color, linestyle='-')

        axs.plot(steps, data_25_percent, color=color, linestyle='-')

    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)

    axs.set_xlim(0, max_step)

    if yticks is not None:

        axs.set_yticks(yticks)
        axs.set_ylim(yticks[0], yticks[-1])
        
    if len(multi_labels) % 3 == 0:
        ncol=3
    else:
        ncol=2

    axs.grid(which='both')
    lgd = axs.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=ncol)

    plt.savefig(output_file + '.' + plot_format, bbox_extra_artists=(lgd,), bbox_inches='tight', format=plot_format)
    
    plt.yscale("log")

    plt.clf()
    plt.close()