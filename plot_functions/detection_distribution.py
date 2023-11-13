import numpy as np
import matplotlib.pyplot as plt
from plot_functions.plot_utils import get_colors
    
    
def plot_detection_distribution(positive_detections, negative_detections, output_file: str):
    
    _, axs = plt.subplots(1, 1)
    
    if len(positive_detections) == 0:
        sensitivity = 1
        
    else:
        sensitivity = np.mean(positive_detections)
        
    miss_rate = 1 - sensitivity
    
    if len(negative_detections) == 0:
        specificity = 1
        
    else:
        specificity = np.mean(negative_detections)
    
    fall_out = 1 - specificity
    
    axs.set_xlabel('Probability')
    
    axs.set_yticks([1, 2, 3, 4])                  
    axs.set_yticklabels(['Sensitivity', 'Specificity', 'Miss rate',  'Fall out'])
    
    position = 0
    
    for score, color in zip([sensitivity, specificity, miss_rate, fall_out], get_colors(4)):
        
        position += 1
        
        axs.barh(position, score, 0.7, color=color)
    
    axs.set_xlim([0, 1])
    
    axs.grid(which='both', zorder=0)
    
    for bars in axs.containers:
        axs.bar_label(bars)
    
    plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()


def plot_multi_detection_distribution(positive_detections, negative_detections, legend_labels, output_file: str):
    
    _, axs = plt.subplots(1, 1)
    
    legends = []
    
    for label in legend_labels:
        if label not in legends:
            legends.append(label)
    
    legend_num = len(legends)
        
    axs.set_xlabel('Probability')
    
    axs.set_yticks([(1 + legend_num) / 2 + (legend_num + 1) * i for i in range(4)])
    axs.set_yticklabels(['Sensitivity', 'Specificity', 'Miss rate', 'Fall out']) 
    
    axs.set_xlim([0, 1])
    
    colors = get_colors(legend_num)
    
    sensitivity_per_legend = []
    specificity_per_legend = []
    
    for legend_index in range(legend_num):
        
        if len(positive_detections[legend_index]) == 0:
            sensitivity = 1
        
        else:
            sensitivity = np.mean(positive_detections[legend_index])
            
        sensitivity_per_legend.append(sensitivity)
        
        if len(negative_detections[legend_index]) == 0:
            specificity = 1
            
        else:
            specificity = np.mean(negative_detections[legend_index])
        
        specificity_per_legend.append(specificity)
        
    position = 0
    
    for sensitivity, color in zip(sensitivity_per_legend, colors):
        
        position += 1
        
        axs.barh(position, sensitivity, 0.7, color=color)
        
    position += 1
    
    
    for specificity, color in zip(specificity_per_legend, colors):
        
        position += 1
        
        axs.barh(position, specificity, 0.7, color=color)
        
    position += 1
    
    
    for sensitivity, color in zip(sensitivity_per_legend, colors):
        
        position += 1
        
        axs.barh(position, 1 - sensitivity, 0.7, color=color)
        
    position += 1
    
    
    for specificity, color in zip(specificity_per_legend, colors):
        
        position += 1
        
        axs.barh(position, 1 - specificity, 0.7, color=color)
    
    
    axs.grid(which='both', zorder=0)
    
    for bars in axs.containers:
        axs.bar_label(bars)
    
    if legends is not None:
    
        if len(legends) % 3 == 0:
            ncol=3
        else:
            ncol=2
        
        lgd = axs.legend(labels=legends, loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=ncol)

        plt.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
    else:
        
        plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()