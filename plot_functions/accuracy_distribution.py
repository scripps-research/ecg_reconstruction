import numpy as np
import matplotlib.pyplot as plt
from plot_functions.plot_utils import get_colors


def plot_accuracy_distribution(accuracy_distribution, parameter_labels, output_file: str):

    _, axs = plt.subplots(1, 1)
    
    if len(accuracy_distribution) == 0:
        accuracy = 1
    else:
        accuracy = np.mean(accuracy_distribution)
        
    error_rate = 1 - accuracy
                
    colors = get_colors(2)
    accuracy_color = colors[0]
    error_color = colors[1]
    
    axs.set_yticks([1, 2])                  
    axs.set_yticklabels(parameter_labels)                
    axs.set_xlabel('Probability')
    
    axs.barh(1, accuracy, 0.7, color=accuracy_color)
    axs.barh(2, error_rate, 0.7, color=error_color)
    
    axs.set_xlim([0, 1])
    
    axs.grid(which='both', zorder=0)
    
    for bars in axs.containers:
        axs.bar_label(bars)
    
    plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()


def plot_multi_accuracy_distribution(accuracy_distributions, model_labels, class_labels, parameter_labels, output_file: str):
    
    _, axs = plt.subplots(1, 1)
    
    unique_classes = []
    unique_models = []
    
    for label in class_labels:
        if label not in unique_classes:
            unique_classes.append(label)
            
    for label in model_labels:
        if label not in unique_models:
            unique_models.append(label)
    
    unique_class_num = len(unique_classes)
    unique_model_num = len(unique_models)
    
    accuracy_scores = []
    error_scores = []
        
    for x in accuracy_distributions:
        if len(x) == 0:                
            accuracy_scores.append(1)
            error_scores.append(0)
        else:
            accuracy_scores.append(np.mean(x))
            error_scores.append(1 - np.mean(x))

    if unique_model_num == 1:
        
        legends = unique_classes
        
        axs.set_xlabel('Probability')
                
        axs.set_yticks([(unique_class_num + 1) / 2 + (unique_class_num + 1) * i for i in range(2)])
        axs.set_yticklabels(parameter_labels)
        
        colors = get_colors(unique_class_num)
        
        position = 0
        
        for accuracy, color in zip(accuracy_scores, colors):
            
            position += 1
            
            axs.barh(position, accuracy, 0.7, color=color)
            
        position += 1
            
        for error, color in zip(error_scores, colors):
            
            position += 1
            
            axs.barh(position, error, 0.7, color=color)
    
    elif unique_class_num == 1:
        
        legends = unique_models
        
        axs.set_xlabel('Probability')
                
        axs.set_yticks([(unique_model_num + 1) / 2 + (unique_model_num + 1) * i for i in range(2)])
        axs.set_yticklabels(parameter_labels)
        
        colors = get_colors(unique_model_num)
        
        position = 0
        
        for accuracy, color in zip(accuracy_scores, colors):
            
            position += 1
            
            axs.barh(position, accuracy, 0.7, color=color)
            
        position += 1
            
        for error, color in zip(error_scores, colors):
            
            position += 1
            
            axs.barh(position, error, 0.7, color=color)
            
    else:
        
        colors = get_colors(unique_model_num)
        legends = unique_models
        
        axs.set_xlabel(parameter_labels[0])            
        axs.set_yticks([(unique_model_num + 1) / 2 + (unique_model_num + 1) * i for i in range(unique_class_num)])
        
        axs.set_yticklabels(unique_classes)
        
        position = 0
        
        for class_label in unique_classes:
            
            class_indexes = np.where(np.array(class_labels) == class_label)[0]
            
            class_accuracies = [accuracy_scores[index] for index in class_indexes]
            
            model_labels_per_class =  [model_labels[index] for index in class_indexes]
            
            class_accuracies = [class_accuracies[index] for index in [unique_models.index(model_class) for model_class in model_labels_per_class]]
            
            for accuracy, color in zip(class_accuracies, colors):
                
                position +=1
            
                axs.barh(position, accuracy, 0.7, color=color)
                
            position += 1
    
    axs.set_xlim([0, 1])
    
    axs.grid(which='both', zorder=0)
    
    for bars in axs.containers:
        axs.bar_label(bars)
    
    if len(legends) % 3 == 0:
        ncol=3
    else:
        ncol=2
    
    lgd = axs.legend(labels=legends, loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=ncol)

    plt.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.clf()
    plt.close()