import numpy as np
import math
import matplotlib.pyplot as plt
from training_functions.classification_functions import get_detection_distributions
from plot_functions.plot_utils import get_colors, get_markers


def plot_roc_curve(detections, groundtruths, threshold_num: int, target_specificity: float, output_file: str, format='png'):
    
    if np.sum(groundtruths) > 0:
        
        sensitivity_values = []
        specificity_values = []
        accuracy_values = []
        
        threshold_values = np.linspace(0, 1, threshold_num)
        
        positive_size = np.sum(groundtruths)
        negative_size = len(groundtruths) - positive_size
        total_size = len(groundtruths)
    
        for threshold in threshold_values:
            
            positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)

            if len(positive_detections) > 0:
                sensitivity_values.append(np.mean(positive_detections))            
            else:        
                sensitivity_values.append(1)
                
            if len(negative_detections) > 0:                    
                specificity_values.append(np.mean(negative_detections))
            else:        
                specificity_values.append(1)
                
            accuracy_values.append((np.sum(positive_detections) + np.sum(negative_detections)) / total_size)
                
        _, axs = plt.subplots(1, 1)
            
        sensitivity_values = np.asarray(sensitivity_values)
        specificity_values = np.asarray(specificity_values)  
        accuracy_values = np.asarray(accuracy_values)
              
        fall_out_values = 1 - specificity_values    
        
        roc_area = (sensitivity_values[1:] + sensitivity_values[:-1]) * (fall_out_values[:-1] - fall_out_values[1:]) / 2
        roc_area = math.fsum(roc_area)
        
        axs.plot(fall_out_values, sensitivity_values, linestyle='-', label='AUC = ' + "{:.3f}".format(int(1000*roc_area)/1000), color='orangered')
    
        specificity_index = np.argmin(np.abs(specificity_values - target_specificity))
        target_sensitivity = sensitivity_values[specificity_index]
        target_threshold = threshold_values[specificity_index]

        axs.set_ylabel('Sensitivity')
        axs.set_xlabel('Fall out')

        axs.set_xlim(0, 1)

        axs.grid()
        axs.legend()

        plt.savefig(output_file + '.' + format, bbox_inches='tight', format=format)

        plt.clf()
        plt.close()
        
        return roc_area, target_threshold, target_sensitivity



def plot_multi_roc_curve(detections_list, groundtruths_list, label_list, threshold_num: int, target_specificity: float, output_file: str, format='png'):
    
    if np.sum(groundtruths_list[0]) > 0:
    
        _, axs = plt.subplots(1, 1)
        threshold_values = np.linspace(0, 1, threshold_num)
        
        specificity_indexes = []
        target_sensitivities = []
        target_thresholds = []
        roc_areas = []
    
        for detections, groundtruths, color, marker, label in zip(detections_list, groundtruths_list, get_colors(len(detections_list)), get_markers(len(detections_list)),  label_list):
            
            sensitivity_values = []
            specificity_values = []
            accuracy_values = []
            
            positive_size = np.sum(groundtruths)
            negative_size = len(groundtruths) - positive_size
            total_size = len(groundtruths)

            for threshold in threshold_values:
                
                positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)
                
                if len(positive_detections) > 0:
                    sensitivity_values.append(np.mean(positive_detections))            
                else:        
                    sensitivity_values.append(1)
                    
                if len(negative_detections) > 0:
                    specificity_values.append(np.mean(negative_detections))            
                else:        
                    specificity_values.append(1)
                    
                accuracy_values.append((np.sum(positive_detections) + np.sum(negative_detections)) / total_size)
                
            sensitivity_values = np.asarray(sensitivity_values)
            specificity_values = np.asarray(specificity_values)
            accuracy_values = np.asarray(accuracy_values)

            fall_out_values = 1 - specificity_values
            
            roc_area = (sensitivity_values[1:] + sensitivity_values[:-1]) * (fall_out_values[:-1] - fall_out_values[1:]) / 2 
            
            roc_area = math.fsum(roc_area)
            roc_areas.append(roc_area)

            axs.plot(fall_out_values, sensitivity_values, linestyle='-', marker=marker, label='AUC = ' + "{:.3f}".format(roc_area) + ' (' + label + ')', color=color, markevery=int(threshold_num / 10))
            specificity_indexes.append(np.argmin(np.abs(specificity_values - target_specificity)))            
            target_sensitivities.append(sensitivity_values[specificity_indexes[-1]])
            target_thresholds.append(threshold_values[specificity_indexes[-1]])
            
        axs.set_ylabel('Sensitivity')
        axs.set_xlabel('Fall out')

        axs.set_xlim(0, 0.25)
        axs.set_ylim(0.75, 1.0)

        axs.grid()

        plt.savefig(output_file + '.' + format, bbox_inches='tight', format=format)

        plt.clf()
        plt.close()
        
        return roc_areas, target_thresholds, target_sensitivities