from plot_functions.roc_curve import plot_multi_roc_curve
from plot_functions.detection_distribution import plot_multi_detection_distribution
from plot_functions.accuracy_distribution import plot_multi_accuracy_distribution
from training_functions.multi_training_manager import MultiTrainingManager
from training_functions.single_classification_manager import ClassificationManager
from training_functions.single_recon_classif_manager import ReconClassifManager
from training_functions.recon_classif_functions import post_process_element, process_element, element_loss_function
from training_functions.classification_functions import get_detection_distributions
from training_functions.classification_functions import process_element as process_classif_element
from training_functions.reconstruction_functions import deprocess_element
from plot_functions.ecg_signal import plot_output_multi_recon_leads, plot_medical_multi_recon_leads
from util_functions.classification_settings import get_classification_default_settings
from util_functions.general import remove_dir, get_twelve_keys, get_parent_folder
from util_functions.load_data_ids import load_dataclass_ids
from plot_functions.continuos_distribution import plot_violin_distribution, plot_box_distribution
import pandas as pd
import random
import itertools
import pathlib
import torch
import copy
import numpy as np


class MultiReconClassifManager(MultiTrainingManager):
    
    def __init__(self,
                 output_name: str,
                 parent_folder: str,
                 device: str,
                 sub_classes: list,
                 input_leads: str,
                 output_leads: str,
                 alpha: float,
                 parallel_classification: bool,
                 data_classes: str,
                 data_size: str,
                 detect_classes,
                 use_residual,
                 epoch_num,
                 batch_size,
                 prioritize_percent,
                 prioritize_size,
                 optimizer_algorithm,
                 learning_rate,
                 weight_decay,
                 momentum,
                 nesterov,
                 manager_labels,
                 ):
        
        self.multi_manager = []
        self.manager_labels = []

        self.output_folder = parent_folder + 'ReconClassif/compare_results/' + output_name + '/'

        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        self.manager_num = len(manager_labels)
        
        if isinstance(sub_classes[0], list):
            sub_classes_list = sub_classes
        else:
            sub_classes_list = [sub_classes] * self.manager_num
        
        if isinstance(input_leads, list):
            input_leads_list = input_leads
        else:
            input_leads_list = [input_leads] * self.manager_num
            
        if isinstance(output_leads, list):
            output_leads_list = output_leads
        else:
            output_leads_list = [output_leads] * self.manager_num
            
        if isinstance(alpha, list):
            alpha_list = alpha
        else:
            alpha_list = [alpha] * self.manager_num
        
        if isinstance(parallel_classification, list):
            parallel_classification_list = parallel_classification
        else:
            parallel_classification_list = [parallel_classification] * self.manager_num
            
        if isinstance(data_classes[0], list):
            data_classes_list = data_classes
        else:
            data_classes_list = [data_classes] * self.manager_num

        if isinstance(data_size, list):
            data_size_list = data_size
        else:
            data_size_list = [data_size] * self.manager_num
            
        if isinstance(detect_classes[0], list):
            detect_classes_list = detect_classes
        else:
            detect_classes_list = [detect_classes] * self.manager_num
            
        if isinstance(use_residual, list):
            use_residual_list = use_residual
        else:
            use_residual_list = [use_residual] * self.manager_num
            
        if isinstance(epoch_num, list):
            epochs_list = epoch_num
        else:
            epochs_list = [epoch_num] * self.manager_num

        if isinstance(batch_size, list):
            batch_size_list = batch_size
        else:
            batch_size_list = [batch_size] * self.manager_num

        if isinstance(prioritize_percent, list):
            prioritize_percent_list = prioritize_percent
        else:
            prioritize_percent_list = [prioritize_percent] * self.manager_num

        if isinstance(prioritize_size, list):
            prioritize_size_list = prioritize_size
        else:
            prioritize_size_list = [prioritize_size] * self.manager_num

        if isinstance(optimizer_algorithm, list):
            optimizer_list = optimizer_algorithm
        else:
            optimizer_list = [optimizer_algorithm] * self.manager_num

        if isinstance(learning_rate, list):
            learning_rate_list = learning_rate
        else:
            learning_rate_list = [learning_rate] * self.manager_num

        if isinstance(weight_decay, list):
            decay_list = weight_decay
        else:
            decay_list = [weight_decay] * self.manager_num

        if isinstance(momentum, list):
            momentum_list = momentum
        else:
            momentum_list = [momentum] * self.manager_num
        
        if isinstance(nesterov, list):
            nesterov_list = nesterov
        else:
            nesterov_list = [nesterov] * self.manager_num
            
        for sub_classes, input_leads, output_leads, alpha, parallel_classification, data_classes, data_size, detect_classes, use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size,\
            optimizer, learning_rate, weight_decay, momentum, nesterov, manager_label in zip(sub_classes_list,
                                                                                             input_leads_list,
                                                                                             output_leads_list,
                                                                                             alpha_list,
                                                                                             parallel_classification_list,
                                                                                             data_classes_list,
                                                                                             data_size_list,
                                                                                             detect_classes_list,
                                                                                             use_residual_list,
                                                                                             epochs_list,
                                                                                             batch_size_list,
                                                                                             prioritize_percent_list,
                                                                                             prioritize_size_list,
                                                                                             optimizer_list,
                                                                                             learning_rate_list,
                                                                                             decay_list,
                                                                                             momentum_list,
                                                                                             nesterov_list,
                                                                                             manager_labels):
                    
            if output_leads is None:
                
                _, input_channel, middle_channel, input_depth, \
                    middle_depth, output_depth, input_kernel, middle_kernel, stride_size, classif_use_residual, \
                        classif_epoch_num, classif_batch_size, classif_prioritize_percent, classif_prioritize_size, \
                            classif_optimizer, classif_learning_rate, classif_weight_decay, classif_momentum, classif_nesterov = get_classification_default_settings()

            
                manager = ClassificationManager(parent_folder,
                                                device,
                                                sub_classes,
                                                input_leads,
                                                parallel_classification,
                                                data_classes,
                                                data_size,
                                                detect_classes,
                                                input_channel,
                                                middle_channel,
                                                input_depth,
                                                middle_depth,
                                                output_depth,
                                                input_kernel,
                                                middle_kernel,
                                                stride_size,
                                                classif_use_residual,
                                                classif_epoch_num,
                                                classif_batch_size,
                                                classif_prioritize_percent,
                                                classif_prioritize_size,
                                                classif_optimizer,
                                                classif_learning_rate,
                                                classif_weight_decay,
                                                classif_momentum,
                                                classif_nesterov)
            
            else:
                
                manager = ReconClassifManager(parent_folder,
                                              device,
                                              sub_classes,
                                              input_leads,
                                              output_leads,
                                              alpha,
                                              parallel_classification,
                                              data_classes,
                                              data_size,
                                              detect_classes,
                                              use_residual,
                                              epoch_num, 
                                              batch_size,
                                              prioritize_percent,
                                              prioritize_size,
                                              optimizer,
                                              learning_rate,
                                              weight_decay,
                                              momentum,
                                              nesterov)
            
            self.manager_labels.append(manager_label)
            
            self.multi_manager.append(manager)
        
    def plot_test_stats(self, threshold_num = 1001, plot_sub_classes = None, plot_format='png'):
        
        pathlib.Path(self.output_folder + 'test_stats/').mkdir(parents=True, exist_ok=True)
        
        loss_ticks = [x / 20 for x in range(-10, 1)]
        r2_ticks = [x / 0.1 for x in range(0, 11)]
        mse_ticks = [ x / 200 for x in range(0, 11)]
        bce_ticks = [ x / 20 for x in range(0, 11)]
        
        multi_losses = []
        multi_mserrors = []
        multi_rsquareds = []
        multi_entropies = []
        
        multi_labels = []
        classification_labels = []
        
        for manager, label in zip(self.multi_manager, self.manager_labels):
            
            if isinstance(manager, ReconClassifManager):
                
                multi_losses.append(manager.test_losses)
                multi_mserrors.append(manager.test_mserrors)
                multi_rsquareds.append(manager.test_rsquareds * 100)
                multi_entropies.append(manager.test_entropies)
                multi_labels.append(label)
                classification_labels.append(label)
                
            elif isinstance(manager, ClassificationManager):
                
                multi_entropies.append(manager.test_entropies)
                classification_labels.append(label)
            
            else:
                raise ValueError
                
        plot_violin_distribution(multi_losses, multi_labels, 'Loss', self.output_folder + 'test_stats/test_losses', yticks=loss_ticks, plot_format=plot_format)
        plot_violin_distribution(multi_mserrors, multi_labels, 'MSE [$mV^2$]', self.output_folder + 'test_stats/test_mserrors', yticks=mse_ticks, plot_format=plot_format)
        plot_violin_distribution(multi_rsquareds, multi_labels, 'R2 [%]', self.output_folder + 'test_stats/test_rsquareds', yticks=r2_ticks, plot_format=plot_format)
        plot_violin_distribution(multi_entropies, classification_labels, 'BCE', self.output_folder + 'test_stats/test_entropies', yticks=bce_ticks, plot_format=plot_format)
        
        plot_box_distribution(multi_losses, multi_labels, 'Loss', self.output_folder + 'test_stats/test_losses_box', yticks=loss_ticks, plot_format=plot_format)
        plot_box_distribution(multi_mserrors, multi_labels, 'MSE [$mV^2$]', self.output_folder + 'test_stats/test_mserrors_box', yticks=mse_ticks, plot_format=plot_format)
        plot_box_distribution(multi_rsquareds, multi_labels, 'R2 [%]', self.output_folder + 'test_stats/test_rsquareds_box', yticks=r2_ticks, plot_format=plot_format)
        plot_box_distribution(multi_entropies, classification_labels, 'BCE', self.output_folder + 'test_stats/test_entropies_box', yticks=bce_ticks, plot_format=plot_format)
        
        detect_classes = self.multi_manager[0].detect_classes
        detect_class_num = len(detect_classes)
        
        if plot_sub_classes is None:
            plot_sub_classes = self.multi_manager[0].sub_classes
                            
        plot_sub_class_num = len(plot_sub_classes)
        manager_num = len(self.multi_manager)
        
        for manager in self.multi_manager:
            
            assert manager.detect_classes == detect_classes
            
            for plot_sub_class in plot_sub_classes:
                
                assert plot_sub_class in manager.sub_classes
        
        best_threshold_per_class = []
        
        for class_index, detect_class in enumerate(detect_classes):
            
            pathlib.Path(self.output_folder + 'test_stats/roc_curves/').mkdir(parents=True, exist_ok=True)
            
            detections_per_manager = []
            groundtruths_per_manager = []
            
            for manager in self.multi_manager:
            
                detections = manager.test_probabilities[:, class_index]
                groundtruths = manager.test_groundtruths[:, class_index]
                
                detections_per_manager.append(detections)
                groundtruths_per_manager.append(groundtruths)
                        
            roc_areas, target_thresholds, target_sensitivities = plot_multi_roc_curve(detections_per_manager,
                                                   groundtruths_per_manager,
                                                   self.manager_labels,
                                                   threshold_num,
                                                   0.9,
                                                   self.output_folder + 'test_stats/roc_curves/' + detect_class,
                                                   format=plot_format)
            
            best_threshold_per_class += target_thresholds                      

        for threshold in best_threshold_per_class:
            
            total_detections_per_class_per_manager = [[] for _ in range(detect_class_num)]
            positive_detections_per_class_per_manager = [[] for _ in range(detect_class_num)]
            negative_detections_per_class_per_manager = [[] for _ in range(detect_class_num)]  
        
            for class_index, detect_class in enumerate(detect_classes):
                
                detections_per_manager = []
                groundtruths_per_manager = []
                
                for manager in self.multi_manager:
                
                    detections = manager.test_probabilities[:, class_index]
                    groundtruths = manager.test_groundtruths[:, class_index]
                    
                    detections_per_manager.append(detections)
                    groundtruths_per_manager.append(groundtruths)
                    
                pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/').mkdir(parents=True, exist_ok=True)
                
                for detections, groundtruths in zip(detections_per_manager, groundtruths_per_manager):
                
                    positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)
                    total_detections = np.concatenate((positive_detections, negative_detections))
                    
                    positive_detections_per_class_per_manager[class_index].append(positive_detections)
                    negative_detections_per_class_per_manager[class_index].append(negative_detections)
                    total_detections_per_class_per_manager[class_index].append(total_detections)
                    
                plot_multi_accuracy_distribution(total_detections_per_class_per_manager[class_index],
                                                self.manager_labels,
                                                [detect_class] * manager_num,
                                                ['Accuracy', 'Error rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/accuracy')
                
                plot_multi_accuracy_distribution(positive_detections_per_class_per_manager[class_index],
                                                self.manager_labels,
                                                [detect_class] * manager_num,
                                                ['Sensitivity', 'Miss rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/sensitivity')
                
                plot_multi_accuracy_distribution(negative_detections_per_class_per_manager[class_index],
                                                self.manager_labels,
                                                [detect_class] * manager_num,
                                                ['Specificity', 'Fall out'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/specificity')
                
                plot_multi_detection_distribution(positive_detections_per_class_per_manager[class_index],
                                                negative_detections_per_class_per_manager[class_index],
                                                self.manager_labels,
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/detection')
                
            model_labels = list(itertools.chain.from_iterable([self.manager_labels for _ in range(detect_class_num)]))
            class_labels = list(itertools.chain.from_iterable([[detect_class for _ in range(manager_num)] for detect_class in detect_classes]))
            
            if detect_class_num > 1:
                
                plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(total_detections_per_class_per_manager)),
                                                model_labels,
                                                class_labels,
                                                ['Accuracy', 'Error rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy_per_class')
                
                plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(positive_detections_per_class_per_manager)),
                                                model_labels,
                                                class_labels,
                                                ['Sensitivity', 'Miss rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity_per_class')
                
                plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(negative_detections_per_class_per_manager)),
                                                model_labels,
                                                class_labels,
                                                ['Specificity', 'Fall out'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity_per_class')
            
                total_detections_per_manager = [np.concatenate([total_detections_per_class_per_manager[class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                positive_detections_per_manager = [np.concatenate([positive_detections_per_class_per_manager[class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                negative_detections_per_manager = [np.concatenate([negative_detections_per_class_per_manager[class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                
                plot_multi_accuracy_distribution(total_detections_per_manager,
                                                self.manager_labels,
                                                ['Overall'] * manager_num,
                                                ['Accuracy', 'Error rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy')
                
                plot_multi_accuracy_distribution(positive_detections_per_manager,
                                                self.manager_labels,
                                                ['Overall'] * manager_num,
                                                ['Sensitivity', 'Miss rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity')
                
                plot_multi_accuracy_distribution(negative_detections_per_manager,
                                                self.manager_labels,
                                                ['Overall'] * manager_num,
                                                ['Specificity', 'Fall out'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity')
                
                plot_multi_detection_distribution(positive_detections_per_manager,
                                                negative_detections_per_manager,
                                                self.manager_labels,
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detection')

            if plot_sub_class_num > 0:
                
                total_detections_per_class_per_sub_class_per_manager = [[[] for _ in range(plot_sub_class_num)] for _ in range(detect_class_num)]
                positive_detections_per_class_per_sub_class_per_manager = [[[] for _ in range(plot_sub_class_num)] for _ in range(detect_class_num)]
                negative_detections_per_class_per_sub_class_per_manager = [[[] for _ in range(plot_sub_class_num)] for _ in range(detect_class_num)]
                
                for sub_class_index, plot_sub_class in enumerate(plot_sub_classes):
                    
                    pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/').mkdir(parents=True, exist_ok=True)
                    
                    for class_index, detect_class in enumerate(detect_classes):
                        
                        pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/').mkdir(parents=True, exist_ok=True)
                        
                        for manager in self.multi_manager:
                            
                            if len(manager.test_probabilities_per_sub_class[manager.sub_classes.index(plot_sub_class)]) > 0:
                                detections = manager.test_probabilities_per_sub_class[manager.sub_classes.index(plot_sub_class)][:, class_index]
                                groundtruths = manager.test_groundtruths_per_sub_class[manager.sub_classes.index(plot_sub_class)][:, class_index]
                            else:
                                detections = np.array([])
                                groundtruths = np.array([])
                        
                            positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)
                            total_detections = np.concatenate((positive_detections, negative_detections))
                            
                            total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index].append(total_detections)                        
                            positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index].append(positive_detections)
                            negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index].append(negative_detections)
                            
                        plot_multi_accuracy_distribution(total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index],
                                                        self.manager_labels,
                                                        [detect_class] * manager_num,
                                                        ['Accuracy', 'Error rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/accuracy')
                        
                        plot_multi_accuracy_distribution(positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index],
                                                        self.manager_labels,
                                                        [detect_class] * manager_num,
                                                        ['Sensitivity', 'Miss rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/sensitivity')
                        
                        plot_multi_accuracy_distribution(negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index],
                                                        self.manager_labels,
                                                        [detect_class] * manager_num,
                                                        ['Specificity', 'Fall out'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/specificity')
                        
                        plot_multi_detection_distribution(positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index],
                                                        negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index],
                                                        self.manager_labels,
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/detection')

                    model_labels = list(itertools.chain.from_iterable([self.manager_labels for _ in range(detect_class_num)]))
                    class_labels = list(itertools.chain.from_iterable([[detect_class for _ in range(manager_num)] for detect_class in detect_classes]))
                    
                    total_detections_per_class_per_manager = [[total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for class_index in range(detect_class_num)] 
                    positive_detections_per_class_per_manager = [[positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for class_index in range(detect_class_num)] 
                    negative_detections_per_class_per_manager = [[negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for class_index in range(detect_class_num)]              
                    
                    if detect_class_num > 1:                
                    
                        plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(total_detections_per_class_per_manager)),
                                                        model_labels,
                                                        class_labels,
                                                        ['Accuracy', 'Error rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/accuracy_per_class')
                        
                        plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(positive_detections_per_class_per_manager)),
                                                        model_labels,
                                                        class_labels,
                                                        ['Sensitivity', 'Miss rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/sensitivity_per_class')
                        
                        plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(negative_detections_per_class_per_manager)),
                                                        model_labels,
                                                        class_labels,
                                                        ['Specificity', 'Fall out'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/specificity_per_class')
                        
                        total_detections_per_manager = [np.concatenate([total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                        positive_detections_per_manager = [np.concatenate([positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                        negative_detections_per_manager = [np.concatenate([negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)]
                        
                        plot_multi_accuracy_distribution(total_detections_per_manager,
                                                        self.manager_labels,
                                                        ['Overall'] * manager_num,
                                                        ['Accuracy', 'Error rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/accuracy')
                        
                        plot_multi_accuracy_distribution(positive_detections_per_manager,
                                                        self.manager_labels,
                                                        ['Overall'] * manager_num,
                                                        ['Sensitivity', 'Miss rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/sensitivity')
                        
                        plot_multi_accuracy_distribution(negative_detections_per_manager,
                                                        self.manager_labels,
                                                        ['Overall'] * manager_num,
                                                        ['Specificity', 'Fall out'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/specificity')
                        
                        plot_multi_detection_distribution(positive_detections_per_manager,
                                                        negative_detections_per_manager,
                                                        self.manager_labels,
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detection')
                    
                model_labels = list(itertools.chain.from_iterable([self.manager_labels for _ in range(plot_sub_class_num)]))
                sub_class_labels = list(itertools.chain.from_iterable([[plot_sub_class for _ in range(manager_num)] for plot_sub_class in plot_sub_classes]))
                    
                for class_index, detect_class in enumerate(detect_classes):
                    
                    total_detections_per_sub_class_per_manager = [[total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                    positive_detections_per_sub_class_per_manager = [[positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                    negative_detections_per_sub_class_per_manager = [[negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                        
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(total_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Accuracy', 'Error rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/accuracy_per_sub_class')
                    
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(positive_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Sensitivity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/sensitivity_per_sub_class')
                    
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(negative_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Specificity', 'Fall out'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/specificity_per_sub_class')
                    
                if detect_class_num > 1:
                    
                    model_labels = list(itertools.chain.from_iterable([self.manager_labels for _ in range(plot_sub_class_num)]))
                    sub_class_labels = list(itertools.chain.from_iterable([[plot_sub_class for _ in range(manager_num)] for plot_sub_class in plot_sub_classes]))
                    
                    total_detections_per_sub_class_per_manager = [[np.concatenate([total_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                    positive_detections_per_sub_class_per_manager = [[np.concatenate([positive_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                    negative_detections_per_sub_class_per_manager = [[np.concatenate([negative_detections_per_class_per_sub_class_per_manager[class_index][sub_class_index][manager_index] for class_index in range(detect_class_num)]) for manager_index in range(manager_num)] for sub_class_index in range(plot_sub_class_num)]
                        
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(total_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Accuracy', 'Error rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy_per_sub_class')
                    
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(positive_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Sensitivity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity_per_sub_class')
                    
                    plot_multi_accuracy_distribution(list(itertools.chain.from_iterable(negative_detections_per_sub_class_per_manager)),
                                                    model_labels,
                                                    sub_class_labels,
                                                    ['Specificity', 'Fall out'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity_per_sub_class')
                    
    def plot_random_example(self, size=1, target_idx=0, plot_format='png'):
    
        remove_dir(self.output_folder + 'random_plot/')
        
        recon_managers = []
        recon_manager_labels = []
        
        for manager, label in zip(self.multi_manager, self.manager_labels):
            if isinstance(manager, ReconClassifManager):
                recon_managers.append(manager)
                recon_manager_labels.append(label)
        
        recon_managers[target_idx].data_loader.shuffle('test')

        batch_ids = recon_managers[target_idx].data_loader.get_random_id('test', size)
        
        twelve_keys = get_twelve_keys()

        for element_id in batch_ids:
                     
            multi_model_leads = []
            multi_output_keys = []
            multi_input_keys = []
            
            twelve_leads = recon_managers[0].data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * recon_managers[0].amplitude + recon_managers[0].min_value for lead in twelve_leads]

            for manager in recon_managers:
                
                recon_input_leads, classif_input_leads, target_probabilities, _, _ = manager.data_loader.load_element(element_id, False)
                
                reconstruction_input, _, _, _ = process_element(recon_input_leads,
                                                                classif_input_leads,
                                                                manager.output_lead_keys,
                                                                manager.classif_input_lead_keys,
                                                                target_probabilities,
                                                                manager.min_value,
                                                                manager.amplitude,
                                                                manager.device)

                reconstruction_output = manager.model.forward(reconstruction_input)

                model_leads = deprocess_element(reconstruction_output)
            
                multi_model_leads.append(model_leads)
            
                multi_output_keys.append(manager.output_lead_keys)
                multi_input_keys.append(manager.input_lead_keys)
                        
            pathlib.Path(self.output_folder + 'random_plot/standard/').mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.output_folder + 'random_plot/medical/').mkdir(parents=True, exist_ok=True)
            
            plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, recon_manager_labels, self.output_folder + 'random_plot/medical/' + str(element_id), plot_format=plot_format)
                            
            plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, recon_manager_labels, self.output_folder + 'random_plot/standard/' + str(element_id), plot_format=plot_format)


    def plot_error_example(self, target_idx=0, plot_format='png'):
    
        remove_dir(self.output_folder + 'error_plot/')
        
        recon_managers = []
        recon_manager_labels = []
        
        for manager, label in zip(self.multi_manager, self.manager_labels):
            if isinstance(manager, ReconClassifManager):
                recon_managers.append(manager)
                recon_manager_labels.append(label)

        print('Reference test set:', recon_manager_labels[target_idx])

        recon_managers[target_idx].data_loader.shuffle('test')

        recon_managers[target_idx].load_test_stats()

        percentiles = [0, 25, 50, 75, 100]

        thresholds = [np.percentile(recon_managers[target_idx].test_losses, percent) for percent in percentiles]

        threshold_num = len(thresholds)

        element_per_threshold = np.zeros(threshold_num - 1)
        
        twelve_keys = get_twelve_keys()

        while np.sum(element_per_threshold) < threshold_num - 1:

            recon_input_leads, classif_input_leads, target_probabilities, _, element_id = recon_managers[target_idx].data_loader.get_next_element('test', False)
            
            twelve_leads = recon_managers[target_idx].data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * recon_managers[target_idx].amplitude + recon_managers[target_idx].min_value for lead in twelve_leads]
            
            reconstruction_input, reconstruction_target, classification_input, classification_target = process_element(recon_input_leads,
                                                                classif_input_leads,
                                                                recon_managers[target_idx].output_lead_keys,
                                                                recon_managers[target_idx].classif_input_lead_keys,
                                                                target_probabilities,
                                                                recon_managers[target_idx].min_value,
                                                                recon_managers[target_idx].amplitude,
                                                                recon_managers[target_idx].device)

            with torch.no_grad():
                
                reconstruction_output = recon_managers[target_idx].model.forward(reconstruction_input)                
                
                classification_input = post_process_element(classification_input,
                                                            reconstruction_output,
                                                            recon_managers[target_idx].classif_input_lead_keys,
                                                            recon_managers[target_idx].output_lead_keys,
                                                            recon_managers[target_idx].min_value,
                                                            recon_managers[target_idx].amplitude)
            
                classification_output = recon_managers[target_idx].classificator.forward(classification_input)
                           
                           
            loss, _, _, _ = element_loss_function(reconstruction_output, reconstruction_target, classification_output, classification_target, recon_managers[target_idx].output_lead_num, recon_managers[target_idx].detect_class_num, recon_managers[target_idx].alpha)

            for i in range(threshold_num - 1):

                if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] == 0:

                    element_per_threshold[i] += 1
                    
                    multi_model_leads = []
                    multi_output_keys = []
                    multi_input_keys = []

                    for manager in recon_managers:
                        
                        recon_input_leads, classif_input_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
            
                        reconstruction_input, classification_input, reconstruction_target, classification_target = process_element(recon_input_leads,
                                                            classif_input_leads,
                                                            manager.output_lead_keys,
                                                            manager.classif_input_lead_keys,
                                                            target_probabilities,
                                                            manager.min_value,
                                                            manager.amplitude,
                                                            manager.device)

                        reconstruction_output = manager.model.forward(reconstruction_input)

                        model_leads = deprocess_element(reconstruction_output)
                        
                        multi_model_leads.append(model_leads)                            
                        multi_output_keys.append(manager.output_lead_keys)
                        multi_input_keys.append(manager.input_lead_keys)
                    
                        pathlib.Path(self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        pathlib.Path(self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        
                    plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, recon_manager_labels, self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                    plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, recon_manager_labels, self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                
                    break
            
            
    def plot_sub_class_example(self, plot_sub_classes = None, target_idx = 0, plot_format = 'png'):
        
        remove_dir(self.output_folder + 'sub_class_plot/')
        
        recon_managers = []
        recon_manager_labels = []
        
        for manager, label in zip(self.multi_manager, self.manager_labels):
            if isinstance(manager, ReconClassifManager):
                recon_managers.append(manager)
                recon_manager_labels.append(label)
                
        percentiles = [0, 25, 50, 75, 100]
        
        threshold_num = len(percentiles)

        print('Reference test set:', recon_manager_labels[target_idx])
        
        recon_managers[target_idx].load_test_stats()
        
        twelve_keys = get_twelve_keys()
        
        if plot_sub_classes is None:
            plot_sub_classes = self.multi_manager[0].sub_classes
            
        for sub_class in plot_sub_classes:
            
            sub_class_idx = self.multi_manager[target_idx].sub_classes.index(sub_class)
            thresholds = [np.percentile(self.multi_manager[target_idx].test_losses_per_sub_class[sub_class_idx], percent) for percent in percentiles]
            
        for sub_class in plot_sub_classes:
            
            sub_class_idx = self.multi_manager[target_idx].sub_classes.index(sub_class)
            thresholds = [np.percentile(self.multi_manager[target_idx].test_losses_per_sub_class[sub_class_idx], percent) for percent in percentiles]
            
            sub_class_map = set(load_dataclass_ids(get_parent_folder(), sub_class))
            test_map = set(self.multi_manager[target_idx].data_loader.test_data_ids)
            
            candidate_list = list(sub_class_map & test_map)
        
            recon_managers[target_idx].data_loader.shuffle('test')
            
            element_per_threshold = np.zeros(threshold_num - 1)
            
            while np.sum(element_per_threshold) < 2 * (threshold_num - 1):
    
                element_id = candidate_list.pop(0)
                
                recon_input_leads, classif_input_leads, target_probabilities, _, _ = self.multi_manager[target_idx].data_loader.load_element(element_id, False)
            
                twelve_leads = recon_managers[target_idx].data_loader.load_element_twelve_leads(element_id)
                twelve_leads = [lead * recon_managers[target_idx].amplitude + recon_managers[target_idx].min_value for lead in twelve_leads]
                
                reconstruction_input, reconstruction_target, classification_input, classification_target = process_element(recon_input_leads,
                                                                    classif_input_leads,
                                                                    recon_managers[target_idx].output_lead_keys,
                                                                    recon_managers[target_idx].classif_input_lead_keys,
                                                                    target_probabilities,
                                                                    recon_managers[target_idx].min_value,
                                                                    recon_managers[target_idx].amplitude,
                                                                    recon_managers[target_idx].device)

                with torch.no_grad():
                    
                    reconstruction_output = recon_managers[target_idx].model.forward(reconstruction_input)
                    
                classification_input = post_process_element(classification_input,
                                                            reconstruction_output,
                                                            recon_managers[target_idx].classif_input_lead_keys,
                                                            recon_managers[target_idx].output_lead_keys,
                                                            recon_managers[target_idx].min_value,
                                                            recon_managers[target_idx].amplitude)
                
                with torch.no_grad():
                
                    classification_output = recon_managers[target_idx].classificator.forward(classification_input)
                            
                loss, _, _ = element_loss_function(reconstruction_output, reconstruction_target, classification_output, classification_target, recon_managers[target_idx].output_lead_num, recon_managers[target_idx].detect_class_num, recon_managers[target_idx].alpha)

                for i in range(threshold_num - 1):
                
                    if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] <= 1:
                    
                        element_per_threshold[i] += 1
                                
                        multi_model_leads = []
                        multi_output_keys = []
                        multi_input_keys = []

                        for manager in recon_managers:
                            
                            recon_input_leads, classif_input_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
                
                            reconstruction_input, reconstruction_target, classification_input, classification_target = process_element(recon_input_leads,
                                                                classif_input_leads,
                                                                manager.output_lead_keys,
                                                                manager.classif_input_lead_keys,
                                                                target_probabilities,
                                                                manager.min_value,
                                                                manager.amplitude,
                                                                manager.device)

                            reconstruction_output = manager.model.forward(reconstruction_input)

                            model_leads = deprocess_element(copy.copy(reconstruction_output))
                            
                            multi_model_leads.append(model_leads)                            
                            multi_output_keys.append(manager.output_lead_keys)
                            multi_input_keys.append(manager.input_lead_keys)
                            
                            classification_input = post_process_element(classification_input,
                                                                    reconstruction_output,
                                                                    manager.classif_input_lead_keys,
                                                                    manager.output_lead_keys,
                                                                    manager.min_value,
                                                                    manager.amplitude)
                    
                            classification_output = manager.classificator.forward(classification_input)
                        
                        pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                            
                        plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, self.manager_labels, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, self.manager_labels, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        
                        break
                    

    def emulate_clinical_test(self, size=1, target_idx=0, sub_classes=None, thresholds=None, sub_class_sizes=None, plot=False):
        
        self.multi_manager[target_idx].data_loader.shuffle('test')
        
        twelve_keys = get_twelve_keys()
        
        multi_dataset = [[] for _ in range(self.manager_num+1)]
        
        dataset_labels = self.manager_labels + ['Original']
        
        if thresholds is None:
            thresholds = [0.5 for _ in range(self.manager_num)]
            
        manager_accuracies = [0 for _ in range(self.manager_num)]
        manager_sensitivities = [0 for _ in range(self.manager_num)]
        manager_specificities = [0 for _ in range(self.manager_num)]
        
        positive_size = 0
        negative_size = 0
        
        positive_negative_list = []
            
        if sub_classes is not None:
            
            dataset_ids = []
            dataset_specifics = []
            
            if sub_class_sizes is None:
                sub_class_sizes = [int(size / len(sub_classes)) for _ in range(len(sub_classes))]                
            else:
                sub_class_sizes = [int(size * x) for x in sub_class_sizes]
            
            test_map = set(self.multi_manager[target_idx].data_loader.test_data_ids)
            
            for sub_class, sub_class_size in zip(sub_classes, sub_class_sizes):
                
                if sub_class == 'other':
                    
                    sub_class_map = copy.copy(test_map)
                    
                else:
                    
                    sub_class_map = set(load_dataclass_ids(get_parent_folder(), sub_class))
                    
                    
                if sub_class != 'st_elevation_or_acute_infarct':
                    
                    for other_sub_class in sub_classes:
                        if other_sub_class != sub_class and other_sub_class != 'other':
                            sub_class_map = sub_class_map - set(load_dataclass_ids(get_parent_folder(), other_sub_class))
                    
                candidate_ids = list(sub_class_map & test_map)
                    
                random.shuffle(candidate_ids)
                
                dataset_ids += candidate_ids[:sub_class_size]
                dataset_specifics += [[sub_class, None] + [None for _ in range(self.manager_num+1)] for _ in range(sub_class_size)]
                
        else:
            
            dataset_ids = self.multi_manager[target_idx].data_loader.get_random_data_ids('test', size)   
            dataset_specifics = [[None for _ in range(self.manager_num + 5)] for _ in range(size)]   

        c = list(zip(dataset_ids, dataset_specifics))

        random.shuffle(c)

        dataset_ids, dataset_specifics = zip(*c)
        
        for signal_index, element_id in enumerate(dataset_ids):
            
            if signal_index % 100 == 0:
                print(signal_index)
            
            index_sequence = list(range(self.manager_num + 1))
            
            original_leads = self.multi_manager[target_idx].data_loader.load_element_twelve_leads(element_id)
            original_diagnosis = self.multi_manager[target_idx].data_loader.extract_element_diagnosis(element_id)
            original_signal = [lead * self.multi_manager[target_idx].amplitude + self.multi_manager[target_idx].min_value for lead in original_leads]
            
            try:
                diagnosis = original_diagnosis[0]
            except:
                print(original_diagnosis)
                diagnosis = original_diagnosis
            
            for i in range(len(original_diagnosis) - 1):
                
                diagnosis += ', ' + original_diagnosis[i+1]
            
            dataset_specifics[signal_index][1] = diagnosis
            
            multi_dataset[index_sequence[-1]].append(original_signal)
            
            for manager_index, manager, threshold in zip(range(self.manager_num), self.multi_manager, thresholds):
                
                if isinstance(manager, ReconClassifManager):
                    
                    recon_input_leads, classif_input_leads, target_probabilities, _, _ = manager.data_loader.load_element(element_id, False)
        
                    reconstruction_input, reconstruction_target, classification_input, classification_target = process_element(recon_input_leads,
                                                        classif_input_leads,
                                                        manager.output_lead_keys,
                                                        manager.classif_input_lead_keys,
                                                        target_probabilities,
                                                        manager.min_value,
                                                        manager.amplitude,
                                                        manager.device)

                    reconstruction_output = manager.model.forward(reconstruction_input)

                    model_leads = deprocess_element(copy.copy(reconstruction_output))
                    
                    classification_input = post_process_element(classification_input,
                                                            reconstruction_output,
                                                            manager.classif_input_lead_keys,
                                                            manager.output_lead_keys,
                                                            manager.min_value,
                                                            manager.amplitude)
            
                    classification_output = manager.classificator.forward(classification_input)
                    
                    reconstructed_signal = []
                
                    for lead_index, lead_key in enumerate(twelve_keys):
                        
                        if lead_key in manager.output_lead_keys:                        
                            reconstructed_signal.append(model_leads[manager.output_lead_keys.index(lead_key)])
                        
                        else:
                            reconstructed_signal.append(np.copy(original_signal[lead_index]))
                    
                elif isinstance(manager, ClassificationManager):
                    
                    input_leads, _, target_probabilities, _, element_id = manager.data_loader.load_element(element_id, False)

                    classification_input, classification_target = process_classif_element(input_leads,
                                                                                  target_probabilities,
                                                                                  manager.device)

                    with torch.no_grad():

                        classification_output = manager.model.forward(classification_input)
                        
                    reconstructed_signal = []
                
                    for lead_index, lead_key in enumerate(twelve_keys):
                        
                        if lead_key in ['III', 'aVL', 'aVR', 'aVF'] + manager.input_lead_keys:
                            reconstructed_signal.append(np.copy(original_signal[lead_index]))          
                            
                        else:
                            
                            reconstructed_signal.append(np.zeros(manager.sample_num))  
                            
                else:
                    raise ValueError
                
                estimate = classification_output[0].item()                
                        
                multi_dataset[index_sequence[manager_index]].append(reconstructed_signal)
                dataset_specifics[signal_index][index_sequence[manager_index]+2] = estimate
                
                groundtruth = classification_target[0].item()
                
                if groundtruth == 1 and estimate > threshold:                        
                    manager_sensitivities[manager_index] += 1             
                    manager_accuracies[manager_index] += 1
                        
                elif groundtruth == 0 and estimate <= threshold:                        
                    manager_specificities[manager_index] += 1             
                    manager_accuracies[manager_index] += 1
                
            if groundtruth == 1:
                positive_size += 1
                positive_negative_list.append(1)
                    
            else:
                positive_negative_list.append(0)
                negative_size += 1    
            
            dataset_specifics[signal_index][index_sequence[-1] + 2] = groundtruth
            
        if plot:
                
            for dataset, label in zip(multi_dataset, dataset_labels):
                
                pathlib.Path(self.output_folder + 'dataset/').mkdir(parents=True, exist_ok=True)
                
                # plot_clinical_dataset(dataset, self.multi_manager[target_idx].sample_num, twelve_keys, label, self.output_folder + 'dataset/' + label)
            
        dataframe = pd.DataFrame(dataset_specifics, columns = ['Class', 'Diagnosis'] + [self.manager_labels[manager_index] for manager_index in range(self.manager_num)] + ['Original'])
        
        dataframe.to_csv(self.output_folder + 'dataset/data_specifics.csv')
        
        for manager_index in range(self.manager_num):
        
            manager_accuracies[manager_index] /= (positive_size + negative_size)
            manager_sensitivities[manager_index] /= positive_size
            manager_specificities[manager_index] /= negative_size        
        
        print('Accuracies: ', manager_accuracies)
        print('Sensitivities: ',manager_sensitivities)
        print('Specificities: ', manager_specificities)
        
        accuracy_effect_sizes = []
        sensitivity_effect_sizes = []
        specificity_effect_sizes = []
        
        var = np.var(positive_negative_list)
        
        for manager_index in range(self.manager_num):
            
            accuracy_effect_sizes.append((manager_accuracies[-1] - manager_accuracies[manager_index]) / var)
            sensitivity_effect_sizes.append((manager_sensitivities[-1] - manager_sensitivities[manager_index]) / var)
            specificity_effect_sizes.append((manager_specificities[-1] - manager_specificities[manager_index]) / var)
        
        print('Accuracy effect size: ', accuracy_effect_sizes)
        print('Sensitivity effect size: ', sensitivity_effect_sizes)
        print('Specificity effect size: ', specificity_effect_sizes)
        
        model_performance = [manager_accuracies,
                             manager_sensitivities,
                             manager_specificities,
                             accuracy_effect_sizes,
                             sensitivity_effect_sizes,
                             specificity_effect_sizes]
        
        dataframe = pd.DataFrame(model_performance, index = ['Accuracy', 'Sensitivity', 'Specificity', 'Accuracy effect size', 'Sensitivity effect size', 'Specificity effect size'], columns = [self.manager_labels[manager_index] for manager_index in range(self.manager_num)])
        
        dataframe.to_csv(self.output_folder + 'dataset/model_performance.csv')