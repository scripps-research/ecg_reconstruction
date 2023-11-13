from plot_functions.roc_curve import plot_multi_roc_curve
from plot_functions.detection_distribution import plot_multi_detection_distribution
from plot_functions.accuracy_distribution import plot_multi_accuracy_distribution
from training_functions.multi_training_manager import MultiTrainingManager
from training_functions.single_classification_manager import ClassificationManager
from training_functions.classification_functions import get_detection_distributions
from plot_functions.continuos_distribution import plot_violin_distribution, plot_box_distribution
import itertools
import pathlib
import numpy as np


class MultiClassificationManager(MultiTrainingManager):
    
    def __init__(self,
                 output_name: str,
                 parent_folder: str,
                 device: str,
                 sub_classes,
                 input_leads: str,
                 parallel_classification: bool,
                 data_classes: str,
                 data_size: str,
                 detect_classes: str,       
                 input_channel: int,
                 middle_channel: int,
                 input_depth: int,
                 middle_depth: int,
                 output_depth: int,
                 input_kernel: int,
                 middle_kernel: int,
                 stride_size: int,
                 use_residual: str,
                 epochs: int,
                 batch_size: int,
                 prioritize_percent: float,
                 prioritize_size: float,
                 optimizer_algorithm: str,
                 learning_rate: float,
                 weight_decay: float,
                 momentum: float,
                 nesterov: bool,
                 manager_labels = None
                 ):
        
        super().__init__(use_residual,
                         epochs,
                         batch_size,
                         prioritize_percent,
                         prioritize_size,
                         optimizer_algorithm,
                         learning_rate,
                         weight_decay,
                         momentum,
                         nesterov,
                         manager_labels)
        
        if manager_labels is None:
            self.manager_labels = []

        self.output_folder = parent_folder + 'Classification/compare_results/' + output_name + '/'

        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        if isinstance(sub_classes[0], list):
            sub_classes_list = sub_classes
        else:
            sub_classes_list = [sub_classes] * self.manager_num
        
        if isinstance(input_leads, list):
            input_leads_list = input_leads
        else:
            input_leads_list = [input_leads] * self.manager_num
            
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

        if isinstance(input_channel, list):
            input_channel_list = input_channel
        else:
            input_channel_list = [input_channel] * self.manager_num            
            
        if isinstance(middle_channel, list):
            middle_channel_list = middle_channel
        else:
            middle_channel_list = [middle_channel] * self.manager_num

        if isinstance(input_depth, list):
            input_depth_list = input_depth
        else:
            input_depth_list = [input_depth] * self.manager_num
            
        if isinstance(middle_depth, list):
            middle_depth_list = middle_depth
        else:
            middle_depth_list = [middle_depth] * self.manager_num
            
        if isinstance(output_depth, list):
            output_depth_list = output_depth
        else:
            output_depth_list = [output_depth] * self.manager_num

        if isinstance(input_kernel, list):
            input_kernel_list = input_kernel
        else:
            input_kernel_list = [input_kernel] * self.manager_num
            
        if isinstance(middle_kernel, list):
            middle_kernel_list = middle_kernel
        else:
            middle_kernel_list = [middle_kernel] * self.manager_num
        
        if isinstance(stride_size, list):
            stride_size_list = stride_size
        else:
            stride_size_list = [stride_size] * self.manager_num
            
        for configuration, configuration_label, sub_classes, input_leads, parallel_classification, data_classes, data_size, detect_classes, input_channel, middle_channel, \
            input_depth, middle_depth, output_depth, input_kernel, middle_kernel, stride_size in zip(self.configurations,
                                                                                                       self.configuration_labels,
                                                                                                       sub_classes_list,
                                                                                                       input_leads_list,
                                                                                                       parallel_classification_list,
                                                                                                       data_classes_list,
                                                                                                       data_size_list,
                                                                                                       detect_classes_list,
                                                                                                       input_channel_list,
                                                                                                       middle_channel_list,
                                                                                                       input_depth_list,
                                                                                                       middle_depth_list,
                                                                                                       output_depth_list,
                                                                                                       input_kernel_list,
                                                                                                       middle_kernel_list,
                                                                                                       stride_size_list):
                    
            use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer_algorithm, learning_rate, weight_decay, momentum, nesterov = configuration
                    
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
                                            use_residual,
                                            epoch_num,
                                            batch_size,
                                            prioritize_percent,
                                            prioritize_size,
                                            optimizer_algorithm,
                                            learning_rate,
                                            weight_decay,
                                            momentum,
                                            nesterov)
            
            self.multi_manager.append(manager)
            
            if manager_labels is None:

                manager_label = ''
                
                if len(input_leads_list) > 1:
                    manager_label += 'input=' + input_leads + ' '
                if len(parallel_classification_list) > 1:
                    manager_label += 'parallel=' + str(parallel_classification) + ' '
                if len(data_classes_list) > 1:
                    manager_label += 'data_class=' + str(data_classes) + ' '
                if len(data_size_list) > 1:
                    manager_label += 'data_size=' + data_size + ' '
                if len(detect_classes_list) > 1:
                    manager_label += 'detect_class=' + str(detect_classes) + ' '
                if len(input_channel_list) > 1 or len(middle_channel_list) > 1:
                    manager_label += 'channel=' + str(input_channel) + '_'  + str(middle_channel) + ' '
                if len(input_depth_list) > 1 or len(middle_depth_list) > 1 or len(output_depth_list) > 1:
                    manager_label += 'depth=' + str(input_depth) + 'x' + str(middle_depth) + 'x' + str(output_depth) + ' '
                if len(input_kernel_list) > 1 or len(middle_kernel_list) > 1:
                    manager_label += 'kernel=' + str(input_kernel) + 'x' + str(middle_kernel) + ' '
                if len(stride_size_list) > 1:
                    manager_label += 'stride=' + str(stride_size) + ' '
                    
                manager_label += configuration_label

                manager_label = manager_label[:-1]
                
                self.manager_labels.append(manager_label)
                
    def load_train_stats(self):
        
        super().load_train_stats()
        
    def load_valid_stats(self):
        
        super().load_valid_stats()
        
    def load_test_stats(self):
        
        super().load_test_stats()
        
    def plot_test_stats(self, threshold_num = 1001, plot_sub_classes = None, plot_format='png'):
        
        multi_entropies = []        
        multi_labels = []
        
        for manager, label in zip(self.multi_manager, self.manager_labels):
                
            multi_entropies.append(manager.test_entropies)
            multi_labels.append(label)
            
        bce_ticks = [ x / 20 for x in range(0, 11)]
            
        plot_violin_distribution(multi_entropies, multi_labels, 'BCE', self.output_folder + 'test_stats/test_entropies', yticks=bce_ticks, plot_format=plot_format)
        plot_box_distribution(multi_entropies, multi_labels, 'BCE', self.output_folder + 'test_stats/test_entropies_box', yticks=bce_ticks, plot_format=plot_format)
        
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
                    
                    if len(detect_classes) > 1:                
                    
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
