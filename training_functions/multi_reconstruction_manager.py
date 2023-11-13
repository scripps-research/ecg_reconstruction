from training_functions.multi_training_manager import MultiTrainingManager
from training_functions.single_reconstruction_manager import ReconstructionManager
from training_functions.reconstruction_functions import element_mse_function, element_r2_function, process_element, deprocess_element
from plot_functions.ecg_signal import plot_output_multi_recon_leads, plot_medical_multi_recon_leads, plot_clinical_dataset
from util_functions.general import get_parent_folder, remove_dir, get_twelve_keys
from plot_functions.continuos_distribution import plot_violin_distribution, plot_box_distribution
from util_functions.load_data_ids import load_dataclass_ids
import pandas as pd
import pathlib
import numpy as np
import random
import torch


class MultiReconstructionManager(MultiTrainingManager):
    
    def __init__(self,
                 output_name: str,
                 parent_folder: str,
                 device: str,
                 sub_classes: list,
                 input_leads: str,
                 output_leads: str,
                 data_class: str,
                 data_size: str,
                 input_channel: int,
                 middle_channel: int,
                 output_channel: int,
                 input_depth: int,
                 middle_depth: int,
                 output_depth: int,
                 input_kernel: int,
                 middle_kernel: int,
                 output_kernel: int,
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

        self.output_folder = parent_folder + 'Reconstruction/compare_results/' + output_name + '/'

        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        if len(sub_classes) > 0 and isinstance(sub_classes[0], list):
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

        if isinstance(data_class[0], list):
            data_class_list = data_class
        else:
            data_class_list = [data_class] * self.manager_num

        if isinstance(data_size, list):
            data_size_list = data_size
        else:
            data_size_list = [data_size] * self.manager_num

        if isinstance(input_channel, list):
            input_channel_list = input_channel
        else:
            input_channel_list = [input_channel] * self.manager_num
            
        if isinstance(middle_channel, list):
            middle_channel_list = middle_channel
        else:
            middle_channel_list = [middle_channel] * self.manager_num
            
        if isinstance(output_channel, list):
            output_channel_list = output_channel
        else:
            output_channel_list = [output_channel] * self.manager_num
            
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
            
        if isinstance(output_kernel, list):
            output_kernel_list = output_kernel
        else:
            output_kernel_list = [output_kernel] * self.manager_num
                
        for configuration, configuration_label, sub_classes, input_leads, output_leads, data_class, data_size,\
            input_channel, middle_channel, output_channel, input_depth, middle_depth, output_depth,\
                input_kernel, middle_kernel, output_kernel in zip(
                    self.configurations,
                    self.configuration_labels,
                    sub_classes_list,
                    input_leads_list,
                    output_leads_list,
                    data_class_list,
                    data_size_list,
                    input_channel_list,
                    middle_channel_list,
                    output_channel_list,
                    input_depth_list,
                    middle_depth_list,
                    output_depth_list,
                    input_kernel_list,
                    middle_kernel_list,
                    output_kernel_list):
                
            use_residual, epochs, batch_size, prioritize_percent, prioritize_size, optimizer_algorithm, learning_rate, weight_decay, momentum, nesterov = configuration
                            
            manager = ReconstructionManager(parent_folder,
                                            device,
                                            sub_classes,
                                            input_leads,
                                            output_leads,
                                            data_class,
                                            data_size,
                                            input_channel,
                                            middle_channel,
                                            output_channel,
                                            input_depth,
                                            middle_depth,
                                            output_depth,
                                            input_kernel,
                                            middle_kernel,
                                            output_kernel,
                                            use_residual,
                                            epochs,
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
                if len(output_leads_list) > 1:
                    manager_label += 'output=' + output_leads + ' '
                if len(data_class_list) > 1:
                    manager_label += 'data_class=' + data_class + ' '
                if len(data_size_list) > 1:
                    manager_label += 'data_size=' + data_size + ' '
                if len(input_channel_list) > 1 or len(middle_channel_list) > 1 or len(output_channel_list) > 1:
                    manager_label += 'channel=' + str(input_channel) + 'x' + str(middle_channel) + 'x' + str(output_channel) + ' '
                if len(input_depth_list) > 1 or len(middle_depth_list) > 1 or len(output_depth_list) > 1:
                    manager_label += 'depth=' + str(input_depth) + 'x' + str(middle_depth) + 'x' + str(output_depth) + ' '
                if len(input_kernel_list) > 1 or len(middle_kernel_list) > 1 or len(output_kernel_list) > 1:
                    manager_label += 'kernel=' + str(input_kernel) + 'x' + str(middle_kernel) + 'x' + str(output_kernel) + ' '
                    
                manager_label += configuration_label

                manager_label = manager_label[:-1]
            
                self.manager_labels.append(manager_label)
                
                
    def plot_test_stats(self, plot_format='png'):
        
        pathlib.Path(self.output_folder + 'test_stats/').mkdir(parents=True, exist_ok=True)
        
        r2_ticks = [x / 0.1 for x in range(0, 11)]
        mse_ticks = [ x / 200 for x in range(0, 11)]

        multi_rsquareds = [manager.test_rsquareds * 100 for manager in self.multi_manager]

        plot_violin_distribution(multi_rsquareds,
                                 self.manager_labels,
                                 'R2 [%]',
                                 self.output_folder + 'test_stats/test_rsquareds',
                                 yticks=r2_ticks, plot_format=plot_format)
        plot_box_distribution(multi_rsquareds, self.manager_labels, 'R2 [%]', self.output_folder + 'test_stats/test_rsquareds_box', yticks=r2_ticks, plot_format=plot_format)
        
        multi_mserrors = [manager.test_mserrors for manager in self.multi_manager]       
        
        plot_violin_distribution(multi_mserrors, self.manager_labels, 'MSE [$mV^2$]', self.output_folder + 'test_stats/test_mserrors', yticks=mse_ticks, plot_format=plot_format)
        plot_box_distribution(multi_mserrors, self.manager_labels, 'MSE [$mV^2$]', self.output_folder + 'test_stats/test_mserrors_box', yticks=mse_ticks, plot_format=plot_format)
                
    def generate_clinical_test(self,
                               size_per_sub_class,
                               sub_classes,
                               target_idx=0,
                               randomized=False,
                               generate=False,
                               split_figures=False):
        
        self.multi_manager[target_idx].data_loader.shuffle('test')
        
        twelve_keys = get_twelve_keys()
        
        original_signals = []
        original_diagnoses = []
        
        multi_dataset = [[] for _ in range(self.manager_num+1)]
            
        dataset_labels = [str(i+1) for i in range(self.manager_num + 1)]
        
        selection_specifics = []
        selection_ids = []
        
        for sub_class, sub_class_size in zip(sub_classes, size_per_sub_class):
            
            selection_specifics += [[sub_class, None, None] + [None for _ in range(self.manager_num + 1)] for _ in range(sub_class_size)]
            
            sub_class_ids = load_dataclass_ids(get_parent_folder(), sub_class)
            
            random.shuffle(sub_class_ids)
            
            selection_ids.extend(sub_class_ids[:sub_class_size])
                
        # RANDOMIZED THE SELECTION IDS
            
        if randomized: 
            
            c = list(zip(selection_ids, selection_specifics)) 

            random.shuffle(c)

            selection_ids, selection_specifics = zip(*c)
        
        # INITIALIZE THE DATASETS
        
        for signal_index, element_id in enumerate(selection_ids):
            
            index_sequence = list(range(self.manager_num + 1))
            
            if randomized:                
                
                random.shuffle(index_sequence)
            
            leads = self.multi_manager[target_idx].data_loader.load_element_twelve_leads(element_id)
            diagnosis = self.multi_manager[target_idx].data_loader.extract_element_diagnosis(element_id)
            signal = [lead * self.multi_manager[target_idx].amplitude + self.multi_manager[target_idx].min_value for lead in leads]
            
            if isinstance(diagnosis, list):
                
                signal_diagnosis = diagnosis[0]
                for x in diagnosis[1:]:                    
                    signal_diagnosis += ', ' + x
            
            else:
                
                signal_diagnosis = diagnosis
            
            selection_specifics[signal_index][1] = element_id   
            selection_specifics[signal_index][2] = signal_diagnosis
                        
            multi_dataset[index_sequence[-1]].append(signal)
            selection_specifics[signal_index][index_sequence[-1] + 3] = 'Original'
            
            original_signals.append(signal)
            original_diagnoses.append(diagnosis)
                
            for manager_index, manager in enumerate(self.multi_manager):
                
                input_leads, output_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
                
                model_input, _ = process_element(input_leads,
                                                   output_leads,                                                   
                                                   manager.min_value,
                                                   manager.amplitude,
                                                   manager.device)

                model_output = manager.model.forward(model_input)

                model_leads = deprocess_element(model_output)
                
                reconstructed_signal = []
                
                for lead_index, lead_key in enumerate(twelve_keys):
                    
                    if lead_key in manager.output_lead_keys:                        
                        reconstructed_signal.append(model_leads[manager.output_lead_keys.index(lead_key)])
                    
                    else:
                        reconstructed_signal.append(np.copy(signal[lead_index]))
                        
                multi_dataset[index_sequence[manager_index]].append(reconstructed_signal)
                selection_specifics[signal_index][index_sequence[manager_index]+3] = self.manager_labels[manager_index]
                
        original_classes = [x[0] for x in selection_specifics]
                
        if generate:
                
            for dataset, label in zip(multi_dataset, dataset_labels):
                
                pathlib.Path(self.output_folder + 'clinical_test/' + label + '/').mkdir(parents=True, exist_ok=True)
                
                plot_clinical_dataset(dataset,
                                      self.multi_manager[target_idx].sample_num,
                                      twelve_keys,
                                      self.output_folder + 'clinical_test/' + label + '/dataset',
                                      label,
                                      split_figures=split_figures)
                
                plot_clinical_dataset(dataset,
                                      self.multi_manager[target_idx].sample_num,
                                      twelve_keys,
                                      self.output_folder + 'clinical_test/' + label + '/dataset',
                                      label,
                                      split_figures=split_figures)
                
        pathlib.Path(self.output_folder + 'clinical_test/original/').mkdir(parents=True, exist_ok=True)

        plot_clinical_dataset(original_signals,
                              self.multi_manager[target_idx].sample_num,
                              twelve_keys,
                              self.output_folder + 'clinical_test/original/dataset',
                              'ORIGINAL',
                              classes=original_classes,
                              diagnoses=original_diagnoses,
                              split_figures=False)
            
        dataframe = pd.DataFrame(selection_specifics, columns = ['Class', 'Element ID', 'Diagnosis'] + dataset_labels)
        
        dataframe.to_csv(self.output_folder + 'clinical_test/specifics.csv')

    def plot_random_example(self, size=1, target_idx=0, plot_format='png'):

        remove_dir(self.output_folder + 'random_plot/')
        
        self.multi_manager[target_idx].data_loader.shuffle('test')

        batch_ids = self.multi_manager[target_idx].data_loader.get_random_data_ids('test', size)
        
        twelve_keys = get_twelve_keys()

        for element_id in batch_ids:
                     
            multi_model_leads = []
            multi_output_keys = []
            multi_input_keys = []
            
            twelve_leads = self.multi_manager[0].data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * self.multi_manager[0].amplitude + self.multi_manager[0].min_value for lead in twelve_leads]

            for manager in self.multi_manager:
                
                input_leads, output_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
                
                model_input, _ = process_element(input_leads,
                                                   output_leads,                                                   
                                                   manager.min_value,
                                                   manager.amplitude,
                                                   manager.device)

                model_output = manager.model.forward(model_input)

                model_leads = deprocess_element(model_output)
            
                multi_model_leads.append(model_leads)
            
                multi_output_keys.append(manager.output_lead_keys)
                multi_input_keys.append(manager.input_lead_keys)
                        
            pathlib.Path(self.output_folder + 'random_plot/standard/').mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.output_folder + 'random_plot/medical/').mkdir(parents=True, exist_ok=True)
            
            plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, self.manager_labels, self.output_folder + 'random_plot/medical/' + str(element_id), plot_format=plot_format)
                            
            plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, self.manager_labels, self.output_folder + 'random_plot/standard/' + str(element_id), plot_format=plot_format)

    def plot_error_example(self, target_idx=0, target_loss='r2', plot_format='png'):

        remove_dir(self.output_folder + 'error_plot/')

        print('Reference test set:', self.manager_labels[target_idx])

        self.multi_manager[target_idx].data_loader.shuffle('test')

        self.multi_manager[target_idx].load_test_stats()

        percentiles = [0, 25, 50, 75, 100]

        percentiles = [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        if target_loss == 'mse':
            thresholds = [np.percentile(self.multi_manager[target_idx].test_mserrors, percent) for percent in percentiles]
            
        elif target_loss == 'r2':
            thresholds = [np.percentile(self.multi_manager[target_idx].test_rsquareds, percent) for percent in percentiles]
            
        else:
            raise ValueError

        threshold_num = len(thresholds)

        element_per_threshold = np.zeros(threshold_num - 1)
        
        twelve_keys = get_twelve_keys()

        while np.sum(element_per_threshold) < threshold_num - 1:

            input_leads, output_leads, _, _, element_id = self.multi_manager[target_idx].data_loader.get_next_element('test', False)
            
            twelve_leads = self.multi_manager[0].data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * self.multi_manager[0].amplitude + self.multi_manager[0].min_value for lead in twelve_leads]

            model_input, model_target = process_element(input_leads,
                                                            output_leads,                                                            
                                                            self.multi_manager[target_idx].min_value,
                                                            self.multi_manager[target_idx].amplitude,
                                                            self.multi_manager[target_idx].device)

            with torch.no_grad():

                model_output = self.multi_manager[target_idx].model.forward(model_input)

                loss, _, _, _ = element_mse_function(model_output,
                                                      model_target,
                                                      self.multi_manager[target_idx].output_lead_num)

            for i in range(threshold_num - 1):

                if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] == 0:

                    element_per_threshold[i] = 1
                    
                    multi_model_leads = []
                    multi_output_keys = []
                    multi_input_keys = []
                    
                    print()

                    for manager, label in zip(self.multi_manager, self.manager_labels):
                        
                        input_leads, output_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
            
                        model_input, _ = process_element(input_leads,
                                                            output_leads,                                                               
                                                            manager.min_value,
                                                            manager.amplitude,
                                                            manager.device)
                        
                        with torch.no_grad():

                            model_output = manager.model.forward(model_input)
                        
                        loss, _, _, _ = element_mse_function(model_output,
                                                      model_target,
                                                      self.multi_manager[target_idx].output_lead_num)
                
                        r2 = element_r2_function(model_output,
                                                    model_target,
                                                    self.multi_manager[target_idx].output_lead_num)
                        
                        print(label, percentiles[i+1], '- MSE:', loss, '- R2:', r2)

                        model_leads = deprocess_element(model_output)
                        
                        multi_model_leads.append(model_leads)                            
                        multi_output_keys.append(manager.output_lead_keys)
                        multi_input_keys.append(manager.input_lead_keys)
                    
                        pathlib.Path(self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        pathlib.Path(self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        
                    plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, self.manager_labels, self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                    plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, self.manager_labels, self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                    
                    break
                
    def plot_sub_class_example(self, plot_sub_classes=None, target_idx=0, target_loss='r2', plot_format='png'):
    
        remove_dir(self.output_folder + 'sub_class_plot/')

        print('Reference test set:', self.manager_labels[target_idx])
        
        percentiles = [0, 25, 50, 75, 100]
        
        threshold_num = len(percentiles)

        self.multi_manager[target_idx].load_test_stats()
        
        twelve_keys = get_twelve_keys()
        
        if plot_sub_classes is None:
            plot_sub_classes = self.multi_manager[0].sub_classes
        
        for sub_class in plot_sub_classes:
            
            print()
            
            print("Subclass", sub_class)
            
            print()
            
            sub_class_idx = self.multi_manager[target_idx].sub_classes.index(sub_class)
            
            if target_loss == 'mse':
                
                thresholds = [np.percentile(self.multi_manager[target_idx].test_mserrors_per_sub_class[sub_class_idx], percent) for percent in percentiles]
                
            elif target_loss == 'r2':
                
                thresholds = [np.percentile(self.multi_manager[target_idx].test_rsquareds_per_sub_class[sub_class_idx], percent) for percent in percentiles]
                
            else:
                raise ValueError
            
            sub_class_map = set(load_dataclass_ids(get_parent_folder(), sub_class))
            test_map = set(self.multi_manager[target_idx].data_loader.test_data_ids)
            
            candidate_list = list(sub_class_map & test_map)
            
            random.shuffle(candidate_list)
        
            self.multi_manager[target_idx].data_loader.shuffle('test')
            
            element_per_threshold = np.zeros(threshold_num - 1)
            
            while np.sum(element_per_threshold) < threshold_num - 1:
                

                element_id = candidate_list.pop(0)
                input_leads, output_leads, _, _, _ = self.multi_manager[target_idx].data_loader.load_element(element_id, False)
                
                twelve_leads = self.multi_manager[0].data_loader.load_element_twelve_leads(element_id)
                twelve_leads = [lead * self.multi_manager[0].amplitude + self.multi_manager[0].min_value for lead in twelve_leads]

                model_input, model_target = process_element(input_leads,
                                                            output_leads,                                                                
                                                            self.multi_manager[target_idx].min_value,
                                                            self.multi_manager[target_idx].amplitude,
                                                            self.multi_manager[target_idx].device)

                with torch.no_grad():

                    model_output = self.multi_manager[target_idx].model.forward(model_input)

                    loss, _, _, _ = element_mse_function(model_output,
                                                          model_target,
                                                          self.multi_manager[target_idx].output_lead_num)

                for i in range(threshold_num - 1):
    
                    if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] == 0:
                        
                        print('Percentiles range:', percentiles[i], percentiles[i+1])                        
                        
                        print('Element diagnoses', self.multi_manager[target_idx].data_loader.extract_element_diagnosis(element_id))
                        
                        print()

                    
                        element_per_threshold[i] = 1
                                
                        multi_output_leads = []
                        multi_model_leads = []
                        multi_output_keys = []
                        multi_input_keys = []

                        for manager in self.multi_manager:
                            
                            input_leads, output_leads, _, _, _ = manager.data_loader.load_element(element_id, False)
                
                            model_input, _ = process_element(input_leads,
                                                            output_leads,                                                        
                                                            manager.min_value,
                                                            manager.amplitude,
                                                            manager.device)

                            model_output = manager.model.forward(model_input)

                            model_leads = deprocess_element(model_output)
                            
                            model_leads = [lead for lead in model_leads]
                            output_leads = [lead * manager.amplitude + manager.min_value for lead in output_leads]
                            
                            multi_output_leads.append(output_leads)
                            multi_model_leads.append(model_leads)                            
                            multi_output_keys.append(manager.output_lead_keys)
                            multi_input_keys.append(manager.input_lead_keys)
                        
                            pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                            pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                            
                        plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, self.manager_labels, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, self.manager_labels, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        
                        break