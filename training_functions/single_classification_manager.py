from load_functions.train_loader import ClassificationDataLoader
from training_functions.single_training_manager import TrainingManager
from learn_functions.generate_model import generate_classificator, generate_optimizer
from training_functions.classification_functions import batch_bce_function, get_detection_distributions, process_batch, process_element, element_bce_function
from plot_functions.detection_distribution import plot_detection_distribution, plot_multi_detection_distribution
from plot_functions.accuracy_distribution import plot_accuracy_distribution, plot_multi_accuracy_distribution
from util_functions.load_data_ids import load_dataclass_ids
from plot_functions.roc_curve import plot_roc_curve, plot_multi_roc_curve
from plot_functions.continuos_distribution import plot_violin_distribution
from training_functions.classification_functions import get_detection_distributions
import torch
import pickle
import pathlib
import numpy as np
import pandas as pd


class ClassificationManager(TrainingManager):
    
    def __init__(self,
                 parent_folder: str,
                 device: str,
                 sub_classes: str,
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
                 prioritize_size: int,
                 optimizer_algorithm: str,
                 learning_rate: float,
                 weight_decay: float,
                 momentum: float,
                 nesterov: bool
                 ):
        
        super().__init__(parent_folder,
                         device,
                         sub_classes,
                         input_leads,
                         epochs,
                         batch_size,
                         prioritize_percent)
        
        self.data_classes = data_classes
        self.data_class_num = len(self.data_classes)
        self.detect_classes = detect_classes
        self.detect_class_num = len(self.detect_classes)
        
        if parallel_classification == 'true':        
            self.parallel_classification = True
        else:
            self.parallel_classification = False
            
        self.test_entropies = []                
        self.test_probabilities = []
        self.test_groundtruths = []
        
        self.test_entropies_per_sub_class = []
        self.test_probabilities_per_sub_class = []
        self.test_groundtruths_per_sub_class = []
        
        for _ in self.sub_classes:
            
            self.test_entropies_per_sub_class.append([])
            self.test_probabilities_per_sub_class.append([])
            self.test_groundtruths_per_sub_class.append([])

        self.data_loader = ClassificationDataLoader(parent_folder,
                                                    data_classes,
                                                    data_size,
                                                    detect_classes,
                                                    batch_size,
                                                    prioritize_percent, 
                                                    prioritize_size,
                                                    self.sample_num,
                                                    self.min_value,
                                                    self.amplitude,
                                                    self.input_lead_keys
                                                    )

        self.model = generate_classificator(self.input_lead_num,
                                            self.detect_class_num,
                                            input_channel,
                                            middle_channel,
                                            input_depth,
                                            middle_depth,
                                            output_depth,
                                            input_kernel,
                                            middle_kernel,
                                            stride_size,
                                            int(self.sample_num / (stride_size ** middle_depth)),
                                            use_residual,
                                            self.device,
                                            self.parallel_classification)
        
        if self.parallel_classification:
            
            self.optimizer = [generate_optimizer(optimizer_algorithm,
                                                 learning_rate,
                                                 weight_decay,
                                                 momentum,
                                                 nesterov,
                                                 parameters) for parameters in self.model.parameters()]
            
        else:
            
            self.optimizer = generate_optimizer(optimizer_algorithm,
                                            learning_rate,
                                            weight_decay,
                                            momentum,
                                            nesterov,
                                            self.model.parameters())
        
        self.output_folder, self.parallel_output_folder = self.get_output_folder(parent_folder,
                                                    input_leads,
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
                                                    epochs,
                                                    batch_size,
                                                    prioritize_percent,
                                                    prioritize_size,
                                                    optimizer_algorithm,
                                                    learning_rate,
                                                    weight_decay,
                                                    momentum,
                                                    nesterov)
        

    def get_output_folder(self,
                          parent_folder: str,
                          input_leads: str,
                          data_classes,                          
                          data_size: str,
                          detect_classes,
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
                          prioritize_size: int,
                          optimizer_algorithm: str,
                          learning_rate: float,
                          weight_decay: float,
                          momentum: float,
                          nesterov: bool):

        output_folder = parent_folder + 'Classification/' + 'input=' + input_leads + '/data_class=' + str(data_classes) + '/data_size=' + data_size + '/detect_class=' + str(detect_classes) +\
            '/channel=' + str(input_channel) + '_' + str(middle_channel) + '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) + '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) +\
                '/stride=' + str(stride_size) + '/'
                
        output_folder += super().get_output_folder(use_residual,
                                                   epochs,
                                                   batch_size,
                                                   prioritize_percent,
                                                   prioritize_size,
                                                   optimizer_algorithm,
                                                   learning_rate,
                                                   weight_decay,
                                                   momentum,
                                                   nesterov)
        
        if self.parallel_classification:
        
            parallel_output_folder = []
            
            for detect_class in detect_classes:
                
                folder = parent_folder + 'Classification/' + 'input=' + input_leads + '/data_class=' + str(data_classes) + '/data_size=' + data_size + '/detect_class=' + str([detect_class]) +\
                '/channel=' + str(input_channel) + '_' + str(middle_channel) + '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) + '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) +\
                    '/stride=' + str(stride_size) + '/'
                    
                folder += super().get_output_folder(use_residual,
                                                        epochs,
                                                        batch_size,
                                                        prioritize_percent,
                                                        prioritize_size,
                                                        optimizer_algorithm,
                                                        learning_rate,
                                                        weight_decay,
                                                        momentum,
                                                        nesterov)
                
                parallel_output_folder.append(output_folder)
        
        else:
            
            parallel_output_folder = None
            

        return output_folder, parallel_output_folder
    
    
    def reset_sub_classes(self, new_sub_classes=None):
            
        self.test_entropies_per_sub_class = np.load(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', allow_pickle=True)
        self.test_probabilities_per_sub_class = np.load(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', allow_pickle=True)
        self.test_groundtruths_per_sub_class = np.load(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', allow_pickle=True)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 
    
            self.sub_classes = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)
        
        if new_sub_classes is None:
            
            self.sub_classes = []
            self.sub_class_num = 0
            
            self.test_entropies_per_sub_class = []
            self.test_probabilities_per_sub_class = []
            self.test_groundtruths_per_sub_class = []
            
        else:
            
            for sub_class in new_sub_classes:
                
                if sub_class in self.sub_classes:
                
                    sub_class_index = self.sub_classes.index(sub_class)
                    del self.sub_classes[sub_class_index]
                    self.sub_class_num -= 1
                    
                    self.test_entropies_per_sub_class = np.delete(self.test_entropies_per_sub_class, sub_class_index, axis=0)
                    self.test_probabilities_per_sub_class = np.delete(self.test_probabilities_per_sub_class, sub_class_index, axis=0)
                    self.test_groundtruths_per_sub_class = np.delete(self.test_groundtruths_per_sub_class, sub_class_index, axis=0)
                
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', self.test_groundtruths_per_sub_class)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
    
    def load_model(self):
        
        if self.parallel_classification:
            
            self.model.load_state_dict([folder + 'state_dict/' for folder in self.parallel_output_folder])
            
        else:
    
            self.model.load_state_dict(self.output_folder + 'state_dict/')
        
        
    def load_dataset(self, train=False, valid=False, test=False):

        self.data_loader.load(train, valid, test, False)
        

    def train_step(self, batch):
        
        model_input, model_target = process_batch(batch,
                                                  self.input_lead_num,
                                                  self.detect_class_num,
                                                  self.device)

        model_output = self.model.forward(model_input)

        batch_bce, bce_per_element = batch_bce_function(model_output,
                                                                 model_target,
                                                                 self.batch_size,
                                                                 self.detect_class_num,
                                                                 self.compute_loss_per_element)

        self.optimizer.zero_grad()
        batch_bce.backward()
        self.optimizer.step()

        self.data_loader.update_priority_weights(bce_per_element)

        self.train_losses.append(batch_bce.item())
                
        return self.data_loader.get_next_batch('train', False)


    def valid_step(self, batch):
        
        model_input, model_target = process_batch(batch, self.input_lead_num, self.detect_class_num, self.device)

        with torch.no_grad():
        
            model_output = self.model.forward(model_input)

        batch_bce, _ = batch_bce_function(model_output,
                                            model_target,
                                            self.batch_size,
                                            self.detect_class_num,
                                            compute_loss_per_element=False)

        self.valid_losses.append(batch_bce.item())

        return self.data_loader.get_next_batch('valid', False)


    def test(self):
        
        self.data_loader.shuffle('test')
        
        sub_class_maps = []
        
        for data_class in self.sub_classes:
            sub_class_maps.append(load_dataclass_ids(self.parent_folder, data_class))

        element = self.data_loader.get_next_element('test', False)
        
        i = 0

        while element is not None:
            
            i+= 1
            
            if i % 10000 == 0:
                print('Testing completed on', i, 'elements!')

            input_leads, _, element_probabilities, _, element_id = element

            model_input, model_target = process_element(input_leads,
                                                        element_probabilities,
                                                        self.device)

            with torch.no_grad():
            
                model_output = self.model.forward(model_input)
                
            bce = element_bce_function(model_output, model_target, self.detect_class_num)
            
            model_probabilities = torch.flatten(torch.cat(model_output)).detach().cpu().numpy()
            
            self.test_entropies.append(bce.item())    
            self.test_element_ids.append(element_id)     
            self.test_probabilities.append(model_probabilities)
            self.test_groundtruths.append(element_probabilities)
            
            for sub_class_index, sub_class_map in enumerate(sub_class_maps):
                    
                if element_id in sub_class_map:
                    
                    self.test_entropies_per_sub_class[sub_class_index].append(bce.item())
                    self.test_probabilities_per_sub_class[sub_class_index].append(model_probabilities)
                    self.test_groundtruths_per_sub_class[sub_class_index].append(element_probabilities)
                            
            element = self.data_loader.get_next_element('test', False)
        
        self.test_entropies = np.asarray(self.test_entropies)        
        self.test_probabilities = np.asarray(self.test_probabilities)
        self.test_groundtruths = np.asarray(self.test_groundtruths)
        
        for sub_class_index in range(self.sub_class_num):
            
            self.test_probabilities_per_sub_class[sub_class_index] = np.asarray(self.test_probabilities_per_sub_class[sub_class_index])
            self.test_groundtruths_per_sub_class[sub_class_index] = np.asarray(self.test_groundtruths_per_sub_class[sub_class_index])
            
        self.test_entropies_per_sub_class = np.asarray(self.test_entropies_per_sub_class, dtype=object)        
        self.test_probabilities_per_sub_class = np.asarray(self.test_probabilities_per_sub_class, dtype=object)
        self.test_groundtruths_per_sub_class = np.asarray(self.test_groundtruths_per_sub_class, dtype=object)

        self.save_test_stats()


    def save_test_stats(self):

        pathlib.Path(self.output_folder + 'test_stats/').mkdir(parents=True, exist_ok=True)
        
        np.save(self.output_folder + 'test_stats/test_entropies.npy', self.test_entropies)
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class) 

        np.save(self.output_folder + 'test_stats/test_probabilities.npy', self.test_probabilities)
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)
        
        np.save(self.output_folder + 'test_stats/test_groundtruths.npy', self.test_groundtruths)
        np.save(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', self.test_groundtruths_per_sub_class)
            
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'wb') as file:
    
            pickle.dump(self.test_element_ids, file, pickle.HIGHEST_PROTOCOL)


    def load_test_stats(self):
    
        self.test_entropies = np.load(self.output_folder + 'test_stats/test_entropies.npy')
        self.test_probabilities = np.load(self.output_folder + 'test_stats/test_probabilities.npy', allow_pickle=True)        
        self.test_groundtruths = np.load(self.output_folder + 'test_stats/test_groundtruths.npy', allow_pickle=True)
            
        self.test_entropies_per_sub_class = np.load(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', allow_pickle=True)            
        self.test_probabilities_per_sub_class = np.load(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', allow_pickle=True)            
        self.test_groundtruths_per_sub_class = np.load(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', allow_pickle=True)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 

            self.sub_classes = pickle.load(file)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'rb') as file: 

            self.test_element_ids = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)
    
    
    def compute_sub_class_losses(self, plot_sub_classes: list):
        
        sub_classes = []        
        sub_class_losses = []
                
        self.test_entropies_per_sub_class = [self.test_entropies_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_probabilities_per_sub_class = [self.test_probabilities_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_groundtruths_per_sub_class = [self.test_groundtruths_per_sub_class[i] for i in range(self.sub_class_num)]
        
        new_sub_classes = []
        new_sub_class_map_list = []
        new_test_entropies_list = []
        new_test_probabilities_list = []
        new_test_groundtruths_list = []
        
        
        if plot_sub_classes is None:
            plot_sub_classes = self.sub_classes
        
        for sub_class in plot_sub_classes:
            
            if sub_class in self.sub_classes:
            
                index = self.sub_classes.index(sub_class)
                
                sub_class_losses.append(self.test_entropies_per_sub_class[index])
                sub_classes.append(sub_class)
                    
            else:
                
                new_sub_classes.append(sub_class)                
                new_sub_class_map_list.append(load_dataclass_ids(self.parent_folder, sub_class))                
                new_test_entropies_list.append([])
                new_test_probabilities_list.append([])
                new_test_groundtruths_list.append([])
                
        for loss, prob, truth, element_id in zip(self.test_entropies, self.test_probabilities, self.test_groundtruths, self.test_element_ids):
                    
            for sub_class_index, sub_class_map in enumerate(new_sub_class_map_list):
                
                if element_id in sub_class_map:
                        
                    new_test_entropies_list[sub_class_index].append(loss)
                    new_test_probabilities_list[sub_class_index].append(prob)
                    new_test_groundtruths_list[sub_class_index].append(truth)
                    
        for sub_class_index, sub_class in enumerate(new_sub_classes):
                
            new_test_entropies = np.asarray(new_test_entropies_list[sub_class_index]) 
            new_test_probabilities = np.asarray(new_test_probabilities_list[sub_class_index])
            new_test_groundtruths = np.asarray(new_test_groundtruths_list[sub_class_index])
            
            self.sub_classes.append(sub_class)                
            self.test_entropies_per_sub_class.append(new_test_entropies)               
            self.test_probabilities_per_sub_class.append(new_test_probabilities)                
            self.test_groundtruths_per_sub_class.append(new_test_groundtruths)
                
            sub_classes.append(sub_class) 
            sub_class_losses.append(new_test_entropies)      
                
        self.test_entropies_per_sub_class = np.asarray(self.test_entropies_per_sub_class, dtype=object)
        self.test_probabilities_per_sub_class = np.asarray(self.test_probabilities_per_sub_class, dtype=object)
        self.test_groundtruths_per_sub_class = np.asarray(self.test_groundtruths_per_sub_class, dtype=object)
                
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)        
        np.save(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', self.test_groundtruths_per_sub_class)
            
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        self.sub_class_num = len(self.sub_classes)
        
        return sub_class_losses, sub_classes
            

    def plot_test_stats(self, threshold_num = 1001, plot_sub_classes = None, target_specificity=0.9):
        
        threshold_per_class = []
        
        bce_ticks = [ x / 20 for x in range(0, 11)]
        
        pathlib.Path(self.output_folder + 'test_stats/roc_curves/').mkdir(parents=True, exist_ok=True)
        
        plot_violin_distribution([self.test_entropies], ['Test set'], 'Cross Entropy', self.output_folder + '/test_stats/test_entropies', yticks = bce_ticks)
    
        sub_class_losses, plot_sub_classes = self.compute_sub_class_losses(plot_sub_classes)            
            
        loss_dict = {'class': [], 'detect': [], 'auc': [], 'sensitivity': []}
        
        if len(plot_sub_classes) > 0:
        
            plot_violin_distribution(sub_class_losses, plot_sub_classes, 'Cross Entropy', self.output_folder + '/test_stats/test_entropies_per_sub_class', yticks = bce_ticks)
        
        plot_sub_class_num = len(plot_sub_classes)
        
        for class_index, detect_class in enumerate(self.detect_classes):
            
            detections = self.test_probabilities[:, class_index]
            groundtruths = self.test_groundtruths[:, class_index]
                        
            roc_area, target_threshold, target_sensitivity = plot_roc_curve(detections,
                                                                   groundtruths,
                                                                   threshold_num,
                                                                   target_specificity,
                                                                   self.output_folder + 'test_stats/roc_curves/' + detect_class)
            
            loss_dict['class'].append('Total')
            loss_dict['detect'].append(detect_class)
            loss_dict['auc'].append(roc_area)
            loss_dict['sensitivity'].append(target_sensitivity)
            
            threshold_per_class += [target_threshold]
            
            if plot_sub_class_num > 0:
                
                detections_per_sub_class = []
                groundtruths_per_sub_class = []
                
                for sub_class in plot_sub_classes:
                    
                    sub_class_index = self.sub_classes.index(sub_class)
                    
                    sub_class_detections = self.test_probabilities_per_sub_class[sub_class_index][:, class_index]
                    sub_class_groundtruths = self.test_groundtruths_per_sub_class[sub_class_index][:, class_index]
                    
                    detections_per_sub_class.append(sub_class_detections)
                    groundtruths_per_sub_class.append(sub_class_groundtruths)
                    
                    positive_detections, negative_detections = get_detection_distributions(sub_class_detections, sub_class_groundtruths, target_threshold)
                       
                    loss_dict['class'].append(sub_class)
                    loss_dict['detect'].append(detect_class)   
                    loss_dict['sensitivity'].append(np.mean(positive_detections))
                
                roc_areas, target_thresholds, target_sensitivities = plot_multi_roc_curve(detections_per_sub_class,
                                                   groundtruths_per_sub_class,
                                                   plot_sub_classes,
                                                   threshold_num,
                                                   target_specificity,
                                                   self.output_folder + 'test_stats/roc_curves/' + detect_class + '_per_sub_class')
                
                for i, sub_class in enumerate(plot_sub_classes):
                             
                    loss_dict['auc'].append(roc_areas[i])
                    
        loss_dataframe = pd.DataFrame.from_dict(loss_dict)
        loss_dataframe.to_csv (self.output_folder + '/test_stats/test_losses.csv', index = False, header=True)
        
        for threshold in threshold_per_class:
            
            positive_detections_per_class = []
            negative_detections_per_class = []
            total_detections_per_class = []
            
            for class_index, detect_class in enumerate(self.detect_classes):
                
                detections = self.test_probabilities[:, class_index]
                groundtruths = self.test_groundtruths[:, class_index]
                
                pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/').mkdir(parents=True, exist_ok=True)
                
                positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)
                total_detections = np.concatenate((positive_detections, negative_detections))            
            
                positive_detections_per_class.append(positive_detections)
                negative_detections_per_class.append(negative_detections)  
                total_detections_per_class.append(total_detections)
                
                plot_accuracy_distribution(total_detections,
                                        ['Accuracy', 'Error rate'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/accuracy')
                
                plot_accuracy_distribution(positive_detections,
                                        ['Sensitivity', 'Miss rate'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/sensitivity')
                
                plot_accuracy_distribution(negative_detections,
                                        ['Specificity', 'Fall out'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/specificity')
                
                plot_detection_distribution(positive_detections,
                                            negative_detections,
                                            self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/detection')
                
            if self.detect_class_num > 1:
                
                plot_multi_accuracy_distribution(total_detections_per_class,
                                                ['Model'] * self.detect_class_num,
                                                self.detect_classes, 
                                                ['Accuracy', 'Error rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy_per_class')
                
                plot_multi_accuracy_distribution(positive_detections_per_class,
                                                ['Model'] * self.detect_class_num,
                                                self.detect_classes, 
                                                ['Sensitivity', 'Miss rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity_per_class')
                
                plot_multi_accuracy_distribution(negative_detections_per_class,
                                                ['Model'] * self.detect_class_num,
                                                self.detect_classes, 
                                                ['Specificity', 'Fall out'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity_per_class')
                
                plot_multi_detection_distribution(positive_detections_per_class,
                                                negative_detections_per_class,
                                                self.detect_classes,
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detection_per_class')
            
                plot_accuracy_distribution(np.concatenate(total_detections_per_class),
                                        ['Accuracy', 'Error rate'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy')
                
                plot_accuracy_distribution(np.concatenate(positive_detections_per_class),
                                        ['Sensitivity', 'Miss rate'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity')
                
                plot_accuracy_distribution(np.concatenate(negative_detections_per_class),
                                        ['Specificity', 'Fall out'],
                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity')
                
                plot_detection_distribution(np.concatenate(positive_detections_per_class),
                                            np.concatenate(negative_detections_per_class),
                                            self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detection')

            if plot_sub_class_num > 0:
                
                total_detections_per_class_per_sub_class = [[] for _ in range(self.detect_class_num)]
                positive_detections_per_class_per_sub_class = [[] for _ in range(self.detect_class_num)]
                negative_detections_per_class_per_sub_class = [[] for _ in range(self.detect_class_num)]
                
                for plot_sub_class_index, plot_sub_class in enumerate(plot_sub_classes): 
                    
                    sub_class_index = self.sub_classes.index(plot_sub_class)
                    
                    pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/').mkdir(parents=True, exist_ok=True)
                    
                    for class_index, detect_class in enumerate(self.detect_classes):
                        
                        pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/').mkdir(parents=True, exist_ok=True)         
                        
                        if len(self.test_probabilities_per_sub_class[sub_class_index]) > 0:
                            detections = self.test_probabilities_per_sub_class[sub_class_index][:, class_index]
                            groundtruths = self.test_groundtruths_per_sub_class[sub_class_index][:, class_index]
                        else:
                            detections = np.array([])
                            groundtruths = np.array([])
                        
                        positive_detections, negative_detections = get_detection_distributions(detections, groundtruths, threshold)                    
                        total_detections = np.concatenate((positive_detections, negative_detections))
                        
                        positive_detections_per_class_per_sub_class[class_index].append(positive_detections)
                        negative_detections_per_class_per_sub_class[class_index].append(negative_detections)                    
                        total_detections_per_class_per_sub_class[class_index].append(total_detections)
                        
                        plot_accuracy_distribution(total_detections,
                                                   ['Accuracy', 'Error rate'],
                                                   self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/accuracy')
                        
                        plot_accuracy_distribution(positive_detections,
                                                   ['Sensitivity', 'Miss rate'],
                                                   self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/sensitivity')
                        
                        plot_accuracy_distribution(negative_detections,
                                                   ['Specificity', 'Fall out'],
                                                   self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/specificity')
                        
                        plot_detection_distribution(positive_detections,
                                                    negative_detections,
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detect_class=' + detect_class + '/detection')
                        
                    pathlib.Path(self.output_folder + 'test_stats/threshold=' + str(threshold) + '/').mkdir(parents=True, exist_ok=True)
                    
                    total_detections_per_class = [total_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]                
                    positive_detections_per_class = [positive_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]
                    negative_detections_per_class = [negative_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]
                    
                    if self.detect_class_num > 1:

                        plot_multi_accuracy_distribution(total_detections_per_class,
                                                        ['Model'] * self.detect_class_num,
                                                        self.detect_classes,
                                                        ['Accuracy', 'Error rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/accuracy_per_class')
                        
                        
                        plot_multi_accuracy_distribution(positive_detections_per_class,
                                                        ['Model'] * self.detect_class_num,
                                                        self.detect_classes,
                                                        ['Sensitivity', 'Miss rate'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/sensitivity_per_class')
                        
                        plot_multi_accuracy_distribution(negative_detections_per_class,
                                                        ['Model'] * self.detect_class_num,
                                                        self.detect_classes,
                                                        ['Specificity', 'Fall out'],
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/specificity_per_class')
                        
                        plot_multi_detection_distribution(positive_detections_per_class,
                                                        negative_detections_per_class,
                                                        self.detect_classes,
                                                        self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class +  '/detection_per_class')

                        plot_accuracy_distribution(np.concatenate(total_detections_per_class),
                                                ['Accuracy', 'Error rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/accuracy')
                        
                        plot_accuracy_distribution(np.concatenate(positive_detections_per_class),
                                                ['Sensitivity', 'Miss rate'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/sensitivity')
                        
                        plot_accuracy_distribution(np.concatenate(negative_detections_per_class),
                                                ['Specificity', 'Fall out'],
                                                self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/specificity')
                        
                        plot_detection_distribution(np.concatenate(positive_detections_per_class),
                                                    np.concatenate(negative_detections_per_class),
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sub_class=' + plot_sub_class + '/detection')
                
                for class_index, detect_class in enumerate(self.detect_classes):
                    
                    plot_multi_accuracy_distribution(total_detections_per_class_per_sub_class[class_index],
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Accuracy', 'Error rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/accuracy_per_sub_class')
                    
                    plot_multi_accuracy_distribution(positive_detections_per_class_per_sub_class[class_index],
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Sensitivity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/sensitivity_per_sub_class')
                    
                    plot_multi_accuracy_distribution(negative_detections_per_class_per_sub_class[class_index],
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Specificity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/specificity_per_sub_class')
                    
                    plot_multi_detection_distribution(positive_detections_per_class_per_sub_class[class_index],
                                                    negative_detections_per_class_per_sub_class[class_index],
                                                    plot_sub_classes,
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detect_class=' + detect_class + '/detection_per_subclass')
                    
                if self.detect_class_num > 1:
                    
                    total_detections_per_sub_class = [np.concatenate([total_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]) for plot_sub_class_index in range(plot_sub_class_num)]
                    positive_detections_per_sub_class = [np.concatenate([positive_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]) for plot_sub_class_index in range(plot_sub_class_num)]
                    negative_detections_per_sub_class = [np.concatenate([negative_detections_per_class_per_sub_class[class_index][plot_sub_class_index] for class_index in range(self.detect_class_num)]) for plot_sub_class_index in range(plot_sub_class_num)]                
                        
                    plot_multi_accuracy_distribution(total_detections_per_sub_class,
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Accuracy', 'Error rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/accuracy_per_sub_class')
                    
                    plot_multi_accuracy_distribution(positive_detections_per_sub_class,
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Sensitivity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/sensitivity_per_sub_class')
                    
                    plot_multi_accuracy_distribution(negative_detections_per_sub_class,
                                                    ['Model'] * self.sub_class_num,
                                                    plot_sub_classes,
                                                    ['Specificity', 'Miss rate'],
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/specificity_per_sub_class')
                    
                    plot_multi_detection_distribution(positive_detections_per_sub_class,
                                                    negative_detections_per_sub_class,
                                                    plot_sub_classes,
                                                    self.output_folder + 'test_stats/threshold=' + str(threshold) + '/detection_per_sub_class')
                    
    def plot_diagnoses(self, percentile_range=[90, 100], total_num=10):

        self.load_test_stats()

        thresholds = [np.percentile(self.test_entropies, percent) for percent in percentile_range]
        
        for k in range(self.detect_class_num):
            
            for j in range(2):

                example_num = 0

                while example_num < total_num:

                    input_leads, _, target_probabilities, _, element_id = self.data_loader.get_next_element('test', False)


                    model_input, model_target = process_element(input_leads,
                                                                target_probabilities,
                                                                self.device)

                    with torch.no_grad():

                        model_output = self.model.forward(model_input)

                    loss = element_bce_function(model_output,
                                                model_target,
                                                self.detect_class_num)

                    if thresholds[0] < loss < thresholds[1] and model_target[k] == j:

                        example_num += 1
                        
                        diagnoses = self.data_loader.extract_element_diagnosis(element_id)
                        
                        print('Processed diagnoses:', diagnoses)
                        print('Target:', model_target)
                        print('Output:', model_output)
                        
                        print()
