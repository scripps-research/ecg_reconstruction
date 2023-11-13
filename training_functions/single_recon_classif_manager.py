from util_functions.classification_settings import get_classification_default_settings
from util_functions.reconstruction_settings import get_reconstruction_default_settings
from util_functions.general import get_lead_keys
from learn_functions.generate_model import generate_reconstructor, generate_classificator, generate_optimizer
from load_functions.train_loader import ReconClassifDataLoader
from training_functions.recon_classif_functions import process_element, process_batch, post_process_element, post_process_batch, element_loss_function, batch_loss_function
from training_functions.single_training_manager import TrainingManager
from plot_functions.detection_distribution import plot_detection_distribution, plot_multi_detection_distribution
from plot_functions.accuracy_distribution import plot_accuracy_distribution, plot_multi_accuracy_distribution
from plot_functions.roc_curve import plot_roc_curve, plot_multi_roc_curve
from plot_functions.continuos_distribution import plot_violin_distribution
from training_functions.classification_functions import get_detection_distributions
from util_functions.load_data_ids import load_dataclass_ids
import torch
import pickle
import pathlib
import numpy as np
import pandas as pd


class ReconClassifManager(TrainingManager):
    
    def __init__(self,
                 parent_folder: str,
                 device: str,
                 sub_classes: list,
                 input_leads: str,
                 output_leads: str,  
                 alpha: float,       
                 parallel_classification: bool,        
                 data_classes: str,
                 data_size: int,
                 detect_classes,
                 use_residual: bool,
                 epochs: int,
                 batch_size: int,
                 prioritize_percent: float,
                 prioritize_size: int,
                 optimizer_algorithm: str,
                 learning_rate: float,
                 weight_decay: float,
                 momentum: float,
                 nesterov: bool,                 
                 ):
        
        
        super().__init__(parent_folder,
                         device,
                         sub_classes,
                         input_leads,
                         epochs,
                         batch_size,
                         prioritize_percent)
        
        self.output_leads = output_leads
        self.output_lead_keys = get_lead_keys(self.output_leads)
        self.output_lead_num = len(self.output_lead_keys)
        self.classif_input_lead_keys = get_lead_keys('full')
        self.classif_input_lead_num = len(self.classif_input_lead_keys)
        
        if parallel_classification == 'true':        
            self.parallel_classification = True
        else:
            self.parallel_classification = False
        
        self.data_classes = data_classes
        self.data_class_num = len(self.data_classes)
        
        self.detect_classes = detect_classes
        self.detect_class_num = len(self.detect_classes)
        
        assert alpha <= 1
        assert alpha >= 0
        
        self.alpha = alpha
        
        self.test_losses = []
        self.test_mserrors = []
        self.test_rsquareds = []
        self.test_entropies = []
        self.test_probabilities = []
        self.test_groundtruths = []
        
        self.test_losses_per_sub_class = []        
        self.test_mserrors_per_sub_class = []
        self.test_rsquareds_per_sub_class = []
        self.test_entropies_per_sub_class = []        
        self.test_probabilities_per_sub_class = []
        self.test_groundtruths_per_sub_class = []
        
        for _ in self.sub_classes:
            
            self.test_losses_per_sub_class.append([])
            self.test_mserrors_per_sub_class.append([])
            self.test_rsquareds_per_sub_class.append([])
            self.test_entropies_per_sub_class.append([])            
            self.test_probabilities_per_sub_class.append([])
            self.test_groundtruths_per_sub_class.append([])

        self.data_loader = ReconClassifDataLoader(parent_folder=parent_folder,
                                                  data_classes=data_classes,
                                                  data_size=data_size,
                                                  detect_classes=detect_classes,
                                                  batch_size=batch_size,
                                                  prioritize_percent=prioritize_percent, 
                                                  prioritize_size=prioritize_size,
                                                  sample_num=self.sample_num,
                                                  min_value=self.min_value,
                                                  amplitude=self.amplitude,
                                                  recon_input_lead_keys=self.input_lead_keys,
                                                  classif_input_lead_keys=self.classif_input_lead_keys)
        
        self.model, self.classificator, self.model_folder = self.get_models(input_leads,
                                                                            output_leads,
                                                                            data_classes,
                                                                            detect_classes)

        
        self.optimizer = generate_optimizer(optimizer_algorithm,
                                            learning_rate,
                                            weight_decay,
                                            momentum,
                                            nesterov,
                                            self.model.parameters())
        
        self.output_folder = self.get_output_folder(input_leads,
                                                    output_leads,
                                                    alpha,
                                                    data_classes,
                                                    data_size,
                                                    detect_classes,
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
        
        try:
            self.load_model()
        except:
            self.save_model()
        

    def get_output_folder(self,
                          input_leads: str,
                          output_leads: str,
                          alpha: float,
                          data_classes,
                          data_size: str,
                          detect_classes,
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
        
        output_folder = self.parent_folder + 'ReconClassif' + '/input=' + input_leads + '/output=' + output_leads + '/data_class=' + str(data_classes) + '/data_size=' + str(data_size) + '/detect_class=' + str(detect_classes) + '/alpha=' + str(alpha) + '/'
                
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

        return output_folder
    
    
    def reset_sub_classes(self, new_sub_classes=None):
            
        self.test_losses_per_sub_class = np.load(self.output_folder + 'test_stats/test_losses_per_sub_class.npy', allow_pickle=True)
        self.test_mserrors_per_sub_class = np.load(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', allow_pickle=True)
        self.test_rsquareds_per_sub_class = np.load(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', allow_pickle=True)
        self.test_entropies_per_sub_class = np.load(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', allow_pickle=True)
        self.test_probabilities_per_sub_class = np.load(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', allow_pickle=True)
        self.test_groundtruths_per_sub_class = np.load(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', allow_pickle=True)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 
    
            self.sub_classes = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)
        
        if new_sub_classes is None:
            
            self.sub_classes = []
            self.sub_class_num = 0
            
            self.test_losses_per_sub_class = []
            self.test_mserrors_per_sub_class = []
            self.test_rsquareds_per_sub_class = []
            self.test_entropies_per_sub_class = []
            self.test_probabilities_per_sub_class = []
            self.test_groundtruths_per_sub_class = []
            
        else:
            
            for sub_class in new_sub_classes:
                
                if sub_class in self.sub_classes:
                
                    sub_class_index = self.sub_classes.index(sub_class)
                    del self.sub_classes[sub_class_index]
                    self.sub_class_num -= 1
                    
                    self.test_losses_per_sub_class = np.delete(self.test_losses_per_sub_class, sub_class_index, axis=0)
                    self.test_mserrors_per_sub_class = np.delete(self.test_mserrors_per_sub_class, sub_class_index, axis=0)
                    self.test_rsquareds_per_sub_class = np.delete(self.test_rsquareds_per_sub_class, sub_class_index, axis=0)
                    self.test_entropies_per_sub_class = np.delete(self.test_entropies_per_sub_class, sub_class_index, axis=0)
                    self.test_probabilities_per_sub_class = np.delete(self.test_probabilities_per_sub_class, sub_class_index, axis=0)
                    self.test_groundtruths_per_sub_class = np.delete(self.test_groundtruths_per_sub_class, sub_class_index, axis=0)
                    
        np.save(self.output_folder + 'test_stats/test_losses_per_sub_class.npy', self.test_losses_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class)        
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_groundtruths_losses_per_sub_class.npy', self.test_groundtruths_per_sub_class)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
        
    
    def get_models(self, input_leads: str, output_leads: str, data_classes, detect_classes):
        
        data_size, input_channel, middle_channel, output_channel, input_depth, \
            middle_depth, output_depth, input_kernel, middle_kernel, output_kernel, use_residual, \
                epoch_num, batch_size, prioritize_percent, prioritize_size, \
                    optimizer, learning_rate, weight_decay, _, _ = get_reconstruction_default_settings()  
        
        reconstructor_folder = self.parent_folder + 'Reconstruction' + '/input=' + input_leads + '/output=' + output_leads + '/data_class=' + str(data_classes) + '/data_size=' + str(data_size) +\
                '/channel=' + str(input_channel) + '_' + str(middle_channel) + '_' + str(output_channel) + '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) + '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) + '_' + str(output_kernel) + '/' 
                
        reconstructor_folder += 'residual=' + use_residual + '/epochs=' + str(epoch_num) + '/batch_size=' + str(batch_size) + '/prioritize_percent=' + str(prioritize_percent) + '/prioritize_size=' + str(prioritize_size) +\
            '/optimizer=' + optimizer + '/lr=' + str(learning_rate) + '/' + 'decay=' + str(weight_decay) + '/'

                    
        reconstructor = generate_reconstructor(input_lead_num=self.input_lead_num,
                                               output_lead_num=self.output_lead_num,
                                               input_channel_per_lead=input_channel,
                                               middle_channel_per_lead=middle_channel,
                                               output_channel_per_lead=output_channel,
                                               block_per_input_network=input_depth,
                                               block_per_middle_network=middle_depth,
                                               block_per_output_network=output_depth,
                                               input_kernel_size=input_kernel,
                                               middle_kernel_size=middle_kernel,
                                               output_kernel_size=output_kernel,
                                               use_residual=use_residual,
                                               device=self.device)
        
        reconstructor.load_state_dict(reconstructor_folder + 'state_dict/')
        
        data_size, input_channel, middle_channel, input_depth, \
            middle_depth, output_depth, input_kernel, middle_kernel, stride_size, use_residual, \
                epoch_num, batch_size, prioritize_percent, prioritize_size, \
                    optimizer, learning_rate, weight_decay, _, _ = get_classification_default_settings()
                    
        classificator = generate_classificator(input_lead_num=self.classif_input_lead_num,
                                                detect_class_num=self.detect_class_num,
                                                input_channel_per_lead=input_channel,
                                                middle_channel_per_class=middle_channel,
                                                block_per_input_network=input_depth,
                                                block_per_middle_network=middle_depth,
                                                block_per_output_network=output_depth,
                                                input_kernel_size=input_kernel,
                                                middle_kernel_size=middle_kernel,                                               
                                                stride_size=stride_size,
                                                average_pool=int(self.sample_num / (stride_size ** middle_depth)),
                                                use_residual=use_residual,
                                                device=self.device,
                                                parallel=self.parallel_classification)
                    
        if self.parallel_classification:
            
            classificator_folder = []
            
            for detect_class in detect_classes:
            
                folder = self.parent_folder + 'Classification' + '/input=' + 'full' + '/data_class=' + str(data_classes) + '/data_size=' + str(data_size) + '/detect_class=' + str([detect_class]) +\
                    '/channel=' + str(input_channel) + '_' + str(middle_channel) + '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) + '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) + '/stride=' + str(stride_size) + '/'
                    
                folder += 'residual=' + use_residual + '/epochs=' + str(epoch_num) + '/batch_size=' + str(batch_size) + '/prioritize_percent=' + str(prioritize_percent) + '/prioritize_size=' + str(prioritize_size) +\
                    '/optimizer=' + optimizer + '/lr=' + str(learning_rate) + '/' + 'decay=' + str(weight_decay) + '/state_dict/'
                    
                classificator_folder.append(folder)
            

        else:
        
            classificator_folder = self.parent_folder + 'Classification' + '/input=' + 'full' + '/data_class=' + str(data_classes) + '/data_size=' + str(data_size) + '/detect_class=' + str(detect_classes) +\
                '/channel=' + str(input_channel) + '_' + str(middle_channel) + '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) + '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) + '/stride=' + str(stride_size) + '/'
                
            classificator_folder += 'residual=' + use_residual + '/epochs=' + str(epoch_num) + '/batch_size=' + str(batch_size) + '/prioritize_percent=' + str(prioritize_percent) + '/prioritize_size=' + str(prioritize_size) +\
                '/optimizer=' + optimizer + '/lr=' + str(learning_rate) + '/' + 'decay=' + str(weight_decay) + '/state_dict/'
            
        classificator.load_state_dict(classificator_folder)
                
        
        return reconstructor, classificator, reconstructor_folder
        
        
    def load_dataset(self, train=False, valid=False, test=False, extract_qrs=False):
    
        self.data_loader.load(train, valid, test, extract_qrs)
        
        
    def reset_model(self):
        
        self.model.load_state_dict(self.model_folder + 'state_dict/')
        
        
    def train_step(self, batch):
        
        reconstruction_input, reconstruction_target, classification_input, classification_target =\
            process_batch(batch, self.input_lead_num, self.classif_input_lead_num, self.output_lead_num, self.detect_class_num, self.output_lead_keys, self.classif_input_lead_keys, self.min_value, self.amplitude, self.device)

        reconstruction_output = self.model.forward(reconstruction_input)
        
        classification_input = post_process_batch(classification_input, reconstruction_output, self.classif_input_lead_keys, self.output_lead_keys, self.min_value, self.amplitude)
        
        classification_output = self.classificator.forward(classification_input)
            
        batch_loss, batch_loss_per_element = batch_loss_function(reconstruction_output,
                                                                reconstruction_target,
                                                                classification_output,
                                                                classification_target,
                                                                self.output_lead_num,
                                                                self.detect_class_num,
                                                                self.alpha,
                                                                self.batch_size,
                                                                self.compute_loss_per_element)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        self.data_loader.update_priority_weights(batch_loss_per_element)

        self.train_losses.append(batch_loss.item())
                
        return self.data_loader.get_next_batch('train', False)


    def valid_step(self, batch):
        
        reconstruction_input, reconstruction_target, classification_input, classification_target =\
            process_batch(batch, self.input_lead_num, self.classif_input_lead_num, self.output_lead_num, self.detect_class_num, self.output_lead_keys,
                          self.classif_input_lead_keys, self.min_value, self.amplitude, self.device)


        with torch.no_grad():
        
            reconstruction_output = self.model.forward(reconstruction_input)
            
            classification_input = post_process_batch(classification_input, reconstruction_output, self.classif_input_lead_keys, self.output_lead_keys, self.min_value, self.amplitude)
            
            classification_output = self.classificator.forward(classification_input)

        batch_loss, _ = batch_loss_function(reconstruction_output,
                                            reconstruction_target,
                                            classification_output,
                                            classification_target,
                                            self.output_lead_num,
                                            self.detect_class_num,
                                            self.alpha,
                                            self.batch_size,
                                            False)

        self.valid_losses.append(batch_loss.item())

        return self.data_loader.get_next_batch('valid', False)
        

    def test(self):
        
        self.data_loader.shuffle('test')
        
        sub_class_maps = []
        
        for data_class in self.sub_classes:
            sub_class_maps.append(load_dataclass_ids(self.parent_folder, data_class))

        element = self.data_loader.get_next_element('test', False)

        while element is not None:

            recon_input_leads, classif_input_leads, target_probabilities, _, element_id = element
            
            reconstruction_input, reconstruction_target, classification_input, classification_target =\
                process_element(recon_input_leads,
                                classif_input_leads,
                                self.output_lead_keys,
                                self.classif_input_lead_keys,
                                target_probabilities,
                                self.min_value,
                                self.amplitude,
                                self.device)

            with torch.no_grad():
                
                reconstruction_output = self.model.forward(reconstruction_input)
                
                classification_input = post_process_element(classification_input,
                                                            reconstruction_output,
                                                            self.classif_input_lead_keys,
                                                            self.output_lead_keys,
                                                            self.min_value,
                                                            self.amplitude)
            
                classification_output = self.classificator.forward(classification_input)
                           
            loss, mse, r2, bce = element_loss_function(reconstruction_output,
                                         reconstruction_target,
                                         classification_output,
                                         classification_target,
                                         self.output_lead_num,
                                         self.detect_class_num,
                                         self.alpha)
            
            model_probabilities =torch.flatten(torch.cat(classification_output)).detach().cpu().numpy()
            
            self.test_losses.append(loss.item())
            self.test_element_ids.append(element_id)
            self.test_mserrors.append(mse.item())
            self.test_rsquareds.append(r2.item())
            self.test_entropies.append(bce.item())            
            self.test_probabilities.append(model_probabilities)
            self.test_groundtruths.append(target_probabilities)
            
            for sub_class_index, sub_class_map in enumerate(sub_class_maps):
                    
                if element_id in sub_class_map:           
                    
                    self.test_losses_per_sub_class[sub_class_index].append(loss.item())
                    self.test_mserrors_per_sub_class[sub_class_index].append(mse.item())
                    self.test_rsquareds_per_sub_class[sub_class_index].append(r2.item())
                    self.test_entropies_per_sub_class[sub_class_index].append(bce.item())
                    self.test_probabilities_per_sub_class[sub_class_index].append(model_probabilities)
                    self.test_groundtruths_per_sub_class[sub_class_index].append(target_probabilities)
                            
            element = self.data_loader.get_next_element('test', False)
        
        self.test_losses = np.asarray(self.test_losses)        
        self.test_mserrors = np.asarray(self.test_mserrors)
        self.test_rsquareds = np.asarray(self.test_rsquareds)
        self.test_entropies = np.asarray(self.test_entropies)
        self.test_probabilities = np.asarray(self.test_probabilities)
        self.test_groundtruths = np.asarray(self.test_groundtruths)
        
        self.test_losses_per_sub_class = np.asarray(self.test_losses_per_sub_class, dtype=object)
        self.test_mserrors_per_sub_class = np.asarray(self.test_mserrors_per_sub_class, dtype=object)
        self.test_rsquareds_per_sub_class = np.asarray(self.test_rsquareds_per_sub_class, dtype=object)                
        
        for sub_class_index in range(self.sub_class_num):
            
            self.test_probabilities_per_sub_class[sub_class_index] = np.asarray(self.test_probabilities_per_sub_class[sub_class_index])
            self.test_groundtruths_per_sub_class[sub_class_index] = np.asarray(self.test_groundtruths_per_sub_class[sub_class_index])
        
        self.test_probabilities_per_sub_class = np.asarray(self.test_probabilities_per_sub_class, dtype=object)
        self.test_groundtruths_per_sub_class = np.asarray(self.test_groundtruths_per_sub_class, dtype=object)
        
        self.save_test_stats()
        

    def save_test_stats(self):
    
        pathlib.Path(self.output_folder + 'test_stats/').mkdir(parents=True, exist_ok=True)
        
        np.save(self.output_folder + 'test_stats/test_losses.npy', self.test_losses)
        np.save(self.output_folder + 'test_stats/test_mserrors.npy', self.test_mserrors)
        np.save(self.output_folder + 'test_stats/test_rsquareds.npy', self.test_rsquareds)
        np.save(self.output_folder + 'test_stats/test_entropies.npy', self.test_entropies)
        np.save(self.output_folder + 'test_stats/test_probabilities.npy', self.test_probabilities)        
        np.save(self.output_folder + 'test_stats/test_groundtruths.npy', self.test_groundtruths)
        
        np.save(self.output_folder + 'test_stats/test_losses_per_sub_class.npy', self.test_losses_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class)         
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class)              
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', self.test_groundtruths_per_sub_class)  
            
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'wb') as file:
    
            pickle.dump(self.test_element_ids, file, pickle.HIGHEST_PROTOCOL)


    def load_test_stats(self):
    
        self.test_losses = np.load(self.output_folder + 'test_stats/test_losses.npy') 
        self.test_mserrors = np.load(self.output_folder + 'test_stats/test_mserrors.npy')        
        self.test_rsquareds = np.load(self.output_folder + 'test_stats/test_rsquareds.npy') 
        self.test_entropies = np.load(self.output_folder + 'test_stats/test_entropies.npy') 
        self.test_probabilities = np.load(self.output_folder + 'test_stats/test_probabilities.npy', allow_pickle=True)
        self.test_groundtruths = np.load(self.output_folder + 'test_stats/test_groundtruths.npy', allow_pickle=True)
        
        self.test_losses_per_sub_class = np.load(self.output_folder + 'test_stats/test_losses_per_sub_class.npy', allow_pickle=True)    
        self.test_mserrors_per_sub_class = np.load(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', allow_pickle=True)    
        self.test_rsquareds_per_sub_class = np.load(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', allow_pickle=True)    
        self.test_entropies_per_sub_class = np.load(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', allow_pickle=True)          
        self.test_probabilities_per_sub_class = np.load(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', allow_pickle=True)
        self.test_groundtruths_per_sub_class = np.load(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', allow_pickle=True)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 

            self.sub_classes = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'rb') as file: 
    
            self.test_element_ids = pickle.load(file)
            
        
    def compute_sub_class_losses(self, plot_sub_classes: list):
        
        sub_classes = []
        sub_class_losses = []
        sub_class_mserrors = []
        sub_class_rsquareds = []
        sub_class_entropies = []
        
        self.test_losses_per_sub_class = [self.test_losses_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_mserrors_per_sub_class = [self.test_mserrors_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_rsquareds_per_sub_class = [self.test_rsquareds_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_entropies_per_sub_class = [self.test_entropies_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_probabilities_per_sub_class = [self.test_probabilities_per_sub_class[i] for i in range(self.sub_class_num)]
        self.test_groundtruths_per_sub_class = [self.test_groundtruths_per_sub_class[i] for i in range(self.sub_class_num)]
        
        if plot_sub_classes is None:
            plot_sub_classes = self.sub_classes
            
        new_sub_classes = []            
        new_sub_class_map_list = []
        new_test_losses_list = []
        new_test_mserrors_list = []
        new_test_rsquareds_list = []
        new_test_entropies_list = []
        new_test_probabilities_list = []
        new_test_groundtruths_list = []
        
        for sub_class in plot_sub_classes:
            
            if sub_class in self.sub_classes:
            
                index = self.sub_classes.index(sub_class)
                sub_classes.append(sub_class)              
                sub_class_losses.append(self.test_losses_per_sub_class[index])
                sub_class_mserrors.append(self.test_mserrors_per_sub_class[index])
                sub_class_rsquareds.append(self.test_rsquareds_per_sub_class[index])
                sub_class_entropies.append(self.test_entropies_per_sub_class[index])
                    
            else:
                
                new_sub_classes.append(sub_class)                 
                new_sub_class_map_list.append(load_dataclass_ids(self.parent_folder, sub_class))              
                new_test_losses_list.append([])
                new_test_mserrors_list.append([])
                new_test_rsquareds_list.append([])
                new_test_entropies_list.append([])
                new_test_probabilities_list.append([])
                new_test_groundtruths_list.append([])
                
        for loss, mse, r2, bce, prob, truth, element_id in zip(self.test_losses, self.test_mserrors, self.test_rsquareds, self.test_entropies, self.test_probabilities, self.test_groundtruths, self.test_element_ids):
                        
            for sub_class_index, sub_class_map in enumerate(new_sub_class_map_list):
                    
                if element_id in sub_class_map:
                    
                    new_test_losses_list[sub_class_index].append(loss)
                    new_test_mserrors_list[sub_class_index].append(mse)
                    new_test_rsquareds_list[sub_class_index].append(r2)
                    new_test_entropies_list[sub_class_index].append(bce)
                    new_test_probabilities_list[sub_class_index].append(prob)
                    new_test_groundtruths_list[sub_class_index].append(truth)
                    
        for sub_class_index, sub_class in enumerate(new_sub_classes):
                
            new_test_losses = np.asarray(new_test_losses_list[sub_class_index])  
            new_test_mserrors = np.asarray(new_test_mserrors_list[sub_class_index])        
            new_test_rsquareds = np.asarray(new_test_rsquareds_list[sub_class_index])  
            new_test_entropies = np.asarray(new_test_entropies_list[sub_class_index])  
            new_test_probabilities = np.asarray(new_test_probabilities_list[sub_class_index])
            new_test_groundtruths = np.asarray(new_test_groundtruths_list[sub_class_index])
            
            self.sub_classes.append(sub_class)                
            self.test_losses_per_sub_class.append(new_test_losses)
            self.test_mserrors_per_sub_class.append(new_test_mserrors) 
            self.test_rsquareds_per_sub_class.append(new_test_rsquareds)                
            self.test_entropies_per_sub_class.append(new_test_entropies)                
            self.test_probabilities_per_sub_class.append(new_test_probabilities)                
            self.test_groundtruths_per_sub_class.append(new_test_groundtruths)
            
            sub_classes.append(sub_class) 
            sub_class_losses.append(new_test_losses) 
            sub_class_mserrors.append(new_test_mserrors) 
            sub_class_rsquareds.append(new_test_rsquareds) 
            sub_class_entropies.append(new_test_entropies)                
                
        self.test_losses_per_sub_class = np.asarray(self.test_losses_per_sub_class, dtype=object)
        self.test_mserrors_per_sub_class = np.asarray(self.test_mserrors_per_sub_class, dtype=object)
        self.test_rsquareds_per_sub_class = np.asarray(self.test_rsquareds_per_sub_class, dtype=object)
        self.test_entropies_per_sub_class = np.asarray(self.test_entropies_per_sub_class, dtype=object)
        self.test_probabilities_per_sub_class = np.asarray(self.test_probabilities_per_sub_class, dtype=object)
        self.test_groundtruths_per_sub_class = np.asarray(self.test_groundtruths_per_sub_class, dtype=object)
                
        np.save(self.output_folder + 'test_stats/test_losses_per_sub_class.npy', self.test_losses_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_entropies_per_sub_class.npy', self.test_entropies_per_sub_class) 
        np.save(self.output_folder + 'test_stats/test_probabilities_per_sub_class.npy', self.test_probabilities_per_sub_class)        
        np.save(self.output_folder + 'test_stats/test_groundtruths_per_sub_class.npy', self.test_groundtruths_per_sub_class)
            
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        self.sub_class_num = len(self.sub_classes)
        
        return sub_classes, sub_class_losses, sub_class_mserrors, sub_class_rsquareds, sub_class_entropies
        

    def plot_test_stats(self, threshold_num = 1001, plot_sub_classes = None, target_specificity=0.9):
        
        threshold_per_class = []
        
        loss_dict = {'class': [], 'detect': [], 'auc': [], 'sensitivity': []}
        
        pathlib.Path(self.output_folder + 'test_stats/roc_curves/').mkdir(parents=True, exist_ok=True)
        
        loss_ticks = [x / 20 for x in range(-10, 1)]
        r2_ticks = [x / 0.1 for x in range(0, 11)]
        mse_ticks = [ x / 200 for x in range(0, 11)]
        bce_ticks = [ x / 20 for x in range(0, 11)]
        
        plot_violin_distribution([self.test_losses], ['Test set'], 'Loss', self.output_folder + '/test_stats/test_losses', yticks=loss_ticks)
        plot_violin_distribution([self.test_mserrors], ['Test set'], 'MSE [$mV^2$]', self.output_folder + '/test_stats/test_mserrors', yticks=mse_ticks)
        plot_violin_distribution([self.test_rsquareds * 100], ['Test set'], 'R2 [%]', self.output_folder + '/test_stats/test_rsquareds', yticks=r2_ticks)
        plot_violin_distribution([self.test_entropies], ['Test set'], 'BCE', self.output_folder + '/test_stats/test_entropies', yticks=bce_ticks)
        
        plot_sub_classes, sub_class_losses, sub_class_mserrors, sub_class_rsquareds, sub_class_entropies = self.compute_sub_class_losses(plot_sub_classes)
        
        if len(plot_sub_classes) > 0:
        
            plot_violin_distribution(sub_class_losses, plot_sub_classes, 'Loss', self.output_folder + '/test_stats/test_losses_per_sub_class', yticks=loss_ticks)
            plot_violin_distribution(sub_class_mserrors, plot_sub_classes, 'MSE [$mV^2$]', self.output_folder + '/test_stats/test_mserrors_per_sub_class', yticks=mse_ticks)
            plot_violin_distribution(sub_class_rsquareds * 100, plot_sub_classes, 'R2 [%]', self.output_folder + '/test_stats/test_rsquareds_per_sub_class', yticks=r2_ticks)
            plot_violin_distribution(sub_class_entropies, plot_sub_classes, 'BCE', self.output_folder + '/test_stats/test_entropies_per_sub_class', yticks=bce_ticks)
        
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
                                                   0.9,
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
            
            plot_sub_class_num = len(plot_sub_classes)

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