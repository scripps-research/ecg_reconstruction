from load_functions.train_loader import ReconstructionDataLoader
from training_functions.single_training_manager import TrainingManager
from util_functions.load_data_ids import load_dataclass_ids
from util_functions.general import remove_dir, remove_dir, get_lead_keys, get_twelve_keys
from learn_functions.generate_model import generate_reconstructor, generate_optimizer
from training_functions.reconstruction_functions import batch_r2_function, element_mse_function, element_r2_function, process_batch, process_element, deprocess_element
from plot_functions.ecg_signal import plot_output_recon_leads, plot_medical_recon_leads
from plot_functions.continuos_distribution import plot_violin_distribution, plot_box_distribution
from plot_functions.bivariate_distribution import plot_2d_distribution
import torch
import pickle
import pathlib
import numpy as np
import pandas as pd


class ReconstructionManager(TrainingManager):
    
    def __init__(self,
                 parent_folder: str,
                 device: str,
                 sub_classes: str,
                 input_leads: str,
                 output_leads: str,
                 data_classes: str,
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
                         prioritize_size)
        
        self.output_lead_keys = get_lead_keys(output_leads)
        self.output_lead_num = len(self.output_lead_keys)
        
        self.test_mserrors = []
        self.test_rsquareds = []
        self.test_mserrors_per_sub_class = []
        self.test_rsquareds_per_sub_class = []
        
        for _ in self.sub_classes:
            self.test_mserrors_per_sub_class.append([])
            self.test_rsquareds_per_sub_class.append([])
            
        self.test_mserrors_per_sample = []
        self.test_distances_per_sample = []
        self.test_leads_per_sample = []

        self.data_loader = ReconstructionDataLoader(parent_folder,
                                                    data_classes,
                                                    data_size,
                                                    batch_size,
                                                    prioritize_percent, 
                                                    prioritize_size,
                                                    self.sample_num,
                                                    self.min_value,
                                                    self.amplitude,
                                                    self.input_lead_keys,    
                                                    self.output_lead_keys,
                                                    )


        self.model = generate_reconstructor(self.input_lead_num,
                                            self.output_lead_num,
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
                                            self.device)
        
        self.optimizer = generate_optimizer(optimizer_algorithm,
                                            learning_rate,
                                            weight_decay,
                                            momentum,
                                            nesterov,
                                            self.model.parameters())
        
        self.output_folder = self.get_output_folder(parent_folder,
                                                    input_leads,
                                                    output_leads,
                                                    data_classes,
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

    def get_output_folder(self,
                          parent_folder: str,
                          input_leads: str,
                          output_leads: str,
                          data_classes,
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
                          prioritize_size: int,
                          optimizer_algorithm: str,
                          learning_rate: float,
                          weight_decay: float,
                          momentum: float,
                          nesterov: bool):

        output_folder = parent_folder + 'Reconstruction' + '/input=' + input_leads + '/output=' + output_leads + '/data_class=' + str(data_classes) + '/data_size=' + data_size +\
            '/channel=' + str(input_channel) + '_' + str(middle_channel) + '_' + str(output_channel) +\
                '/depth=' + str(input_depth) + '_' + str(middle_depth) + '_' + str(output_depth) +\
                    '/kernel=' + str(input_kernel) + '_' + str(middle_kernel) + '_' + str(output_kernel) + '/'
                
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
        
        self.test_mserrors_per_sub_class = np.load(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', allow_pickle=True)        
        self.test_rsquareds_per_sub_class = np.load(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', allow_pickle=True)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 
    
            self.sub_classes = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)
        
        if new_sub_classes is None:
            
            self.sub_classes = []
            self.sub_class_num = 0
            
            self.test_mserrors_per_sub_class = []
            self.test_rsquareds_per_sub_class = []         
            
        else:
            
            for sub_class in new_sub_classes:
                
                if sub_class in self.sub_classes:
                
                    sub_class_index = self.sub_classes.index(sub_class)
                    del self.sub_classes[sub_class_index]
                    self.sub_class_num -= 1
                    
                    self.test_mserrors_per_sub_class = np.delete(self.test_mserrors_per_sub_class, sub_class_index, axis=0)
                    self.test_rsquareds_per_sub_class = np.delete(self.test_rsquareds_per_sub_class, sub_class_index, axis=0)
        
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class)
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
        
    def load_dataset(self, train=False, valid=False, test=False, extract_qrs=False):

        self.data_loader.load(train, valid, test, extract_qrs)
        

    def train_step(self, batch):
        
        model_input, model_target = process_batch(batch,
                                                  self.input_lead_num,
                                                  self.output_lead_num,
                                                  self.min_value,
                                                  self.amplitude,
                                                  self.device)

        model_output = self.model.forward(model_input)

        # batch_loss, loss_per_element = batch_loss_function(model_output,
        #                                                          model_target,
        #                                                          self.output_lead_num,
        #                                                          self.batch_size,
        #                                                          self.compute_loss_per_element)
        
        batch_loss, loss_per_element = batch_r2_function(model_output,
                                                         model_target,
                                                         self.output_lead_num,
                                                         self.batch_size,
                                                         self.compute_loss_per_element)

        self.optimizer.zero_grad(set_to_none=True)
        
        batch_loss.backward()
                
        self.optimizer.step()

        self.data_loader.update_priority_weights(loss_per_element)

        self.train_losses.append(batch_loss.item())
                
        return self.data_loader.get_next_batch('train', False)


    def valid_step(self, batch):
        
        model_input, model_target = process_batch(batch,
                                                  self.input_lead_num,
                                                  self.output_lead_num,                                                  
                                                  self.min_value,
                                                  self.amplitude,
                                                  self.device)

        with torch.no_grad():
        
            model_output = self.model.forward(model_input)

        # batch_loss, _ = batch_mse_function(model_output,
        #                                     model_target,
        #                                     self.output_lead_num,
        #                                     self.batch_size,
        #                                     False)
        
        batch_loss, _ = batch_r2_function(model_output,
                                          model_target,
                                          self.output_lead_num,
                                          self.batch_size,
                                          False)

        self.valid_losses.append(batch_loss.item())

        return self.data_loader.get_next_batch('valid', False)


    def test(self, compute_loss_per_sample = False):
        
        sub_class_maps = []
        
        for data_class in self.sub_classes:
            sub_class_maps.append(load_dataclass_ids(self.parent_folder, data_class))
        
        self.data_loader.shuffle('test')

        element = self.data_loader.get_next_element('test', compute_loss_per_sample)

        j = 0

        while element is not None:

            input_leads, output_leads, _, qrs_times, element_id = element

            j += 1
            
            if j % 1000 == 0:
                
                print('Completed test over', j, 'element!')

            model_input, model_target = process_element(input_leads,
                                                        output_leads,                                                        
                                                        self.min_value,
                                                        self.amplitude,
                                                        self.device)

            with torch.no_grad():
            
                model_output = self.model.forward(model_input)

            mse, mse_per_sample, distance_per_sample, lead_per_sample = element_mse_function(model_output,
                                                                                                model_target,
                                                                                                self.output_lead_num,
                                                                                                qrs_times = qrs_times,
                                                                                                sample_num = self.sample_num)
            
            r2 = element_r2_function(model_output,
                                     model_target,
                                     self.output_lead_num)               
                
            if compute_loss_per_sample:

                self.test_mserrors_per_sample += mse_per_sample
                self.test_distances_per_sample += distance_per_sample
                self.test_leads_per_sample += lead_per_sample
            
            self.test_mserrors.append(mse.item())
            self.test_rsquareds.append(r2.item())
            self.test_element_ids.append(element_id)
            
            for data_class_idx, data_class_map in enumerate(sub_class_maps):
                
                if element_id in data_class_map:
                    self.test_mserrors_per_sub_class[data_class_idx].append(mse.item())
                    self.test_rsquareds_per_sub_class[data_class_idx].append(r2.item())

            element = self.data_loader.get_next_element('test', compute_loss_per_sample)
            
        self.test_mserrors = np.asarray(self.test_mserrors)
        self.test_mserrors_per_sub_class = np.asarray(self.test_mserrors_per_sub_class, dtype=object)
        
        self.test_rsquareds = np.asarray(self.test_rsquareds)
        self.test_mserrors_per_sub_class = np.asarray(self.test_rsquareds_per_sub_class, dtype=object)
            
        self.test_mserrors_per_sample = np.asarray(self.test_mserrors_per_sample)
        self.test_distances_per_sample = np.asarray(self.test_distances_per_sample)
        self.test_leads_per_sample = np.asarray(self.test_leads_per_sample)

        self.save_test_stats(compute_loss_per_sample)


    def save_test_stats(self, compute_loss_per_sample = False):

        pathlib.Path(self.output_folder + 'test_stats/').mkdir(parents=True, exist_ok=True)

        np.save(self.output_folder + 'test_stats/test_mserrors.npy', self.test_mserrors)                        
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class)
        
        np.save(self.output_folder + 'test_stats/test_rsquareds.npy', self.test_rsquareds)                        
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class)
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'wb') as file:
    
            pickle.dump(self.test_element_ids, file, pickle.HIGHEST_PROTOCOL)
        
        if compute_loss_per_sample:
    
            np.save(self.output_folder + 'test_stats/test_mserrors_per_sample.npy', self.test_mserrors_per_sample)
            np.save(self.output_folder + 'test_stats/test_distances_per_sample.npy', self.test_distances_per_sample)
            np.save(self.output_folder + 'test_stats/test_leads_per_sample.npy', self.test_leads_per_sample)


    def load_test_stats(self, compute_loss_per_sample = False):
        
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'rb') as file: 
    
            self.sub_classes = pickle.load(file)
            
        self.sub_class_num = len(self.sub_classes)

        self.test_mserrors = np.load(self.output_folder + 'test_stats/test_mserrors.npy')            
        self.test_mserrors_per_sub_class = np.load(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', allow_pickle=True)
        
        self.test_rsquareds = np.load(self.output_folder + 'test_stats/test_rsquareds.npy')            
        self.test_rsquareds_per_sub_class = np.load(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', allow_pickle=True)
            
        with open(self.output_folder + 'test_stats/test_element_ids.pkl', 'rb') as file: 
    
            self.test_element_ids = pickle.load(file)

        if compute_loss_per_sample:

            self.test_mserrors_per_sample = np.load(self.output_folder + 'test_stats/test_mserrors_per_sample.npy')
            self.test_distances_per_sample = np.load(self.output_folder + 'test_stats/test_distances_per_sample.npy')
            self.test_leads_per_sample = np.load(self.output_folder + 'test_stats/test_leads_per_sample.npy')
            
            
    def compute_sub_class_losses(self, plot_sub_classes: list):
        
        sub_classes = []   
        sub_class_mserrors = []
        sub_class_rsquareds = []
        
        if plot_sub_classes is None:
            plot_sub_classes = self.sub_classes
            
        self.test_mserrors_per_sub_class = [self.test_mserrors_per_sub_class[i] for i in range(len(self.sub_classes))]
        self.test_rsquareds_per_sub_class = [self.test_rsquareds_per_sub_class[i] for i in range(len(self.sub_classes))]
        
        new_sub_classes = []
        new_sub_class_map_list = []
        new_test_mserrors_list = []
        new_test_rsquareds_list = []
        
        for sub_class in plot_sub_classes:
            
            if sub_class in self.sub_classes:
            
                index = self.sub_classes.index(sub_class)
                
                sub_classes.append(sub_class)                 
                sub_class_mserrors.append(self.test_mserrors_per_sub_class[index])
                sub_class_rsquareds.append(self.test_rsquareds_per_sub_class[index])
                    
            else:
                
                new_sub_classes.append(sub_class)                
                new_sub_class_map_list.append(load_dataclass_ids(self.parent_folder, sub_class))                 
                new_test_mserrors_list.append([])
                new_test_rsquareds_list.append([])
                
        for mse, r2, element_id in zip(self.test_mserrors, self.test_rsquareds, self.test_element_ids):
                    
            for sub_class_index, sub_class_map in enumerate(new_sub_class_map_list):
                
                if element_id in sub_class_map:
                        
                    new_test_mserrors_list[sub_class_index].append(mse)
                    new_test_rsquareds_list[sub_class_index].append(r2)
                    
        for sub_class_index, sub_class in enumerate(new_sub_classes):
                
            new_test_mserrors = np.asarray(new_test_mserrors_list[sub_class_index])
            new_test_rsquareds = np.asarray(new_test_rsquareds_list[sub_class_index])
            
            self.sub_classes.append(sub_class)
            self.test_mserrors_per_sub_class.append(new_test_mserrors)
            self.test_rsquareds_per_sub_class.append(new_test_rsquareds)
                
            sub_classes.append(sub_class) 
            sub_class_mserrors.append(new_test_mserrors)       
            sub_class_rsquareds.append(new_test_rsquareds)                  
      
        self.test_mserrors_per_sub_class = np.asarray(self.test_mserrors_per_sub_class, dtype=object)        
        self.test_rsquareds_per_sub_class = np.asarray(self.test_rsquareds_per_sub_class, dtype=object)
                           
        np.save(self.output_folder + 'test_stats/test_mserrors_per_sub_class.npy', self.test_mserrors_per_sub_class)                           
        np.save(self.output_folder + 'test_stats/test_rsquareds_per_sub_class.npy', self.test_rsquareds_per_sub_class)
            
        with open(self.output_folder + 'test_stats/sub_classes.pkl', 'wb') as file:
    
            pickle.dump(self.sub_classes, file, pickle.HIGHEST_PROTOCOL)
            
        self.sub_class_num = len(self.sub_classes)
        
        return sub_class_mserrors, sub_class_rsquareds, sub_classes
    
    
    def plot_test_stats(self, plot_sub_classes = None, gamma = 0.1, loss_threshold = 0.1, time_threshold = 0.5, density = 249, compute_loss_per_sample = False):    
        
        mse_ticks = [ x / 200 for x in range(0, 11)]  
        
        plot_violin_distribution([self.test_mserrors], ['Test set'], 'Loss [$mV^2$]', self.output_folder + '/test_stats/test_mserrors', yticks = mse_ticks)
        plot_box_distribution([self.test_mserrors], ['Test set'], 'Loss [$mV^2$]', self.output_folder + '/test_stats/test_mserrors_box', yticks = mse_ticks)
        
        rsquared_ticks = [x / 10 for x in range(0, 11)]
        
        plot_violin_distribution([self.test_rsquareds], ['Test set'], 'Loss [%]', self.output_folder + '/test_stats/test_rsquareds', yticks = rsquared_ticks)
        plot_box_distribution([self.test_rsquareds], ['Test set'], 'Loss [%]', self.output_folder + '/test_stats/test_rsquareds_box', yticks = rsquared_ticks)
        
        loss_dict = {'class': [], 'mse': [], 'r2': []}
        loss_dict['class'].append('Total')
        loss_dict['mse'].append(np.mean(self.test_mserrors))
        loss_dict['r2'].append(np.mean(self.test_rsquareds) * 100)
        
        sub_class_mserrors, sub_class_rsquareds, plot_sub_classes = self.compute_sub_class_losses(plot_sub_classes)
        
        for mserrors, rsquareds, sub_class in zip(sub_class_mserrors, sub_class_rsquareds, plot_sub_classes):
            loss_dict['class'].append(sub_class)
            loss_dict['mse'].append(np.mean(mserrors))
            loss_dict['r2'].append(np.mean(rsquareds) * 100)
            
        loss_dataframe = pd.DataFrame.from_dict(loss_dict)
        loss_dataframe.to_csv (self.output_folder + '/test_stats/test_losses.csv', index = False, header=True)
            
        if len(plot_sub_classes) > 0:
        
            plot_violin_distribution(sub_class_mserrors, plot_sub_classes, 'Loss', self.output_folder + '/test_stats/test_mserrors_per_sub_class', yticks = mse_ticks)
            plot_box_distribution(sub_class_mserrors, plot_sub_classes, 'Loss', self.output_folder + '/test_stats/test_mserrors_per_sub_class_box', yticks = mse_ticks)
            
            plot_violin_distribution(sub_class_rsquareds, plot_sub_classes, 'Loss', self.output_folder + '/test_stats/test_rsquareds_per_sub_class', yticks = rsquared_ticks)
            plot_box_distribution(sub_class_rsquareds, plot_sub_classes, 'Loss', self.output_folder + '/test_stats/test_rsquareds_per_sub_class_box', yticks = rsquared_ticks)

        if compute_loss_per_sample:

            mserrors_per_sample_target_indexes = np.where(self.test_mserrors_per_sample < loss_threshold)

            distances_per_sample_target_indexes = np.where(np.abs(self.test_distances_per_sample) < time_threshold)

            target_indexes = np.intersect1d(mserrors_per_sample_target_indexes, distances_per_sample_target_indexes)

            test_mserrors_per_sample = self.test_mserrors_per_sample[target_indexes]
            test_distances_per_sample = self.test_distances_per_sample[target_indexes] 
            test_lead_per_sample = self.test_leads_per_sample[target_indexes]

            plot_2d_distribution(test_distances_per_sample, test_mserrors_per_sample,
                'Time [s]', 'Loss [V^2]', self.output_folder + 'test_stats/mserrors_vs_distances', xbins=density, ybins=density, gamma=gamma, xlim=[-.5, .5])

            plot_2d_distribution(test_lead_per_sample, test_mserrors_per_sample,
                'Lead', 'Loss [V^2]', self.output_folder + 'test_stats/mserrors_vs_leads', xticks=self.output_lead_keys, xbins=self.input_lead_num, ybins=density, gamma=gamma)

            for lead_index, lead_key in enumerate(self.output_lead_keys):

                target_indexes = np.where(test_lead_per_sample==lead_index)

                plot_2d_distribution(test_distances_per_sample[target_indexes], test_mserrors_per_sample[target_indexes],
                    'Time [s]', 'Loss [V^2]', self.output_folder + 'test_stats/mserrors_vs_distances_' + lead_key, xbins=density, ybins=density, gamma=gamma, xlim=[-.5, .5])


    def plot_random_example(self, size=1, batch_ids=None, plot_format='png'):
    
        remove_dir(self.output_folder + 'random_plot/')
        
        if batch_ids is None:
            
            batch_ids = self.data_loader.get_random_data_ids('test', size)
        
        twelve_keys = get_twelve_keys()

        for element_id in batch_ids:
            
            twelve_leads = self.data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * self.amplitude + self.min_value for lead in twelve_leads]
                
            input_leads, output_leads, _, _, _ = self.data_loader.load_element(element_id, False)
            
            model_input, _ = process_element(input_leads,
                                               output_leads,                                               
                                               self.min_value,
                                               self.amplitude,
                                               self.device)

            model_output = self.model.forward(model_input)

            model_leads = deprocess_element(model_output)
                        
            pathlib.Path(self.output_folder + 'random_plot/standard/').mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.output_folder + 'random_plot/medical/').mkdir(parents=True, exist_ok=True)
            
            plot_medical_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.input_lead_keys, self.output_folder + 'random_plot/medical/' + str(element_id), plot_format=plot_format)     
            plot_output_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.output_folder + 'random_plot/standard/' + str(element_id), plot_format=plot_format)

    def plot_error_example(self, plot_format='png'):

        remove_dir(self.output_folder + 'error_plot/')

        self.load_test_stats()

        percentiles = [0, 25, 50, 75, 100]

        thresholds = [np.percentile(self.test_mserrors, percent) for percent in percentiles]

        threshold_num = len(thresholds)

        element_per_threshold = np.zeros(threshold_num - 1)
        
        twelve_keys = get_twelve_keys()

        while np.sum(element_per_threshold) < threshold_num - 1:

            input_leads, output_leads, _, _, element_id = self.data_loader.get_next_element('test', False)
            
            twelve_leads = self.data_loader.load_element_twelve_leads(element_id)
            twelve_leads = [lead * self.amplitude + self.min_value for lead in twelve_leads]

            model_input, model_target = process_element(input_leads,
                                                            output_leads,                                                            
                                                            self.min_value,
                                                            self.amplitude,
                                                            self.device)

            with torch.no_grad():

                model_output = self.model.forward(model_input)

                loss, _, _, _ = element_mse_function(model_output,
                                                      model_target,
                                                      self.output_lead_num)

            for i in range(threshold_num - 1):

                if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] == 0:

                    element_per_threshold[i] += 1
                        
                    input_leads, output_leads, _, _, _ = self.data_loader.load_element(element_id, False)
        
                    model_input, _ = process_element(input_leads,
                                                        output_leads,
                                                        self.min_value,
                                                        self.amplitude,
                                                        self.device)

                    model_output = self.model.forward(model_input)

                    model_leads = deprocess_element(model_output)
                
                    pathlib.Path(self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                    pathlib.Path(self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        
                    plot_medical_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.input_lead_keys, self.output_folder + 'error_plot/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                    plot_output_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.output_folder + 'error_plot/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                    
                    break
                
    def plot_sub_class_example(self, plot_sub_classes, plot_format='png'):
    
        remove_dir(self.output_folder + 'sub_class_plot/')
        
        percentiles = [0, 25, 50, 75, 100]
        
        twelve_keys = get_twelve_keys()
            
        threshold_num = len(percentiles)
        
        for sub_class in plot_sub_classes:
            
            sub_class_idx = self.sub_classes.index(sub_class)
            
            thresholds = [np.percentile(self.test_mserrors_per_sub_class[sub_class_idx], percent) for percent in percentiles]

            element_per_threshold = np.zeros(threshold_num - 1)
            
            sub_class_map = set(load_dataclass_ids(self.parent_folder, sub_class))
            test_map = set(self.data_loader.test_data_ids)
            
            candidate_list = list(sub_class_map & test_map)

            while np.sum(element_per_threshold) < threshold_num - 1:

                element_id = candidate_list.pop(0)
                input_leads, output_leads, _, _, _ = self.data_loader.load_element(element_id, False)
                
                twelve_leads = self.data_loader.load_element_twelve_leads(element_id)
                twelve_leads = [lead * self.amplitude + self.min_value for lead in twelve_leads]

                model_input, model_target = process_element(input_leads,
                                                                output_leads,
                                                                self.min_value,
                                                                self.amplitude,
                                                                self.device)

                with torch.no_grad():

                    model_output = self.model.forward(model_input)

                    loss, _, _, _ = element_mse_function(model_output,
                                                          model_target,
                                                          self.output_lead_num)

                for i in range(threshold_num - 1):
    
                    if thresholds[i] < loss < thresholds[i+1] and element_per_threshold[i] == 0:
                        
                        element_per_threshold[i] = 1
                            
                        input_leads, output_leads, _, _, _ = self.data_loader.load_element(element_id, False)
            
                        model_input, _ = process_element(input_leads,
                                                        output_leads,
                                                        self.min_value,
                                                        self.amplitude,
                                                        self.device)

                        model_output = self.model.forward(model_input)

                        model_leads = deprocess_element(model_output)
                        
                        model_leads = [lead for lead in model_leads]
                        output_leads = [lead * self.amplitude + self.min_value for lead in output_leads]
                    
                        pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                        pathlib.Path(self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/').mkdir(parents=True, exist_ok=True)
                            
                        plot_medical_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.input_lead_keys, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/medical/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        plot_output_recon_leads(twelve_leads, twelve_keys, model_leads, self.output_lead_keys, self.output_folder + 'sub_class_plot/sub_class=' + sub_class + '/standard/percent=' + str(percentiles[i+1]) + '/' + str(element_id), plot_format=plot_format)
                        
                        break