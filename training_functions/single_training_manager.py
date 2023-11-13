from util_functions.general import get_lead_keys, get_value_range
from plot_functions.time_series import plot_time_series
from plot_functions.bivariate_distribution import plot_2d_distribution
import numpy as np
import torch
import pathlib


class TrainingManager(object):
    
    def __init__(self,
                 parent_folder: str,
                 device: str,
                 sub_classes: list,
                 input_leads: str,
                 epochs: int,
                 batch_size: int,
                 prioritize_percent: float,
                 ):
        
        self.parent_folder = parent_folder
        self.device = torch.device(device)

        self.input_lead_keys = get_lead_keys(input_leads)
        self.min_value, self.amplitude, self.sample_num = get_value_range()
        self.input_lead_num = len(self.input_lead_keys)
        
        self.epochs = epochs
        self.batch_size = batch_size          
        self.compute_loss_per_element = prioritize_percent > 0
        
        self.current_epoch = 0
        self.validation_score = np.infty
        self.max_validation_timer = 10
        self.validation_timer = self.max_validation_timer

        self.train_losses = []
        self.valid_losses = []      
        self.test_element_ids = []

        self.biases_vs_depths = None
        self.weights_vs_depths = None
        self.biases_vs_leads = None
        self.weights_vs_leads = None
        
        self.sub_classes = sub_classes            
        self.sub_class_num = len(self.sub_classes)
        
        self.model = None        
        self.data_loader = None        
        self.output_folder = None
        
        
    def get_output_folder(self, 
                          use_residual,
                          epochs,
                          batch_size,
                          prioritize_percent,
                          prioritize_size,
                          optimizer_algorithm,
                          learning_rate,
                          weight_decay,
                          momentum,
                          nesterov):
        
        output_folder = 'residual=' + use_residual + '/epochs=' + str(epochs) + '/batch_size=' + str(batch_size) + '/prioritize_percent=' + str(prioritize_percent) + '/prioritize_size=' + str(prioritize_size) +\
            '/optimizer=' + optimizer_algorithm + '/lr=' + str(learning_rate) + '/'
            
        if optimizer_algorithm == 'adam':
            output_folder += 'decay=' + str(weight_decay) + '/'

        elif optimizer_algorithm == 'sgd':
            output_folder += 'decay=' + str(weight_decay) + 'momentum=' + str(momentum) + 'nest=' + str(nesterov) + '/'

        else:
            raise ValueError
        
        return output_folder
    

    def release_dataset(self):

        self.data_loader.release_dataset()


    def reset_model(self):

        self.model.reset()


    def load_model(self):

        self.model.load_state_dict(self.output_folder + 'state_dict/')

    def save_model(self):

        self.model.save_state_dict(self.output_folder + 'state_dict/')
        

    def train_step(self, batch):
        
        return None
    
    def valid_step(self, batch):
        
        return None
        

    def train(self):

        while self.current_epoch < self.epochs:

            print("Training epoch... ", self.current_epoch)

            self.data_loader.shuffle(subset='train')

            batch = self.data_loader.get_next_batch('train', False)

            while batch is not None:

                batch = self.train_step(batch)

            self.validate()

            self.current_epoch = self.current_epoch + 1

            self.save_train_stats()

            self.plot_train_stats()

            self.save_valid_stats()

            self.plot_valid_stats()
            
            if self.validation_timer == 0:
                break


    def save_train_stats(self):

        pathlib.Path(self.output_folder + 'train_stats/').mkdir(parents=True, exist_ok=True)
        np.save(self.output_folder + 'train_stats/train_losses.npy', self.train_losses)
        np.save(self.output_folder + 'train_stats/current_epoch.npy', self.current_epoch)


    def load_train_stats(self):

        self.train_losses = list(np.load(self.output_folder + 'train_stats/train_losses.npy'))
        self.current_epoch = np.load(self.output_folder + 'train_stats/current_epoch.npy')

    
    def plot_train_stats(self, yticks=None, point_num=100):

        plot_time_series(np.asarray(self.train_losses), self.current_epoch, yticks, point_num, 'Epoch', 'Loss', self.output_folder + '/train_stats/train_losses')


    def validate(self):

        self.data_loader.shuffle(subset='valid')

        batch = self.data_loader.get_next_batch('valid', False)
        
        new_validation_score = 0

        while batch is not None:

            batch = self.valid_step(batch)
            
            new_validation_score += self.valid_losses[-1]
            
        if self.validation_score > new_validation_score:
            
            self.validation_score = new_validation_score
            
            self.save_model()
            
            self.validation_timer = self.max_validation_timer
            
        else:
            
            self.validation_timer -= 1            

    
    def save_valid_stats(self):

        pathlib.Path(self.output_folder + 'valid_stats/').mkdir(parents=True, exist_ok=True)
        np.save(self.output_folder + 'valid_stats/valid_losses.npy', self.valid_losses)

    
    def load_valid_stats(self):

        self.valid_losses = list(np.load(self.output_folder + 'valid_stats/valid_losses.npy'))
        self.current_epoch = np.load(self.output_folder + 'train_stats/current_epoch.npy')


    def plot_valid_stats(self, yticks=None, point_num=100):

        plot_time_series(np.asarray(self.valid_losses), self.current_epoch, yticks, point_num, 'Epoch', 'Loss', self.output_folder + '/valid_stats/valid_losses') 


    def compute_model_stats(self):

        self.biases_vs_depths, self.weights_vs_depths, self.biases_vs_leads, self.weights_vs_leads = self.model.compute_model_stats()
        self.save_model_stats()


    def save_model_stats(self):

        self.biases_vs_depths = np.asarray(self.biases_vs_depths)
        self.weights_vs_depths = np.asarray(self.weights_vs_depths)
        self.biases_vs_leads = np.asarray(self.biases_vs_leads)
        self.weights_vs_leads = np.asarray(self.weights_vs_leads)

        pathlib.Path(self.output_folder + '/model_stats/').mkdir(parents=True, exist_ok=True)

        np.save(self.output_folder + 'model_stats/biases_vs_depths.npy', self.biases_vs_depths)
        np.save(self.output_folder + 'model_stats/weights_vs_depths.npy', self.weights_vs_depths)
        np.save(self.output_folder + 'model_stats/biases_vs_leads.npy', self.biases_vs_leads)
        np.save(self.output_folder + 'model_stats/weights_vs_leads.npy', self.weights_vs_leads)


    def load_model_stats(self):

        self.biases_vs_depths = np.load(self.output_folder + 'model_stats/biases_vs_depths.npy')
        self.weights_vs_depths = np.load(self.output_folder + 'model_stats/weights_vs_depths.npy')
        self.biases_vs_leads = np.load(self.output_folder + 'model_stats/biases_vs_leads.npy')
        self.weights_vs_leads = np.load(self.output_folder + 'model_stats/weights_vs_leads.npy')


    def plot_model_stats(self, gamma=0.15, density=100):

        pathlib.Path(self.output_folder + '/model_stats/').mkdir(parents=True, exist_ok=True)

        xticks = np.arange(1, np.max(self.weights_vs_depths[1])+2, dtype=np.int).tolist()

        plot_2d_distribution(self.weights_vs_depths[1], self.weights_vs_depths[0],
            'Depth', 'Value', self.output_folder + '/model_stats/weights_vs_depths', xticks=xticks, xbins=len(xticks), ybins=density, gamma=gamma)

        plot_2d_distribution(self.weights_vs_leads[1], self.weights_vs_leads[0],
            'Lead', 'Value', self.output_folder + '/model_stats/weights_vs_leads', xticks=self.input_lead_keys, xbins=self.input_lead_num, ybins=density, gamma=gamma)

        xticks = np.arange(1, np.max(self.biases_vs_depths[1])+2, dtype=np.int).tolist()

        plot_2d_distribution(self.biases_vs_depths[1], self.biases_vs_depths[0],
            'Depth', 'Value', self.output_folder + '/model_stats/biases_vs_depths', xticks=xticks, xbins=len(xticks), ybins=density, gamma=gamma)

        plot_2d_distribution(self.biases_vs_leads[1], self.biases_vs_leads[0],
            'Lead', 'Value', self.output_folder + '/model_stats/biases_vs_leads', xticks=self.input_lead_keys, xbins=self.input_lead_num, ybins=density, gamma=gamma)
