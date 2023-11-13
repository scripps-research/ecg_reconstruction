from plot_functions.time_series import plot_multi_time_series
import pathlib


class MultiTrainingManager(object):
    
    def __init__(self,
                 use_residual,
                 epochs,
                 batch_size,
                 prioritize_percent,
                 prioritize_size,
                 optimizer_algorithm,
                 learning_rate,
                 weight_decay,
                 momentum,
                 nesterov,
                 manager_labels = None):

        self.multi_manager = []
        self.manager_labels = manager_labels
        self.manager_num = len(manager_labels)
        
        self.output_folder = None
            
        if isinstance(use_residual, list):
            use_residual_list = use_residual
        else:
            use_residual_list = [use_residual] * self.manager_num

        if isinstance(epochs, list):
            epochs_list = epochs
        else:
            epochs_list = [epochs] * self.manager_num

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
                
        self.configurations = zip(use_residual_list,
                                  epochs_list,
                                  batch_size_list,
                                  prioritize_percent_list,
                                  prioritize_size_list,
                                  optimizer_list,
                                  learning_rate_list,
                                  decay_list,
                                  momentum_list,
                                  nesterov_list)
        
        if self.manager_labels is None:

            configuration_sizes = [len(x) for x in [use_residual_list,
                                                        epochs_list,
                                                        batch_size_list,
                                                        prioritize_percent_list,
                                                        prioritize_size_list,
                                                        optimizer_list,
                                                        learning_rate_list,
                                                        decay_list,
                                                        momentum_list,
                                                        nesterov_list]]
            
            self.configuration_labels = []
            
            for configuration in self.configurations:
                
                use_residual, epochs, batch_size, prioritize_percent, prioritize_size, optimizer_algorithm, learning_rate, weight_decay, momentum, nesterov = configuration

                configuration_label = ''
                
                if configuration_sizes[0] > 1:
                    configuration_label += 'use_residual=' + str(use_residual) + '_'
                if configuration_sizes[1] > 1:
                    configuration_label += 'epochs=' + str(epochs) + '_'
                if configuration_sizes[2] > 1:
                    configuration_label += 'batch=' + str(batch_size) + '_'
                if configuration_sizes[3] > 1:
                    configuration_label += 'prioritize_percent=' + str(prioritize_percent) + '_'
                if configuration_sizes[4] > 1:
                    configuration_label += 'prioritize_size=' + str(prioritize_size) + '_'
                if configuration_sizes[5] > 1:
                    configuration_label += 'optimizer=' + optimizer_algorithm + '_'
                if configuration_sizes[6] > 1:
                    configuration_label += 'lr=' + str(learning_rate) + '_'
                if configuration_sizes[7] > 1:
                    configuration_label += 'decay=' + str(weight_decay) + '_'
                if configuration_sizes[8] > 1:
                    configuration_label += 'momentum=' + str(momentum) + '_'
                if configuration_sizes[9] > 1:
                    configuration_label += 'nest=' + str(nesterov) + '_'
                    
                self.configuration_labels.append(configuration_label)
        
        else:
            
            self.configuration_labels = [None] * self.manager_num

    def load_model(self):

        for manager in self.multi_manager:

            manager.load_model()

    def train(self):

        for manager, manager_label in zip(self.multi_manager, self.manager_labels):
            
            print('Training system', manager_label)
            
            manager.reset_model()

            manager.load_dataset(train=True, valid=True)

            manager.train()

            manager.release_dataset()
            
            manager.plot_train_stats()
            
            manager.plot_valid_stats()

            manager.load_dataset(test=True)
            
            print()
            
            print('Testing system', manager_label)

            manager.test()

            manager.release_dataset()
            
            manager.plot_test_stats()
            
            manager.compute_model_stats()
            
            manager.plot_model_stats()
            
            print()
            
    def test(self):
    
        for manager, manager_label in zip(self.multi_manager, self.manager_labels):
            
            print('Testing system', manager_label)
            
            manager.load_model()

            manager.load_dataset(test=True)

            manager.test()

            manager.release_dataset()
            
            manager.plot_test_stats()
            
            print()
            
    def eval(self):
        
        for manager in self.multi_manager:
        
            manager.load_model()
            
    def load_train_stats(self):
    
        for manager in self.multi_manager:
    
            manager.load_train_stats()
            
    def load_valid_stats(self):

        for manager in self.multi_manager:

            manager.load_valid_stats()
            
    def load_test_stats(self):

        for manager in self.multi_manager:            

            manager.load_test_stats()

    def plot_train_stats(self, loss_ticks=None, point_num=100, plot_format='png'):

        pathlib.Path(self.output_folder + 'train_stats/').mkdir(parents=True, exist_ok=True)

        multi_losses = [manager.train_losses for manager in self.multi_manager]
        multi_epochs = [int(manager.current_epoch) for manager in self.multi_manager]

        plot_multi_time_series(multi_losses, self.manager_labels, multi_epochs, loss_ticks, point_num, 'Epoch', 'Loss', self.output_folder + 'train_stats/train_losses', plot_format=plot_format)
        
    def plot_valid_stats(self, loss_ticks=None, point_num=100, plot_format='png'):

        pathlib.Path(self.output_folder + 'valid_stats/').mkdir(parents=True, exist_ok=True)

        multi_losses = [manager.valid_losses for manager in self.multi_manager]
        multi_epochs = [int(manager.current_epoch) for manager in self.multi_manager]

        plot_multi_time_series(multi_losses, self.manager_labels, multi_epochs, loss_ticks, point_num, 'Epoch', 'Loss', self.output_folder + 'valid_stats/valid_losses', plot_format=plot_format)
    