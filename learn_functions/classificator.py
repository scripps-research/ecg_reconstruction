from learn_functions.convolutional_network import ParallelConvolutionalNetwork, ConvolutionalNetwork
from learn_functions.linear_network import ParallelLinearNetwork
import torch
import numpy as np


class ParallelClassificator(object):
    
    def __init__(self,
                 input_lead_num: int,
                 detect_class_num: int,
                 input_channel_per_lead: int, 
                 middle_channel_per_class: int, 
                 block_per_input_network: int,
                 block_per_middle_network: int,
                 block_per_output_network: int,  
                 input_kernel_size: int,
                 middle_kernel_size: int,
                 stride_size: int,
                 average_pool: int,
                 activation_function: str,
                 use_residual_block: bool,
                 device
                 ):
        
        self.classificators = []
        
        for _ in range(detect_class_num):
            
            self.classificators.append(Classificator(input_lead_num,
                                                     1,
                                                     input_channel_per_lead,
                                                     middle_channel_per_class,
                                                     block_per_input_network,
                                                     block_per_middle_network,
                                                     block_per_output_network,
                                                     input_kernel_size,
                                                     middle_kernel_size, 
                                                     stride_size,
                                                     average_pool,
                                                     activation_function, 
                                                     use_residual_block,
                                                     device))
            
    def reset(self):
    
        for classificator in self.classificators:
            classificator.reset()

    def parameters(self):

        return [classificator.parameters() for classificator in self.classificators]

    def named_parameters(self):

        return [classificator.named_parameters() for classificator in self.classificators]

    def compute_model_stats(self):
        
        biases_vs_depths = []
        weights_vs_depths = []
        biases_vs_leads = []
        weights_vs_leads = []
        
        for classificator in self.classificators:
            
            x, y, z, w = classificator.compute_model_stats()
            
            biases_vs_depths.append(x)
            weights_vs_depths.append(y)
            biases_vs_leads.append(z)
            weights_vs_leads.append(w)
        
        return biases_vs_depths,  weights_vs_depths, biases_vs_leads, weights_vs_leads

    def save_state_dict(self, path_list: str):
        
        for classificator, path in zip(self.classificators, path_list):
            
            classificator.save_state_dict(path)
            
    def load_state_dict(self, path_list: str):
        
        for classificator, path in zip(self.classificators, path_list):
            
            classificator.load_state_dict(path)
    
    def forward(self, input):
        
        output = []
        
        for classificator in self.classificators:
            
            output += classificator.forward([torch.clone(x) for x in input])
        
        return output


class Classificator(object):

    """
    class Reconstructor

    """

    def __init__(self,
                 input_lead_num: int,
                 detect_class_num: int,
                 input_channel_per_lead: int, 
                 middle_channel_per_class: int, 
                 block_per_input_network: int,
                 block_per_middle_network: int,
                 block_per_output_network: int,  
                 input_kernel_size: int,
                 middle_kernel_size: int,
                 stride_size: int,
                 average_pool: int,
                 activation_function: str,
                 use_residual_block: bool,
                 device
                 ):

        self.input_network = ParallelConvolutionalNetwork(network_num=input_lead_num,
                                                          input_channel=1,
                                                          output_channel=input_channel_per_lead,
                                                          block_num=block_per_input_network,
                                                          kernel_size=input_kernel_size,
                                                          inner_activation=activation_function,
                                                          output_activation=activation_function,
                                                          use_residual_block=use_residual_block,
                                                          device=device)

        self.middle_network = ConvolutionalNetwork(input_channel=input_channel_per_lead * input_lead_num,
                                                   output_channel=middle_channel_per_class * detect_class_num,
                                                   block_num=block_per_middle_network,
                                                   kernel_size=middle_kernel_size,
                                                   stride_size=stride_size,
                                                   inner_activation=activation_function,
                                                   output_activation=activation_function,
                                                   use_residual_block=use_residual_block,
                                                   average_pool=average_pool,
                                                   device=device)

        self.output_network = ParallelLinearNetwork(network_num=detect_class_num,
                                                    input_size=middle_channel_per_class * detect_class_num,
                                                    output_size=1,
                                                    block_num=block_per_output_network,
                                                    inner_activation=activation_function,
                                                    output_activation='sigmoid',
                                                    device=device)

    def reset(self):

        self.input_network.reset()
        self.middle_network.reset()
        self.output_network.reset()

    def parameters(self):

        parameters = self.input_network.parameters() + self.middle_network.parameters() + self.output_network.parameters()

        return parameters

    def named_parameters(self):

        named_parameters = {}

        named_parameters = {}

        for i, net in enumerate([self.input_network, self.middle_network, self.output_network]):

            named_parameters[i] = net.named_parameters()

        return named_parameters


    def compute_model_stats(self):

        biases_vs_depths = [[], []]
        weights_vs_depths = [[], []]
        biases_vs_leads = [[], []]
        weights_vs_leads = [[], []]

        named_parameters = self.named_parameters()

        lead_idx = -1

        for i in named_parameters[0].keys():

            lead_idx += 1

            depth_idx = -1

            for j in named_parameters[0][i].keys():                

                for conv in ['in', 'out']:

                    depth_idx += 1
                    
                    if len(named_parameters[0][i][j][conv]['bias']) > 0:

                        block_biases = np.concatenate([np.reshape(param, -1) for param in named_parameters[0][i][j][conv]['bias']])
                        block_weights = np.concatenate([np.reshape(param, -1) for param in named_parameters[0][i][j][conv]['weight']])

                        biases_vs_depths[0].extend(block_biases)
                        weights_vs_depths[0].extend(block_weights)
                        biases_vs_depths[1].extend([depth_idx] * len(block_biases))
                        weights_vs_depths[1].extend([depth_idx] * len(block_weights))

                        biases_vs_leads[0].extend(block_biases)
                        weights_vs_leads[0].extend(block_weights)
                        biases_vs_leads[1].extend([lead_idx] * len(block_biases))
                        weights_vs_leads[1].extend([lead_idx] * len(block_weights))

        for i in named_parameters[1].keys():            

            for j in named_parameters[1][i].keys():

                depth_idx += 1
                
                if len(named_parameters[1][i][j]['bias']) > 0:

                    block_biases = np.concatenate([np.reshape(param, -1) for param in named_parameters[1][i][j]['bias']])
                    block_weights = np.concatenate([np.reshape(param, -1) for param in named_parameters[1][i][j]['weight']])

                    biases_vs_depths[0].extend(block_biases)
                    weights_vs_depths[0].extend(block_weights)
                    biases_vs_depths[1].extend([depth_idx] * len(block_biases))
                    weights_vs_depths[1].extend([depth_idx] * len(block_weights))

        last_depth_idx = depth_idx

        for i in named_parameters[2].keys():    

            depth_idx = last_depth_idx

            for j in named_parameters[2][i].keys():
                
                for w in named_parameters[2][i][j].keys():

                    depth_idx += 1
                    
                    if len(named_parameters[2][i][j][w]['bias']) > 0:

                        block_biases = np.concatenate([np.reshape(param, -1) for param in named_parameters[2][i][j][w]['bias']])
                        block_weights = np.concatenate([np.reshape(param, -1) for param in named_parameters[2][i][j][w]['weight']])

                        biases_vs_depths[0].extend(block_biases)
                        weights_vs_depths[0].extend(block_weights)
                        biases_vs_depths[1].extend([depth_idx] * len(block_biases))
                        weights_vs_depths[1].extend([depth_idx] * len(block_weights))

        return biases_vs_depths, weights_vs_depths, biases_vs_leads, weights_vs_leads

    def save_state_dict(self, path: str):

        self.input_network.save_state_dict(path + 'input/')
        self.middle_network.save_state_dict(path + 'middle/')
        self.output_network.save_state_dict(path + 'output/')

    def load_state_dict(self, path: str):

        self.input_network.load_state_dict(path + 'input/')
        self.middle_network.load_state_dict(path + 'middle/')
        self.output_network.load_state_dict(path + 'output/')
    
    def forward(self, x):

        x = self.input_network.forward(x)

        x = torch.cat(x, dim=1)

        x = self.middle_network.forward(x)
        
        x = x.view(x.size(0), -1)

        x = self.output_network.forward(x)
        
        return x