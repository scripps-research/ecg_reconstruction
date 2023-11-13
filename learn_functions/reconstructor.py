from learn_functions.convolutional_network import ParallelConvolutionalNetwork, SymmetricConvolutionalNetwork
import torch
import numpy as np


class Reconstructor(object):

    """
    class Reconstructor

    """

    def __init__(self,
                 input_lead_num: int,
                 output_lead_num: int,
                 input_channel_per_lead: int,
                 middle_channel_per_lead: int,
                 output_channel_per_lead: int,
                 block_per_input_network: int,
                 block_per_middle_network: int,
                 block_per_output_network: int,
                 input_kernel_size: int,
                 middle_kernel_size: int,
                 output_kernel_size: int,
                 activation_function: str,
                 use_residual_block: bool,
                 device
                 ):

        self.output_lead_num = output_lead_num
        self.output_channel_per_lead = output_channel_per_lead
        
        self.input_network = ParallelConvolutionalNetwork(network_num=input_lead_num,
                                         input_channel=1,
                                         output_channel=input_channel_per_lead,
                                         block_num=block_per_input_network,
                                         kernel_size=input_kernel_size,
                                         inner_activation=activation_function,
                                         output_activation=activation_function,
                                         use_residual_block=use_residual_block,
                                         device=device)

        self.middle_network = SymmetricConvolutionalNetwork(input_channel=input_channel_per_lead*input_lead_num,
                                           middle_channel=middle_channel_per_lead * int((input_lead_num + output_lead_num) / 2),
                                           output_channel=output_channel_per_lead*output_lead_num,
                                           block_num=block_per_middle_network,
                                           kernel_size=middle_kernel_size,
                                           inner_activation=activation_function,
                                           output_activation=activation_function,
                                           use_residual_block=use_residual_block,
                                           device=device)

        self.output_network = ParallelConvolutionalNetwork(network_num=output_lead_num,
                                          input_channel=output_channel_per_lead * output_lead_num,
                                          output_channel=1,
                                          block_num=block_per_output_network,
                                          kernel_size=output_kernel_size,
                                          inner_activation=activation_function,
                                          output_activation=None,
                                          use_residual_block=use_residual_block,
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

                for conv in ['in', 'out']:

                    depth_idx += 1

                    block_biases = np.concatenate([np.reshape(param, -1) for param in named_parameters[1][i][j][conv]['bias']])
                    block_weights = np.concatenate([np.reshape(param, -1) for param in named_parameters[1][i][j][conv]['weight']])

                    biases_vs_depths[0].extend(block_biases)
                    weights_vs_depths[0].extend(block_weights)
                    biases_vs_depths[1].extend([depth_idx] * len(block_biases))
                    weights_vs_depths[1].extend([depth_idx] * len(block_weights))

        last_depth_idx = depth_idx

        for i in named_parameters[2].keys():    

            depth_idx = last_depth_idx

            for j in named_parameters[2][i].keys():

                for conv in ['in', 'out']:

                    depth_idx += 1

                    block_biases = np.concatenate([np.reshape(param, -1) for param in named_parameters[2][i][j][conv]['bias']])
                    block_weights = np.concatenate([np.reshape(param, -1) for param in named_parameters[2][i][j][conv]['weight']])

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
    
    def forward(self, input_leads):

        leads = self.input_network.forward(input_leads)

        leads = torch.cat(leads, dim=1)

        leads = self.middle_network.forward(leads)

        leads = [torch.clone(leads) for _ in range(self.output_lead_num)]

        output_leads = self.output_network.forward(leads)
 
        return [torch.squeeze(lead) for lead in output_leads]