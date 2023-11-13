from learn_functions.convolutional_block import ConvolutionalBlock
import torch

class ConvolutionalNetwork(object):
    """
    class ConvolutionalNetwork

    """

    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 block_num: int,
                 kernel_size: int,
                 stride_size: int,
                 inner_activation: str,
                 output_activation: str,
                 use_residual_block: bool,
                 average_pool: bool,
                 device
                 ):

        input = input_channel

        self.blocks = []

        for _ in range(block_num - 1):

            output = int((input + output_channel) / 2)

            block = ConvolutionalBlock(input,
                                       output,
                                       kernel_size,
                                       stride_size,
                                       inner_activation,
                                       inner_activation,
                                       use_residual_block,
                                       1,
                                       device)

            self.blocks.append(block)
            
            input = output

        block = ConvolutionalBlock(input,
                                   output_channel,
                                   kernel_size,
                                   stride_size,
                                   inner_activation,
                                   output_activation,
                                   use_residual_block,
                                   average_pool,
                                   device)
        
        self.blocks.append(block)

    def reset(self): 

        for block in self.blocks:

            block.reset()

    def parameters(self):

        parameters = []

        for block in self.blocks:

            parameters += list(block.parameters())

        return parameters

    def named_parameters(self):

        named_parameters = {}

        for i, block in enumerate(self.blocks):

            named_parameters[i] = block.named_parameters()

        return named_parameters

    def save_state_dict(self, path: str):

        for idx, block in enumerate(self.blocks):

            block.save_state_dict(path + str(idx) + '/')
    
    def load_state_dict(self, path: str):

        for idx, block in enumerate(self.blocks):

            block.load_state_dict(path + str(idx) + '/')

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :type x:
        :return:
        :rtype:
        """

        for block in self.blocks:

            x = block.forward(x)

        return x


class ParallelConvolutionalNetwork(object):

    """
    class ParallelNetwork

    """

    def __init__(self,
                 network_num: int,
                 input_channel: int,
                 output_channel: int,
                 block_num: int,
                 kernel_size: int,
                 inner_activation: str,
                 output_activation: str,
                 use_residual_block: bool,
                 device
                 ):

        self.networks = []
        
        stride_size = 1
        average_pool = 1

        for _ in range(network_num):

            self.networks.append(ConvolutionalNetwork(input_channel,
                                                 output_channel,
                                                 block_num,
                                                 kernel_size,
                                                 stride_size,
                                                 inner_activation,
                                                 output_activation,
                                                 use_residual_block,
                                                 average_pool,
                                                 device))

    def reset(self):

        for net in self.networks:

            net.reset()

    def parameters(self):

        parameters = []

        for net in self.networks:

            parameters += list(net.parameters())

        return parameters

    def named_parameters(self):

        named_parameters = {}

        for i, net in enumerate(self.networks):

            named_parameters[i] = net.named_parameters()

        return named_parameters

    def save_state_dict(self, path: str):

        for idx, net in enumerate(self.networks):

            net.save_state_dict(path + str(idx) + '/')
    
    def load_state_dict(self, path: str):

        for idx, net in enumerate(self.networks):

            net.load_state_dict(path + str(idx) + '/')
    
    def forward(self, x_vector):

        output_vector = []

        for i, x in enumerate(x_vector):

            output_vector.append(self.networks[i].forward(x))
 
        return output_vector


class SymmetricConvolutionalNetwork(object):

    """
    class SymmetricConvolutionalNetwork

    """

    def __init__(self,
                 input_channel: int,
                 middle_channel: int,
                 output_channel: int,
                 block_num: int,
                 kernel_size: int,
                 inner_activation: str,
                 output_activation: str,
                 use_residual_block: bool,
                 device
                 ):
        
        assert block_num % 2 == 0
        
        block_num = int(block_num / 2)
        
        stride_size = 1
        average_pool = 1

        self.input_network = ConvolutionalNetwork(input_channel,
                                             middle_channel,
                                             block_num,
                                             kernel_size,
                                             stride_size,
                                             inner_activation,
                                             output_activation,
                                             use_residual_block,
                                             average_pool,
                                             device)

        self.output_network = ConvolutionalNetwork(middle_channel,
                                              output_channel,
                                              block_num,
                                              kernel_size,
                                              stride_size,
                                              inner_activation,
                                              output_activation,
                                              use_residual_block,
                                              average_pool,
                                              device)

    def reset(self):

        self.input_network.reset()
        self.output_network.reset()

    def parameters(self):

        parameters = list(self.input_network.parameters()) + list(self.output_network.parameters())

        return parameters

    def named_parameters(self):

        named_parameters = {}

        for i, net in enumerate([self.input_network, self.output_network]):

            named_parameters[i] = net.named_parameters()

        return named_parameters
    
    def save_state_dict(self, path: str):

        self.input_network.save_state_dict(path + 'input/')
        self.output_network.save_state_dict(path + 'output/')

    def load_state_dict(self, path: str):

        self.input_network.load_state_dict(path + 'input/')
        self.output_network.load_state_dict(path + 'output/')
    
    def forward(self, x):

        x = self.input_network.forward(x)
        x = self.output_network.forward(x)
 
        return x
