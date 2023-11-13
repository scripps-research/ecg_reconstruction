from learn_functions.linear_block import LinearBlock
import torch

class LinearNetwork(object):
    """
    class LinearNetwork

    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 block_num: int,
                 inner_activation: str,
                 output_activation: str,
                 device
                 ):

        input = input_size

        self.blocks = []

        for _ in range(block_num - 1):

            output = int((input + output_size) / 2)

            block = LinearBlock(input,
                                output,
                                inner_activation,
                                device)

            self.blocks.append(block)
            
            input = output

        block = LinearBlock(input,
                            output_size,
                            output_activation,
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
    
    
class ParallelLinearNetwork(object):
    
    """
    class ParallelNetwork

    """

    def __init__(self,
                 network_num: int,
                 input_size: int,
                 output_size: int,
                 block_num: int,
                 inner_activation: str,
                 output_activation: str,
                 device
                 ):

        self.networks = []

        for _ in range(network_num):

            self.networks.append(LinearNetwork(input_size,
                                               output_size,
                                               block_num,
                                               inner_activation,
                                               output_activation,
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
    
    def forward(self, x):

        output_vector = []

        for network in self.networks:

            output_vector.append(network.forward(x))
 
        return output_vector