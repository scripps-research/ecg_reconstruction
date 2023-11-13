import numpy as np
import torch
import torch.nn as nn
import pathlib

class LinearBlock(nn.Module):
    """
    class ResidualBlock

    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 output_activation: str,
                 device
                 ):

        super(LinearBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.output_activation = output_activation

        self.device = device

        if self.output_activation == None:
            self.output_activation_function = None
        elif self.output_activation == 'relu':
            self.output_activation_function = torch.relu
        elif self.output_activation == 'leaky_relu':
            self.output_activation_function = torch.nn.functional.leaky_relu
        elif self.output_activation == 'sigmoid':
                self.output_activation_function = torch.sigmoid
        else:
            raise ValueError

        self.linear = nn.Linear(in_features=self.input_size,
                                       out_features=self.output_size).to(self.device)

    def reset(self):

        if self.output_activation is None:
            torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        else:
            torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity=self.output_activation)

        torch.nn.init.zeros_(self.linear.bias)
        torch.nn.init.zeros_(self.linear.bias)

    def parameters(self):
    
        parameters = list(self.linear.parameters())

        return parameters

    def named_parameters(self):

        named_parameters = {'linear': {param_name: [] for param_name in ['bias', 'weight']}}

        for param_name, param in self.linear.named_parameters():

            param = np.squeeze(param.cpu().detach().numpy())

            named_parameters['linear'][param_name].append(param)

        return named_parameters

    def load_state_dict(self, path: str):

        super().load_state_dict(torch.load(path + 'state_dict', map_location=self.device))

        super().eval()

    def save_state_dict(self, path: str):

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        torch.save(super().state_dict(), path + 'state_dict')

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :type x:s
        :return:
        :rtype:
        """

        if self.output_activation_function is not None:            
            x = self.output_activation_function(self.linear(x))        
        else:
            x = self.linear(x)

        return x
