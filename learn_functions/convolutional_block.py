from sqlite3 import SQLITE_DROP_INDEX
import numpy as np
import torch
import torch.nn as nn
import pathlib

class ConvolutionalBlock(nn.Module):
    """
    class ResidualBlock

    """

    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size: int,
                 stride_size: int,
                 inner_activation: str,
                 output_activation: str,
                 use_residual_block: bool,
                 average_pool: int,
                 device
                 ):

        super(ConvolutionalBlock, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel

        self.use_residual_block = use_residual_block

        self.inner_activation = inner_activation
        self.output_activation = output_activation

        self.device = device

        if self.inner_activation == 'relu':
            self.inner_activation_function = torch.relu  
        elif self.inner_activation == 'leaky_relu':
            self.inner_activation_function = torch.nn.functional.leaky_relu
        elif self.inner_activation == 'sigmoid':
            self.inner_activation_function = torch.sigmoid
        else:
            raise ValueError

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
        
        if stride_size > 1:
            stride = stride_size
            if kernel_size % 2 == 1:
                padding = int(kernel_size/2)
            else:
                raise ValueError
                
        else:
            stride = 1
            padding = 'same'

        self.inner_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=self.output_channel,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding='same').to(self.device)   

        if self.use_residual_block:
        
            self.res_conv = nn.Conv1d(in_channels=self.input_channel,
                                        out_channels=self.output_channel,
                                        kernel_size=1,
                                        stride=stride,
                                        padding='valid').to(self.device)

        else:

            self.res_conv = None

        self.out_conv = nn.Conv1d(in_channels=self.output_channel,
                                  out_channels=self.output_channel,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding).to(self.device)

        self.reset()   
        
        if self.use_residual_block:
            self.models = [self.inner_conv, self.out_conv, self.res_conv]
            self.model_names = ['in', 'out', 'res']
        else:
            self.models = [self.inner_conv, self.out_conv]
            self.model_names = ['in', 'out']
            
        if average_pool > 1:
            
            self.average_pool = torch.nn.AvgPool1d(average_pool)
            
        else:
            
            self.average_pool = None

    def reset(self):

        torch.nn.init.kaiming_uniform_(self.inner_conv.weight, nonlinearity=self.inner_activation)
        torch.nn.init.zeros_(self.inner_conv.bias)

        if self.use_residual_block:

            if self.input_channel != self.output_channel:

                if self.output_activation is None:
                    torch.nn.init.kaiming_uniform_(self.res_conv.weight, nonlinearity='linear')
                else:
                    torch.nn.init.kaiming_uniform_(self.res_conv.weight, nonlinearity=self.output_activation)

                torch.nn.init.zeros_(self.res_conv.bias)

        if self.output_activation is None:
            torch.nn.init.kaiming_uniform_(self.out_conv.weight, nonlinearity='linear')
        else:
            torch.nn.init.kaiming_uniform_(self.out_conv.weight, nonlinearity=self.output_activation)

        torch.nn.init.zeros_(self.out_conv.bias)

    def parameters(self):

        parameters = []
        
        for model in self.models:

            parameters += list(model.parameters())

        return parameters

    def named_parameters(self):

        named_parameters = {model_name: {param_name: [] for param_name in ['bias', 'weight']} for model_name in self.model_names}

        for model_name, model in zip(self.model_names, [self.inner_conv, self.out_conv, self.res_conv]):

            for param_name, param in model.named_parameters():

                param = np.squeeze(param.cpu().detach().numpy())

                named_parameters[model_name][param_name].append(param)

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

        if self.use_residual_block:
            residual = self.res_conv(x)

        x = self.inner_activation_function(self.inner_conv(x))

        x = self.out_conv(x)

        if self.use_residual_block:
            x += residual

        if self.output_activation_function is not None:
            
            x = self.output_activation_function(x)
            
        if self.average_pool is not None:
            
            x = self.average_pool(x)

        return x
