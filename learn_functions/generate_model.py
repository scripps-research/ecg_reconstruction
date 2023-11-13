from learn_functions.reconstructor import Reconstructor
from learn_functions.classificator import Classificator, ParallelClassificator
import torch


def generate_reconstructor(input_lead_num: int,
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
                           use_residual: bool,
                           device):

    activation_function = 'relu'
    
    if use_residual == 'true':
        use_residual_block = True
    elif use_residual == 'false':
        use_residual_block = False
    else:
        raise ValueError

    model = Reconstructor(input_lead_num,
                          output_lead_num,
                          input_channel_per_lead,
                          middle_channel_per_lead,
                          output_channel_per_lead,
                          block_per_input_network,
                          block_per_middle_network,
                          block_per_output_network,
                          input_kernel_size,
                          middle_kernel_size, 
                          output_kernel_size,
                          activation_function, 
                          use_residual_block,
                          device)

    return model


def generate_classificator(input_lead_num: int,
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
                           use_residual: bool,
                           device: str,
                           parallel: bool):
    
    activation_function = 'relu'
    
    if use_residual == 'true':
        use_residual_block = True
    elif use_residual == 'false':
        use_residual_block = False
    else:
        raise ValueError
    
    if parallel:
        
        model = ParallelClassificator(input_lead_num,
                                       detect_class_num,
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
                                       device)
        
    else:

        model = Classificator(input_lead_num,
                            detect_class_num,
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
                            device)

    return model


def generate_optimizer(optimizer, learning_rate, weight_decay, momentum, nesterov, model_parameters):

    if optimizer == 'adam':

        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay, eps=0.01)

    elif optimizer == 'sgd':

        if nesterov == 'true':
            nesterov = True
        elif nesterov == 'false':
            nesterov = False
        else: raise ValueError
    

        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    
    else:
        raise ValueError

    return optimizer