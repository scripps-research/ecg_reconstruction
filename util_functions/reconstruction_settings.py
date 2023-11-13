from util_functions.general import get_parent_folder, get_data_classes
from training_functions.single_reconstruction_manager import ReconstructionManager
from training_functions.multi_reconstruction_manager import MultiReconstructionManager
    

def get_reconstruction_default_settings():
    
    data_size = 'max'
        
    input_channel = 32
        
    middle_channel = 32
        
    output_channel = 32
        
    input_depth = 3
        
    middle_depth = 2
        
    output_depth = 3
        
    input_kernel = 17
        
    middle_kernel = 17
        
    output_kernel = 17
        
    use_residual = 'true'
        
    epoch_num = 200
        
    batch_size = 16
        
    prioritize_percent = 0
    
    prioritize_size = 0
        
    optimizer = 'adam'
        
    learning_rate = 0.000003
        
    weight_decay = 0.001
        
    momentum = 0.9
        
    nesterov = 'true'
    
    return data_size, input_channel, middle_channel, output_channel, input_depth, \
        middle_depth, output_depth, input_kernel, middle_kernel, output_kernel, use_residual, \
            epoch_num, batch_size, prioritize_percent, prioritize_size, \
                optimizer, learning_rate, weight_decay, momentum, nesterov
    
def get_reconstruction_settings(args):
    
    parent_folder = get_parent_folder()
        
    data_classes = get_data_classes(args['dataset'])
        
    sub_classes = []
        
    if args['device'] is not None:
        device = args['device']        
    else:
        device = 'cuda:0'
        
    if args['input_leads'] is not None:
        input_leads = args['input_leads']
    else:
        input_leads = 'limb'
         
    if args['output_leads'] is not None:
        output_leads = args['output_leads']
    else:
        output_leads = 'precordial'
    
    data_size, input_channel, middle_channel, output_channel, input_depth, \
        middle_depth, output_depth, input_kernel, middle_kernel, output_kernel, use_residual, \
            epoch_num, batch_size, prioritize_percent, prioritize_size, \
                optimizer, learning_rate, weight_decay, momentum, nesterov = get_reconstruction_default_settings()
    
    if args['data_size'] is not None:
        data_size = args['data_size']
    else:
        assert data_size is not None
        
    if args['input_channel'] is not None:
        input_channel = args['input_channel']
        
    if args['middle_channel'] is not None:
        middle_channel = args['middle_channel']
        
    if args['output_channel'] is not None:
        output_channel = args['output_channel']
        
    if args['input_depth'] is not None:
        input_depth = args['input_depth']
        
    if args['middle_depth'] is not None:
        middle_depth = args['middle_depth']
        
    if args['output_depth'] is not None:
        output_depth = args['output_depth']
        
    if args['input_kernel'] is not None:
        input_kernel = args['input_kernel']
        
    if args['middle_kernel'] is not None:
        middle_kernel = args['middle_kernel']
        
    if args['output_kernel'] is not None:
        output_kernel = args['output_kernel']
        
    if args['use_residual'] is not None:
        use_residual = args['use_residual']
        
    if args['epoch_num'] is not None:
        epoch_num = args['epoch_num']
        
    if args['batch_size'] is not None:
        batch_size = args['batch_size']
        
    if args['prioritize_percent'] is not None:
        prioritize_percent = args['prioritize_percent']
        prioritize_size = args['prioritize_size']
        
    if args['optimizer'] is not None:
        optimizer = args['optimizer']
        
    if args['learning_rate'] is not None:
        learning_rate = args['learning_rate']
        
    if args['weight_decay'] is not None:
        weight_decay = args['weight_decay']
        
    if args['momentum'] is not None:
        momentum = args['momentum']
        
    if args['nesterov'] is not None:
        nesterov = args['nesterov']
        
    return parent_folder, device, sub_classes, input_leads, output_leads, data_classes, data_size, input_channel, middle_channel, output_channel, input_depth, middle_depth, output_depth, input_kernel, middle_kernel, output_kernel,\
        use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov
        
        
def get_reconstruction_manager(args):
    
    parent_folder, device, sub_classes, input_leads, output_leads, data_classes, data_size, input_channel, middle_channel, output_channel,\
        input_depth, middle_depth, output_depth, input_kernel, middle_kernel, output_kernel, use_residual, epoch_num, batch_size,\
            prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov = get_reconstruction_settings(args)
            

    manager = ReconstructionManager(parent_folder,
                                    device,
                                    sub_classes,
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
                                    epoch_num,
                                    batch_size,
                                    prioritize_percent,
                                    prioritize_size,
                                    optimizer,
                                    learning_rate,
                                    weight_decay,
                                    momentum,
                                    nesterov)
    
    return manager, sub_classes


def get_multi_reconstruction_manager(args):
    
    parent_folder, device, sub_classes, input_leads, output_leads, data_classes, data_size, input_channel, middle_channel, output_channel, input_depth, middle_depth, output_depth, input_kernel, middle_kernel, output_kernel,\
        use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov = get_reconstruction_settings(args)

    output_name = args['output_name']

    if output_name == 'paper_compare_input':
        
        input_leads = ['limb', 'limb+v1', 'limb+v2', 'limb+v3', 'limb+v4', 'limb+v5', 'limb+v6']
        
        manager_labels = ['I + II', 'I + II + V1', 'I + II + V2', 'I + II + V3', 'I + II + V4', 'I + II + V5', 'I + II + V6']
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = []
        
        output_name += '/'
        
    elif output_name == 'compare_input_v2':
        
        input_leads = ['limb', 'limb+v1', 'limb+v2', 'limb+v3',]
        
        manager_labels = ['I + II', 'I + II + V1', 'I + II + V2', 'I + II + V3']
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = []
        
        output_name += '/'
        
    elif output_name == 'compare_input_v3':
        
        input_leads = ['limb+v2', 'limb+v3', 'limb+v4']
        
        manager_labels = ['I + II + V2', 'I + II + V3', 'I + II + V4']
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = []
        
        output_name += '/'
        
    elif 'clinical_test' in output_name:
        
        input_leads = ['limb', 'limb+v3']
        
        epoch_num = [200, 200]
        
        manager_labels = ['I + II', 'I + II + V3']
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = ['acute_infarct', 'other']
        
        output_name += '/'
        
    else:
        raise ValueError

    multi_manager = MultiReconstructionManager(output_name,
                                               parent_folder,
                                               device,
                                               sub_classes,
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
                                               epoch_num,
                                               batch_size,
                                               prioritize_percent,
                                               prioritize_size,
                                               optimizer,
                                               learning_rate,
                                               weight_decay,
                                               momentum,
                                               nesterov,
                                               manager_labels)
    
    return  multi_manager, sub_classes
