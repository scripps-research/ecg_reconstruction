from util_functions.general import get_parent_folder, get_data_classes, get_detect_classes
from training_functions.single_classification_manager import ClassificationManager
from training_functions.multi_classification_manager import MultiClassificationManager


def get_classification_default_settings():
    
    data_size = 'max'
        
    input_channel = 16
    
    middle_channel = 128
        
    input_depth = 3
        
    middle_depth = 6
        
    output_depth = 3
        
    input_kernel = 17
        
    middle_kernel = 17
    
    stride_size = 3
        
    use_residual = 'true'
        
    epoch_num = 200
        
    batch_size = 16
        
    prioritize_percent = 0
    
    prioritize_size = 0
        
    optimizer = 'adam'
        
    learning_rate = 0.000001
        
    weight_decay = 0.001
        
    momentum = 0.9
        
    nesterov = 'true'
    
    return data_size, input_channel, middle_channel, input_depth, \
        middle_depth, output_depth, input_kernel, middle_kernel, stride_size, use_residual, \
            epoch_num, batch_size, prioritize_percent, prioritize_size, \
                optimizer, learning_rate, weight_decay, momentum, nesterov
            

def get_classification_settings(args):
    
    parent_folder = get_parent_folder()    
    
    data_classes = get_data_classes(args['dataset'])
    
    detect_classes = get_detect_classes(args['detectset'])
        
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda:0'
        
    sub_classes = []
    
    if args['input_leads'] is not None:
        input_leads = args['input_leads']
    else:
        input_leads = 'limb'
        
    if args['parallel'] is not None:
        parallel_classification = args['parallel']
    else:
        if len(detect_classes) > 1:
            parallel_classification = 'true'
        else:
            parallel_classification = 'false'
    
    data_size, input_channel, middle_channel, input_depth, \
        middle_depth, output_depth, input_kernel, middle_kernel, stride_size, use_residual, \
            epoch_num, batch_size, prioritize_percent, prioritize_size, \
                optimizer, learning_rate, weight_decay, momentum, nesterov = get_classification_default_settings()
         
    if args['data_size'] is not None:
        data_size = args['data_size']
        
    if args['input_channel'] is not None:
        input_channel = args['input_channel']
        
    if args['middle_channel'] is not None:
        middle_channel = args['middle_channel']
        
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
        
    if args['stride_size'] is not None:
        stride_size = args['stride_size']
        
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
        
    return parent_folder, device, sub_classes, input_leads, parallel_classification, data_classes, data_size, detect_classes, input_channel, middle_channel, input_depth, middle_depth, output_depth, input_kernel, middle_kernel, stride_size,\
        use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov


def get_classification_manager(args):
    
    parent_folder, device, sub_classes, input_leads, parallel_classification, data_classes, data_size, detect_classes, input_channel, middle_channel, input_depth, middle_depth, output_depth, input_kernel, middle_kernel, stride_size,\
        use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov = get_classification_settings(args)    

    manager = ClassificationManager(parent_folder,
                                    device,
                                    sub_classes,
                                    input_leads,
                                    parallel_classification,
                                    data_classes,                                    
                                    data_size,     
                                    detect_classes,                     
                                    input_channel,
                                    middle_channel,
                                    input_depth,
                                    middle_depth,
                                    output_depth,
                                    input_kernel,
                                    middle_kernel,
                                    stride_size,                          
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


def get_multi_classification_manager(args):    
    
    parent_folder, device, sub_classes, input_leads, parallel_classification, data_classes, data_size, detect_classes, input_channel, middle_channel, input_depth, middle_depth, output_depth, input_kernel, middle_kernel, stride_size,\
            use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov = get_classification_settings(args)
            
    output_name = args['output_name']

    if output_name == 'compare_input':
        
        manager_labels = ['I + II',
                          'I + II + V3',
                          'Full 12-lead']
        
        input_leads = ['limb', 'limb+v3', 'full']
        
        parallel_classification = ['false' for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = []
        
        output_name += '/'
        
    elif output_name == 'compare_lr':
        
        manager_labels = ['I + II + V3 with lr=5x10^-6',
                          'I + II + V3 with lr=1x10^-6',
                          'Full 12-lead with lr=5x10^-6',
                          'Full 12-lead with lr=1x10^-6']
        
        input_leads = ['limb+v3', 'limb+v3', 'full', 'full']
        
        learning_rate = [0.000005, 0.000001, 0.000005, 0.000001]
        
        parallel_classification = ['false' for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
        sub_classes = ['st_elevation_or_acute_infarct', 'prior_infarct', 'normal_ecg']
        
        output_name += '/'
        
    else:
        raise ValueError

    multi_manager = MultiClassificationManager(output_name,
                                            parent_folder,
                                            device,
                                            sub_classes,
                                            input_leads,
                                            parallel_classification,
                                            data_classes,                                            
                                            data_size,
                                            detect_classes,
                                            input_channel,
                                            middle_channel,
                                            input_depth,
                                            middle_depth,
                                            output_depth,
                                            input_kernel,
                                            middle_kernel,
                                            stride_size,
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
    
    return multi_manager, sub_classes
