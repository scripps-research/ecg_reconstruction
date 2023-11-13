from util_functions.general import get_parent_folder, get_data_classes, get_detect_classes
from training_functions.single_recon_classif_manager import ReconClassifManager
from training_functions.multi_recon_classif_manager import MultiReconClassifManager


def get_recon_classif_default_settings():
    
    data_size = 'max'
    
    use_residual = 'true'
        
    epoch_num = 0
        
    batch_size = 16
        
    prioritize_percent = 0
    
    prioritize_size = 0
        
    optimizer = 'adam'
        
    learning_rate = 0.000003
        
    weight_decay = 0.001
        
    momentum = 0.9
        
    nesterov = 'true'
    
    return data_size, use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, \
                optimizer, learning_rate, weight_decay, momentum, nesterov

    
def get_recon_classif_settings(args):
    
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
        input_leads = 'full'
    
    if args['output_leads'] is not None:
        output_leads = args['output_leads']
    else:
        output_leads = 'precordial'
        
    if args['alpha'] is not None:
        alpha = args['alpha']
    else:
        alpha = 0.5
        
    if args['parallel'] is not None:
        parallel_classification = args['parallel']
    else:
        if len(detect_classes) > 1:
            parallel_classification = 'true'
        else:
            parallel_classification = 'false'
        
    data_size, use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, \
        optimizer, learning_rate, weight_decay, momentum, nesterov = get_recon_classif_default_settings()  
    
    if args['data_size'] is not None:
        data_size = args['data_size']
        
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
        
    return parent_folder, device, sub_classes, input_leads, output_leads, alpha, parallel_classification, data_classes, data_size, detect_classes,\
        use_residual, epoch_num, batch_size,prioritize_percent, prioritize_size, optimizer, learning_rate, weight_decay, momentum, nesterov
    

def get_recon_classif_manager(args):
    
    parent_folder, device, sub_classes, input_leads, output_leads, alpha, parallel_classification, data_classes, data_size, detect_classes,\
        use_residual, epoch_num, batch_size,prioritize_percent, prioritize_size, optimizer_algorithm, learning_rate, weight_decay, momentum, nesterov = get_recon_classif_settings(args)

    manager = ReconClassifManager(parent_folder,
                                  device,
                                  sub_classes,
                                  input_leads,
                                  output_leads,
                                  alpha,
                                  parallel_classification,
                                  data_classes,                                  
                                  data_size,
                                  detect_classes,
                                  use_residual,
                                  epoch_num,
                                  batch_size,
                                  prioritize_percent,
                                  prioritize_size,
                                  optimizer_algorithm,
                                  learning_rate,
                                  weight_decay,
                                  momentum,
                                  nesterov)
    
    return manager, sub_classes


def get_multi_recon_classif_manager(args):
    
    parent_folder, device, sub_classes, input_leads, output_leads, alpha, parallel_classification, data_classes, data_size, detect_classes,\
        use_residual, epoch_num, batch_size, prioritize_percent, prioritize_size, optimizer_algorithm, learning_rate, weight_decay, momentum, nesterov = get_recon_classif_settings(args)
    
    parent_folder = get_parent_folder()

    output_name = args['output_name']
    
    if output_name == 'paper_compare_detection':
        
        manager_labels = ['I + II', 'I + II + V3', 'Original',]
        
        input_leads = ['limb', 'limb+v3', 'full']
            
        output_leads = ['precordial', 'precordial', None]
            
        alpha = [0.5, 0.5, None]
        
        epoch_num = [200, 200, 200]
        
        sub_classes = [[] for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        detect_classes = [['acute_infarct'] for _ in range(len(manager_labels))]
        
    elif output_name == 'compare_detection_v2':
        
        manager_labels = ['I + II', 'I + II + V3', 'Full 12-lead',]
        
        input_leads = ['limb', 'limb+v3', 'full']
            
        output_leads = ['precordial', 'precordial', None]
            
        alpha = [0.5, 0.5, None]
        
        epoch_num = [0, 0, 200]
        
        sub_classes = []
        
        data_classes = [['st_elevation_or_infarct', 'non_st_elevation_or_infarct'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
    
    elif output_name == 'compare_detection_limb+v3':
        
        manager_labels = ['Full 12-lead', 'I + II + V3', 'I + II + V3 + recon', 'I + II + V3 + transf']
        
        input_leads = ['full', 'limb+v3', 'limb+v3', 'limb+v3']
            
        output_leads = [None, None, 'precordial', 'precordial']
            
        alpha = [None, None, 0.95, 0.95]
        
        epoch_num = [100, 100, 0, 100]
        
        sub_classes = []
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
    elif output_name == 'compare_detection_limb':
        
        manager_labels = ['Full 12-lead', 'I + II', 'I + II + recon', 'I + II + transf']
        
        input_leads = ['full', 'limb', 'limb', 'limb']
            
        output_leads = [None, None, 'precordial', 'precordial']
            
        alpha = [None, None, 0.95, 0.95]
        
        epoch_num = [100, 100, 0, 100]
        
        sub_classes = []
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
    elif output_name == 'compare_reconstruction_limb+v3':
        
        manager_labels = ['I + II + V3 + recon', 'I + II + V3 + transf']
        
        parallel_classification = ['true' for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
        input_leads = ['limb+v3' for _ in range(len(manager_labels))] 
            
        output_leads = ['precordial' for _ in range(len(manager_labels))]
        
        sub_classes = []
            
        alpha = [0.95, 0.95]
            
        epoch_num = [0, 100]
        
    elif output_name == 'compare_reconstruction_limb':
        
        manager_labels = ['I + II + recon', 'I + II + transf']
        
        parallel_classification = ['true' for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
        input_leads = ['limb' for _ in range(len(manager_labels))] 
            
        output_leads = ['precordial' for _ in range(len(manager_labels))]
        
        sub_classes = []
            
        alpha = [0.95, 0.95]
            
        epoch_num = [0, 100]
        
    elif output_name == 'compare_reconstruction':
        
        manager_labels = ['I + II + V3 + recon', 'I + II + V + transf', 'I + II + recon', 'I + II + transf']
        
        parallel_classification = ['true' for _ in range(len(manager_labels))]
        
        data_classes = [['st_elevation_or_infarct', 'other'] for _ in range(len(manager_labels))]
        
        detect_classes = [['st_elevation_or_acute_infarct'] for _ in range(len(manager_labels))]
        
        input_leads = ['limb+v3', 'limb+v3', 'limb', 'limb'] 
            
        output_leads = ['precordial' for _ in range(len(manager_labels))]
        
        sub_classes = []
            
        alpha = [0.95, 0.95, 0.95, 0.95]
            
        epoch_num = [0, 100, 0, 100]
        
    else:
        
        raise ValueError

    multi_manager = MultiReconClassifManager(output_name,
                                            parent_folder,
                                            device,
                                            sub_classes,
                                            input_leads,
                                            output_leads,
                                            alpha,
                                            parallel_classification,
                                            data_classes,                                            
                                            data_size,
                                            detect_classes,
                                            use_residual,
                                            epoch_num,
                                            batch_size,
                                            prioritize_percent,
                                            prioritize_size,
                                            optimizer_algorithm,
                                            learning_rate,
                                            weight_decay,
                                            momentum,
                                            nesterov,
                                            manager_labels)
    
    return multi_manager, sub_classes
