import argparse
from util_functions.recon_classif_settings import get_multi_recon_classif_manager

"""
This script makes it possible to design and train multiple Deep Learning
models, each taking a subset of ECG leads as input and generating a full 12-lead
ECG as output. The reconstruction loss is given by the probability of 
detecting specific clinical labels from the reconstructed signals. 

"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-device', '--device', type=str, default=None)
    parser.add_argument('-input', '--input_leads', type=str, default=None)
    parser.add_argument('-output', '--output_leads', type=str, default=None)
    parser.add_argument('-dataset', '--dataset', type=str, default=None)
    parser.add_argument('-detectset', '--detectset', type=str, default=None)
    parser.add_argument('-data_size', '--data_size', type=str, default=None)
    parser.add_argument('-use_residual', '--use_residual', type=str, default=None)
    parser.add_argument('-epoch', '--epoch_num', type=int, default=None)
    parser.add_argument('-batch', '--batch_size', type=int, default=None)
    parser.add_argument('-ppercent', '--prioritize_percent', type=float, default=None)
    parser.add_argument('-psize', '--prioritize_size', type=int, default=None)
    parser.add_argument('-optim', '--optimizer', type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, default=None)
    parser.add_argument('-mom', '--momentum', type=float, default=None)
    parser.add_argument('-decay', '--weight_decay', type=float, default=None)
    parser.add_argument('-nest', '--nesterov', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=None)
    parser.add_argument('-parallel', '--parallel', type=str, default=None)
    parser.add_argument('-format', '--plot_format', type=str, default='png')

    parser.add_argument('-name', '--output_name', type=str, default=None)
    parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
    parser.add_argument('-test', '--test', action='store_const', const=True, default=False)
    parser.add_argument('-train', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-eval', '--eval', action='store_const', const=True, default=False)
    parser.add_argument('-plot_train', '--plot_train', action='store_const', const=True, default=False)
    parser.add_argument('-plot_test', '--plot_test', action='store_const', const=True, default=False)
    parser.add_argument('-plot_random', '--plot_random', action='store_const', const=True, default=False)
    parser.add_argument('-plot_error', '--plot_error', action='store_const', const=True, default=False)
    parser.add_argument('-plot_sub_class', '--plot_sub_class', action='store_const', const=True, default=False)
    parser.add_argument('-clinical_test', '--clinical_test', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())

    train_ticks = [x * .05 / 10 for x in range(0, 11)]
    valid_ticks = [x * .05 / 10 for x in range(0, 11)]

    multi_manager, sub_classes = get_multi_recon_classif_manager(args)

    sub_classes = []

    if args['plot']:
        
        print('Plot...')
        
        multi_manager.load_train_stats()
        
        multi_manager.load_valid_stats()

        multi_manager.plot_train_stats(loss_ticks = train_ticks, plot_format = args['plot_format'])
        
        multi_manager.plot_valid_stats(loss_ticks = valid_ticks, plot_format = args['plot_format'])
        
        multi_manager.load_test_stats()

        multi_manager.plot_test_stats(plot_sub_classes = sub_classes, plot_format = args['plot_format'])
        
        multi_manager.load_model()
        
        multi_manager.plot_error_example(plot_format = args['plot_format'])
        
        multi_manager.plot_random_example(plot_format = args['plot_format'])
        
        multi_manager.plot_sub_class_example(plot_sub_classes = sub_classes, plot_format = args['plot_format'])
        
    else:
        
        if args['train']:
            
            print('Train...')
            
            multi_manager.train()
            
        if args['test']:
            
            print('Test...')
            
            multi_manager.test()
            
        if args['eval']:
            
            print('Evaluate...')
            
            multi_manager.eval()
            
        if args['plot_train']:
            
            print('Plot train performance...')
            
            multi_manager.load_train_stats()
            
            multi_manager.load_valid_stats()
            
            multi_manager.plot_train_stats(loss_ticks = train_ticks, plot_format = args['plot_format'])
            
            multi_manager.plot_valid_stats(loss_ticks = valid_ticks, plot_format = args['plot_format'])
            
        if args['plot_test']:
            
            print('Plot test performance...')
            
            multi_manager.load_test_stats()

            multi_manager.plot_test_stats(plot_sub_classes = sub_classes, plot_format = args['plot_format'])
        
        if args['plot_random']:
            
            print('Plot random example...')
            
            multi_manager.load_model()
        
            multi_manager.plot_random_example(plot_format = args['plot_format'])
            
        if args['plot_error']:
            
            print('Plot error example...')
            
            multi_manager.load_model()
        
            multi_manager.plot_error_example(plot_format = args['plot_format'])
            
        if args['plot_sub_class']:
            
            print('Plot error examples...')

            multi_manager.load_model()

            multi_manager.plot_sub_class_example(plot_sub_classes = sub_classes, plot_format = args['plot_format'])
            
        if args['clinical_test']:
            
            print('Clinical test...')

            multi_manager.load_model()

            multi_manager.emulate_clinical_test(size=6000, thresholds = [0.123, 0.319, 0.509], sub_classes=['st_elevation_or_acute_infarct', 'prior_infarct', 'non_st_elevation_or_infarct'], sub_class_sizes=[0.5, 0.25, 0.25])
