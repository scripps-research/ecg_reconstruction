import argparse
from util_functions.reconstruction_settings import get_multi_reconstruction_manager

"""
This script makes it possible to design and train multiple Deep Learning
models, each taking a subset of ECG leads as input and generating a full 12-lead
ECG as output. The reconstruction loss is given by the mathematical difference
between the recostructed signal and the original 12-lead ECG.

"""

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-device', '--device', type=str, default=None)
    parser.add_argument('-input', '--input_leads', type=str, default=None)
    parser.add_argument('-output', '--output_leads', type=str, default=None)
    parser.add_argument('-dataset', '--dataset', type=str, default=None)
    parser.add_argument('-data_size', '--data_size', type=str, default=None)
    parser.add_argument('-input_channel', '--input_channel', type=int, default=None)
    parser.add_argument('-middle_channel', '--middle_channel', type=int, default=None)
    parser.add_argument('-output_channel', '--output_channel', type=int, default=None)
    parser.add_argument('-input_depth', '--input_depth', type=int, default=None)
    parser.add_argument('-middle_depth', '--middle_depth', type=int, default=None)
    parser.add_argument('-output_depth', '--output_depth', type=int, default=None)
    parser.add_argument('-input_kernel', '--input_kernel', type=int, default=None)
    parser.add_argument('-middle_kernel', '--middle_kernel', type=int, default=None)
    parser.add_argument('-output_kernel', '--output_kernel', type=int, default=None)
    parser.add_argument('-residual', '--use_residual', type=str, default=None)
    parser.add_argument('-epoch', '--epoch_num', type=int, default=None)
    parser.add_argument('-batch', '--batch_size', type=int, default=None)
    parser.add_argument('-ppercent', '--prioritize_percent', type=float, default=None)
    parser.add_argument('-psize', '--prioritize_size', type=int, default=None)
    parser.add_argument('-optim', '--optimizer', type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, default=None)
    parser.add_argument('-mom', '--momentum', type=float, default=None)
    parser.add_argument('-decay', '--weight_decay', type=float, default=None)
    parser.add_argument('-nest', '--nesterov', type=str, default=None)

    parser.add_argument('-format', '--plot_format', type=str, default='png')
    parser.add_argument('-name', '--output_name', type=str, default=None)
    parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
    parser.add_argument('-test', '--test', action='store_const', const=True, default=False)
    parser.add_argument('-train', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-eval', '--eval', action='store_const', const=True, default=False)
    parser.add_argument('-plot_test', '--plot_test', action='store_const', const=True, default=False)
    parser.add_argument('-plot_train', '--plot_train', action='store_const', const=True, default=False)
    parser.add_argument('-plot_random', '--plot_random', action='store_const', const=True, default=False)
    parser.add_argument('-plot_error', '--plot_error', action='store_const', const=True, default=False)
    parser.add_argument('-plot_sub_class', '--plot_sub_class', action='store_const', const=True, default=False)
    parser.add_argument('-generate', '--generate', action='store_const', const=True, default=False)

    train_ticks = [x * .02  for x in range(-40, -29)]
    valid_ticks = [x * .02  for x in range(-40, -29)]

    args = vars(parser.parse_args())

    multi_manager, sub_classes = get_multi_reconstruction_manager(args)

    if args['plot']:
        
        print('Plot all...')

        multi_manager.load_train_stats()

        multi_manager.plot_train_stats(loss_ticks=train_ticks, plot_format=args['plot_format'])

        multi_manager.load_valid_stats()

        multi_manager.plot_valid_stats(loss_ticks=valid_ticks, plot_format=args['plot_format'])

        multi_manager.load_test_stats()

        multi_manager.plot_test_stats(plot_format=args['plot_format'])
        
        multi_manager.load_model()

        multi_manager.plot_random_example(plot_format=args['plot_format'])

        multi_manager.plot_error_example(plot_format=args['plot_format'])
        
        multi_manager.plot_sub_class_example(plot_sub_classes=sub_classes, plot_format=args['plot_format'])
        
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

            multi_manager.plot_train_stats(loss_ticks=train_ticks, plot_format=args['plot_format'])

            multi_manager.plot_valid_stats(loss_ticks=valid_ticks, plot_format=args['plot_format'])

        if args['plot_test']:
            
            print('Plot test performance...')
            
            multi_manager.load_test_stats()

            multi_manager.plot_test_stats(plot_format=args['plot_format'])

        if args['plot_random']:

            print('Plot random examples...')

            multi_manager.load_model()

            multi_manager.plot_random_example(plot_format=args['plot_format'])

        if args['plot_error']:
        
            print('Plot error examples...')

            multi_manager.load_model()

            multi_manager.plot_error_example(plot_format=args['plot_format'])
            
        if args['plot_sub_class']:
            
            print('Plot error examples...')

            multi_manager.load_model()

            multi_manager.plot_sub_class_example(plot_sub_classes=sub_classes, plot_format=args['plot_format'])
            
        if args['generate']:
            
            print('Generate clinical test...')

            multi_manager.load_model()

            multi_manager.generate_clinical_test(size_per_sub_class=[119, 119],
                                                sub_classes=['test_positives', 'test_negatives'],
                                                generate=True,
                                                randomized=True,
                                                split_figures=True)
