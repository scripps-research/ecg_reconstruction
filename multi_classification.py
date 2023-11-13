import argparse
from util_functions.classification_settings import get_multi_classification_manager

"""
This script makes it possible to design and train multiple Deep Learning
models, each taking a subset of ECG leads as input and detecting
specific clinical labels associated with the the target ECG signal.  

"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-device', '--device', type=str, default=None)
    parser.add_argument('-input', '--input_leads', type=str, default=None)
    parser.add_argument('-dataset', '--dataset', type=str, default=None)
    parser.add_argument('-detectset', '--detectset', type=str, default=None)
    parser.add_argument('-data_size', '--data_size', type=str, default=None)
    parser.add_argument('-input_channel', '--input_channel', type=int, default=None)
    parser.add_argument('-middle_channel', '--middle_channel', type=int, default=None)
    parser.add_argument('-input_depth', '--input_depth', type=int, default=None)
    parser.add_argument('-middle_depth', '--middle_depth', type=int, default=None)
    parser.add_argument('-output_depth', '--output_depth', type=int, default=None)
    parser.add_argument('-input_kernel', '--input_kernel', type=int, default=None)
    parser.add_argument('-middle_kernel', '--middle_kernel', type=int, default=None)
    parser.add_argument('-stride_size', '--stride_size', type=int, default=None)
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
    parser.add_argument('-parallel', '--parallel', type=str, default=None)

    parser.add_argument('-format', '--plot_format', type=str, default='png')
    parser.add_argument('-name', '--output_name', type=str, default=None)
    parser.add_argument('-test', '--test', action='store_const', const=True, default=False)
    parser.add_argument('-train', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-eval', '--eval', action='store_const', const=True, default=False)
    parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
    parser.add_argument('-plot_train', '--plot_train', action='store_const', const=True, default=False)
    parser.add_argument('-plot_test', '--plot_test', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())

    train_ticks = [x * .5 / 10 for x in range(0, 11)]
    valid_ticks = [x * .5 / 10 for x in range(0, 11)]

    multi_manager, sub_classes = get_multi_classification_manager(args)

    if args['plot']:
        
        print('Plot all...')

        multi_manager.load_train_stats()

        multi_manager.plot_train_stats(loss_ticks=train_ticks, plot_format=args['plot_format'])

        multi_manager.load_valid_stats()

        multi_manager.plot_valid_stats(loss_ticks=valid_ticks, plot_format=args['plot_format'])

        multi_manager.load_test_stats()

        multi_manager.plot_test_stats(plot_sub_classes = sub_classes, plot_format=args['plot_format'])
        
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

            multi_manager.plot_test_stats(plot_sub_classes = sub_classes, plot_format=args['plot_format'])