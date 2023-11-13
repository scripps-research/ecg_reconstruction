from util_functions.recon_classif_settings import get_recon_classif_manager
import argparse

"""
This script makes it possible to design and train a Deep Learning
model taking a subset of ECG leads as input and generating a full 12-lead
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

    parser.add_argument('-train', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-plot_train', '--plot_train', action='store_const', const=True, default=False)
    parser.add_argument('-test', '--test', action='store_const', const=True, default=False)
    parser.add_argument('-plot_test', '--plot_test', action='store_const', const=True, default=False)
    parser.add_argument('-eval', '--evaluate', action='store_const', const=True, default=False)
    parser.add_argument('-plot_model', '--plot_model', action='store_const', const=True, default=False)
    parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
    parser.add_argument('-plot_random', '--plot_random', action='store_const', const=True, default=False)
    parser.add_argument('-plot_error', '--plot_error', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())

    manager, sub_classes = get_recon_classif_manager(args)

    if args['plot']:
        
        print('Plot all...')

        manager.load_train_stats()

        manager.plot_train_stats()

        manager.load_valid_stats()

        manager.plot_valid_stats()

        manager.load_test_stats()

        manager.plot_test_stats()

        manager.load_model_stats()
        
        manager.plot_model_stats()

        manager.load_model()
        
    else:

        if args['train']:

            print('Train...')

            manager.reset_model()

            manager.load_dataset(train=True, valid=True)

            manager.train()

            manager.release_dataset()

            manager.plot_train_stats()

            manager.plot_valid_stats()
            
        elif args['plot_train']:
            
            print('Plot train...')
            
            manager.load_train_stats()

            manager.plot_train_stats()

            manager.load_valid_stats()

            manager.plot_valid_stats()

        if args['test']:

            print('Test...')
            
            manager.load_model()

            manager.load_dataset(test=True)

            manager.test()

            manager.release_dataset()

            manager.plot_test_stats(plot_sub_classes=sub_classes)
            
        elif args['plot_test']:
        
            print('Plot test...')
            
            manager.load_test_stats()

            manager.plot_test_stats(plot_sub_classes=sub_classes)
            
        if args['evaluate']:

            print('Evaluate...')

            manager.load_model()

            manager.compute_model_stats()
            
            manager.plot_model_stats()
            
        elif args['plot_model']:
        
            print('Plot model...')

            manager.load_model_stats()
        
            manager.plot_model_stats()




