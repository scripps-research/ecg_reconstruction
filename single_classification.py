from util_functions.classification_settings import get_classification_manager
import argparse

"""
This script makes it possible to design and train a Deep Learning
model taking a subset of ECG leads as input and detecting
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

    parser.add_argument('-train', '--train', action='store_const', const=True, default=False)
    parser.add_argument('-test', '--test', action='store_const', const=True, default=False)
    parser.add_argument('-eval', '--evaluate', action='store_const', const=True, default=False)
    parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
    parser.add_argument('-plot_model', '--plot_model', action='store_const', const=True, default=False)
    parser.add_argument('-plot_train', '--plot_train', action='store_const', const=True, default=False)
    parser.add_argument('-plot_test', '--plot_test', action='store_const', const=True, default=False)
    parser.add_argument('-plot_diagn', '--plot_diagn', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())
        
    manager, sub_classes = get_classification_manager(args)

    sub_classes = ['young',
                'old',
                'very_old',
                'female',
                'male',
                'caucasian',
                'non_caucasian',
                'normal_sinus_rhythm',
                'abnormal_sinus_rhythm',
                'atrial_arrhythmia',
                'ventricular_arrhythmia',
                'conduction_abnormality',
                'repolarization_abnormality',
                'cardiac_hypertrophy',
                'axis_deviation',
                'ischemia',
                'prior_infarct',
                'acute_infarct',
                'anterior_acute_infarct',
                'septal_acute_infarct',
                'lateral_acute_infarct',
                'anteroseptal_acute_infarct',
                'anterolateral_acute_infarct',
                'inferolateral_acute_infarct',
                'inferior_or_posterior_acute_infarct',
                'nonspecific_localization_acute_infarct']


    if args['plot']:
        
        print('Plot all...')

        manager.load_train_stats()

        manager.plot_train_stats()

        manager.load_valid_stats()

        manager.plot_valid_stats()

        manager.load_test_stats()

        manager.plot_test_stats(plot_sub_classes = sub_classes)

        manager.load_model_stats()
        
        manager.plot_model_stats()
        
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

        if args['test']:

            print('Test...')

            manager.load_model()

            manager.load_dataset(test=True)

            manager.test()

            manager.release_dataset()

            manager.plot_test_stats(plot_sub_classes = sub_classes)
            
        elif args['plot_test']:
            
            print('Plot test...')
            
            manager.load_test_stats()

            manager.plot_test_stats(plot_sub_classes = sub_classes)
            
        if args['evaluate']:

            print('Evaluate...')

            manager.load_model()

            manager.compute_model_stats()
            
            manager.plot_model_stats()
        
        elif args['plot_model']:
            
            print('Plot model...')
            
            manager.load_model_stats()

            manager.plot_model_stats()
            
        elif args['plot_diagn']:
            
            print('Plot diagnonses...')
            
            manager.load_model()

            manager.plot_diagnoses()

