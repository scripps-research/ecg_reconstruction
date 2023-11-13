import argparse
import pathlib
from attr import dataclass
import numpy as np
from clean_dataclass import clean_dataclass
from analyze_dataclass import analyze_dataclass
from plot_functions.continuos_distribution import plot_violin_distribution
from util_functions.general import get_parent_folder
from util_functions.load_data_ids import *

def process_dataclass(data_class: str, clean: bool, analyze: bool, reset: bool): 
    
    """
    This function makes it possible to (i) discern the elements of a given data_class
    between corrupted and cleaned elements and (ii) compute the conditional probability
    for the elements of the data_class to be associated with the different
    clinical labels used to describe the dataset
    
    The IDs of the corrupted/cleaned elements are stored in the folder ./../Data/Feature_map/Dataclass/data_class/'
    
    The statistical results are stored in the folder ./../Data/Analysis/Dataclass/data_class/Data/
    
    The figures describing the results are stored in the folder ./../Data/Analysis/Dataclass/data_class/Plot/
    
    """
    
    parent_folder = get_parent_folder()
        
    stats_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Data/'
    plot_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Plot/'  
    
    pathlib.Path(plot_folder).mkdir(parents=True, exist_ok=True) 
    
    class_ids = load_dataclass_ids(parent_folder, data_class)
    class_patient_ids = load_dataclass_patient_ids(parent_folder, data_class)
    
    print('Number of class data:', len(class_ids))
    
    print('Number of class individuals:', len(class_patient_ids))  

    if clean:
        
        clean_dataclass(data_class)
        
    class_clean_ids = load_dataclass_clean_ids(parent_folder, data_class)
    class_corrupted_ids = load_dataclass_corrupted_ids(parent_folder, data_class)
    
    class_clean_patient_ids = load_dataclass_clean_patient_ids(parent_folder, data_class)
    class_corrupted_patient_ids = load_dataclass_corrupted_patient_ids(parent_folder, data_class)

    class_clean_ages = np.load(stats_folder + 'clean_ages.npy')
    class_clean_acquisition_date = np.load(stats_folder + 'clean_acquisition_date.npy')
    class_clean_max_values = np.load(stats_folder + 'clean_max_values.npy')
    class_clean_min_values = np.load(stats_folder + 'clean_min_values.npy')   
    
    class_corrupted_ages = np.load(stats_folder + 'corrupted_ages.npy')
    class_corrupted_acquisition_date = np.load(stats_folder + 'corrupted_acquisition_date.npy')
    class_corrupted_max_values = np.load(stats_folder + 'corrupted_max_values.npy')
    class_corrupted_min_values = np.load(stats_folder + 'corrupted_min_values.npy') 
    
    clean_data_size = len(class_clean_ids)
    corrupted_data_size = len(class_corrupted_ids)
    class_data_size = len(class_ids)
    
    clean_patient_size = len(class_clean_patient_ids)
    corrupted_patient_size = len(class_corrupted_patient_ids)
    class_patient_size = len(class_patient_ids)

    print('Number of data: ',  class_data_size)
    print('Number of clean data: ', clean_data_size)
    print('Number of corrupted data: ', corrupted_data_size)

    print()
    
    print('Number of individuals: ',  class_patient_size)
    print('Number of clean individuals: ', clean_patient_size)
    print('Number of corrupted individuals: ', corrupted_patient_size)

    print()
    
    data_values = []
    data_value_labels = []
    
    ages = []
    acquisition_dates = []
    acquisition_date_labels = []
    
    if clean_data_size > 0:

        print('CLEAN DATA STATS')

        print('Max values')
        print('99th percentile: ', np.percentile(class_clean_max_values, 99))
        print('95th percentile: ', np.percentile(class_clean_max_values, 95))
        print('75th percentile: ', np.percentile(class_clean_max_values, 75))
        print('Median: ', np.median(class_clean_max_values))
        print('Mean: ', np.mean(class_clean_max_values))

        print('Min values')
        print('1st percentile: ', np.percentile(class_clean_min_values, 1))
        print('5th percentile: ', np.percentile(class_clean_min_values, 5))
        print('25th percentile: ', np.percentile(class_clean_min_values, 25))
        print('Median: ', np.median(class_clean_min_values))
        print('Mean: ', np.mean(class_clean_min_values))
    
        data_values += [class_clean_max_values, class_clean_min_values]
        data_value_labels += ['Max values (clean)', 'Min value (clean)']
        
        ages += [class_clean_ages]        
        acquisition_dates += [class_clean_acquisition_date]        
        acquisition_date_labels += ['Clean data']

        print()
        
    if corrupted_data_size > 0:

        print('CORRUPTED DATA STATS')

        print('Max values')
        print('99th percentile: ', np.percentile(class_corrupted_max_values, 99))
        print('95th percentile: ', np.percentile(class_corrupted_max_values, 95))
        print('75th percentile: ', np.percentile(class_corrupted_max_values, 75))
        print('Median: ', np.median(class_corrupted_max_values))
        print('Mean: ', np.mean(class_corrupted_max_values))

        print('Min values')
        print('1st percentile: ', np.percentile(class_corrupted_min_values, 1))
        print('5th percentile: ', np.percentile(class_corrupted_min_values, 5))
        print('25th percentile: ', np.percentile(class_corrupted_min_values, 25))
        print('Median: ', np.median(class_corrupted_min_values))
        print('Mean: ', np.mean(class_corrupted_min_values))
    
        data_values += [class_corrupted_max_values, class_corrupted_min_values]
        data_value_labels += ['Max values (corrupted)', 'Min value (corrupted)']
        
        ages += [class_corrupted_ages]       
        acquisition_dates += [class_corrupted_acquisition_date]        
        acquisition_date_labels += ['Corrupted data']

        print()
        
    if clean_data_size > 0 and corrupted_data_size > 0:

        print('Plot voltage distribution...')

        plot_violin_distribution(data_values, data_value_labels, 'Value', plot_folder + 'data_values', ylims=[-4, 4])
        
        print('Plot age distribution...')

        plot_violin_distribution(ages, acquisition_date_labels, 'Age', plot_folder + 'age_distribution')

        print('Plot acquisition distribution...')

        plot_violin_distribution(acquisition_dates, acquisition_date_labels, 'Acquisition date', plot_folder + 'acquisition_distribution')
        
        print()
        
    if reset:
            
        reset_dataclass_training_ids(parent_folder, data_class)
            
    train_ids, valid_ids, test_ids = load_class_training_ids(parent_folder, data_class)
    train_patient_ids, valid_patient_ids, test_patient_ids = load_class_training_patient_ids(parent_folder, data_class)

    print('Number of train, valid, test data',
          len(train_ids), len(valid_ids), len(test_ids),
          'whose sum is',  len(train_ids) + len(valid_ids) + len(test_ids))
    
    print()
    
    print('Number of train, valid, test individuals',
          len(train_patient_ids), len(valid_patient_ids), len(test_patient_ids),
          'whose sum is',  len(train_patient_ids) + len(valid_patient_ids) + len(test_patient_ids))
    
    print()
    
    if analyze:
        
        analyze_dataclass(data_class, analyze=True)
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_class', '--data_class', type=str)
    parser.add_argument('-multi_class', '--multi_class', action='store_const', const=True, default=False)
    parser.add_argument('-clean', '--clean', action='store_const', const=True, default=False)
    parser.add_argument('-analyze', '--analyze', action='store_const', const=True, default=False)
    parser.add_argument('-reset', '--reset', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())
    
    data_class = args['data_class']    
    
    clean = args['clean']
    analyze = args['analyze']    
    reset = args['reset']
    
    process_dataclass(data_class, clean, analyze, reset)
