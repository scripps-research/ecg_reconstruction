from util_functions.load_data_ids import *
from util_functions.general import get_parent_folder
from load_functions.load_leads import extract_twelve_leads, load_data
from plot_functions.continuos_distribution import plot_violin_distribution
from util_functions.general import get_collection, get_parent_folder
from util_functions.diagnosis_map import diagnosis_map
import numpy as np
import pickle
import pathlib
import argparse

if __name__ == '__main__':
    
    """
    This script makes it possible to (i) discern the elements of the overall dataset between
    corrupted and cleaned, and (ii) divide the cleaned elements in three groups, which
    will be used as training, validation and testing sets for the analysis. 
    
    The IDs of the corrupted/cleaned elements are stored in the folder ./../Data/Feature_map/Dataset/'
    
    The statistical results are stored in the folder ./../Data/Analysis/Dataset/Data/
    
    The figures describing the results are stored in the folder ./../Data/Analysis/Dataset/Plot/
    
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-clean', '--clean', action='store_const', const=True, default=False)
    parser.add_argument('-reset', '--reset', action='store_const', const=True, default=False)
    
    parser.add_argument('-valid', '--valid_percent', type=float, default=.15)
    parser.add_argument('-test', '--test_percent', type=float, default=.15)
    
    args = vars(parser.parse_args())

    parent_folder = get_parent_folder()
    valid_percent = args['valid_percent']
    test_percent = args['test_percent']
    
    stats_folder = parent_folder + 'Analysis/Dataset/Data/'
    plot_folder = parent_folder + 'Analysis/Dataset/Plot/'
    
    pathlib.Path(stats_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plot_folder).mkdir(parents=True, exist_ok=True)
    
    with open(parent_folder + 'Feature_map/Dataset/map.pkl', 'rb') as file:
        
        dataset_ids = pickle.load(file)
        
    with open(parent_folder + 'Feature_map/Dataset/patient_map.pkl', 'rb') as file:
        
        dataset_patient_ids = pickle.load(file)

    if args['clean']:
        
        dataset_size = len(dataset_ids)
        threshold = 0.01
        
        corrupted_ids = []        
        corrupted_patient_ids = []        
        corrupted_ages = []      
        corrupted_acquisition_date = []
        corrupted_max_values = []
        corrupted_min_values = []
        
        clean_ids = []        
        clean_patient_ids = []  
        clean_ages = []         
        clean_acquisition_date = []
        clean_max_values = []
        clean_min_values = []
        
        poor_data_quality_ids = load_dataclass_ids(parent_folder, 'poor_data_quality')
        
        subset_size = 100000
        
        for j in range(int(dataset_size / subset_size) + 1):
            
            if j == int(dataset_size / subset_size):
                
                subset_ids = dataset_ids[j*subset_size:]
                
            else:
                
                subset_ids = dataset_ids[j*subset_size:(j+1)*subset_size]

            subset = load_data(get_collection(), subset_ids)
            
            for i, element in enumerate(subset): 

                if i % 1000 == 0:
                    print('Processed ', i, ' data!')

                element_id = element['_id']
                patient_id = element['RestingECG']['PatientDemographics']['PatientID']
                
                try:
                    
                    age = int(element['RestingECG']['PatientDemographics']['PatientAge'])
                    acquisition_date = element['RestingECG']['TestDemographics']['AcquisitionDate']                
                    twelve_leads, _ = extract_twelve_leads(element)
                    
                    if element_id in poor_data_quality_ids or age < 18:
                        
                        clean_element = False
                    
                    else:
                    
                        clean_element = True
            
                    wave_sample = len(twelve_leads[0])
                    
                    max_value, min_value = 0, 0

                    for lead in twelve_leads:
                        
                        new_max_value, new_min_value = np.max(lead), np.min(lead)
                        
                        if new_max_value > max_value:
                            max_value = new_max_value
                            
                        if new_min_value < min_value:
                            min_value = new_min_value
                            
                        if clean_element:

                            if np.max(np.abs(lead[0:int(wave_sample/4)])) < threshold:

                                clean_element = False

                            elif np.max(np.abs(lead[int(3 * wave_sample/4):])) < threshold:

                                clean_element = False
                                
                    if clean_element:
                        clean_ids.append(element_id)
                        clean_patient_ids.append(patient_id)
                        clean_ages.append(age)
                        clean_acquisition_date.append(int(str(acquisition_date)[:4]))
                        clean_max_values.append(max_value)
                        clean_min_values.append(min_value)
                        
                    else:
                        corrupted_ids.append(element_id)
                        corrupted_patient_ids.append(patient_id)
                        corrupted_ages.append(age)
                        corrupted_acquisition_date.append(int(str(acquisition_date)[:4]))
                        corrupted_max_values.append(max_value)
                        corrupted_min_values.append(min_value)
                        
                except:
                    
                    corrupted_ids.append(element_id)
                    corrupted_patient_ids.append(patient_id)
                    
        clean_patient_ids = list(set(clean_patient_ids))         
        corrupted_patient_ids = list(set(corrupted_patient_ids))
        
        save_dataset_clean_ids(parent_folder, clean_ids)
        save_dataset_corrupted_ids(parent_folder, corrupted_ids)
        
        save_dataset_clean_patient_ids(parent_folder, clean_patient_ids)
        save_dataset_corrupted_patient_ids(parent_folder, corrupted_patient_ids)
            
        np.save(stats_folder + 'clean_max_values.npy', np.asarray(clean_max_values))
        np.save(stats_folder + 'clean_min_values.npy', np.asarray(clean_min_values))
        np.save(stats_folder + 'clean_ages.npy', np.asarray(clean_ages))
        np.save(stats_folder + 'clean_acquisition_date.npy', np.asarray(clean_acquisition_date))
            
        np.save(stats_folder + 'corrupted_max_values.npy', np.asarray(corrupted_max_values))
        np.save(stats_folder + 'corrupted_min_values.npy', np.asarray(corrupted_min_values))
        np.save(stats_folder + 'corrupted_ages.npy', np.asarray(corrupted_ages))
        np.save(stats_folder + 'corrupted_acquisition_date.npy', np.asarray(corrupted_acquisition_date))
        
    else:
        
        clean_ids = load_dataset_clean_ids(parent_folder)            
        clean_patient_ids = load_dataset_clean_patient_ids(parent_folder)
        
        corrupted_ids = load_dataset_corrupted_ids(parent_folder)            
        corrupted_patient_ids = load_dataset_corrupted_patient_ids(parent_folder)
        
        clean_max_values = np.load(stats_folder + 'clean_max_values.npy')
        clean_min_values = np.load(stats_folder + 'clean_min_values.npy')         
        clean_ages = np.load(stats_folder + 'clean_ages.npy')
        clean_acquisition_date = np.load(stats_folder + 'clean_acquisition_date.npy')
        
        corrupted_max_values = np.load(stats_folder + 'corrupted_max_values.npy')
        corrupted_min_values = np.load(stats_folder + 'corrupted_min_values.npy')
        corrupted_ages = np.load(stats_folder + 'corrupted_ages.npy')
        corrupted_acquisition_date = np.load(stats_folder + 'corrupted_acquisition_date.npy')
            
    dataset_size = len(dataset_ids)    
    clean_data_size = len(clean_ids)
    corrupted_data_size = len(corrupted_ids)
    
    dataset_patient_size = len(dataset_patient_ids)    
    clean_patient_size = len(clean_patient_ids)
    corrupted_patient_size = len(corrupted_patient_ids)

    print('Number of data:',  dataset_size)
    print('Number of clean data:', clean_data_size)
    print('Number of corrupted data:', corrupted_data_size)

    print()

    print('Number of individuals:',  dataset_patient_size)
    print('Number of clean individuals:', clean_patient_size)
    print('Number of corrupted individuals:', corrupted_patient_size)

    print()
    
    data_values = []
    data_value_labels = []
    
    ages = []
    acquisition_dates = []
    acquisition_date_labels = []
    
    if clean_data_size > 0:

        print('CLEAN DATA STATS')

        print('Max values')
        print('99th percentile: ', np.percentile(clean_max_values, 99))
        print('95th percentile: ', np.percentile(clean_max_values, 95))
        print('75th percentile: ', np.percentile(clean_max_values, 75))
        print('Median: ', np.median(clean_max_values))
        print('Mean: ', np.mean(clean_max_values))

        print('Min values')
        print('1st percentile: ', np.percentile(clean_min_values, 1))
        print('5th percentile: ', np.percentile(clean_min_values, 5))
        print('25th percentile: ', np.percentile(clean_min_values, 25))
        print('Median: ', np.median(clean_min_values))
        print('Mean: ', np.mean(clean_min_values))
    
        data_values += [clean_max_values, clean_min_values]
        data_value_labels += ['Max values (clean)', 'Min value (clean)']
        
        ages += [clean_ages]        
        acquisition_dates += [clean_acquisition_date]        
        acquisition_date_labels += ['Clean data']

        print()
        
    if corrupted_data_size > 0:

        print('CORRUPTED DATA STATS')

        print('Max values')
        print('99th percentile: ', np.percentile(corrupted_max_values, 99))
        print('95th percentile: ', np.percentile(corrupted_max_values, 95))
        print('75th percentile: ', np.percentile(corrupted_max_values, 75))
        print('Median: ', np.median(corrupted_max_values))
        print('Mean: ', np.mean(corrupted_max_values))

        print('Min values')
        print('1st percentile: ', np.percentile(corrupted_min_values, 1))
        print('5th percentile: ', np.percentile(corrupted_min_values, 5))
        print('25th percentile: ', np.percentile(corrupted_min_values, 25))
        print('Median: ', np.median(corrupted_min_values))
        print('Mean: ', np.mean(corrupted_min_values))
    
        data_values += [corrupted_max_values, corrupted_min_values]
        data_value_labels += ['Max values (corrupted)', 'Min value (corrupted)']
        
        ages += [corrupted_ages]         
        acquisition_dates += [corrupted_acquisition_date]        
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
        
    if args['reset']:
        
        clean_patient_ids = []
        
        for j in range(int(clean_data_size / subset_size) + 1):
            
            if j == int(clean_data_size / subset_size):
                
                subset_ids = clean_ids[j*subset_size:]
                
            else:
                
                subset_ids = clean_ids[j*subset_size:(j+1)*subset_size]

            subset = load_data(get_collection(), subset_ids)
            
            for i, element in enumerate(subset): 

                if i % 1000 == 0:
                    print('Processed ', i, ' data!')
                    
                clean_patient_ids.append(element['RestingECG']['PatientDemographics']['PatientID'])
            
        clean_ids = [element_id for _, element_id in sorted(zip(clean_patient_ids, clean_ids))]
        
        clean_patient_ids = sorted(clean_patient_ids)

        valid_size = int(clean_data_size * valid_percent)
        test_size = int(clean_data_size * test_percent)
        
        valid_ids, test_ids = [], []

        test_ids += clean_ids[:test_size]
        test_patient_ids = clean_patient_ids[:test_size]
        
        clean_ids = clean_ids[test_size:]
        clean_patient_ids = clean_patient_ids[test_size:]
        
        while clean_patient_ids[0] == test_patient_ids[-1]:
            
            test_ids.append(clean_ids.pop(0))
            test_patient_ids.append(clean_patient_ids.pop(0))
            
        valid_ids += clean_ids[:valid_size]
        valid_patient_ids = clean_patient_ids[:valid_size]
        
        clean_ids = clean_ids[valid_size:]
        clean_patient_ids = clean_patient_ids[valid_size:]
        
        while clean_patient_ids[0] == valid_patient_ids[-1]:
            
            valid_ids.append(clean_ids.pop(0))
            valid_patient_ids.append(clean_patient_ids.pop(0))
            
        train_ids = clean_ids
        train_patient_ids = clean_patient_ids
        
        train_patient_ids = list(set(train_patient_ids))
        valid_patient_ids = list(set(valid_patient_ids))
        test_patient_ids = list(set(test_patient_ids))
        
        save_dataset_training_ids(parent_folder, train_ids, valid_ids, test_ids)
        save_dataset_training_patient_ids(parent_folder, train_patient_ids, valid_patient_ids, test_patient_ids)
            
    else:
        
        train_ids, valid_ids, test_ids = save_dataset_training_ids(parent_folder)
        train_patient_ids, valid_patient_ids, test_patient_ids = save_dataset_training_patient_ids(parent_folder)

    print('Number of train, valid, test data',
          len(train_ids), len(valid_ids), len(test_ids),
          'whose sum is',  len(train_ids) + len(valid_ids) + len(test_ids))
    
    print()
    
    print('Number of train, valid, test individuals',
          len(train_patient_ids), len(valid_patient_ids), len(test_patient_ids),
          'whose sum is',  len(train_patient_ids) + len(valid_patient_ids) + len(test_patient_ids))
    
    print()