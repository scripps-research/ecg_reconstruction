from load_functions.load_leads import extract_twelve_leads, load_data
from util_functions.general import get_parent_folder
from util_functions.general import get_collection
from util_functions.load_data_ids import *
import argparse
import pathlib


def clean_dataclass(data_class: str):
    
    """
    This function makes it possible to discern the elements of a given data_class
    between corrupted and cleaned elements, gathering also statistical information
    about the differences (in terms of ECG values, acquisition date, ...)
    between the two groups of elements (corrupted vs cleaned)
    
    The IDs of the corrupted/cleaned elements are stored in the folder ./../Data/Feature_map/Dataclass/data_class/'

    The statistical results are stored in the folder ./../Data/Analysis/Dataclass/data_class/Data/'
    
    """
    
    parent_folder = get_parent_folder()    
    
    class_ids = load_dataclass_ids(parent_folder, data_class)
            
    if data_class == 'poor_data_quality':
        
        class_clean_ids = []                
        class_corrupted_ids = class_ids
        
    else:
    
        dataset_clean_ids = load_dataset_clean_ids(parent_folder)            
        dataset_corrupted_ids = load_dataset_corrupted_ids(parent_folder)
        
        class_clean_ids = list(set(class_ids) & set(dataset_clean_ids))
        class_corrupted_ids = list(set(class_ids) & set(dataset_corrupted_ids))
        
    save_dataclass_clean_ids(parent_folder, data_class, class_clean_ids)
    save_dataclass_corrupted_ids(parent_folder, data_class, class_corrupted_ids)
        
    class_corrupted_patient_ids = []
    class_corrupted_ages = []                    
    class_corrupted_acquisition_date = []
    class_corrupted_max_values = []
    class_corrupted_min_values = []
    
    class_clean_patient_ids = []
    class_clean_ages = []
    class_clean_acquisition_date = []
    class_clean_max_values = []
    class_clean_min_values = []
            
    class_size = len(class_ids)
    
    total_processed_data = 0
    
    subset_size = 100000  
    
    for j in range(int(class_size / subset_size) + 1):
        
        if j == int(class_size / subset_size):
            
            subset_ids = class_ids[j*subset_size:]
            
        else:
            
            subset_ids = class_ids[j*subset_size:(j+1)*subset_size]

        subset = load_data(get_collection(), subset_ids)
        
        for i, element in enumerate(subset): 

            if i % 1000 == 0:
                print('Processed ', total_processed_data, ' data!')
                
            total_processed_data += 1

            element_id = element['_id']      
            patient_id = element['RestingECG']['PatientDemographics']['PatientID']
                
            if element_id in class_clean_ids:
                
                class_clean_patient_ids.append(patient_id) 
                
                age = int(element['RestingECG']['PatientDemographics']['PatientAge'])
                acquisition_date = element['RestingECG']['TestDemographics']['AcquisitionDate']                
                twelve_leads, _ = extract_twelve_leads(element)
                
                class_clean_ages.append(age)          
                class_clean_acquisition_date.append(int(str(acquisition_date)[:4]))
                
                for lead in twelve_leads:
        
                    class_clean_max_values.append(np.max(lead))
                    class_clean_min_values.append(np.min(lead))
                    
            else:
                    
                class_corrupted_patient_ids.append(patient_id)
                
                try:
                    
                    age = int(element['RestingECG']['PatientDemographics']['PatientAge'])
                    acquisition_date = element['RestingECG']['TestDemographics']['AcquisitionDate']             
                    twelve_leads, _ = extract_twelve_leads(element)
                    
                    class_corrupted_ages.append(age) 
                    class_corrupted_acquisition_date.append(int(str(acquisition_date)[:4]))
                
                    for lead in twelve_leads:
    
                        class_corrupted_max_values.append(np.max(lead))
                        class_corrupted_min_values.append(np.min(lead))
                        
                except:
                    
                    pass
                
    class_clean_patient_ids = list(set(class_clean_patient_ids))                    
    class_corrupted_patient_ids = list(set(class_corrupted_patient_ids))
    
    save_dataclass_clean_patient_ids(parent_folder, data_class, class_clean_patient_ids)
    save_dataclass_corrupted_patient_ids(parent_folder, data_class, class_corrupted_patient_ids)
    
    stats_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Data/'
    pathlib.Path(stats_folder).mkdir(parents=True, exist_ok=True)
    
    np.save(stats_folder + 'clean_ages.npy', np.asarray(class_clean_ages))
    np.save(stats_folder + 'clean_acquisition_date.npy', np.asarray(class_clean_acquisition_date))
    np.save(stats_folder + 'clean_max_values.npy', np.asarray(class_clean_max_values))
    np.save(stats_folder + 'clean_min_values.npy', np.asarray(class_clean_min_values))            
    
    np.save(stats_folder + 'corrupted_ages.npy', np.asarray(class_corrupted_ages))  
    np.save(stats_folder + 'corrupted_acquisition_date.npy', np.asarray(class_corrupted_acquisition_date))
    np.save(stats_folder + 'corrupted_max_values.npy', np.asarray(class_corrupted_max_values))
    np.save(stats_folder + 'corrupted_min_values.npy', np.asarray(class_corrupted_min_values))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_class', '--data_class', type=str)

    args = vars(parser.parse_args())

    data_class = args['data_class']
    
    clean_dataclass(data_class)