import argparse
import pathlib

from util_functions.diagnosis_hierarchy import diagnosis_hierarchy, diagnosis_subtraction
from process_dataclass import process_dataclass
from load_functions.load_leads import load_data
from util_functions.general import get_collection, get_parent_folder
from util_functions.load_data_ids import load_dataclass_ids, load_dataclass_patient_ids, save_dataclass_ids, save_dataclass_patient_ids, load_dataset_ids, load_dataset_patient_ids

collection = get_collection()

parser = argparse.ArgumentParser()

parser.add_argument('-data_class', '--data_class', type=str, default=None)
parser.add_argument('-clean', '--clean', action='store_const', const=True, default=False)
parser.add_argument('-analyze', '--analyze', action='store_const', const=True, default=False)
parser.add_argument('-reset', '--reset', action='store_const', const=True, default=False)
parser.add_argument('-clean_input', '--clean_input', action='store_const', const=True, default=False)
parser.add_argument('-analyze_input', '--analyze_input', action='store_const', const=True, default=False)
parser.add_argument('-reset_input', '--reset_input', action='store_const', const=True, default=False)

"""

This script makes it possible to define a new data_class from those already appearing
in the dataset. The new dataclass could be defined as the union or intersection of multiple
clinical labels, according to what is defined by the object "diagnosis_hierarchy" and "diagnosis_substraction".

The IDs of the class elements are stored in the folder ./../Data/Feature_map/Dataclass/data_class/'

The statistical results are stored in the folder ./../Data/Analysis/Dataclass/data_class/Data/'

"""

args = vars(parser.parse_args())

data_class = args['data_class']
clean = args['clean']
analyze = args['analyze']
reset = args['reset']

clean_input = args['clean_input']
analyze_input = args['analyze_input']
reset_input = args['reset_input']

parent_folder = get_parent_folder()

diagnosis_map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'
diagnosis_stats_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Data/'
    
if data_class in diagnosis_hierarchy.keys():
    
    input_data_classes = diagnosis_hierarchy[data_class]
    subtract_data_classes = []
    
    unite = True
    subtract = False
    
elif data_class in diagnosis_subtraction.keys():
    
    input_data_classes = diagnosis_subtraction[data_class][0]
    subtract_data_classes = diagnosis_subtraction[data_class][1]
    
    unite = False
    subtract = True
    

else:
    raise ValueError

pathlib.Path(diagnosis_map_folder).mkdir(parents=True, exist_ok=True)
pathlib.Path(diagnosis_stats_folder).mkdir(parents=True, exist_ok=True)

data_ids = []
patient_ids = []

subtract_data_ids = []

for input_data_class in input_data_classes:
    
    print('Input class: ', input_data_class)
    
    print()
    
    if input_data_class == 'dataset':
        
        new_data_ids = load_dataset_ids(parent_folder)
        new_patient_ids = load_dataset_patient_ids(parent_folder)
        
        data_ids.append(new_data_ids)
        patient_ids.append(new_patient_ids)
        
    else:
    
        new_data_ids = load_dataclass_ids(parent_folder, input_data_class)
        new_patient_ids = load_dataclass_patient_ids(parent_folder, input_data_class)
        
        data_ids.append(new_data_ids)
        patient_ids.append(new_patient_ids)
    
        if clean_input or reset_input:
            
            print('Process input class...')
            
            print()
            
            print(clean_input)
            
            process_dataclass(input_data_class, clean_input, analyze_input, reset_input)

if unite:

    data_ids = list(set.union(*[set(x) for x in data_ids]))
    patient_ids = list(set.union(*[set(x) for x in patient_ids]))
        
elif subtract:
    
    for subtract_data_class in subtract_data_classes:
        
        new_data_ids = load_dataclass_ids(parent_folder, subtract_data_class)
        
        subtract_data_ids.append(new_data_ids)
    
    data_ids = set.union(*[set(x) for x in data_ids])
    subtract_data_ids = set.union(*[set(x) for x in subtract_data_ids])
    
    data_ids = list(data_ids - subtract_data_ids)
    
    data_size = len(data_ids)
    subset_size = 100000
    
    patient_ids = []
    
    for j in range(int(data_size / subset_size) + 1):
            
        if j == int(data_size / subset_size):
            
            subset_ids = data_ids[j*subset_size:]
            
        else:
            
            subset_ids = data_ids[j*subset_size:(j+1)*subset_size]

        subset = load_data(get_collection(), subset_ids)
        
        for i, element in enumerate(subset): 

            if i % 1000 == 0:
                print('Processed ', i, ' data!')
                
            patient_ids.append(element['RestingECG']['PatientDemographics']['PatientID'])
            
    patient_ids = list(set(patient_ids))    
    
else:
    
    raise ValueError

print('Save output class maps...')

print('Number of class data:', len(data_ids))

print('Number of class individuals:', len(patient_ids))

print()

save_dataclass_ids(parent_folder, data_class, data_ids)
save_dataclass_patient_ids(parent_folder, data_class, patient_ids)
        
if clean or reset or analyze:
    
    print('Process output class...')

    print()

    process_dataclass(data_class, clean, analyze, reset)
