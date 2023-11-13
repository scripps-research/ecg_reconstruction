from util_functions.general import get_parent_folder
import pickle
import numpy as np
import pathlib


def compute_feature_map(labels, data, patient_data, folder, save_total=False):
    
    """
    This function makes it possible to compute the list of element IDs
    associated with each of the labels provided in input
    
    """
    
    total_map = []
    total_patient_map = []

    for label in labels:

        label_map = list(set(data[label]))
        label_patient_map = list(set(patient_data[label]))
        
        if save_total:
            total_map += label_map
            total_patient_map += label_patient_map
        
        label_folder = folder + str(label).replace(' ', '_') + '/'

        pathlib.Path(label_folder).mkdir(parents=True, exist_ok=True)

        with open(label_folder + 'map.pkl', 'wb') as file:
    
            pickle.dump(label_map, file)
        
        with open(label_folder + 'patient_map.pkl', 'wb') as file: 
                
            pickle.dump(label_patient_map, file, pickle.HIGHEST_PROTOCOL)
            
    if save_total:
        
        total_map = list(set(total_map))
            
        with open(get_parent_folder() + 'Feature_map/Dataset/map.pkl', 'wb') as file:
        
            pickle.dump(total_map, file)
        
        total_patient_map = list(set(total_patient_map))
        
        with open(get_parent_folder() + 'Feature_map/Dataset/patient_map.pkl', 'wb') as file: 
                
            pickle.dump(total_patient_map, file, pickle.HIGHEST_PROTOCOL)


def compute_feature_matrix(labels, data, folder):
    
    """
    This function makes it possible to compute a binary matrix
    associating each element of the dataset with each of the labels provided in input
    
    """

    data_size, label_size = len(data), len(labels)

    matrix = np.zeros((data_size, label_size))

    for i in range(data_size):
        for j in range(label_size):

            label = labels[j]

            if label in data[i]:

                matrix[i, j] = 1

    np.save(folder + '_matrix.npy', matrix)
    np.save(folder + '_labels.npy', labels)