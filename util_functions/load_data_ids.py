import pickle
import copy
import numpy as np


def load_dataset_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataset_clean_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'clean_map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataset_corrupted_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'corrupted_map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataset_patient_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def load_dataset_clean_patient_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'clean_patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def load_dataset_corrupted_patient_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'corrupted_patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def save_dataset_ids(parent_folder: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'map.pkl', 'wb') as file:
        
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataset_clean_ids(parent_folder: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'clean_map.pkl', 'wb') as file:
        
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataset_corrupted_ids(parent_folder: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'corrupted_map.pkl', 'wb') as file: 
        
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataset_patient_ids(parent_folder: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'patient_map.pkl', 'wb') as file: 
        
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataset_clean_patient_ids(parent_folder: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'clean_patient_map.pkl', 'wb') as file: 
    
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataset_corrupted_patient_ids(parent_folder: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'

    with open(map_folder + 'corrupted_patient_map.pkl', 'wb') as file: 
    
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)


# DATACLASS


def load_dataclass_ids(parent_folder: str, data_class: str):

    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataclass_clean_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'clean_map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataclass_corrupted_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'corrupted_map.pkl', 'rb') as file: 
    
        data_ids = pickle.load(file)
    
    return data_ids

def load_dataclass_patient_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def load_dataclass_clean_patient_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'clean_patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def load_dataclass_corrupted_patient_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'corrupted_patient_map.pkl', 'rb') as file: 
    
        patient_ids = pickle.load(file)
    
    return patient_ids

def save_dataclass_ids(parent_folder: str, data_class: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'map.pkl', 'wb') as file: 
    
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)
        
def save_dataclass_clean_ids(parent_folder: str, data_class: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'clean_map.pkl', 'wb') as file: 
    
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataclass_corrupted_ids(parent_folder: str, data_class: str, data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'corrupted_map.pkl', 'wb') as file: 
    
        pickle.dump(data_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataclass_patient_ids(parent_folder: str, data_class: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'patient_map.pkl', 'wb') as file: 
    
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataclass_clean_patient_ids(parent_folder: str, data_class: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'clean_patient_map.pkl', 'wb') as file: 
    
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)

def save_dataclass_corrupted_patient_ids(parent_folder: str, data_class: str, patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'

    with open(map_folder + 'corrupted_patient_map.pkl', 'wb') as file: 
    
        pickle.dump(patient_ids, file, pickle.HIGHEST_PROTOCOL)


# TRAINING DATA


def load_dataset_training_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'
    
    with open(map_folder + 'train_map.pkl', 'rb') as file: 
            
        train_data_ids = pickle.load(file)

    with open(map_folder + 'valid_map.pkl', 'rb') as file: 
            
        valid_data_ids = pickle.load(file)
            
    with open(map_folder + 'test_map.pkl', 'rb') as file:
            
        test_data_ids = pickle.load(file)
        
    return train_data_ids, valid_data_ids, test_data_ids

def save_dataset_training_ids(parent_folder: str, train_data_ids: list, valid_data_ids: list, test_data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'
    
    with open(map_folder + 'train_map.pkl', 'wb') as file: 
            
        pickle.dump(train_data_ids, file, pickle.HIGHEST_PROTOCOL)

    with open(map_folder + 'valid_map.pkl', 'wb') as file: 
            
        pickle.dump(valid_data_ids, file, pickle.HIGHEST_PROTOCOL)
            
    with open(map_folder + 'test_map.pkl', 'wb') as file:
            
        pickle.dump(test_data_ids, file, pickle.HIGHEST_PROTOCOL)

def load_dataset_training_patient_ids(parent_folder: str):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'
    
    with open(map_folder + 'train_patient_map.pkl', 'rb') as file: 
            
        train_patient_ids = pickle.load(file)

    with open(map_folder + 'valid_patient_map.pkl', 'rb') as file: 
            
        valid_patient_ids = pickle.load(file)
            
    with open(map_folder + 'test_patient_map.pkl', 'rb') as file:
            
        test_patient_ids = pickle.load(file)
        
    return train_patient_ids, valid_patient_ids, test_patient_ids

def save_dataset_training_patient_ids(parent_folder: str, train_patient_ids: list, valid_patient_ids: list, test_patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataset/'
    
    with open(map_folder + 'train_patient_map.pkl', 'wb') as file:
        
        pickle.dump(train_patient_ids, file, pickle.HIGHEST_PROTOCOL)

    with open(map_folder + 'valid_patient_map.pkl', 'wb') as file:
        
        pickle.dump(valid_patient_ids, file, pickle.HIGHEST_PROTOCOL)
            
    with open(map_folder + 'test_patient_map.pkl', 'wb') as file:
        
        pickle.dump(test_patient_ids, file, pickle.HIGHEST_PROTOCOL)

def load_class_training_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'
    
    with open(map_folder + 'train_map.pkl', 'rb') as file: 
            
        train_data_ids = pickle.load(file)

    with open(map_folder + 'valid_map.pkl', 'rb') as file: 
            
        valid_data_ids = pickle.load(file)
            
    with open(map_folder + 'test_map.pkl', 'rb') as file:
            
        test_data_ids = pickle.load(file)
        
    return train_data_ids, valid_data_ids, test_data_ids

def save_dataclass_training_ids(parent_folder: str, data_class: str, train_data_ids: list, valid_data_ids: list, test_data_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'
    
    with open(map_folder + 'train_map.pkl', 'wb') as file: 
            
        pickle.dump(train_data_ids, file, pickle.HIGHEST_PROTOCOL)

    with open(map_folder + 'valid_map.pkl', 'wb') as file: 
            
        pickle.dump(valid_data_ids, file, pickle.HIGHEST_PROTOCOL)
            
    with open(map_folder + 'test_map.pkl', 'wb') as file:
            
        pickle.dump(test_data_ids, file, pickle.HIGHEST_PROTOCOL)

def load_class_training_patient_ids(parent_folder: str, data_class: str):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'
    
    with open(map_folder + 'train_patient_map.pkl', 'rb') as file: 
            
        train_patient_ids = pickle.load(file)

    with open(map_folder + 'valid_patient_map.pkl', 'rb') as file: 
            
        valid_patient_ids = pickle.load(file)
            
    with open(map_folder + 'test_patient_map.pkl', 'rb') as file:
            
        test_patient_ids = pickle.load(file)
        
    return train_patient_ids, valid_patient_ids, test_patient_ids

def save_dataclass_training_patient_ids(parent_folder: str, data_class: str, train_patient_ids: list, valid_patient_ids: list, test_patient_ids: list):
    
    map_folder = parent_folder + 'Feature_map/Dataclass/' + data_class + '/'
    
    with open(map_folder + 'train_patient_map.pkl', 'wb') as file:
        
        pickle.dump(train_patient_ids, file, pickle.HIGHEST_PROTOCOL)

    with open(map_folder + 'valid_patient_map.pkl', 'wb') as file:
        
        pickle.dump(valid_patient_ids, file, pickle.HIGHEST_PROTOCOL)
            
    with open(map_folder + 'test_patient_map.pkl', 'wb') as file:
        
        pickle.dump(test_patient_ids, file, pickle.HIGHEST_PROTOCOL)


def reset_dataclass_training_ids(parent_folder: str, data_class: str):
        
    class_clean_ids = load_dataclass_clean_ids(parent_folder, data_class)
    class_clean_patient_ids = load_dataclass_clean_patient_ids(parent_folder, data_class)
    
    dataset_train_ids, dataset_valid_ids, dataset_test_ids = load_dataset_training_ids(parent_folder)
    dataset_train_patient_ids, dataset_valid_patient_ids, dataset_test_patient_ids = load_dataset_training_patient_ids(parent_folder)

    class_train_ids = list(set(class_clean_ids) & set(dataset_train_ids))
    class_valid_ids = list(set(class_clean_ids) & set(dataset_valid_ids))
    class_test_ids = list(set(class_clean_ids) & set(dataset_test_ids))
    
    class_train_patient_ids = list(set(class_clean_patient_ids) & set(dataset_train_patient_ids))
    class_valid_patient_ids = list(set(class_clean_patient_ids) & set(dataset_valid_patient_ids))
    class_test_patient_ids = list(set(class_clean_patient_ids) & set(dataset_test_patient_ids))
    
    save_dataclass_training_ids(parent_folder, data_class, class_train_ids, class_valid_ids, class_test_ids)
    save_dataclass_training_patient_ids(parent_folder, data_class, class_train_patient_ids, class_valid_patient_ids, class_test_patient_ids)
    

def load_learning_ids(parent_folder: str, data_classes, data_size: str, valid_percent=0.15, test_percent=0.15):
    
    data_class_num = len(data_classes)
    
    train_ids_per_class = []
    valid_ids_per_class = []
    test_ids_per_class = []
    
    for data_class in data_classes:
        
        if data_class == 'other':        
            class_train_ids, class_valid_ids, class_test_ids = load_dataset_training_ids(parent_folder)
        else:
            class_train_ids, class_valid_ids, class_test_ids = load_class_training_ids(parent_folder, data_class)
          
        train_ids_per_class.append(class_train_ids)            
        valid_ids_per_class.append(class_valid_ids)            
        test_ids_per_class.append(class_test_ids)
    
    if 'other' in data_classes:
        
        dataset_index = data_classes.index('other')
        
        dataset_train_ids = set(copy.copy(train_ids_per_class[dataset_index]))
        dataset_valid_ids = set(copy.copy(valid_ids_per_class[dataset_index]))
        dataset_test_ids = set(copy.copy(test_ids_per_class[dataset_index]))
        
        for class_index in range(data_class_num):
            
            if class_index != dataset_index:
                
                class_train_ids = set(train_ids_per_class[class_index])
                class_valid_ids = set(valid_ids_per_class[class_index])
                class_test_ids = set(test_ids_per_class[class_index])
                
                dataset_train_ids = dataset_train_ids - set(class_train_ids)
                dataset_valid_ids = dataset_valid_ids - set(class_valid_ids)
                dataset_test_ids = dataset_test_ids - set(class_test_ids)
                
        train_ids_per_class[dataset_index] = list(dataset_train_ids)
        valid_ids_per_class[dataset_index] = list(dataset_valid_ids)
        test_ids_per_class[dataset_index] = list(dataset_test_ids)
        
    train_size_per_class = [len(class_train_ids) for class_train_ids in train_ids_per_class]
    valid_size_per_class = [len(class_valid_ids) for class_valid_ids in valid_ids_per_class]
    test_size_per_class = [len(class_test_ids) for class_test_ids in test_ids_per_class]
        
    train_ids = []
    valid_ids = []
    test_ids = []
    
    train_ids_per_class = [x for _, x in sorted(zip(train_size_per_class, train_ids_per_class))]
    train_classes = [x for _, x in sorted(zip(train_size_per_class, data_classes))]
    valid_ids_per_class = [x for _, x in sorted(zip(valid_size_per_class, valid_ids_per_class))]
    valid_classes = [x for _, x in sorted(zip(valid_size_per_class, data_classes))]
    test_ids_per_class = [x for _, x in sorted(zip(test_size_per_class, test_ids_per_class))]
    test_classes = [x for _, x in sorted(zip(test_size_per_class, data_classes))]
    
    if data_size != 'max':
        
        data_size = int(data_size)
        
        class_valid_size = int(valid_percent * data_size / data_class_num)
        class_test_size = int(test_percent * data_size / data_class_num)   
        class_train_size = int(data_size / data_class_num)  - (class_valid_size + class_test_size)
        
    else:
        
        class_train_size = np.min(train_size_per_class)
        class_valid_size = np.min(valid_size_per_class)
        class_test_size = np.min(test_size_per_class)
        
    for class_index, data_class in enumerate(train_classes):
        
        train_size = len(train_ids)
        
        actual_class_train_size = len(list(set(train_ids_per_class[class_index]) & set(train_ids)))  
        
        train_ids += train_ids_per_class[class_index]
        
        train_ids = list(dict.fromkeys(train_ids))[:train_size+class_train_size-actual_class_train_size]
        
    for class_index, data_class in enumerate(valid_classes):
        
        valid_size = len(valid_ids)
        
        actual_class_valid_size = len(list(set(valid_ids_per_class[class_index]) & set(valid_ids)))  
        
        valid_ids += valid_ids_per_class[class_index]
        
        valid_ids = list(dict.fromkeys(valid_ids))[:valid_size+class_valid_size-actual_class_valid_size]
        
    for class_index, data_class in enumerate(test_classes):
        
        test_size = len(test_ids)
        
        actual_class_test_size = len(list(set(test_ids_per_class[class_index]) & set(test_ids)))  
        
        test_ids += test_ids_per_class[class_index]
        
        test_ids = list(dict.fromkeys(test_ids))[:test_size+class_test_size-actual_class_test_size]
    
    return train_ids, valid_ids, test_ids
