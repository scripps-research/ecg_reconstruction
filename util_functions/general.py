import json
import shutil
import pymongo


def get_parent_folder():
    
    parent_folder = "./../Data/"
    
    return parent_folder


def remove_dir(folder: str):
    
    try:
        shutil.rmtree(folder)        
    except:
        pass


def get_collection(database_params_file=None):
    
    if database_params_file is None:
        
        raise ValueError('Database parameters are not defined')
    
    database_params = json.load(open(database_params_file, 'r'))
    client = pymongo.MongoClient(
        database_params['host'], 
        database_params['port'], 
        username=database_params['username'], 
        password=database_params['password'])
    database = client[database_params['collection']]
    collection = database[database_params['collection']]

    return collection


def get_twelve_keys():

    return ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_lead_keys(leads: str):
    
    if leads == 'limb':
    
        keys = ['I', 'II']
        
    elif leads == 'limb+comb(v3+v4)':
        
        keys = ['I', 'II', ['V3', 'V4']]
        
    elif leads == 'limb+v2+v4':
        
        keys = ['I', 'II', 'V2', 'V4']
        
    elif leads == 'full_limb':
        
        keys = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF']
        
    elif leads == 'limb+v1':
        
        keys = ['I', 'II', 'V1']
        
    elif leads == 'limb+v2':
        
        keys = ['I', 'II', 'V2']
        
    elif leads == 'limb+v3':
        
        keys = ['I', 'II', 'V3']
        
    elif leads == 'limb+v4':
        
        keys = ['I', 'II', 'V4']
        
    elif leads == 'limb+v5':
        
        keys = ['I', 'II', 'V5']
        
    elif leads == 'limb+v6':
        
        keys = ['I', 'II', 'V6']
        
    elif leads == 'precordial':
        
        keys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    elif leads == 'full':
    
        keys = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    else:
        raise ValueError

    return keys


def get_data_classes(dataset: str):
    
    if dataset == 'infarct+other':
        data_classes = ['st_elevation_or_infarct', 'other']
    elif dataset == 'infarct+noninfarct':
        data_classes = ['st_elevation_or_infarct', 'non_st_elevation_or_infarct']
    else:
        data_classes = [None]
    
    return data_classes


def get_detect_classes(detect_class: str):
    
    detect_classes = [detect_class]
    
    return detect_classes


def get_value_range():
        
    min_value = -2.5
    amplitude = 5.0
    wave_sample = 2500

    return min_value, amplitude, wave_sample