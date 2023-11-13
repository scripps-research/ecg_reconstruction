import pandas as pd
from util_functions.diagnosis_map import diagnosis_map
from classify_functions.extract_demographics import compute_demographic_features
from classify_functions.extract_diagnosis import compute_diagnosis_features


def classify_dataset(data):

    """
    This function makes it possible to associate all the imput elements
    with the clinical and demographic features of the dataset, and compute
    some statistical results regarding the frequency of each feature in 
    the dataset.
    
    
    """

    feature_dataset = []

    diagnosis_labels = list(diagnosis_map.keys()) + ['unknown']
    demographic_labels = ['young', 'old', 'very_old', 'unknown_age', 'male', 'female', 'unknown_gender', 'caucasian', 'non_caucasian', 'unknown_race']

    feature_map = {'Demographic': {label: [] for label in demographic_labels}, 'Diagnosis': {label: [] for label in diagnosis_labels}}
    feature_patient_map = {'Demographic': {label: [] for label in demographic_labels}, 'Diagnosis': {label: [] for label in diagnosis_labels}}

    diagnosis_categories = []

    for key in diagnosis_map.keys():
        diagnosis_categories.append(diagnosis_map[key][0])

    diagnosis_classifier = {}

    for category in diagnosis_categories + ['unknown', 'discarded', 'full']:
        diagnosis_classifier[category] = {'statement': [], 'occurences': [], 'proposed label': []}

    for element in data: 
        
        save_element = True

        try:

            measure_id = element['MeasureID']
            patient_id = element['PatientID']
            raw_diagnosis = element['Diagnosis']
            raw_demographic = element['Demographic']
            
        except:
            
            save_element = False
            
        if save_element:

            diagnosis_features = compute_diagnosis_features(raw_diagnosis, diagnosis_classifier)  

            demographic_features = compute_demographic_features(raw_demographic)  

            feature_dataset.append({'Demographic': demographic_features, 'Diagnosis': diagnosis_features})

            for feature in diagnosis_features:
                feature_map['Diagnosis'][feature].append(measure_id)
                feature_patient_map['Diagnosis'][feature].append(patient_id)
            
            for feature in demographic_features:
                feature_map['Demographic'][feature].append(measure_id)
                feature_patient_map['Demographic'][feature].append(patient_id)

    feature_dataset = pd.DataFrame(feature_dataset)

    feature_labels = {
        'Demographic': demographic_labels,
        'Diagnosis': diagnosis_labels}

    return feature_dataset, feature_map, feature_patient_map, feature_labels, diagnosis_classifier