from explore_functions.distribution_plot import distribution_plot
from explore_functions.compute_statistics import compute_statistics
from classify_functions.classify_dataset import classify_dataset
from explore_functions.compute_features import compute_feature_matrix, compute_feature_map
from util_functions.general import get_collection, get_parent_folder
import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib
import argparse

"""   

This script carries out an explorative analysis on the overall dataset.
Specifically, it makes it possible to (i) load the elements of the dataset, 
(ii) associate the elements with the clinical labels describing them,
(iii) compute some statistical information regarding each label of the dataset,
including the correlation of each couple of labels, (iv) plot the results
regarding the statistical information

The IDs of the elements associated with each clinical label are stored in the folder ./../Data/Feature_map/'

The statistical results are stored in the folder ./../Data/Exploration/'

"""


collection = get_collection()

parser = argparse.ArgumentParser()

parser.add_argument('-classify', '--classify', action='store_const', const=True, default=False)
parser.add_argument('-process', '--process', action='store_const', const=True, default=False)
parser.add_argument('-plot', '--plot', action='store_const', const=True, default=False)
parser.add_argument('-data_size', '--data_size', type=int, default=-1)
parser.add_argument('-variable_size', '--variable_size', type=int, default=20)

args = vars(parser.parse_args())

classify_data = args['classify']
process_data = args['process']
plot_data = args['plot']
data_size = args['data_size']
variable_size = args['variable_size']

parent_folder = get_parent_folder()

feature_map_folder = parent_folder + 'Feature_map/'
exploration_folder = parent_folder + 'Exploration/' 

pathlib.Path(feature_map_folder).mkdir(parents=True, exist_ok=True)

pathlib.Path(exploration_folder + 'Data/').mkdir(parents=True, exist_ok=True)
pathlib.Path(exploration_folder + 'Process/').mkdir(parents=True, exist_ok=True)
pathlib.Path(exploration_folder + 'Plot/').mkdir(parents=True, exist_ok=True)

filter = {
    'RestingECG.TestDemographics.DataType': 1,
    'RestingECG.TestDemographics.Priority': 1,
    'RestingECG.TestDemographics.TestReason': 1,
    'RestingECG.Order.ReasonForTest': 1,
    'RestingECG.PatientDemographics': 1,
    'RestingECG.Diagnosis': 1,
    'RestingECG.OriginalDiagnosis': 1
}

if classify_data:

    print('Data classification...')

    if data_size == -1:
        dataset = tqdm(collection.find({}, filter))
    else:
        dataset = tqdm(collection.find({}, filter, limit=data_size))

    dataset, feature_map, feature_patient_map, feature_labels, diagnosis_classifier = classify_dataset(dataset)

    diagnosis_data = dataset['Diagnosis']
    demographic_data = dataset['Demographic']

    diagnosis_map = feature_map['Diagnosis']
    diagnosis_patient_map = feature_patient_map['Diagnosis']
    
    demographic_map = feature_map['Demographic']
    demographic_patient_map = feature_patient_map['Demographic']

    diagnosis_labels = feature_labels['Diagnosis']
    demographic_labels = feature_labels['Demographic']

    compute_feature_map(diagnosis_labels, diagnosis_map, diagnosis_patient_map, feature_map_folder + 'Dataclass/', save_total=True)
    compute_feature_map(demographic_labels, demographic_map, demographic_patient_map, feature_map_folder + 'Dataclass/')

    for key in diagnosis_classifier.keys():

        str_key = key.replace(' ', '_')
    
        pd.DataFrame.from_dict(data=diagnosis_classifier[key]).to_csv(feature_map_folder + str_key + '_classification.csv', header=True, index=False)

    compute_feature_matrix(diagnosis_labels, diagnosis_data, exploration_folder + 'Data/diagnosis',)
    compute_feature_matrix(demographic_labels, demographic_data, exploration_folder + 'Data/demographic')

if process_data:

    print('Data processing...')

    diagnosis_matrix = np.load(exploration_folder + 'Data/diagnosis_matrix.npy')
    diagnosis_labels = np.load(exploration_folder + 'Data/diagnosis_labels.npy')

    variables, variable_means, covariance, correlation, eigenvalues, eigenvectors, conditional_probability = compute_statistics(diagnosis_matrix, diagnosis_labels, sort=True)

    np.save(exploration_folder + 'Process/diagnosis_labels.npy', variables)
    np.save(exploration_folder + 'Process/diagnosis_means.npy', variable_means)
    np.save(exploration_folder + 'Process/diagnosis_covariance.npy', covariance)
    np.save(exploration_folder + 'Process/diagnosis_correlation.npy', correlation)
    np.save(exploration_folder + 'Process/diagnosis_eigenvalues.npy', eigenvalues)
    np.save(exploration_folder + 'Process/diagnosis_eigenvectors.npy', eigenvectors)
    np.save(exploration_folder + 'Process/diagnosis_conditional_prob.npy', conditional_probability)

    demographic_matrix = np.load(exploration_folder + 'Data/demographic_matrix.npy')
    demographic_labels = np.load(exploration_folder + 'Data/demographic_labels.npy')

    variables, variable_means, covariance, correlation, eigenvalues, eigenvectors, conditional_probability = compute_statistics(demographic_matrix, demographic_labels)

    np.save(exploration_folder + 'Process/demographic_labels.npy', variables)
    np.save(exploration_folder + 'Process/demographic_means.npy', variable_means)
    np.save(exploration_folder + 'Process/demographic_covariance.npy', covariance)
    np.save(exploration_folder + 'Process/demographic_correlation.npy', correlation)
    np.save(exploration_folder + 'Process/demographic_eigenvalues.npy', eigenvalues)
    np.save(exploration_folder + 'Process/demographic_eigenvectors.npy', eigenvectors)
    np.save(exploration_folder + 'Process/demographic_conditional_prob.npy', conditional_probability)

if plot_data:

    print('Data plotting...')

    variables = np.load(exploration_folder + 'Process/diagnosis_labels.npy')
    variable_means = np.load(exploration_folder + 'Process/diagnosis_means.npy')
    covariance = np.load(exploration_folder + 'Process/diagnosis_covariance.npy')
    correlation = np.load(exploration_folder + 'Process/diagnosis_correlation.npy')
    eigenvalues = np.load(exploration_folder + 'Process/diagnosis_eigenvalues.npy')
    eigenvectors = np.load(exploration_folder + 'Process/diagnosis_eigenvectors.npy')
    conditional_probability = np.load(exploration_folder + 'Process/diagnosis_conditional_prob.npy')

    distribution_plot(
        variables,
        len(variables),
        variable_size,
        variable_means,
        covariance,
        correlation,
        eigenvalues,
        eigenvectors,   
        conditional_probability,         
        'Diagnosis',
        exploration_folder + 'Plot/diagnosis')

    variables = np.load(exploration_folder + 'Process/demographic_labels.npy')
    variable_means = np.load(exploration_folder + 'Process/demographic_means.npy')
    covariance = np.load(exploration_folder + 'Process/demographic_covariance.npy')
    correlation = np.load(exploration_folder + 'Process/demographic_correlation.npy')
    eigenvalues = np.load(exploration_folder + 'Process/demographic_eigenvalues.npy')
    eigenvectors = np.load(exploration_folder + 'Process/demographic_eigenvectors.npy')
    conditional_probability = np.load(exploration_folder + 'Process/demographic_conditional_prob.npy')

    distribution_plot(
        variables,
        len(variables),
        variable_size,
        variable_means,
        covariance,
        correlation,
        eigenvalues,
        eigenvectors,
        conditional_probability,            
        'Demographic',
        exploration_folder + 'Plot/demographic')
