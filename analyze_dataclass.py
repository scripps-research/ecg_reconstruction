import numpy as np
import matplotlib.pyplot as plt
import pathlib
import argparse
from explore_functions.compute_statistics import sort_data_matrix, compute_conditional_probability
from util_functions.load_data_ids import load_dataclass_ids
from util_functions.general import get_parent_folder
from util_functions.diagnosis_hierarchy import diagnosis_hierarchy


def analyze_dataclass(data_class: str, var_size = 30, analyze=False):
    
    """
    This function makes it possible to compute the conditional probability
    for the elements of a given data_class to be associated with the different
    clinical labels used to describe the dataset
    
    The statistical results are stored in the folder ./../Data/Analysis/Dataclass/data_class/Data/
    
    The figures describing the figures are stored in the folder ./../Data/Analysis/Dataclass/data_class/Plot/
    
    """

    parent_folder = get_parent_folder()

    analysis_data_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Data/'        
    analysis_plot_folder = parent_folder + 'Analysis/Dataclass/' + data_class + '/Plot/'
    
    pathlib.Path(analysis_data_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(analysis_plot_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        
        diagnosis_labels = np.load(analysis_data_folder + 'diagnosis_labels.npy')   
        conditional_probability = np.load(analysis_data_folder + 'diagnosis_conditional_prob.npy')
        
    except:
        
        analyze = True
            
    if analyze:

        if data_class not in diagnosis_labels:
            
            exploration_data_folder = parent_folder + 'Exploration/Data/'
            
            diagnosis_labels = np.load(exploration_data_folder + 'diagnosis_labels.npy')
            diagnosis_matrix = np.load(exploration_data_folder + 'diagnosis_matrix.npy')
            
            input_indexes = [np.argwhere(diagnosis_labels == input_data_class)[0][0] for input_data_class in diagnosis_hierarchy[data_class]]
            
            new_column = np.zeros(len(diagnosis_matrix))
            
            for index in input_indexes:
                
                new_column += diagnosis_matrix[:, index]
                
            new_column[np.where(new_column > 0)] = 1
            
            diagnosis_matrix = np.append(diagnosis_matrix, new_column.reshape((-1, 1)), axis=1)
            
            diagnosis_labels = np.append(diagnosis_labels, data_class)
            
            print('Generating conditional probabilities...')
            
            print()
            
            diagnosis_matrix, diagnosis_labels = sort_data_matrix(diagnosis_matrix, diagnosis_labels)        
            conditional_probability = compute_conditional_probability(diagnosis_matrix, len(diagnosis_matrix), len(diagnosis_labels))  
            
            print()
            
        else:
            
            exploration_process_folder = parent_folder + 'Exploration/Process/'
            
            diagnosis_labels = np.load(exploration_process_folder + 'diagnosis_labels.npy')
            conditional_probability = np.load(exploration_process_folder + 'diagnosis_conditional_prob.npy')
            
        np.save(analysis_data_folder + 'diagnosis_labels.npy', diagnosis_labels)   
        np.save(analysis_data_folder + 'diagnosis_conditional_prob.npy', conditional_probability)        
    
    var_index = np.argwhere(np.asarray(diagnosis_labels) == data_class)[0][0]
    conditional_probability = conditional_probability[var_index].reshape(len(conditional_probability))
    conditional_probability = np.delete(conditional_probability, var_index)
    diagnosis_labels = np.delete(diagnosis_labels, var_index)

    diagnosis_labels = [diagnosis_labels[idx] for idx in (-conditional_probability).argsort()]

    conditional_probability[::-1].sort()
        
    variable_num = var_size
    conditional_probability = conditional_probability[:variable_num]
    diagnosis_labels = diagnosis_labels[:variable_num]

    _, ax = plt.subplots()

    ax.bar(np.arange(variable_num), conditional_probability)

    ax.set_xticks(np.arange(variable_num))
    ax.set_xticklabels(diagnosis_labels, rotation=90)

    ax.set_yticks(np.linspace(0,1,11)) 

    ax.set_ylim([0,1])

    ax.set_ylabel('Conditional probability')
    ax.set_xlabel('Diagnosis')

    ax.grid()

    plt.savefig(analysis_plot_folder + 'conditional_probability.png', format='png', bbox_inches='tight')

    plt.cla()
    plt.clf()
        
    data_size = len(load_dataclass_ids(parent_folder, data_class))

    _, ax = plt.subplots()

    ax.bar(np.arange(variable_num), conditional_probability * data_size)

    ax.set_xticks(np.arange(variable_num))
    ax.set_xticklabels(diagnosis_labels, rotation=90)

    ax.set_yticks(np.linspace(0,data_size,11)) 

    ax.set_ylim([0,data_size])

    ax.set_ylabel('Subclass sizes')
    ax.set_xlabel('Diagnosis')

    ax.grid()

    plt.savefig(analysis_plot_folder + 'subclass_sizes.png', format='png', bbox_inches='tight')

    plt.cla()
    plt.clf()
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_class', '--data_class', type=str)
    parser.add_argument('-var_size', '--var_size', type=int, default=30)

    args = vars(parser.parse_args())

    data_class = args['data_class']
    
    var_size = args['var_size']
    
    analyze_dataclass(data_class, var_size)