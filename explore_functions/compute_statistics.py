import numpy as np


def sort_data_matrix(data_matrix, variables):
    
    variable_means = np.mean(data_matrix, axis=0)
    data_matrix = data_matrix[:, (-variable_means).argsort()]
    
    variables = np.asarray([variables[idx] for idx in (-variable_means).argsort()])  
    
    return data_matrix, variables


def compute_mean_and_std(data_matrix):
    
    data_matrix_2 = data_matrix ** 2    
    variable_means = np.mean(data_matrix, axis=0)
    data_matrix_2_means = np.mean(data_matrix_2, axis=0)
    standard_deviations = np.sqrt(data_matrix_2_means- variable_means ** 2)
    
    return variable_means, standard_deviations


def compute_covariance_matrix(data_matrix, sample_num: int, variable_num: int, variable_means):
    
    covariance_matrix = np.zeros((variable_num, variable_num))

    for k in range(sample_num):
        
        if k % 100000 == 0:
            print("Processing element ", k)

        sample = data_matrix[k]

        for i in range(variable_num):
            for j in range(i, variable_num):

                delta = (sample[i] - variable_means[i]) * (sample[j] - variable_means[j]) / sample_num

                covariance_matrix[i, j] += delta

                if j != i:

                    covariance_matrix[j, i] += delta
                    
    return covariance_matrix


def compute_correlation_matrix(variable_num: int, covariance_matrix, standard_deviations):

    correlation_matrix = np.zeros((variable_num, variable_num))    

    for i in range(variable_num):
        for j in range(i, variable_num):

            correlation_matrix[i,j] = covariance_matrix[i, j] / (standard_deviations[i] * standard_deviations[j])

            if j != i:
                correlation_matrix[j, i] = correlation_matrix[i,j]
                
    return correlation_matrix


def compute_eigenvalues(covariance_matrix):
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idxs =  np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[idxs])
    eigenvectors = np.real(eigenvectors[:, idxs])
    
    return eigenvalues, eigenvectors


def compute_conditional_probability(data_matrix: np.ndarray, sample_num: int, variable_num: int):
    
    conditional_probability = np.zeros((variable_num, variable_num))

    sample_per_variable = np.asarray([np.sum(data_matrix[:, i]) for i in range(variable_num)])

    for k in range(sample_num):

        sample = data_matrix[k]

        for i in range(variable_num):
            if sample[i] == 1:   
                for j in range(i, variable_num):   
                    if sample[j] == 1:
                        conditional_probability[i, j] += 1
                        if i != j:
                            conditional_probability[j, i] += 1

    for i in range(variable_num):
        if sample_per_variable[i] > 0:
            conditional_probability[i] /= sample_per_variable[i]
        
    return conditional_probability


def compute_statistics(data_matrix, variables, sort=False):

    sample_num, variable_num = len(data_matrix), len(variables)
    
    if sort:
        data_matrix, variables = sort_data_matrix(data_matrix, variables)
        
    variable_means, standard_deviations = compute_mean_and_std(data_matrix)
    
    covariance_matrix = compute_covariance_matrix(data_matrix, sample_num, variable_num, variable_means)

    correlation_matrix = compute_correlation_matrix(variable_num, covariance_matrix, standard_deviations)

    eigenvalues, eigenvectors = compute_eigenvalues(covariance_matrix)
    
    conditional_probability = compute_conditional_probability(data_matrix, sample_num, variable_num)
        
    return variables, variable_means, covariance_matrix, correlation_matrix, eigenvalues, eigenvectors, conditional_probability    
    
