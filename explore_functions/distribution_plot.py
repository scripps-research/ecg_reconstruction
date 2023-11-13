import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def distribution_plot(
    variables: list, 
    variable_num: int,
    max_variable_num: int,
    data_means: np.ndarray,
    data_covariance: np.ndarray,
    data_correlation: np.ndarray,
    pca_eigenvalues: np.ndarray,
    pca_eigenvectors: np.ndarray,   
    conditional_probability: np.ndarray,     
    label: str,
    output_file: str,
    plot_format='png'):

    if max_variable_num < variable_num:

        data_means = data_means[:max_variable_num]
        data_covariance = data_covariance[:max_variable_num, :max_variable_num]
        data_correlation = data_correlation[:max_variable_num, :max_variable_num]
        pca_eigenvalues = pca_eigenvalues[:max_variable_num]
        pca_eigenvectors = pca_eigenvectors[:max_variable_num, :max_variable_num]
        conditional_probability = conditional_probability[:max_variable_num, :max_variable_num]
        variables = variables[:max_variable_num]
        variable_num = max_variable_num

    data_covariance = np.flip(data_covariance, axis=0)
    data_correlation = np.flip(data_correlation, axis=0)
    conditional_probability = np.transpose(conditional_probability)
    conditional_probability = np.flip(conditional_probability, axis=0)

    fig, ax = plt.subplots()

    ax.bar(np.arange(variable_num), data_means)

    ax.set_xticks(np.arange(variable_num))

    ax.set_xticklabels(variables, rotation=90)

    ax.set_ylabel('Mean value')
    ax.set_xlabel(label)

    ax.grid()

    plt.savefig(output_file + '_means.png', format='png', bbox_inches='tight')

    if plot_format != 'png':

        plt.savefig(output_file + '_means.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()

    max_cov, min_cov = np.max(data_covariance), np.min(data_covariance)

    max_corr, min_corr = 1, -1

    fig, ax = plt.subplots()

    im = ax.imshow(data_covariance, vmin=min_cov, vmax=max_cov, cmap='plasma')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks(np.arange(variable_num))
    ax.set_yticks(np.arange(variable_num))

    ax.set_xticklabels(variables, rotation='vertical')
    ax.set_yticklabels(np.flip(variables))

    plt.savefig(output_file + '_covariance.png', format='png', bbox_inches='tight')
    if plot_format != 'png':
        plt.savefig(output_file + '_covariance.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()

    im = ax.imshow(data_correlation, vmin=min_corr, vmax=max_corr, cmap='plasma')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks(np.arange(variable_num))
    ax.set_yticks(np.arange(variable_num))

    ax.set_xticklabels(variables, rotation='vertical')
    ax.set_yticklabels(np.flip(variables))

    plt.savefig(output_file + '_correlation.png', format='png', bbox_inches='tight')
    if plot_format != 'png':
        plt.savefig(output_file + '_correlation.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()

    im = ax.imshow(pca_eigenvectors, vmin=min_corr, vmax=max_corr, cmap='plasma')

    plt.gca().invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks(np.arange(variable_num))
    ax.set_yticks(np.arange(variable_num))

    ax.set_xticklabels(np.arange(variable_num))
    ax.set_yticklabels(variables)   

    ax.set_ylabel(label)
    ax.set_xlabel('Principal component') 

    plt.savefig(output_file + '_eigenvectors.png', format='png', bbox_inches='tight')
    if plot_format != 'png':
        plt.savefig(output_file + '_eigenvectors.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()

    plt.close()

    fig, ax = plt.subplots()

    ax.plot(np.arange(variable_num), pca_eigenvalues)

    ax.set_xticks(np.arange(variable_num))

    ax.set_ylabel('Eigenvalue')
    ax.set_xlabel('Principal component')

    ax.grid()

    plt.savefig(output_file + '_eigenvalues.png', format='png', bbox_inches='tight')

    if plot_format != 'png':

        plt.savefig(output_file + '_eigenvalues.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()

    fig, ax = plt.subplots()

    im = ax.imshow(conditional_probability, vmin=0, vmax=np.max(conditional_probability), cmap='plasma')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xticks(np.arange(variable_num))
    ax.set_yticks(np.arange(variable_num))

    ax.set_xticklabels(variables, rotation='vertical')
    ax.set_yticklabels(np.flip(variables))

    plt.savefig(output_file + '_conditional_probability.png', format='png', bbox_inches='tight')
    if plot_format != 'png':
        plt.savefig(output_file + '_conditional_probability.' + plot_format, format=plot_format, bbox_inches='tight')

    plt.cla()
    plt.clf()
        
                    