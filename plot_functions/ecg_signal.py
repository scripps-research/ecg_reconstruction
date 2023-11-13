
from plot_functions.plot_utils import get_colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.backends.backend_pdf import PdfPages
    
    
def plot_element_leads(leads, keys, element_id: str, output_file: str):
    
    times = 10 * np.arange(len(leads[0])) / (len(leads[0])-1)

    wave_sample = len(leads[0])
    
    lead_num = len(leads)

    rhythm_waveform = np.zeros((lead_num, wave_sample))

    for index, lead in enumerate(leads):

        rhythm_waveform[index] = lead
        
    colors = get_colors(lead_num)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for index, lead_key, color in zip(range(lead_num), keys, colors):

        lead_data = rhythm_waveform[index]

        ax.plot(times, lead_data, label=lead_key, color=color)
        ax.set_ylabel(lead_key)

    ax.set_xlim((0, 10))
        
    ax.grid()
    
    ax.legend()
        
    ax.set_xlabel('Time [s]')
    
    fig.subplots_adjust(hspace=.5)
    
    fig.suptitle(element_id, fontsize=15, y=0.92)

    plt.savefig(output_file, bbox_inches='tight')

    plt.clf()
    plt.close()
    
    
def plot_medical_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_output_keys, multi_input_keys, multi_labels, plot_name: str, plot_format='png'):
        
    manager_num = len(multi_model_leads)
    
    times = 10 * np.arange(len(twelve_leads[0])) / (len(twelve_leads[0])-1)
    
    fig, axis = plt.subplots(4, manager_num + 1, figsize=(20 * (manager_num + 1), 20))
    
    time_ranges = [[0,625], [625, 1250], [1250, 1875], [1875, 2500]]
    subplot_keys = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
    
    multi_model_output_keys = multi_output_keys + [[]]
    multi_model_input_keys = multi_input_keys + [twelve_keys]
    multi_model_labels = multi_labels + ['Full 12-lead']
    multi_model_leads = multi_model_leads + [ twelve_leads]
    
    for model_leads, output_keys, input_keys, model_label in zip(multi_model_leads, multi_model_output_keys, multi_model_input_keys, multi_model_labels):
        
        output_keys = copy.copy(output_keys)
        
        flatten_input_keys = []
        
        for key in input_keys:
            
            if isinstance(key, list):
                
                flatten_input_keys += key
            
            else:
                flatten_input_keys.append(key)
                
        input_keys = flatten_input_keys
        
        model_index = multi_model_labels.index(model_label)
        
        for key in input_keys:
            
            if key in output_keys:
                output_keys.remove(key)
        
        for lead, lead_key in zip(twelve_leads, twelve_keys):
            
            if lead_key in subplot_keys[0]:
                key_index = 0
                axis[key_index, model_index].set_title(model_label, fontsize=25, pad=20)
                time_range = time_ranges[subplot_keys[0].index(lead_key)]
            elif lead_key in subplot_keys[1]:
                key_index = 1
                time_range = time_ranges[subplot_keys[1].index(lead_key)]
            elif lead_key in subplot_keys[2]:
                key_index = 2
                time_range = time_ranges[subplot_keys[2].index(lead_key)]
            else:
                raise ValueError
        
            ax = axis[key_index, model_index]
        
            if lead_key in output_keys:
                model_lead = model_leads[output_keys.index(lead_key)]
                ax.plot(times[time_range[0]: time_range[1]], model_lead[time_range[0]: time_range[1]], color='b', linestyle='-')
            elif lead_key not in input_keys:
                ax.plot(times[time_range[0]: time_range[1]], lead[time_range[0]: time_range[1]], color='b', linestyle='-')
            else:
                ax.plot(times[time_range[0]: time_range[1]], lead[time_range[0]: time_range[1]], color='k', linestyle='-')
                
    for model_index in range(manager_num + 1):
        
        ax = axis[3, model_index]
        
        ax.plot(times, twelve_leads[1], color='k', linestyle='-')
        
        for plot_index in range(4):
            
            ax = axis[plot_index, model_index]
            
            ax.xaxis.set_major_locator(MultipleLocator(.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            
            ax.xaxis.set_minor_locator(MultipleLocator(.04))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            
            ax.grid(which='major', color='r', alpha=0.3)
            ax.grid(which='minor', color='r', alpha=0.1)
            
            ax.set_xlim((0, 10))  
            ax.set_ylim((-1.5, 1.5))
            if model_index == 0:
                ax.set_ylabel('Voltage [mV]', fontsize=20)
            
            if plot_index == 3:
                ax.text(.1, 1.1, 'II', fontsize=25)  
                ax.set_xlabel('Time [s]', fontsize=20)
                ax.set_xticks(list(np.linspace(0, 10, num=51)))
                ax.set_xticklabels([0, '', '', '', '', 1, '', '', '', '', 2, '', '', '', '', 3, '', '', '', '', 4, '', '', '', '', 5, '', '', '', '', 6, '', '', '', '', 7, '', '', '', '', 8, '', '', '', '', 9, '', '', '', '', 10])
            else:
                ax.axes.xaxis.set_ticklabels([])
                
                titles = subplot_keys[plot_index]
                
                ax.text(.1, 1.1, titles[0], fontsize=25)
                ax.text(2.6, 1.1, titles[1], fontsize=25)
                ax.text(5.1, 1.1, titles[2], fontsize=25)
                ax.text(7.6, 1.1, titles[3], fontsize=25)

    fig.subplots_adjust(hspace=.2, wspace=.1)

    if plot_format != 'png':

        plt.savefig(plot_name + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.savefig(plot_name + '.png', bbox_inches='tight', format='png')

    plt.close('all')
        
def plot_output_multi_recon_leads(twelve_leads, twelve_keys, multi_model_leads, multi_model_keys, multi_model_labels, plot_name: str, plot_format='png'):
    
    manager_num = len(multi_model_leads)
    
    times = 10 * np.arange(len(twelve_leads[0])) / (len(twelve_leads[0])-1)
    
    output_keys = []
    
    for keys in multi_model_keys:
        for key in keys:
            if key not in output_keys:
                output_keys.append(key)
    
    output_keys = copy.copy(output_keys)  
    
    fig, axis = plt.subplots(len(output_keys), manager_num+1, figsize=(20 * (manager_num+1), 20))      
    
    multi_model_leads = multi_model_leads + [[twelve_leads[twelve_keys.index(key)] for key in output_keys]]
    multi_model_keys = multi_model_keys + [output_keys]
    multi_model_labels = multi_model_labels + ['Full 12-lead']
    
    for model_leads, model_keys, model_label in zip(multi_model_leads, multi_model_keys, multi_model_labels):
        
        for model_lead, lead_key in zip(model_leads, model_keys):
        
            ax = axis[output_keys.index(lead_key), multi_model_labels.index(model_label)]
            ax.plot(times, model_lead, color='k', linestyle='-')
            ax.text(0.1, 1.1, lead_key, fontsize=25)
            ax.set_ylim((-1.5, 1.5))
            ax.set_xlim((0, 10))
            
            ax.xaxis.set_major_locator(MultipleLocator(.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            
            ax.xaxis.set_minor_locator(MultipleLocator(.04))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            
            ax.grid(which='major', color='r', alpha=0.3)
            ax.grid(which='minor', color='r', alpha=0.1)
            
            if lead_key == output_keys[-1]:
                ax.set_xlabel('Time [s]', fontsize=20)
                ax.set_xticks(list(np.linspace(0, 10, num=51)))
                ax.set_xticklabels([0, '', '', '', '', 1, '', '', '', '', 2, '', '', '', '', 3, '', '', '', '', 4, '', '', '', '', 5, '', '', '', '', 6, '', '', '', '', 7, '', '', '', '', 8, '', '', '', '', 9, '', '', '', '', 10])
            else:
                ax.axes.xaxis.set_ticklabels([])
                if lead_key == output_keys[0]:
                    ax.set_title(model_label, fontsize=25, pad=20)
                    
            if model_label == multi_model_labels[0]:
                ax.set_ylabel('Voltage [mV]', fontsize=20)    

    fig.subplots_adjust(hspace=.2, wspace=.1)

    if plot_format != 'png':

        plt.savefig(plot_name + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.savefig(plot_name + '.png', bbox_inches='tight', format='png')

    plt.close('all')
    

def plot_medical_recon_leads(twelve_leads, twelve_keys, model_leads, model_output_keys, model_input_keys, plot_name: str, plot_format='png'):
    
    times = 10 * np.arange(len(twelve_leads[0])) / (len(twelve_leads[0])-1)

    fig, axis = plt.subplots(4, 2, figsize=(20 * 2, 20))
    
    time_ranges = [[0,625], [625, 1250], [1250, 1875], [1875, 2500]]
    subplot_keys = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
    
    multi_model_output_keys = [model_output_keys, []]
    multi_model_input_keys = [model_input_keys, twelve_keys]
    multi_model_labels = ['Reconstruction', 'Full 12-lead']
    multi_model_leads = [model_leads, twelve_leads]
    
    for model_leads, output_keys, input_keys, model_label in zip(multi_model_leads, multi_model_output_keys, multi_model_input_keys, multi_model_labels):
        
        output_keys = copy.copy(output_keys)
            
        flatten_input_keys = []
        
        for key in input_keys:
            
            if isinstance(key, list):
                
                flatten_input_keys += key
            
            else:
                flatten_input_keys.append(key)
                
        input_keys = flatten_input_keys
        
        model_index = multi_model_labels.index(model_label)
        
        for key in input_keys:
            if key in output_keys:
                output_keys.remove(key)
        
        for lead, lead_key in zip(twelve_leads, twelve_keys):
            
            if lead_key in subplot_keys[0]:
                key_index = 0
                axis[key_index, model_index].set_title(model_label, fontsize=25, pad=20)
                time_range = time_ranges[subplot_keys[0].index(lead_key)]
            elif lead_key in subplot_keys[1]:
                key_index = 1
                time_range = time_ranges[subplot_keys[1].index(lead_key)]
            elif lead_key in subplot_keys[2]:
                key_index = 2
                time_range = time_ranges[subplot_keys[2].index(lead_key)]
            else:
                raise ValueError
        
            ax = axis[key_index, model_index]
        
            if lead_key in output_keys:
                model_lead = model_leads[output_keys.index(lead_key)]
                ax.plot(times[time_range[0]: time_range[1]], model_lead[time_range[0]: time_range[1]], color='b', linestyle='-')
            elif lead_key not in input_keys:
                ax.plot(times[time_range[0]: time_range[1]], lead[time_range[0]: time_range[1]], color='b', linestyle='-')
            else:
                ax.plot(times[time_range[0]: time_range[1]], lead[time_range[0]: time_range[1]], color='k', linestyle='-')
                
    for model_index in range(2):
        
        ax = axis[3, model_index]
        
        ax.plot(times, twelve_leads[1], color='k', linestyle='-')
        
        for plot_index in range(4):
            
            ax = axis[plot_index, model_index]
            
            ax.xaxis.set_major_locator(MultipleLocator(.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            
            ax.xaxis.set_minor_locator(MultipleLocator(.04))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            
            ax.grid(which='major', color='r', alpha=0.3)
            ax.grid(which='minor', color='r', alpha=0.1)
            
            ax.set_xlim((0, 10))  
            ax.set_ylim((-1.5, 1.5))
            ax.set_ylabel('Voltage [mV]', fontsize=20)
            
            if plot_index == 3:
                ax.text(.1, 1.1, 'II', fontsize=25)  
                ax.set_xlabel('Time [s]', fontsize=20)
                ax.set_xticks(list(np.linspace(0, 10, num=51)))
                ax.set_xticklabels([0, '', '', '', '', 1, '', '', '', '', 2, '', '', '', '', 3, '', '', '', '', 4, '', '', '', '', 5, '', '', '', '', 6, '', '', '', '', 7, '', '', '', '', 8, '', '', '', '', 9, '', '', '', '', 10])
            else:
                ax.axes.xaxis.set_ticklabels([])
                
                titles = subplot_keys[plot_index]
                
                ax.text(.1, 1.1, titles[0], fontsize=25)
                ax.text(2.6, 1.1, titles[1], fontsize=25)
                ax.text(5.1, 1.1, titles[2], fontsize=25)
                ax.text(7.6, 1.1, titles[3], fontsize=25)

    fig.subplots_adjust(hspace=.2, wspace=.1)

    if plot_format != 'png':

        plt.savefig(plot_name + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.savefig(plot_name + '.png', bbox_inches='tight', format='png')

    plt.close('all')
        
def plot_output_recon_leads(twelve_leads, twelve_keys, model_leads, model_keys, plot_name: str, plot_format='png'):
    
    times = 10 * np.arange(len(twelve_leads[0])) / (len(twelve_leads[0])-1)

    output_keys = copy.copy(model_keys)   
    
    fig, axis = plt.subplots(len(output_keys), 2, figsize=(20 * 2, 20)) 
    
    multi_model_leads = [model_leads, [twelve_leads[twelve_keys.index(key)] for key in output_keys]]
    multi_model_keys = [model_keys, output_keys]
    multi_model_labels = ['Reconstruction', 'Full 12-lead']
    
    for model_leads, model_keys, model_label in zip(multi_model_leads, multi_model_keys, multi_model_labels):
        
        for model_lead, lead_key in zip(model_leads, model_keys):
        
            ax = axis[output_keys.index(lead_key), multi_model_labels.index(model_label)]
            ax.plot(times, model_lead, color='k', linestyle='-')
            ax.text(0.1, 1.1, lead_key, fontsize=25)
            ax.set_ylim((-1.5, 1.5))
            ax.set_xlim((0, 10))
            
            ax.xaxis.set_major_locator(MultipleLocator(.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            
            ax.xaxis.set_minor_locator(MultipleLocator(.04))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            
            ax.grid(which='major', color='r', alpha=0.3)
            ax.grid(which='minor', color='r', alpha=0.1)
            
            if lead_key == output_keys[-1]:
                ax.set_xlabel('Time [s]', fontsize=20)
                ax.set_xticks(list(np.linspace(0, 10, num=51)))
                ax.set_xticklabels([0, '', '', '', '', 1, '', '', '', '', 2, '', '', '', '', 3, '', '', '', '', 4, '', '', '', '', 5, '', '', '', '', 6, '', '', '', '', 7, '', '', '', '', 8, '', '', '', '', 9, '', '', '', '', 10])
            else:
                ax.axes.xaxis.set_ticklabels([])
                if lead_key == output_keys[0]:
                    ax.set_title(model_label, fontsize=25, pad=20)
                    
            if model_label == multi_model_labels[0]:
                ax.set_ylabel('Voltage [mV]', fontsize=20)    

    fig.subplots_adjust(hspace=.2, wspace=.1)

    if plot_format != 'png':

        plt.savefig(plot_name + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.savefig(plot_name + '.png', bbox_inches='tight', format='png')

    plt.close('all')
    
    
def plot_clinical_dataset(dataset_signals: list, 
                          signal_sample: int,
                          twelve_keys: list,
                          output_file: str,
                          dataset_label: str,
                          classes=None,
                          diagnoses=None,
                          split_figures=False):
    
    with PdfPages(output_file + '.pdf') as dataset_output:
    
        times = 10 * np.arange(signal_sample) / (signal_sample-1)
        
        time_ranges = [[0,625], [625, 1250], [1250, 1875], [1875, 2500]]
        subplot_keys = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
        
        for signal_index, twelve_leads in enumerate(dataset_signals):
            
            print('Generating signal:', signal_index)
            
            if split_figures:
                
                fig, axis = plt.subplots(4, 1, figsize=(40, 30), constrained_layout=True)
                
            else:
                
                if diagnoses is None:            
                    fig, axis = plt.subplots(5, 1, figsize=(40, 30), constrained_layout=True)
                
                else:
                    fig, axis = plt.subplots(5, 1, figsize=(40, 30), constrained_layout=True)
            
            for lead, lead_key in zip(twelve_leads, twelve_keys):
                
                if lead_key in subplot_keys[0]:
                    key_index = 0
                    time_range = time_ranges[subplot_keys[0].index(lead_key)]
                elif lead_key in subplot_keys[1]:
                    key_index = 1
                    time_range = time_ranges[subplot_keys[1].index(lead_key)]
                elif lead_key in subplot_keys[2]:
                    key_index = 2
                    time_range = time_ranges[subplot_keys[2].index(lead_key)]
                else:
                    raise ValueError
            
                axis[key_index].plot(times[time_range[0]: time_range[1]], lead[time_range[0]: time_range[1]], color='k', linestyle='-')
            
            axis[3].plot(times, twelve_leads[1], color='k', linestyle='-')
                
            for plot_index in range(4):
                    
                ax = axis[plot_index]
                ax.xaxis.set_major_locator(MultipleLocator(.2))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                
                ax.xaxis.set_minor_locator(MultipleLocator(.04))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    
                ax.grid(which='major', color='r', alpha=0.3)
                ax.grid(which='minor', color='r', alpha=0.1)
                    
                ax.set_xlim((0, 10))  
                ax.set_ylim((-2, 2))
                ax.set_ylabel('Voltage [mV]', fontsize=25)
                ax.tick_params(axis='y', which='major', labelsize=25)
                    
                if plot_index == 3:
                    ax.text(.1, 1.1, 'II', fontsize=25)  
                    ax.set_xlabel('Time [s]', fontsize=25)
                    ax.set_xticks(list(np.linspace(0, 10, num=51)))
                    ax.set_xticklabels([0, '', '', '', '', 1, '', '', '', '', 2, '', '', '', '', 3, '', '', '', '', 4, '', '', '', '', 5, '', '', '', '', 6, '', '', '', '', 7, '', '', '', '', 8, '', '', '', '', 9, '', '', '', '', 10], fontsize=25)
                else:
                    ax.axes.xaxis.set_ticklabels([])            
                    titles = subplot_keys[plot_index]            
                    ax.text(.1, 1.1, titles[0], fontsize=25)
                    ax.text(2.6, 1.1, titles[1], fontsize=25)
                    ax.text(5.1, 1.1, titles[2], fontsize=25)
                    ax.text(7.6, 1.1, titles[3], fontsize=25)
                    
            if classes is not None:
                
                if classes[signal_index] == 'clinical_positives':
                
                    axis[0].set_title('ECG ' + str(signal_index + 1) + ' - STEMI', fontsize=35)
                    
                elif classes[signal_index] == 'other':
                    
                    axis[0].set_title('ECG ' + str(signal_index + 1) + ' - NON STEMI', fontsize=35)
                    
                elif classes[signal_index] == 'other*':
                    
                    axis[0].set_title('ECG ' + str(signal_index + 1) + ' - NON STEMI (POSSIBLE INFARCT)', fontsize=35)
                    
                else:
                    
                    axis[0].set_title('ECG ' + str(signal_index + 1) + ' - ' + classes[signal_index], fontsize=35)
                
            else:
                    
                axis[0].set_title('ECG ' + str(signal_index + 1) + ' - DATASET ' + str(dataset_label), fontsize=35)
                    
            if split_figures:
                
                plt.savefig(output_file + '_' + str(signal_index) + '.jpg', bbox_inches='tight', format='jpg')
                
            else:
                
                if diagnoses is None:
                
                    axis[4].axis('tight')
                    axis[4].axis('off')
                    table = axis[4].table(cellText=[['SELECT ONE OF THE FOLLOWING DIAGNOSES:', 'ACUTE ST ELEVATION INFARCT', 'NO ACUTE ST ELEVATION INFARCT', 'UNABLE TO DETERMINE'], ['NOTES:', '', '', '']], cellLoc='center', loc='center')
                    
                    table.scale(1, 5)
                    table.set_fontsize(50)
                    
                    table[(0, 1)].set_facecolor('#FF6347')
                    table[(0, 2)].set_facecolor("#ADFF2F")
                    table[(0, 3)].set_facecolor("#FFA500")
                    
                else:
                    
                    table_content = []
                    
                    for diagnosis in diagnoses[signal_index]:
                        table_content.append([diagnosis])
                    
                    axis[4].axis('off')
                    table = axis[4].table(cellText=table_content, cellLoc='center', loc='center')
                    
                    axis[4].axis('tight')
                    
                    table.scale(1, 5)
                    table.set_fontsize(20)

            dataset_output.savefig()