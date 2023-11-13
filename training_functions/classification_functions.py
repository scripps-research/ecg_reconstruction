import torch
import numpy as np
import matplotlib.pyplot as plt

bce_loss = torch.nn.BCELoss()


def get_detection_distributions(detections, groundtruths, threshold):
    
    positive_detections = np.copy(detections)
    positive_detections[positive_detections > threshold] = 1
    positive_detections[positive_detections < 1] = 0
    
    negative_detections = 1 - np.copy(positive_detections)
    
    positive_detections = positive_detections[np.where(groundtruths == 1)]
    negative_detections = negative_detections[np.where(groundtruths == 0)]    
    
    return positive_detections, negative_detections


def process_element(input_leads, target_probabilities, device):

    model_input = [torch.unsqueeze(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0), 0) for lead in input_leads]

    model_target = [torch.unsqueeze(torch.FloatTensor([probability]).float().to(device).detach(), dim=1) for probability in target_probabilities] 

    return model_input, model_target


def process_batch(batch, input_lead_num: int, detect_class_num: int, device):

    model_input = [[] for _ in range(input_lead_num)]
    model_target = [[] for _ in range(detect_class_num)]

    for input_leads, _, target_probabilities, _, _ in batch:

        for lead_idx, lead in enumerate(input_leads):

            model_input[lead_idx].append(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0))
            
        for class_idx, probability in enumerate(target_probabilities):

            model_target[class_idx].append(probability)

    model_input = [torch.stack(lead) for lead in model_input]
    model_target = [torch.unsqueeze(torch.FloatTensor(probability).to(device), 1).detach() for probability in model_target]

    return model_input, model_target


def element_bce_function(model_output, model_target, detect_class_num: int):
    
    element_loss = 0
    
    for output, target in zip(model_output, model_target):
        
        element_loss += bce_loss(output, target) / detect_class_num

    return element_loss


def batch_bce_function(model_output, model_target, batch_size: int, detect_class_num: int, compute_loss_per_element: bool):
    
    loss_per_element = np.zeros(batch_size)
    
    batch_loss = 0
    
    for probability_index in range(detect_class_num):
    
        if compute_loss_per_element:
            
            for element_index in range(batch_size):

                probability_loss = bce_loss(model_output[probability_index][element_index].detach(), model_target[probability_index][element_index])
                
                loss_per_element[element_index] += probability_loss.cpu().numpy()
                
                batch_loss += probability_loss / detect_class_num
                
        else:

            batch_loss += bce_loss(model_output[probability_index], model_target[probability_index]) / detect_class_num
        
    return batch_loss, loss_per_element


def plot_element_classif_leads(leads, lead_keys, element_class, model_class, element_id: str, plot_name: str, plot_format='png'):
    
    times = 10 * np.arange(len(leads[0])) / (len(leads[0])-1)

    wave_sample = len(leads[0])
    
    lead_num = len(leads)

    rhythm_waveform = np.zeros((lead_num, wave_sample))

    for idx, lead in enumerate(leads):

        rhythm_waveform[idx] = lead

    fig, axis = plt.subplots(lead_num, 1, figsize=(10, 20))

    for idx, lead_key in enumerate(lead_keys):

        lead_data = rhythm_waveform[idx]

        axis[idx].plot(times, lead_data)
        axis[idx].set_ylabel(lead_key)

        axis[idx].set_xlim((0, 10))
        
        axis[idx].grid()
        
    axis[-1].set_xlabel('Time [s]')
    
    fig.subplots_adjust(hspace=.5)
    
    fig.suptitle(element_id, fontsize = 15, y = 1.2)
    
    plt.suptitle('True class: ' + element_class + ' - Model class: ' + model_class, fontsize = 15, y = 0.9)
    
    if plot_format != 'png':
    
        plt.savefig(plot_name + '.' + plot_format, bbox_inches='tight', format=plot_format)

    plt.savefig(plot_name + '.png', bbox_inches='tight', format='png')

    plt.clf()
    plt.close()
