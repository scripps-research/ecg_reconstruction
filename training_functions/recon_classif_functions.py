import torch
import numpy as np


bce_loss = torch.nn.BCELoss()

def process_element(recon_input_leads, classif_input_leads, recon_output_lead_keys, classif_input_lead_keys, output_probabilities, min_value: float, amplitude: float, device):
    
    reconstruction_input = [torch.unsqueeze(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0), 0) for lead in recon_input_leads]
    classification_input = [torch.unsqueeze(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0), 0) for lead in classif_input_leads]
    
    reconstruction_target = [torch.from_numpy(classif_input_leads[classif_input_lead_keys.index(lead_key)] * amplitude + min_value).float().to(device).detach() for lead_key in recon_output_lead_keys]        
    classification_target = [torch.unsqueeze(torch.FloatTensor([probability]).float().to(device).detach(), dim=1) for probability in output_probabilities] 


    return reconstruction_input, reconstruction_target, classification_input, classification_target


def process_batch(batch, recon_input_lead_num: int, classif_input_lead_num: int, recon_output_lead_num: int, detect_class_num: int, recon_output_lead_keys, classif_input_lead_keys, min_value: float, amplitude: float, device):

    reconstruction_input = [[] for _ in range(recon_input_lead_num)]
    classification_input = [[] for _ in range(classif_input_lead_num)]
    reconstruction_target = [[] for _ in range(recon_output_lead_num)]
    classification_target = [[] for _ in range(detect_class_num)]
    
    for recon_input_leads, classif_input_leads, output_probabilities, _, _ in batch:

        for lead_idx, lead in enumerate(recon_input_leads):

            reconstruction_input[lead_idx].append(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0))
            
        for lead_idx, lead in enumerate(classif_input_leads):
    
            classification_input[lead_idx].append(torch.unsqueeze(torch.from_numpy(lead).float().to(device), 0))
            
        for lead_idx, lead_key in enumerate(recon_output_lead_keys):
            
            reconstruction_target[lead_idx].append(torch.from_numpy(classif_input_leads[classif_input_lead_keys.index(lead_key)] * amplitude + min_value).float().to(device).detach())

        for class_idx, probability in enumerate(output_probabilities):
    
            classification_target[class_idx].append(probability)

    reconstruction_input = [torch.stack(lead) for lead in reconstruction_input]
    classification_input = [torch.stack(lead) for lead in classification_input]
    
    reconstruction_target = [torch.stack(lead).detach() for lead in reconstruction_target]    
    classification_target = [torch.unsqueeze(torch.FloatTensor(probability).to(device), 1).detach() for probability in classification_target]


    return reconstruction_input, reconstruction_target, classification_input, classification_target


def post_process_element(classification_input,
                         reconstruction_output,
                         classif_input_lead_keys,
                         recon_output_lead_keys,
                         min_value,
                         amplitude):
    
    for lead, key in zip(reconstruction_output, recon_output_lead_keys):
        
        classification_input[classif_input_lead_keys.index(key)] = torch.unsqueeze(torch.unsqueeze((lead - min_value) / amplitude, 0), 0)

    return classification_input


def post_process_batch(classification_input,
                       reconstruction_output,
                       classif_input_lead_keys,
                       recon_output_lead_keys,
                       min_value,
                       amplitude):
    
    for key in recon_output_lead_keys:
        
        classification_input[classif_input_lead_keys.index(key)] = []
    
    for output_leads, key in zip(reconstruction_output, recon_output_lead_keys):
    
        for lead in output_leads:
            
            classification_input[classif_input_lead_keys.index(key)].append(torch.unsqueeze((lead - min_value) / amplitude, 0))
            
    for recon_key in recon_output_lead_keys:
        
        classification_input[classif_input_lead_keys.index(recon_key)] = torch.stack(classification_input[classif_input_lead_keys.index(recon_key)])

    return classification_input


def element_loss_function(reconstruction_output, reconstruction_target, classification_output, classification_target, recon_output_lead_num: int, detect_class_num: int, alpha: float):
    
    mse = 0
    r2 = 0
    bce = 0

    for output_lead, target_lead in zip(reconstruction_output, reconstruction_target):

        mse += (((output_lead - target_lead) ) ** 2).mean() / recon_output_lead_num
        r2 += ( 1 - mse / ((( target_lead - target_lead.mean())) ** 2 ).mean() ) / recon_output_lead_num
        
    bce = 0
    
    for output, target in zip(classification_output, classification_target):
        
        bce += bce_loss(output, target) / detect_class_num
                
    loss =  - r2 * alpha + bce * (1 - alpha)

    return loss, mse, r2, bce


def batch_loss_function(reconstruction_output, reconstruction_target, classification_output, classification_target, recon_output_lead_num: int, detect_class_num: int, alpha: float, batch_size: int, compute_loss_per_element: bool):

    batch_r2 = 0

    r2_per_element = np.zeros(batch_size)

    for output_lead, target_lead in zip(reconstruction_output, reconstruction_target):

        if compute_loss_per_element:

            lead_ssr = torch.sum( ( (output_lead - target_lead) ) ** 2, 1)
        
            lead_sst = torch.sum( ( (target_lead - target_lead.mean()) ) ** 2, 1)
            
            lead_r2 = 1 - lead_ssr / lead_sst

            batch_r2 += lead_r2.mean() / recon_output_lead_num

            r2_per_element += lead_r2.detach().cpu().numpy() / recon_output_lead_num

        else:

            lead_ssr = torch.sum( ( (output_lead - target_lead) ) ** 2, 1)
        
            lead_sst = torch.sum( ( (target_lead - target_lead.mean()) ) ** 2, 1)
            
            lead_r2 = 1 - lead_ssr / lead_sst

            batch_r2 += lead_r2.mean() / recon_output_lead_num
    
    batch_bce = 0

    bce_per_element = np.zeros(batch_size)            
    
    for probability_index in range(detect_class_num):
    
        if compute_loss_per_element:
            
            for element_index in range(batch_size):

                element_loss = bce_loss(classification_output[probability_index][element_index].detach(), classification_target[probability_index][element_index])
                bce_per_element[element_index] += element_loss.cpu().numpy()
                batch_bce += element_loss / detect_class_num
                
        else:

            batch_bce += bce_loss(classification_output[probability_index], classification_target[probability_index])

    batch_loss = - batch_r2 * alpha + batch_bce * (1 - alpha)
    
    loss_per_element = - r2_per_element * alpha + bce_per_element * (1 - alpha)

    return batch_loss, loss_per_element
