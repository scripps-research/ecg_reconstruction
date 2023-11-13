import numpy as np
from util_functions.general import get_twelve_keys


def load_element(collection, element_id):

    return collection.find_one({"ElementID": element_id})


def load_element_twelve_leads(collection, element_id: str, sample_num: int, min_value: float, amplitude: float):
    
    element = load_element(collection, element_id)

    twelve_leads, _ = extract_twelve_leads(element, False)
    
    twelve_leads, _ = process_leads(twelve_leads, get_twelve_keys(), [], sample_num, min_value, amplitude)

    return twelve_leads


def load_element_leads(collection, element_id, data_ids_per_detect_class, input_keys, output_keys, sample_num: int, min_value: float, amplitude: float, extract_qrs: bool):

    element = load_element(collection, element_id)
    
    target_probabilities = extract_probabilities(element_id, data_ids_per_detect_class)

    twelve_leads, qrs_times = extract_twelve_leads(element, extract_qrs)
    
    input_leads, output_leads = process_leads(twelve_leads, input_keys, output_keys, sample_num, min_value, amplitude)

    return (input_leads, output_leads, target_probabilities, qrs_times, element_id)


def load_data(collection, data_ids):

    dataset = []

    max_query_len = 1000
    i = 0

    while (i+1) * max_query_len < len(data_ids):

        dataset.extend(list(collection.find({'ElementID': { "$in": data_ids[i * max_query_len:(i+1) * max_query_len]}})))
        i += 1

    dataset.extend(list(collection.find({'ElementID': { "$in": data_ids[i * max_query_len:]}})))

    return dataset


def load_data_leads(collection, data_ids, data_ids_per_detect_class, input_keys, output_keys, sample_num: int, min_value: float, amplitude: float, extract_qrs: bool):
    
    data = []
    
    max_query_len = 1000
    j = 0
    
    while (j+1) * max_query_len < len(data_ids):

        for element in list(collection.find({'ElementID': { "$in": data_ids[j * max_query_len:(j+1) * max_query_len]}})):
            
            element_id = element['ElementID']
            
            target_probabilities = extract_probabilities(element_id, data_ids_per_detect_class)
            
            twelve_leads, qrs_times = extract_twelve_leads(element, extract_qrs)

            input_leads, output_leads = process_leads(twelve_leads, input_keys, output_keys, sample_num, min_value, amplitude)

            data.append((input_leads, output_leads, target_probabilities, qrs_times, element_id))
            
        j += 1
            
        print('Loaded ', j * max_query_len, ' data!')
    
    for element in list(collection.find({'ElementID': { "$in": data_ids[j * max_query_len:]}})):
        
        element_id = element['ElementID']
        
        target_probabilities = extract_probabilities(element_id, data_ids_per_detect_class)
        
        twelve_leads, qrs_times = extract_twelve_leads(element, extract_qrs)
    
        input_leads, output_leads = process_leads(twelve_leads, input_keys, output_keys, sample_num, min_value, amplitude)

        data.append((input_leads, output_leads, target_probabilities, qrs_times, element_id))
            
    print('Loaded', len(data), 'data!')

    return data


def process_leads(twelve_leads, input_keys, output_keys, sample_num, min_value, amplitude):

    twelve_keys = get_twelve_keys()
    
    actual_sample_num = len(twelve_leads[0])

    if sample_num != actual_sample_num:

        x = (np.arange(actual_sample_num) + 1) / actual_sample_num
        new_x = (np.arange(sample_num) + 1) / sample_num

        twelve_leads = [np.interp(new_x, x, lead) for lead in twelve_leads]
        
    input_leads = []
        
    for key in input_keys:
        
        if key in twelve_keys:
            
            input_leads.append(np.copy(twelve_leads[twelve_keys.index(key)]))
                               
        else:
            
            input_leads.append(generate_lead(twelve_leads, twelve_keys, key))

    output_leads = [np.copy(twelve_leads[i]) for i in [twelve_keys.index(key) for key in output_keys]]
        
    input_leads = [normalize_lead(lead, min_value, amplitude) for lead in input_leads]
    
    output_leads = [normalize_lead(lead, min_value, amplitude) for lead in output_leads]
    
    return input_leads, output_leads
  

def extract_twelve_leads(element, extract_qrs=False):

    twelve_leads = element['lead']
    
    twelve_keys = get_twelve_keys()

    twelve_leads = [twelve_leads[key].to_numpy() / 1000 for key in twelve_keys]
        
    if extract_qrs:
    
        qrs_times = element['QRS']

        qrs_times = np.array([int(qrs['Time']) for qrs in qrs_times])

        if qrs_times[-1] > 5000:

            qrs_times = (qrs_times/4).astype(int)

        elif qrs_times[-1] > 2500:

            qrs_times = (qrs_times/2).astype(int)
            
    else:
        
        qrs_times = None
    
    return twelve_leads, qrs_times


def extract_probabilities(element_id, data_ids_per_detect_class):
    
    target_probabilities = np.zeros(len(data_ids_per_detect_class))
            
    for class_index, detect_class_ids in enumerate(data_ids_per_detect_class):
        if element_id in detect_class_ids:
            target_probabilities[class_index] = 1
    
    return target_probabilities


def generate_lead(twelve_leads, twelve_keys, keys):
    
    assert len(keys) == 2
    
    alpha = np.random.uniform(1)
    
    new_lead = alpha * np.copy(twelve_leads[twelve_keys.index(keys[0])]) + (1-alpha) * np.copy(twelve_leads[twelve_keys.index(keys[1])])
    
    return new_lead


def normalize_lead(lead, min_value, amplitude):

    lead -= min_value
    lead /= amplitude

    lead[lead > 1] = 1
    lead[lead < 0] = 0

    return lead


def denormalize_lead(lead, min_value, amplitude):

    lead *= amplitude
    lead += min_value

    return lead

