from classify_functions.process_diagnosis import format_diagnosis, process_diagnosis
from util_functions.diagnosis_map import diagnosis_map

 
def compute_diagnosis_features(raw_dignosis, diagnosis_classifier):
    
    """
    This function makes it possible to etract the clinical features
    of the input element
    
    """

    element_labels = []
            
    statement_list, discarded_statement_list = format_diagnosis(raw_dignosis)

    for statement in statement_list:
        
        processed_statement = process_diagnosis(statement)

        unknwon_statement = True

        statement_labels = []

        for label in diagnosis_map.keys():
            
            diagnosis_keys = diagnosis_map[label][1]
            false_diagnosis_keys =  diagnosis_map[label][2]

            if any(x in processed_statement for x in diagnosis_keys) and not any(x in processed_statement for x in false_diagnosis_keys):

                unknwon_statement = False
                element_labels.append(label)
                statement_labels.append(label)

                diagnosis_category = diagnosis_map[label][0]

                if statement not in diagnosis_classifier[diagnosis_category]['statement']:
                    diagnosis_classifier[diagnosis_category]['statement'].append(statement)
                    diagnosis_classifier[diagnosis_category]['proposed label'].append([label])
                    diagnosis_classifier[diagnosis_category]['occurences'].append(1)

                else:
                    statement_idx = diagnosis_classifier[diagnosis_category]['statement'].index(statement)
                    diagnosis_classifier[diagnosis_category]['occurences'][statement_idx] += 1
                    if label not in diagnosis_classifier[diagnosis_category]['proposed label'][statement_idx]:
                        diagnosis_classifier[diagnosis_category]['proposed label'][statement_idx].append(label)
        
        if unknwon_statement:        

            if statement not in diagnosis_classifier['unknown']['statement']:
                diagnosis_classifier['unknown']['statement'].append(statement)
                diagnosis_classifier['unknown']['proposed label'].append('')
                diagnosis_classifier['unknown']['occurences'].append(1)
            else:
                statement_idx = diagnosis_classifier['unknown']['statement'].index(statement)
                diagnosis_classifier['unknown']['occurences'][statement_idx] += 1

            if statement not in diagnosis_classifier['full']['statement']:
                diagnosis_classifier['full']['statement'].append(statement)
                diagnosis_classifier['full']['proposed label'].append('')
                diagnosis_classifier['full']['occurences'].append(1)           
            else:
                statement_idx = diagnosis_classifier['full']['statement'].index(statement)
                diagnosis_classifier['full']['occurences'][statement_idx] += 1
        
        else:

            if statement not in diagnosis_classifier['full']['statement']:
                diagnosis_classifier['full']['statement'].append(statement)
                diagnosis_classifier['full']['occurences'].append(1)                   

                statement_dict = {}

                for label in statement_labels:

                    diagnosis_category, diagnosis_keys, false_diagnosis_keys =\
                        diagnosis_map[label][0], diagnosis_map[label][1], diagnosis_map[label][2]

                    if diagnosis_category not in statement_dict.keys():
                        statement_dict[diagnosis_category] = [label]
                    else:
                        statement_dict[diagnosis_category].append(label)

                statement_labels = []

                for key in statement_dict:

                    statement_labels.append(key + ': ')
                    
                    for label in statement_dict[key]:

                        statement_labels[-1] += label + ', '
                
                    statement_labels[-1] = statement_labels[-1][:-2]

                diagnosis_classifier['full']['proposed label'].append(statement_labels)

            else:
                statement_idx = diagnosis_classifier['full']['statement'].index(statement)
                diagnosis_classifier['full']['occurences'][statement_idx] += 1

    for statement in discarded_statement_list:

        if statement not in diagnosis_classifier['discarded']['statement']:
            diagnosis_classifier['discarded']['statement'].append(statement)
            diagnosis_classifier['discarded']['proposed label'].append('discarded')
            diagnosis_classifier['discarded']['occurences'].append(1)
        else:
            statement_idx = diagnosis_classifier['discarded']['statement'].index(statement)
            diagnosis_classifier['discarded']['occurences'][statement_idx] += 1
        
        if statement not in diagnosis_classifier['full']['statement']:
            diagnosis_classifier['full']['statement'].append(statement)
            diagnosis_classifier['full']['proposed label'].append('discarded')
            diagnosis_classifier['full']['occurences'].append(1)           
        else:
            statement_idx = diagnosis_classifier['full']['statement'].index(statement)
            diagnosis_classifier['full']['occurences'][statement_idx] += 1

    if len(element_labels) == 0:
        element_labels.append('unknown')
    
    return element_labels