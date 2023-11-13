import copy


def process_diagnosis(text: str):
    
    text = str(copy.copy(text))
    
    text = text.replace(' inf ', ' inferior ')
    text = text.replace(' post ', ' posterior ')
    text = text.replace(' lat ', ' lateral ')
    text = text.replace(' ant ', ' anterior ')
    
    text = text.replace('antero - septal', 'anteroseptal')
    text = text.replace('antero septal', 'anteroseptal')
    text = text.replace('anterior - septal', 'anteroseptal')
    text = text.replace('anterior septal', 'anteroseptal')
    
    text = text.replace('antero - lateral', 'anterolateral')
    text = text.replace('antero lateral', 'anterolateral')
    text = text.replace('anterior - lateral', 'anterolateral')
    text = text.replace('anterior lateral', 'anterolateral')
    
    text = text.replace('infero - lateral', 'inferolateral')
    text = text.replace('infero lateral', 'inferolateral')
    text = text.replace('inferior - lateral', 'inferolateral')   
    text = text.replace('inferior lateral', 'inferolateral')
    
    text = text.replace('infero - posterior', 'inferoposterior')
    text = text.replace('infero posterior', 'inferoposterior')
    text = text.replace('inferior - posterior', 'inferoposterior')
    text = text.replace('inferior posterior', 'inferoposterior')
    
    text = text.replace('postero - lateral', 'posterolateral')
    text = text.replace('postero lateral', 'posterolateral')
    text = text.replace('posterior - lateral', 'posterolateral')   
    text = text.replace('posterior lateral', 'posterolateral')
    
    text = text.replace('anteriorly', 'in anterior leads')
    text = text.replace('inferiorly', 'in inferior leads')
    text = text.replace('laterally', 'in lateral leads')
    text = text.replace('posteriorly', 'in posterior leads')
    text = text.replace('anterolaterally', 'in anterolateral leads')
    text = text.replace('inferolaterally', 'in inferolateral leads')
    text = text.replace('posterolaterally', 'in posterolateral leads')
    text = text.replace('anteroseptally', 'in anteroseptal leads')
    text = text.replace('inferoposteriorly', 'in inferoposterior leads')
      
    text = text.replace('injury or acute infarct', 'acute infarct')    
    text = text.replace('injury', 'infarct')    
    text = text.replace('subendocardial infarct', 'infarct')
    text = text.replace('subepicardial infarct', 'infarct')
    text = text.replace('myocardial infarct', 'infarct')
    text = text.replace('infarct pattern', 'infarct')
    text = text.replace('( infarct current )', 'infarct')
    text = text.replace('cardiac infarct', 'infarct')
    text = text.replace('rv infarct', 'infarct')
    text = text.replace('myocardial infarct', 'infarct')
    
    text = text.replace('and not infarct', '')
    text = text.replace('or less likely infarct', '')
    text = text.replace('less likely infarct', '')
    text = text.replace('than acute infarct', '')
    text = text.replace('( masked by fascicular block )', '')
    
    text = text.replace('acute and evolving', 'acute')            
    text = text.replace('evolving', 'acute')
    text = text.replace('in evolution', 'acute') 
    text = text.replace('acute or recent', 'acute')
    text = text.replace('acute extensive', 'acute')
    text = text.replace('age acute', 'acute')
    text = text.replace('( acute )', 'acute')
    text = text.replace('possibly acute', 'acute')
    
    text = text.replace('prior', 'old')
    text = text.replace('( age undetermined )', 'old')
    text = text.replace('age undetermined', 'old')
    text = text.replace('are undetermined', 'old')
    text = text.replace('( old )', 'old')
    text = text.replace('- old', 'old')
    
    text = text.replace('elevated st', 'st elevation')
    text = text.replace('st wave elevation', 'st elevation')
    text = text.replace('st more elevated', 'st elevation')
    text = text.replace('st now elevated', 'st elevation')
    
    text = text.replace('old posterior infarct', 'posterior infarct old')
    text = text.replace('old inferior infarct', 'inferior infarct old')
    text = text.replace('old septal infarct', 'septal infarct old')
    text = text.replace('old lateral infarct', 'lateral infarct old')
    text = text.replace('old anteroseptal infarct', 'anteroseptal infarct old')
    text = text.replace('old inferoposterior infarct', 'inferoposterior infarct old')
    text = text.replace('old inferolateral infarct', 'inferolateral infarct old')
    text = text.replace('old posterolateral infarct', 'posterolateral infarct old')
    text = text.replace('old anterolateral infarct', 'anterolateral infarct old')
        
    if 'st elevation' in text:
        
        if 'infarct' in text:
            index = text.index('infarct')
            if text[index-4:index-1] != 'old' and text[index+8:index+11] != 'old' and text[index-6:index-1] != 'acute':
                text = text[:index] + 'acute infarct' + text[index+7:]
                
        elif 'early repolarization' in text or 'pericarditis' in text:
            
            index = text.index('st elevation')
            
            text = text[:index] + 'st elevation due to early repolarization or pericarditis' + text[index+12:]
        
        if 'anterior leads' in text:            
            
            if 'anterior st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'anterior ' + text[index:]
            
            if 'infarct' in text and 'anterior infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'anterior ' + text[index:]
                
        elif 'inferior leads' in text:
            
            if 'inferior st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'inferior ' + text[index:]
            
            if 'infarct' in text and 'inferior infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'inferior ' + text[index:]
            
        elif 'posterior leads' in text:
            
            if 'posterior st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'posterior ' + text[index:]
                
            if 'infarct' in text and 'posterior infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'posterior ' + text[index:]          
            
        elif 'lateral leads' in text:
            
            if 'lateral st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'lateral ' + text[index:]
                
            if 'infarct' in text and 'lateral infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'lateral ' + text[index:]  
                
        elif 'septal leads' in text:
            
            if 'septal st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'septal ' + text[index:]
                
            if 'infarct' in text and 'septal infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'septal ' + text[index:] 
            
        elif 'anterolateral leads' in text:
            
            if 'anterolateral st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'anterolateral ' + text[index:]
                
            if 'infarct' in text and 'septal infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'septal ' + text[index:] 
            
        elif 'inferolateral leads' in text:
            
            if 'inferolateral st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'inferolateral ' + text[index:]
                
            if 'infarct' in text and 'inferolateral infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'inferolateral ' + text[index:] 
                
        elif 'posterolateral leads' in text:
            
            if 'posterolateral st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'posterolateral ' + text[index:]
                
            if 'infarct' in text and 'posterolateral infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'posterolateral ' + text[index:] 
            
        elif 'anteroseptal leads' in text:
            
            if 'anteroseptal st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'anteroseptal ' + text[index:]
                
            if 'infarct' in text and 'anteroseptal infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'anteroseptal ' + text[index:] 
                         
        elif 'inferosterior leads' in text:
            
            if 'inferoposterior st elevation' not in text:
                index = text.index('st elevation')
                text = text[:index] + 'inferoposterior ' + text[index:]
            
            if 'infarct' in text and 'inferoposterior infarct' not in text:                
                index = text.index('infarct')
                text = text[:index] + 'inferoposterior ' + text[index:] 
            
    return text


def format_diagnosis(raw_diagnosis, maintain_number=False):

    if not isinstance(raw_diagnosis, list):
        raw_diagnosis = [raw_diagnosis]
        
    months = ['- jan -',
              '- feb -',
              '- mar -',
              '- apr -',
              '- may -',
              '- jun -',
              '- jul -',
              '- aug -',
              '- sep -',
              '- oct -',
              '- nov -',
              '- dec -']

    diagnosis_list = []
    discarded_list = []

    old_text = ''
    
    for i, raw_diagnosis_line in enumerate(raw_diagnosis):
        
        if raw_diagnosis_line['StmtText'] == 'When compared with ECG of':
            
            raw_diagnosis = raw_diagnosis[:i]
            
            break

    for raw_diagnosis_line in raw_diagnosis:
        
        if 'StmtFlag' in raw_diagnosis_line.keys():
            flag = raw_diagnosis_line['StmtFlag']
            if isinstance(flag, str):
                flags = [flag]
            else:
                flags = flag
        else:
            flags = []

        raw_text = raw_diagnosis_line['StmtText']

        if raw_text is None:
            raw_text = ''
            
        else:
    
            raw_text = raw_text.lower()

        if 'ENDSLINE' not in flags: 

            old_text = old_text + ' ' + raw_text

        else:
            raw_text = old_text + ' ' + raw_text
            
            formatted_text_splitted = clean_text(raw_text)
            
            old_text = ''

            original_text_splitted = copy.copy(formatted_text_splitted)

            for idx, x in enumerate(formatted_text_splitted):
                if x.isdigit():
                    formatted_text_splitted[idx] = 'xxxx'
                    while len(original_text_splitted[idx]) < 4:
                        original_text_splitted[idx] = '0' + original_text_splitted[idx]

            original_text = ' '.join(original_text_splitted)

            formatted_text = ' '.join(formatted_text_splitted)
            
            for month in months:                
                if month in formatted_text:
                    formatted_text = formatted_text.replace(month, ' yyyy ')
                    
            formatted_text = formatted_text.replace('xxxx yyyy xxxx', 'xxxx')            
            formatted_text = formatted_text.replace('yyyy xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx yyyy', 'xxxx')
            formatted_text = formatted_text.replace('xxxx xxxx xxxx xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx xxxx xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx - xxxx - xxxx - xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx - xxxx - xxxx', 'xxxx')
            formatted_text = formatted_text.replace('xxxx - xxxx', 'xxxx')
            
            formatted_text_list = split_text(formatted_text)
            originial_text_list = split_text(original_text)
            
            for formatted_text, original_text in zip(formatted_text_list, originial_text_list):     
                
                if len(formatted_text) > 1:
                    
                    if evaluate_text(formatted_text):
                        if maintain_number:
                            diagnosis_list.append(' ' + original_text + ' ')
                        else:
                            diagnosis_list.append(' ' + formatted_text + ' ')
                            
                    else:
                        
                        discarded_list.append(' ' + formatted_text + ' ')
                        
    for idx, text in enumerate(diagnosis_list):
        
        if text == ' no ':
            try:
                diagnosis_list[idx + 1] = ' no' + diagnosis_list[idx + 1]
            except:
                diagnosis_list.append(' no significant change ')

    return diagnosis_list, discarded_list


def clean_text(text: str):
    
    text = text.replace(':', ' ')
    text = text.replace('?', ' ')
    text = text.replace('*', ' ')
    text = text.replace(',', ' ')
    text = text.replace('\'', ' ')
    text = text.replace(';', ' ')
    text = text.replace('\"', ' ')
    text = text.replace('.', ' ')
    text = text.replace('-', ' - ')
    text = text.replace('/', ' - ')
    text = text.replace('%', ' % ')
    text = text.replace('>', ' > ')
    text = text.replace('<', ' < ')
    text = text.replace('&', ' and ')
    text = text.replace('=', ' = ')
    text = text.replace('mv', ' mv ')
    text = text.replace('ms', ' ms ')
    text = text.replace('wall', '')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    
    text = ' ' + ' '.join(text.split()) + ' '
    
    text = text.replace(' - - ', ' - ')       
    text = text.replace(' identfied ', ' identified ')    
    text = text.replace(' recengt ', ' recent ')
    text = text.replace(' inferoposterolateral ', ' inferoposterior ')
    text = text.replace(' inferio ', ' inferior ')
    text = text.replace(' posterio ', ' posterior ')  
    text = text.replace(' antgerior ', ' anterior ')    
    text = text.replace(' iscehmia ', ' ischemia ')        
    text = text.replace(' possiably ', ' possibly ')
    text = text.replace(' changes ', ' change ')
    text = text.replace(' replace ', ' have replaced ')
    text = text.replace(' replaces ', ' has replaced ')
    text = text.replace(' indeterminate ', ' undetermined ')
    text = text.replace(' subendo inj ', ' subendocardial injury ')
    text = text.replace(' chornic pulm ', ' chronic pulmonary ')
    text = text.replace(' arrhytmia ', ' arrhythmia ')
    text = text.replace(' arrythmia ', ' arrhythmia ')
    text = text.replace(' rythm ', ' rhythm ')
    text = text.replace(' rhytm ', ' rhythm ')
    text = text.replace(' complexes ', ' complex ')
    text = text.replace(' beats ', ' complex ')
    text = text.replace(' beat ', ' complex ')
    text = text.replace(' abnormalities ', ' abnormality ')
    text = text.replace(' ekg ', ' ecg ')
    text = text.replace(' ekgs ', ' ecg ')
    text = text.replace(' ecgs ', ' ecg ')
    text = text.replace(' elevations ', ' elevation ')
    text = text.replace(' depressions ', ' depression ')
    text = text.replace(' fasicular ', ' fascicular ')
    text = text.replace(' can not ruleout ', ' cannot rule out ')
    text = text.replace(' elevationnc ', ' elevation c ')
    text = text.replace(' c - w ', ' consistent with ')
    text = text.replace(' r - o ', ' rule out ')
    text = text.replace(' suggestive of ', ' consistent with ')
    text = text.replace(' has shifted ', ' shifted ')
    text = text.replace(' wwave ', ' wave ')
    text = text.replace(' waves ', ' wave ') 
    text = text.replace(' nodal ', ' node ')
    text = text.replace(' stage ', ' degree ')
    text = text.replace(' setpal ', ' septal ')
    text = text.replace(' anterseptal ', ' anteroseptal ') 
    text = text.replace(' inferro ', ' inferior ')
    text = text.replace(' re - entry ', ' reentrant ')
    text = text.replace(' atrial - paced ', ' atrial paced ')
    text = text.replace(' a - paced ', ' atrial paced ')
    text = text.replace(' ventricular - paced ', ' ventricular paced ')
    text = text.replace(' v - paced ', ' ventricular paced ')
    text = text.replace(' dual - paced ', ' dual paced ')
    text = text.replace(' pattern of bigeminy ', ' bigeminy pattern ')
    text = text.replace(' pattern of trigeminy ', ' trigeminy pattern ')
    text = text.replace(' pattern of quadrigeminy ', ' quadrigeminy pattern ')
    text = text.replace(' pacs ', ' pac ') 
    text = text.replace(' pvcs ', ' pvc ')
    text = text.replace(' atri ', ' atrial ')
    text = text.replace(' atria ', ' atrial ')
    text = text.replace(' atrio - ventricular ', ' av ')
    text = text.replace(' a - v ', ' av ')
    text = text.replace(' av dual - paced ', ' dual paced ')
    text = text.replace(' heart block ', ' av block ')
    text = text.replace(' hemiblock ', ' fascicular block ')
    text = text.replace(' sinoatrial ', ' sa ')
    text = text.replace(' sinuatrial ', ' sa ')
    text = text.replace(' sino - atrial ', ' sa ')
    text = text.replace(' sinus - atrial ', ' sa ')   
    text = text.replace(' s - a ', ' sa ')    
    text = text.replace(' st - wave ', ' st wave ')
    text = text.replace(' t - wave ', ' t wave ')
    text = text.replace(' st - t ', ' st and t ')    
    text = text.replace(' is now present ', ' now present ')
    text = text.replace(' are now present ', ' now present ')
    text = text.replace(' [now present] ', ' now present ')
    text = text.replace(' is now ', ' now ')
    text = text.replace(' are now ', ' now ')
    text = text.replace(' acute coronary syndrome ( acs ) ', ' acute coronary syndrome ')
    text = text.replace(' ( acs ) ', ' acute coronary syndrome ')
    text = text.replace(' acs ', ' acute coronary syndrome ')                
    text = text.replace(' nonstemi ', ' non-stemi ')
    text = text.replace(' no stemi ', ' non-stemi ')
    text = text.replace(' non stemi ', ' non-stemi ')
    text = text.replace(' not stemi ', ' non-stemi ')
    text = text.replace(' than stemi ', ' non-stemi ')
    text = text.replace(' ami ', ' acute myocardial infarct ')
    text = text.replace(' a mi ', ' acute myocardial infarct ')
    text = text.replace(' mi ', ' myocardial infarct ')
    text = text.replace(' injury ( acute infarct ) ', ' acute infarct ')    
    text = text.replace(' infarction ', ' infarct ')    
    text = text.replace(' last previous tracing ', ' last ecg ')
    text = text.replace(' last previous record ', ' last ecg ')
    text = text.replace(' last previous ecg ', ' last ecg ')
    text = text.replace(' last old tracing ', ' last ecg ')
    text = text.replace(' last old record ', ' last ecg ')
    text = text.replace(' last old ecg ', ' last ecg ')
    text = text.replace(' previous tracing ', ' last ecg ')
    text = text.replace(' previous record ', ' last ecg ')
    text = text.replace(' previous ecg ', ' last ecg ')
    text = text.replace(' previous old tracing ', ' last ecg ')
    text = text.replace(' previous old record ', ' last ecg ')
    text = text.replace(' previous old ecg ', ' last ecg ')
    text = text.replace(' old tracing ', ' last ecg ')
    text = text.replace(' old record ', ' last ecg ')
    text = text.replace(' old ecg ', ' last ecg ')
    text = text.replace(' last record ', ' last ecg ')
    text = text.replace(' last tracing ', ' last ecg ')
    text = text.replace(' no longer depressed ', ' no longer seen ')
    text = text.replace(' no longer elevated ', ' no longer seen ')
    text = text.replace(' no longer inverted ', ' no longer seen ')
    text = text.replace(' no longer evident ', ' no longer seen ')
    text = text.replace(' no longer present ', ' no longer seen ')
    text = text.replace(' no longer identified ', ' no longer seen ')
    text = text.replace(' no longer noted ', ' no longer seen ')  
    
    text = ' ' + ' '.join(text.split()) + ' '
    
    return text.split()


def split_text(text: str):
    
    intro_delimiters = [
    'now present in anterior leads',
    'now present in inferior leads',
    'now present in lateral leads',
    'now present in posterior leads',
    'now present in anterolateral leads',
    'now present in inferolateral leads',
    'now present in precordial leads',
    'now present in septal leads',
    'now present in anteroseptal leads',
    'now present in its place',
    'no longer seen in anterior leads',
    'no longer seen in inferior leads',
    'no longer seen in lateral leads',
    'no longer seen in posterior leads',
    'no longer seen in anterolateral leads',
    'no longer seen in inferolateral leads',
    'no longer seen in precordial leads',
    'no longer seen in septal leads',
    'no longer seen in anteroseptal leads',
    'no longer seen in its place',
    'replaced by',
    'resolved',
    'disappeared']
    
    out_delimiters = [
    'has replaced',
    'have replaced']
    
    text_list = []
    
    text = text.strip()
    
    while any(x in text for x in intro_delimiters):

        delimiter_index = 1000
        pattern_index = None

        for index, delimiter in enumerate(intro_delimiters):

            if delimiter in text and text.index(delimiter) < delimiter_index:

                pattern_index = index
                delimiter_index = text.index(delimiter)

        illegal_intro = text[:delimiter_index + len(intro_delimiters[pattern_index])].strip()
        text_list.append(illegal_intro)
        text = text[len(illegal_intro):].strip()
        
    while any(x in text for x in out_delimiters):
    
        delimiter_index = 0

        for index, delimiter in enumerate(out_delimiters):

            if delimiter in text and text.index(delimiter) > delimiter_index:

                delimiter_index = text.index(delimiter)

        illegal_out = text[delimiter_index:].strip()
        text_list.append(illegal_out)
        text = text[:-len(illegal_out)].strip()
        
    text_list.append(text)
    
    return text_list
        


def evaluate_text(text: str):
    
    illegal_patterns = [
    'replaced',
    'resolved',
    'disappeared',
    'no longer',
    'no significant',
    'when compared']
    
    illegal_intro_patterns = ['no ']

    if any(x in text for x in illegal_patterns):       
        return False
    else:
        if any(text[:len(x)] == x for x in illegal_intro_patterns):
            return False
        else:
            return True
