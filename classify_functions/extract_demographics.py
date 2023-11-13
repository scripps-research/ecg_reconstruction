
def compute_demographic_features(element):
    
    """
    This function makes it possible to etract the demographic features
    of the input element
    
    """

    demographic_features = []

    try:
        age = int(element['PatientAge'])
        if age <= 60:    
            age = 'young'
        elif age <= 80:    
            age = 'old'
        else:    
            age = 'very_old'
    except:
        age = 'unknown_age'

    demographic_features.append(age)

    try:
        gender = element['Gender']
        if gender == 'MALE':
            gender = 'male'
        elif gender == 'FEMALE':
            gender = 'female'
        else:
            gender = 'unknown_gender'
    except:
        gender = 'unknown_gender'

    demographic_features.append(gender)        

    try:
        race = element['Race']
        if race == 'CAUCASIAN':
            race = 'caucasian'
        elif race == 'UNKNOWN':
            race = 'unknown_race'
        else:
            race = 'non_caucasian'
    except:
        race = 'unknown_race'

    demographic_features.append(race)
    
    return demographic_features