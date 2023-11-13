
diagnosis_hierarchy = {

    # OTHER
    
    # low voltage
    # pericarditis
    
    # HYPERTROPHY

    'cardiac_hypertrophy': ['left_atrial_hypertrophy',
                            'right_atrial_hypertrophy',
                            'left_atrial_hypertrophy',
                            'right_atrial_hypertrophy',
                            'biatrial_hypertrophy',
                            'biventricular_hypertrophy'],
    
    # SINUS RHYTHM    
    
    'sinus_rhythm': ['normal_sinus_rhythm',
                     'sinus_arrhythmia',
                     'sinus_bradycardia',
                     'sinus_tachycardia'],
    
    'abnormal_sinus_rhythm': ['sinus_arrhythmia',
                              'sinus_bradycardia',
                              'sinus_tachycardia'],

    # OTHER RHYTHM
    
    'other_rhythm': ['junctional_rhythm',
                     'idioventricular_rhythm',
                     'fusion_rhythm',
                     'pacemaker_rhythm'],
    
    # ATRIAL ARRHYTHMIA

    'atrial_arrhythmia': ['atrial_nonspecific_arrhythmia',
                          'atrial_bradycardia',
                          'atrial_tachycardia',
                          'premature_atrial_complex',
                          'atrial_flutter',
                          'atrial_fibrillation',
                          'atrial_bigeminy',
                          'atrial_trigeminy',
                          'atrial_quadrigeminy',
                          'wolff_parkinson_white'],
    
    'ventricular_arrhythmia': ['ventricular_nonspecific_arrhythmia',
                               'ventricular_bradycardia',
                               'ventricular_tachycardia',
                               'premature_ventricular_complex',
                               'ventricular_flutter',
                               'ventricular_fibrillation',
                               'ventricular_bigeminy',
                               'ventricular_trigeminy',
                               'ventricular_quadrigeminy',
                               'brugada_syndrome'],

    # CONDUCTION SYSTEM DISEASES
    
    'conduction_abnormality': ['aberrant_condution',
                                  'first_degree_av_block',
                                  'second_degree_av_block',
                                  'third_degree_av_block',
                                  'variable_av_block',
                                  'sinoatrial_block',
                                  'nonspecific_intraventricular_block',                                  
                                  'right_bundle_branch_block',
                                  'incomplete_right_bundle_branch_block',
                                  'left_bundle_branch_block',
                                  'incomplete_left_bundle_branch_block',                                  
                                  'left_anterior_fascicular_block',
                                  'left_posterior_fascicular_block',
                                  'bifascicular_block',
                                  'trifascicular_block',                                  
                                  'right_ventricular_conduction_delay',
                                  'left_ventricular_conduction_delay',
                                  'intraventricular_conduction_delay',
                                  'wide_qrs',
                                  'short_pr',
                                  'long_qt'],
    
    
    # AXIS DEVIATION
    
    'axis_deviation': ['right_axis_deviation',
                       'left_axis_deviation',
                       'extreme_axis_deviation'],
    
    # REPOLARIZATION ABNORMALITY
    
    'repolarization_abnormality': ['t_abnormality',
                                   'st_abnormality',
                                   'nonspecific_wave_abnormality',
                                   'early_repolarization',
                                   'st_depression',
                                   'st_elevation',
                                   't_inversion',
                                   't_flattening'],

    # INJURIES
    
    # 'anterior_st_elevation_or_acute_infarct': ['anterior_st_elevation',
    #                                            'anterior_acute_infarct'],
    
    # 'lateral_st_elevation_or_acute_infarct': ['lateral_st_elevation',
    #                                           'lateral_acute_infarct'],
    
    # 'septal_st_elevation_or_acute_infarct': ['septal_st_elevation',
    #                                          'septal_acute_infarct'],
    
    # 'anteroseptal_st_elevation_or_acute_infarct': ['anteroseptal_st_elevation',
    #                                                 'anteroseptal_acute_infarct'],
    
    # 'anterolateral_st_elevation_or_acute_infarct': ['anterolateral_st_elevation',
    #                                                 'anterolateral_acute_infarct'],
    
    # 'inferolateral_st_elevation_or_acute_infarct': ['inferolateral_st_elevation',
    #                                                 'inferolateral_acute_infarct'],
    
    'inferior_or_posterior_acute_infarct': ['inferoposterior_acute_infarct',
                                            'inferior_acute_infarct',
                                            'posterior_acute_infarct'],
    
    
    'st_elevation_or_acute_infarct': ['anterolateral_st_elevation',
                                      'inferolateral_st_elevation',
                                      'posterolateral_st_elevation',
                                      'anteroseptal_st_elevation',
                                      'inferoposterior_st_elevation',
                                      'anterior_st_elevation',
                                      'lateral_st_elevation',
                                      'inferior_st_elevation',
                                      'posterior_st_elevation',
                                      'septal_st_elevation',
                                      'generic_st_elevation',
                                      'anterolateral_acute_infarct',
                                      'inferolateral_acute_infarct',
                                      'anteroseptal_acute_infarct',
                                      'posterolateral_acute_infarct',
                                      'inferoposterior_acute_infarct',
                                      'inferior_acute_infarct',
                                      'posterior_acute_infarct',
                                      'anterior_acute_infarct',
                                      'lateral_acute_infarct',
                                      'septal_acute_infarct',
                                      'generic_acute_infarct'],
    
    'st_elevation_or_infarct': ['anterolateral_st_elevation',
                                'inferolateral_st_elevation',
                                'posterolateral_st_elevation',
                                'anteroseptal_st_elevation',
                                'inferoposterior_st_elevation',
                                'anterior_st_elevation',
                                'lateral_st_elevation',
                                'inferior_st_elevation',
                                'posterior_st_elevation',
                                'septal_st_elevation',
                                'generic_st_elevation',
                                'anterolateral_infarct',
                                'inferolateral_infarct',
                                'anteroseptal_infarct',
                                'posterolateral_infarct',
                                'inferoposterior_infarct',
                                'inferior_infarct',
                                'posterior_infarct',
                                'anterior_infarct',
                                'lateral_infarct',
                                'septal_infarct',
                                'generic_infarct'],
    
    'st_elevation': ['anterolateral_st_elevation',
                     'inferolateral_st_elevation',
                     'posterolateral_st_elevation',
                     'anteroseptal_st_elevation',
                     'inferoposterior_st_elevation',
                     'anterior_st_elevation',
                     'lateral_st_elevation',
                     'inferior_st_elevation',
                     'posterior_st_elevation',
                     'septal_st_elevation',
                     'generic_st_elevation'],
    
     'acute_infarct': ['anterolateral_acute_infarct',
                       'inferolateral_acute_infarct',
                       'anteroseptal_acute_infarct',
                       'posterolateral_acute_infarct',
                       'inferoposterior_acute_infarct',
                       'inferior_acute_infarct',
                       'posterior_acute_infarct',
                       'anterior_acute_infarct',
                       'lateral_acute_infarct',
                       'septal_acute_infarct',
                       'generic_acute_infarct'],
    
    'prior_infarct': ['anterolateral_prior_infarct',
                      'inferolateral_prior_infarct',
                      'anteroseptal_prior_infarct',
                      'posterolateral_prior_infarct',
                      'inferoposterior_prior_infarct',
                      'inferior_prior_infarct',
                      'posterior_prior_infarct',
                      'anterior_prior_infarct',
                      'lateral_prior_infarct',
                      'septal_prior_infarct',
                      'generic_prior_infarct'],
    
    'infarct': ['anterolateral_infarct',
                'inferolateral_infarct',
                'anteroseptal_infarct',
                'posterolateral_infarct',
                'inferoposterior_infarct',
                'inferior_infarct',
                'posterior_infarct',
                'anterior_infarct',
                'lateral_infarct',
                'septal_infarct',
                'generic_infarct'],
    
    'ischemia': ['anterolateral_ischemia',
                 'inferolateral_ischemia',
                 'anteroseptal_ischemia',
                 'posterolateral_ischemia',
                 'inferoposterior_ischemia',
                 'inferior_ischemia',
                 'posterior_ischemia',
                 'anterior_ischemia',
                 'lateral_ischemia',
                 'septal_ischemia',
                 'generic_ischemia']
    
}

diagnosis_subtraction = {
    
    'nonspecific_localization_acute_infarct': [['generic_acute_infarct'],
                              ['anterolateral_acute_infarct',
                               'inferolateral_acute_infarct',
                               'anteroseptal_acute_infarct',
                               'posterolateral_acute_infarct',
                               'inferoposterior_acute_infarct',
                               'anterior_acute_infarct',
                               'lateral_acute_infarct',
                               'septal_acute_infarct',
                               'posterior_acute_infarct',
                               'inferior_acute_infarct']],
    
    'non_st_elevation_or_infarct': [['dataset'],
                    ['st_elevation_or_infarct']],
    
}