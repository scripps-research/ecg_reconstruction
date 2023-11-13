diagnosis_map = {

    # GENERAL

    'abnormal_ecg':
        
    ['general',
    [' abnormal ecg'],
    []
    ],

     'normal_ecg':
        
    ['general',
    [' normal ecg'],
    ['otherwise normal ecg']
    ],

     'borderline_ecg':
        
    ['general',
    [' borderline ecg'],
    []
    ],

     'pediatric_ecg':
        
    ['general',
    [' pediatric ecg',
     ' pediatric analysis'],
    []
    ],

    'poor_data_quality':
        
    ['general',
    [' poor data quality',
     ' low technical quality',
     ' questionable change',
     ' no ecg analysis possible',
     ' lead reversal interpretation assumes no reversal',
     ' current undetermined rhythm precludes rhythm comparison'],
    []
    ],
    

    # OTHER

    'low_voltage':
        
    ['other',
    [' low voltage'],
    []
    ],
    
    'pericarditis':
        
    ['other',
    [' pericarditis'],
    []
    ],
    
    # HYPERTROPHY

    'left_atrial_hypertrophy':
        
    ['hypertrophy',
    [' left atrial hypertrophy',
     ' left atrial enlargement',
     ' lae '],
    []
    ],   

    'right_atrial_hypertrophy':
        
    ['hypertrophy',
    [' right atrial hypertrophy',
     ' right atrial enlargement',
     ' rae '],
    []
    ],
    
    'biatrial_hypertrophy':
        
    ['hypertrophy',
    [' biatrial hypertrophy',
     ' biatrial enlargement'],
    []
    ],

    'right_ventricular_hypertrophy':
        
    ['hypertrophy',
    [' right ventricular hypertrophy',
     ' right ventricular enlargement',
     ' rvh '],
    []
    ],

    'left_ventricular_hypertrophy':
        
    ['hypertrophy',
    [' left ventricular hypertrophy',
     ' left ventricular enlargement',
     ' lvh '],
    []
    ],
    
    'biventricular_hypertrophy':
        
    ['hypertrophy',
    [' biventricular hypertrophy', 
     ' biventricular enlargement'],
    []
    ],

    # SINUS RHTHM

    'normal_sinus_rhythm':
        
    ['sinus_rhythm',
    [' sinus rhythm',
     ' regular rhythm',
     ' rhythm now sinus'],
    [' sinus arrhythmia',
     ' sinus tachycardia',
     ' sinus bradycardia']
    ], 

    'sinus_arrhythmia':
        
    ['sinus_rhythm',
    [' sinus arrhythmia'],
    []
    ],      

    'sinus_bradycardia':
        
    ['sinus_rhythm',
    [' sinus bradycardia'],
    []
    ],   

    'sinus_tachycardia':
        
    ['sinus_rhythm', 
    [' sinus tachycardia'],
    []
    ],
    
    # OTHER RHYTHM
    
    'idioventricular_rhythm':
        
    ['other_rhythm',
    [' idioventricular rhythm',
     ' idioventricular escape',
     ' idioventricular tachycardia',
     ' premature idioventricular complex'],
    []
    ],
    
    'junctional_rhythm':
        
    ['other_rhythm',
    [' junctional rhythm',
     ' junctional escape',
     ' junctional tachycardia',
     ' premature junctional complex',
     ' pjc '],
    []
    ],

    'fusion_rhythm':
        
    ['other_rhythm',
    [' fusion complex'],
    []
    ],

    'pacemaker_rhythm':
        
    ['other_rhythm',
    [' atrial paced rhythm',
     ' atrial paced complex',
     ' ventricular paced rhythm',
     ' ventricular paced complex',
     ' dual paced rhythm',
     ' dual paced complex',
     ' pacing',
     ' pacemaker'],
    []
    ],
    

    # ATRIAL ARRHYTMIA
    
    'wolff_parkinson_white':
        
    ['atrial_arrhythmia',
    [' wolff - parkinson - white',
     ' wpw '],
    []
    ],

    'atrial_nonspecific_arrhythmia':
        
    ['atrial_arrhythmia',
    [' atrial arrhythmia',
     ' supraventricular arrhythmia'],
    []
    ],

    'atrial_bradycardia':
        
    ['atrial_arrhythmia',
    [' atrial bradycardia',
     ' supraventricular bradycardia'],
    []
    ],

    'atrial_tachycardia':
        
    ['atrial_arrhythmia',
    [' atrial tachycardia',
     ' supraventricular tachycardia',
     ' av node reentrant tachycardia',
     ' av reentrant tachycardia',
     ' svt ',
     ' at ',
     ' psvt ',
     ' pat ',
     ' avnrt ',
     ' avrt '],
    []
    ],
    
    'atrial_flutter':
        
    ['atrial_arrhythmia',
    [' atrial flut',
     ' aflut'],
    []
    ],

    'atrial_fibrillation':
        
    ['atrial_arrhythmia',
    [' atrial fib',
     ' atrial are fibrillating',
     ' atrial flut - fib',
     ' afib',
     ' psvf ',
     ' paf '],
    []
    ],
    
    'premature_atrial_complex':
        
    ['atrial_arrhythmia',
    [' premature atrial complex',
     ' atrial premature complex',
     ' premature supraventricular complex',
     ' supraventricular premature complex',
     ' premature complex ventricular or supraventricular',
     ' atrial extrasystole',
     ' supraventricular extrasystole',
     ' pac '],
    []
    ],

    'atrial_bigeminy':
        
    ['atrial_arrhythmia',
    [' supraventricular bigeminy',
     ' atrial bigeminy',
     ' supraventricular complex in a bigeminy pattern',
     ' atrial complex in a bigeminy pattern'],
    []
    ],

    'atrial_trigeminy':
        
    ['atrial_arrhythmia',
    [' supraventricular trigeminy',
     ' atrial trigeminy',
     ' supraventricular complex in a trigeminy pattern',
     ' atrial complex in a trigeminy pattern'],
    []
    ],
    
    'atrial_quadrigeminy':
        
    ['atrial_arrhythmia',
    [' supraventricular quadrigeminy',
     ' atrial quadrigeminy',
     ' supraventricular complex in a quadrigeminy pattern',
     ' atrial complex in a quadrigeminy pattern'],
    []
    ],

    # VETRINCULAR ARRHYTMIAS
    
    'brugada_syndrome':
        
    ['ventricular_arrhythmia',
    [' brugada'],
    []
    ], 

    'ventricular_nonspecific_arrhythmia':
        
    ['ventricular_arrhythmia',
    [' ventricular arrhythmia'],
    []
    ], 

    'ventricular_bradycardia':
        
    ['ventricular_arrhythmia',
    [' ventricular bradycardia'],
    []
    ], 

    'ventricular_tachycardia':
        
    ['ventricular_arrhythmia', 
    [' ventricular tachycardia',
     ' ventricular outflow tract',
     ' vt ',
     ' rvot ',
     ' lvot '],
    []
    ],
    
    'ventricular_flutter':
        
    ['ventricular_arrhythmia',
    [' ventricular flut',
     ' vflut'],
    []
    ],

    'ventricular_fibrillation':
        
    ['ventricular_arrhythmia',
    [' ventricular fib',
     ' ventricular flut - fib',
     ' vfib'],    
    []
    ],

    'premature_ventricular_complex':
        
    ['ventricular_arrhythmia',
    [' premature ventricular complex',
     ' ventricular premature complex',
     ' ventricular extrasystole',
     ' premature ventricular or aberrantly conducted complex',    
     ' premature complex ventricular or aberrant supraventricular',
     ' pvc '],
    []
    ],

    'ventricular_bigeminy':
        
    ['ventricular_arrhythmia',
    [' ventricular bigeminy',
     ' ventricular complex in a bigeminy pattern'],
    []
    ],

    'ventricular_trigeminy':
        
    ['ventricular_arrhythmia',
    [' ventricular trigeminy',
     ' ventricular complex in a trigeminy pattern'],
    []
    ],
    
    'ventricular_quadrigeminy':
        
    ['ventricular_arrhythmia',
    [' ventricular quadrigeminy',
     ' ventricular complex in a quadrigeminy pattern'],
    []
    ],
    
    
    # CONDUCTION ABNORMALITY
 
    'first_degree_av_block':
        
    ['conduction_abnormality',
    [' 1st degree av block',
     ' first degree av block'],
    []
    ],

    'second_degree_av_block':
        
    ['conduction_abnormality',
    [' 2nd degree av block',
     ' second degree av block'],
    []
    ],
    
    'third_degree_av_block':
        
    ['conduction_abnormality',
    [' 3rd degree av block',
     ' third degree av block',
     ' complete av block'],
    []
    ],

    'variable_av_block':
        
    ['conduction_abnormality',
    [' variable av block'],
    []
    ],
 
    'sinoatrial_block':
        
    ['conduction_abnormality',
    [' sa block',
     ' sa node exit block',
     ' sa node block'],
    []
    ],
    
    'nonspecific_intraventricular_block':
        
    ['conduction_abnormality',
    [' nonspecific intraventricular block'],
    []
    ], 

    'right_bundle_branch_block':
        
    ['conduction_abnormality',
    [' right bundle branch block',
     ' rbbb '],
    ['incomplete right bundle branch block',
     'incomplete rbbb']
    ],

    'incomplete_right_bundle_branch_block':
        
    ['conduction_abnormality',
    [' incomplete right bundle branch block',
     ' incomplete rbbb'],
    []
    ],

    'left_bundle_branch_block':
        
    ['conduction_abnormality',
    [' left bundle branch block',
     ' lbbb '],
    ['incomplete left bundle branch block',
     'incomplete lbbb']
    ],

    'incomplete_left_bundle_branch_block':
        
    ['conduction_abnormality', 
    [' incomplete left bundle branch block',
     ' incomplete lbbb'],
    []
    ],

    'left_anterior_fascicular_block':
        
    ['conduction_abnormality',     
    [' left anterior fascicular block',
     ' lafb '],
    []
    ],

    'left_posterior_fascicular_block':        
    ['conduction_abnormality',
    [' left posterior fascicular block',
     ' lpfb '],
    []
    ],

    'bifascicular_block':
        
    ['conduction_abnormality',
    [' bifascicular block'],
    []
    ],
    
    'trifascicular_block':
        
    ['conduction_abnormality',
    [' trifascicular block'],
    []
    ],

    'intraventricular_conduction_delay':
        
    ['conduction_abnormality',
    [' intraventricular conduction delay',
     ' ivcd '],
    []
    ], 
    
    'right_ventricular_conduction_delay':
        
    ['conduction_abnormality',
    [' right ventricular conduction delay',
     ' rvcd '],
    []
    ], 
    
    'left_ventricular_conduction_delay':
        
    ['conduction_abnormality',
    [' left ventricular conduction delay',
     ' lvcd '],
    []
    ],  
    
    'aberrant_condution':
        
    ['conduction_abnormality',
    [' aberrant conduction'],
    []
    ],
    
    'wide_qrs':
        
    ['conduction_abnormality',
    [' qrs widening',
     ' wide qrs '],
    []
    ],

    
    'short_pr':
        
    ['conduction_abnormality',
    [' short pr'],
    []
    ],

    'long_qt':
        
    ['conduction_abnormality',
    [' prolonged qt',
     ' qt interval long'],
    []
    ],


    # QRS AXIS DEVIATION

    'right_axis_deviation':
        
    ['axis_deviation',
    [' right axis deviation',
     ' right superior axis deviation',
     ' right inferior axis deviation',
     ' axis shifted right',
     ' axis shifted more right',
     ' axis shifted slightly right',
     ' axis shifted from left to right',
     ' rightward axis',
     ' rad '],
    []
    ], 

    'left_axis_deviation':
        
    ['axis_deviation',
    [' left axis deviation',
     ' left superior axis deviation',
     ' left inferior axis deviation',
     ' axis shifted left',
     ' axis shifted more left',
     ' axis shifted slightly left',
     ' axis shifted from right to left',
     ' leftward axis',
     ' lad '],
    []
    ],
    
    'extreme_axis_deviation':
        
    ['axis_deviation',
    [' extreme axis'],
    []
    ],


    # ACUTE INFARCT
    
    'anterior_acute_infarct':
        
    ['acute_infarct',     
    [' anterior stemi ',
     ' acute anterior infarct',
     ' acute infarct in anterior',
     ' anterior acute infarct',
     ' anterior infarct acute'],  
    []
    ],

    'lateral_acute_infarct':
        
    ['acute_infarct',     
    [' lateral stemi ',
     ' acute lateral infarct',
     ' acute infarct in lateral',
     ' lateral acute infarct',
     ' lateral infarct acute'],  
    []
    ],
    
    'inferior_acute_infarct':
        
    ['acute_infarct',     
    [' inferior stemi ',
     ' acute inferior infarct',
     ' acute infarct in inferior',
     ' inferior acute infarct',
     ' inferior infarct acute'],
    []
    ],

    'posterior_acute_infarct':
        
    ['acute_infarct',     
    [' posterior stemi ',     
     ' acute posterior infarct',
     ' acute infarct in posterior', 
     ' posterior acute infarct',
     ' posterior infarct acute'],
    []
    ],
    
    'septal_acute_infarct':
        
    ['acute_infarct',     
    [' septal stemi ',
     ' acute septal infarct',
     ' acute infarct in septal',
     ' septal acute infarct',
     ' septal infarct acute'],  
    []
    ],
    
    'anteroseptal_acute_infarct':
        
    ['acute_infarct',     
    [' anteroseptal stemi ',
     ' acute anteroseptal infarct',
     ' acute infarct in anteroseptal',
     ' anteroseptal acute infarct',
     ' anteroseptal infarct acute'],  
    []
    ],
    
    'anterolateral_acute_infarct':
        
    ['acute_infarct',     
    [' anterolateral stemi ',
     ' acute anterolateral infarct',
     ' acute infarct in anterolateral',
     ' anterolateral acute infarct',
     ' anterolateral infarct acute'],
    []
    ],

    'inferolateral_acute_infarct':
        
    ['acute_infarct',     
    [' inferolateral stemi ',
     ' acute inferolateral infarct',
     ' acute infarct in inferolateral',
     ' inferolateral acute infarct',
     ' inferolateral infarct acute'],  
    []
    ],
    
    'posterolateral_acute_infarct':
        
    ['acute_infarct',     
    [' posterolateral stemi ',
     ' acute posterolateral infarct',
     ' acute infarct in posterolateral',
     ' posterolateral acute infarct',
     ' posterolateral infarct acute'],  
    []
    ],
    
    'inferoposterior_acute_infarct':
        
    ['acute_infarct',     
    [' inferoposterior stemi ',
     ' acute inferoposterior infarct',
     ' acute infarct in inferoposterior',
     ' inferoposterior acute infarct',
     ' inferoposterior infarct acute'],  
    []
    ],
    
    
    'generic_acute_infarct':
        
    ['acute_infarct',
    [' stemi ',
     ' acute infarct',
     ' infarct acute'],    
    [' anterior stemi ',
     ' acute anterior infarct',
     ' acute infarct in anterior',
     ' anterior acute infarct',
     ' anterior infarct acute',
     ' lateral stemi ',
     ' acute lateral infarct',
     ' acute infarct in lateral',
     ' lateral acute infarct',
     ' lateral infarct acute',
     ' inferior stemi ',
     ' acute inferior infarct',
     ' acute infarct in inferior',
     ' inferior acute infarct',
     ' inferior infarct acute',
     ' posterior stemi ',     
     ' acute posterior infarct',
     ' acute infarct in posterior', 
     ' posterior acute infarct',
     ' posterior infarct acute',
     ' septal stemi ',
     ' acute septal infarct',
     ' acute infarct in septal',
     ' septal acute infarct',
     ' septal infarct acute',
     ' anteroseptal stemi ',
     ' acute anteroseptal infarct',
     ' acute infarct in anteroseptal',
     ' anteroseptal acute infarct',
     ' anteroseptal infarct acute',
     ' anterolateral stemi ',
     ' acute anterolateral infarct',
     ' acute infarct in anterolateral',
     ' anterolateral acute infarct',
     ' anterolateral infarct acute',
     ' inferolateral stemi ',
     ' acute inferolateral infarct',
     ' acute infarct in inferolateral',
     ' inferolateral acute infarct',
     ' inferolateral infarct acute',
     ' posterolateral stemi ',
     ' acute posterolateral infarct',
     ' acute infarct in posterolateral',
     ' posterolateral acute infarct',
     ' posterolateral infarct acute',
     ' inferoposterior stemi ',
     ' acute inferoposterior infarct',
     ' acute infarct in inferoposterior',
     ' inferoposterior acute infarct',
     ' inferoposterior infarct acute']
    ],
    
    # PRIOR INFARCT
    
    'anterior_prior_infarct':
        
    ['prior_infarct',     
    [' anterior infarct',
     ' infarct in anterior'],
    ['acute anterior infarct',
     'acute infarct in anterior',
     'anterior infarct acute']
    ],
    
    'lateral_prior_infarct':
        
    ['prior_infarct',     
    [' lateral infarct',
     ' infarct in lateral'],
    ['acute lateral infarct',
     'acute infarct in lateral',
     'lateral infarct acute']
    ],
    
    'inferior_prior_infarct':
        
    ['prior_infarct',     
    [' inferior infarct',
     ' infarct in inferior'],
    ['acute inferior infarct',
     'acute infarct in inferior',
     'inferior infarct acute']
    ],
    
    'posterior_prior_infarct':
        
    ['prior_infarct',     
    [' posterior infarct',
     ' infarct in posterior'],
    ['acute posterior infarct',
     'acute infarct in posterior',
     'posterior infarct acute']
    ],
    
    'septal_prior_infarct':
        
    ['prior_infarct',     
    [' septal infarct',
     ' infarct in septal'],
    ['acute septal infarct',
     'acute infarct in septal',
     'septal infarct acute']
    ],
    
    'anteroseptal_prior_infarct':
        
    ['prior_infarct',     
    [' anteroseptal infarct',
     ' infarct in anteroseptal'],
    ['acute anteroseptal infarct',
     'acute infarct in anteroseptal',
     'anteroseptal infarct acute']
    ],
    
    'anterolateral_prior_infarct':
        
    ['prior_infarct',     
    [' anterolateral infarct',
     ' infarct in anterolateral'],
    ['acute anterolateral infarct',
     'acute infarct in anterolateral',
     'anterolateral infarct acute']
    ],
    
    'inferolateral_prior_infarct':
        
    ['prior_infarct',     
    [' inferolateral infarct',
     ' infarct in inferolateral'],
    ['acute inferolateral infarct',
     'acute infarct in inferolateral',
     'inferolateral infarct acute']
    ],
    
    'posterolateral_prior_infarct':
        
    ['prior_infarct',     
    [' posterolateral infarct',
     ' infarct in posterolateral'],
    ['acute posterolateral infarct',
     'acute infarct in posterolateral',
     'posterolateral infarct acute']
    ],
    
    'inferoposterior_prior_infarct':
        
    ['prior_infarct',     
    [' inferoposterior infarct',
     ' infarct in inferoposterior'],
    ['acute inferoposterior infarct',
     'acute infarct in inferoposterior',
     'inferoposterior infarct acute']
    ],
    
    'generic_prior_infarct':
        
    ['prior_infarct',
    [' infarct'],
    ['acute infarct',
     'infarct acute',          
     'anterior infarct',
     'infarct in anterior',     
     'lateral infarct',
     'infarct in lateral',     
     'septal infarct',
     'infarct in septal',     
     'inferior infarct',
     'infarct in inferior',     
     'posterior infarct',
     'infarct in posterior',          
     'anteroseptal infarct',
     'infarct in anteroseptal',   
     'anterolateral infarct',
     'infarct in anterolateral',         
     'inferolateral infarct',
     'infarct in inferolateral',       
     'posterolateral infarct',
     'infarct in posterolateral',
     'inferoposterior infarct',
     'infarct in inferoposterior']
    ],
    
    # INFARCT
    
    'anterior_infarct':
        
    ['infarct',     
    [' anterior stemi ',
     ' anterior infarct',
     ' infarct in anterior',
     ' anterior acute infarct'],
    []
    ],
    
    'lateral_infarct':
        
    ['infarct',     
    [' lateral stemi ',
     ' lateral infarct',
     ' infarct in lateral',
     ' lateral acute infarct'],
    []
    ],
    
    'inferior_infarct':
        
    ['infarct',     
    [' inferior stemi ',
     ' inferior infarct',
     ' infarct in inferior',
     ' inferior acute infarct'],
    []
    ],
    
    'posterior_infarct':
        
    ['infarct',     
    [' posterior stemi ',
     ' posterior infarct',
     ' infarct in posterior',
     ' posterior acute infarct'],
    []
    ],
    
    'septal_infarct':
        
    ['infarct',     
    [' septal stemi ',
     ' septal infarct',
     ' infarct in septal',
     ' septal acute infarct'],
    []
    ],
    
    'anteroseptal_infarct':
        
    ['infarct',     
    [' anteroseptal stemi ',
     ' anteroseptal infarct',
     ' infarct in anteroseptal',
     ' anteroseptal acute infarct'],
    []
    ],
    
    'anterolateral_infarct':
        
    ['infarct',     
    [' anterolateral stemi ',
     ' anterolateral infarct',
     ' infarct in anterolateral',
     ' anterolateral acute infarct'],
    []
    ],
    
    'inferolateral_infarct':
        
    ['infarct',     
    [' inferolateral stemi ',
     ' inferolateral infarct',
     ' infarct in inferolateral',
     ' inferolateral acute infarct'],
    []
    ],
    
    'posterolateral_infarct':
        
    ['infarct',     
    [' posterolateral stemi ',
     ' posterolateral infarct',
     ' infarct in posterolateral',
     ' posterolateral acute infarct'],
    []
    ],
    
    'inferoposterior_infarct':
        
    ['infarct',     
    [' inferoposterior stemi ',
     ' inferoposterior infarct',
     ' infarct in inferoposterior',
     ' inferoposterior acute infarct'],
    []
    ],
    
    'generic_infarct':
        
    ['infarct',
    [' stemi',
     ' infarct',
     ' injury'],
    ['anterior infarct',
     'infarct in anterior',
     'anterior stemi ',     
     'anterior acute infarct',  
     'lateral infarct',
     'infarct in lateral',     
     'lateral stemi ',      
     'lateral acute infarct', 
     'septal infarct',
     'infarct in septal', 
     'septal stemi ',       
     'septal acute infarct',   
     'inferior infarct',
     'infarct in inferior',      
     'inferior stemi ',     
     'inferior acute infarct',         
     'posterior stemi ',      
     'posterior acute infarct',   
     'posterior infarct',
     'infarct in posterior',           
     'anteroseptal infarct',
     'infarct in anteroseptal',
     'anterolateral infarct',
     'infarct in anterolateral',     
     'posterolateral infarct',
     'infarct in posterolateral',
     'inferolateral infarct',
     'infarct in inferolateral',
     'inferoposterior infarct',
     'infarct in inferoposterior']
    ],
    
    # ISCHEMIA
    
    'anterolateral_ischemia':
        
    ['ischemia',
    [' anterolateral ischemia',
     ' anterolateral acute ischemia',
     ' anterolateral infarct or ischemia'],
    []
    ],

    'inferolateral_ischemia':
        
    ['ischemia',
    [' inferolateral ischemia',
     ' inferolateral acute ischemia',
     ' inferolateral infarct or ischemia'],
    []
    ],
    
    'posterolateral_ischemia':
        
    ['ischemia',
    [' posterolateral ischemia',
     ' posterolateral acute ischemia',
     ' posterolateral infarct or ischemia'],
    []
    ],
    
    'inferoposterior_ischemia':
        
    ['ischemia',
    [' inferoposterior ischemia',
     ' inferoposterior acute ischemia',
     ' inferoposterior infarct or ischemia'],
    []
    ],

    'anteroseptal_ischemia':
        
    ['ischemia',
    [' anteroseptal ischemia',
     ' anteroseptal acute ischemia',
     ' anteroseptal infarct or ischemia'],
    []
    ],
    
    'septal_ischemia':
        
    ['ischemia',
    [' septal ischemia',
     ' septal acute ischemia',
     ' septal infarct or ischemia'],
    []
    ], 

    'anterior_ischemia':
        
    ['ischemia', 
    [' anterior ischemia',
     ' anterior acute ischemia',
     ' anterior infarct or ischemia'],
    []
    ],

    'lateral_ischemia':
        
    ['ischemia',
    [' lateral ischemia',
     ' lateral acute ischemia',
     ' lateral infarct or ischemia'],
    [],
    ],

    'inferior_ischemia':
        
    ['ischemia',
    [' inferior ischemia',
     ' inferior acute ischemia',
     ' inferior infarct or ischemia'],
    []
    ],
    
    'posterior_ischemia':
        
    ['ischemia',
    [' posterior ischemia',
     ' posterior acute ischemia',
     ' posterior infarct or ischemia'],
    []
    ],

    'generic_ischemia':
        
    ['ischemia',
    [' ischemia'],
    ['anterior ischemia',
     'lateral ischemia',
     'septal ischemia',
     'inferior ischemia',
     'posterior ischemia',
     'anterior acute ischemia',
     'lateral acute ischemia',
     'septal acute ischemia',
     'inferior acute ischemia',
     'posterior acute ischemia',
     'anterior infarct or ischemia',
     'lateral infarct or ischemia',
     'septal infarct or ischemia',
     'inferior infarct or ischemia',
     'posterior infarct or ischemia']
    ],
    
    
    # ST ELEVATION
    
    'anterior_st_elevation':
        
    ['st_elevation',
    [' anterior st elevation'],
    [],
    ],
    
    'anterolateral_st_elevation':
        
    ['st_elevation',
    [' anterolateral st elevation'],
    [],
    ],
    
    'inferolateral_st_elevation':
        
    ['st_elevation',
    [' inferolateral st elevation'],
    [],
    ],
    
    'posterolateral_st_elevation':
        
    ['st_elevation',
    [' posterolateral st elevation'],
    [],
    ],
    
    'lateral_st_elevation':
        
    ['st_elevation',
    [' lateral st elevation'],
    [],
    ],
    
    'septal_st_elevation':
        
    ['st_elevation',
    [' septal st elevation'],
    [],
    ],
    
    'anteroseptal_st_elevation':
        
    ['st_elevation',
    [' anteroseptal st elevation'],
    [],
    ],
    
    'inferior_st_elevation':
        
    ['st_elevation',
    [' inferior st elevation'],
    [],
    ],
    
    'posterior_st_elevation':
        
    ['st_elevation',
    [' posterior st elevation'],
    [],
    ],
    
    'inferoposterior_st_elevation':
        
    ['st_elevation',
    [' inferoposterior st elevation'],
    [],
    ],
    
    'generic_st_elevation':
        
    ['st_elevation',
    [' st elevation'],
    ['inferior st elevation',
     'posterior st elevation',
     'septal st elevation',
     'lateral st elevation',
     'anterior st elevation'],
    ],
    
    'st_elevation_dueto_erpc':
        
    ['st_elevation',
    [' st elevation due to early repolarization or pericarditis'],
    []
    ],
    
    # REPOLARIZATION ABNORMALITY    
    
    'st_depression':
        
    ['repolarization_abnormality',
    [' depressed st',
     ' st depression',
     ' st wave depression',
     ' st more depressed',
     ' st now depressed'],
    [],
    ],    
    
    'st_abnormality':
        
    ['repolarization_abnormality',
    [' st wave abnormality',
     ' st wave change',
     ' st abnormality',
     ' st change',
     ' st and t wave abnormality',
     ' st and t wave change',
     ' st and t abnormality',
     ' st and t change',
     ' change in st segment'],
    []
    ],    

    't_abnormality':
        
    ['repolarization_abnormality',
    [' t wave abnormality',
     ' t wave change',
     ' t abnormality',    
     ' t change',    
     ' change in t segment'],
    []
    ],

    'nonspecific_wave_abnormality':
        
    ['repolarization_abnormality',
    [' repolarization abnormality',
     ' repolarization change'],
    []
    ],
    
    
    'early_repolarization':
        
    ['repolarization_abnormality',
    [' early repolarization'],
    []
    ],
    
    't_inversion':
        
    ['repolarization_abnormality',
    [' inverted t ',
     ' t inversion',
     ' t wave inversion',
     ' t wave now inverted'],
    [],
    ],
    
    't_flattening':
        
    ['repolarization_abnormality',
    [' flattened t ',
     ' t wave flattening',
     ' t flattening',
     ' t wave now flat'],
    [],
    ], 

    }