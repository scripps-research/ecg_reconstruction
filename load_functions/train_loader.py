from load_functions.data_loader import DataLoader
from load_functions.load_leads import load_data_leads, load_element_leads


class ReconstructionDataLoader(DataLoader):
    def __init__(self,
                 parent_folder: str,
                 data_classes: str,
                 data_size: int,
                 batch_size: int,
                 prioritize_percent: float,
                 prioritize_size: int,
                 sample_num: int,
                 min_value: float,
                 amplitude: float,
                 input_lead_keys,
                 output_lead_keys):

        self.input_lead_keys = input_lead_keys
        self.output_lead_keys = output_lead_keys
        
        super().__init__(parent_folder, data_classes, data_size, [], batch_size, prioritize_percent, prioritize_size, sample_num, min_value, amplitude)

    def load_element(self, element_id, extract_qrs: bool):

        return load_element_leads(self.collection, element_id, self.data_ids_per_detect_class, self.input_lead_keys, self.output_lead_keys, self.sample_num, self.min_value, self.amplitude, extract_qrs)
    
    def load_data(self, data_ids, extract_qrs: bool):
    
        return load_data_leads(self.collection, data_ids, self.data_ids_per_detect_class, self.input_lead_keys, self.output_lead_keys, self.sample_num, self.min_value, self.amplitude, extract_qrs)
        
        
class ClassificationDataLoader(DataLoader):
    def __init__(self,
                 parent_folder: str,
                 data_classes,
                 data_size: int,
                 detect_classes,
                 batch_size: int,
                 prioritize_percent: float,
                 prioritize_size: int,
                 sample_num: int,
                 min_value: float,
                 amplitude: float,
                 lead_keys):

        self.lead_keys = lead_keys
        
        super().__init__(parent_folder, data_classes, data_size, detect_classes, batch_size, prioritize_percent, prioritize_size, sample_num, min_value, amplitude)
        
    def load_element(self, element_id, extract_qrs: bool):

        return load_element_leads(self.collection, element_id, self.data_ids_per_detect_class, self.lead_keys, [], self.sample_num, self.min_value, self.amplitude, extract_qrs)
    
    def load_data(self, data_ids, extract_qrs: bool):
    
        return load_data_leads(self.collection, data_ids, self.data_ids_per_detect_class, self.lead_keys, [], self.sample_num, self.min_value, self.amplitude, extract_qrs)
    
    
class ReconClassifDataLoader(DataLoader):
    def __init__(self,
                 parent_folder: str,
                 data_classes,
                 data_size: int,
                 detect_classes,
                 batch_size: int,
                 prioritize_percent: float,
                 prioritize_size: int,
                 sample_num: int,
                 min_value: float,
                 amplitude: float,
                 recon_input_lead_keys,
                 classif_input_lead_keys):

        self.recon_input_lead_keys = recon_input_lead_keys
        self.classif_input_lead_keys = classif_input_lead_keys
        
        super().__init__(parent_folder, data_classes, data_size, detect_classes, batch_size, prioritize_percent, prioritize_size, sample_num, min_value, amplitude)

    def load_element(self, element_id, extract_qrs: bool):

        return load_element_leads(self.collection, element_id, self.data_ids_per_detect_class, self.recon_input_lead_keys, self.classif_input_lead_keys, self.sample_num, self.min_value, self.amplitude, extract_qrs)

    def load_data(self, data_ids, extract_qrs: bool):
    
        return load_data_leads(self.collection, data_ids, self.data_ids_per_detect_class, self.recon_input_lead_keys, self.classif_input_lead_keys, self.sample_num, self.min_value, self.amplitude, extract_qrs)
