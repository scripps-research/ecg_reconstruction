import random
import numpy as np
from util_functions.general import get_collection
from util_functions.load_data_ids import load_dataclass_ids, load_learning_ids
from classify_functions.process_diagnosis import format_diagnosis
from load_functions.load_leads import load_element_twelve_leads
from itertools import chain


class DataLoader(object):
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
                 amplitude: float):

        self.collection = get_collection()
        
        self.parent_folder = parent_folder
        self.data_classes = data_classes

        self.train_data_ids, self.valid_data_ids, self.test_data_ids =\
            load_learning_ids(self.parent_folder, self.data_classes, data_size)
            
        self.train_data_size = len(self.train_data_ids)
        self.valid_data_size = len(self.valid_data_ids)
        self.test_data_size = len(self.test_data_ids)
        
        self.train_data_index = 0
        self.valid_data_index = 0
        self.test_data_index = 0
        
        self.detect_classes = detect_classes            
        self.detect_class_num = len(self.detect_classes)
        
        self.batch_size = batch_size            

        self.data_ids_per_detect_class = []
        
        self.train_data_ids_per_detect_class = []
        self.valid_data_ids_per_detect_class = []
        self.test_data_ids_per_detect_class = []
        
        self.train_data_size_per_detect_class = []
        self.valid_data_size_per_detect_class = []
        self.test_data_size_per_detect_class = []
        
        remaining_train_ids = set(self.train_data_ids)
        remaining_valid_ids = set(self.valid_data_ids)
        remaining_test_ids = set(self.test_data_ids)
    
        for data_class in detect_classes:
            
            class_ids = load_dataclass_ids(parent_folder, data_class)
            
            self.data_ids_per_detect_class.append(class_ids)
            
            train_ids = list(set(self.train_data_ids) & set(class_ids))
            valid_ids = list(set(self.valid_data_ids) & set(class_ids))
            test_ids = list(set(self.test_data_ids) & set(class_ids))
            
            remaining_train_ids = remaining_train_ids.difference(set(class_ids))
            remaining_valid_ids = remaining_valid_ids.difference(set(class_ids))
            remaining_test_ids = remaining_test_ids.difference(set(class_ids))
            
            self.train_data_ids_per_detect_class.append(train_ids)
            self.valid_data_ids_per_detect_class.append(valid_ids)
            self.test_data_ids_per_detect_class.append(test_ids)
            
            self.train_data_size_per_detect_class.append(len(train_ids))
            self.valid_data_size_per_detect_class.append(len(valid_ids))
            self.test_data_size_per_detect_class.append(len(test_ids))
            
        remaining_train_ids = list(remaining_train_ids)
        remaining_valid_ids = list(remaining_valid_ids)
        remaining_test_ids = list(remaining_test_ids)
        
        if len(remaining_train_ids) > 0:
            
            self.train_data_ids_per_detect_class.append(remaining_train_ids)
            self.valid_data_ids_per_detect_class.append(remaining_valid_ids)
            self.test_data_ids_per_detect_class.append(remaining_test_ids)
            
            self.train_data_size_per_detect_class.append(len(remaining_train_ids))
            self.valid_data_size_per_detect_class.append(len(remaining_valid_ids))
            self.test_data_size_per_detect_class.append(len(remaining_test_ids))
            
            self.learn_class_num = self.detect_class_num + 1
            
        else:
            
            self.learn_class_num = self.detect_class_num
            
        train_sorted_indexes = np.argsort(self.train_data_size_per_detect_class)
        
        for i, index in enumerate(train_sorted_indexes):
            
            train_data_ids = set(self.train_data_ids_per_detect_class[index])
                
            for other_index in train_sorted_indexes[i+1:]:
                self.train_data_ids_per_detect_class[other_index] = list(set(self.train_data_ids_per_detect_class[other_index]) - train_data_ids)
                
        valid_sorted_indexes = np.argsort(self.valid_data_size_per_detect_class)
        
        for i, index in enumerate(valid_sorted_indexes):
            
            valid_data_ids = set(self.valid_data_ids_per_detect_class[index])
                
            for other_index in valid_sorted_indexes[i+1:]:
                self.valid_data_ids_per_detect_class[other_index] = list(set(self.valid_data_ids_per_detect_class[other_index]) - valid_data_ids)
                
        test_sorted_indexes = np.argsort(self.test_data_size_per_detect_class)
        
        for i, index in enumerate(test_sorted_indexes):
            
            test_data_ids = set(self.test_data_ids_per_detect_class[index])
                
            for other_index in test_sorted_indexes[i+1:]:
                self.test_data_ids_per_detect_class[other_index] = list(set(self.test_data_ids_per_detect_class[other_index]) - test_data_ids)
                
        self.train_data_size_per_detect_class = [len(data_ids) for data_ids in self.train_data_ids_per_detect_class]
        self.valid_data_size_per_detect_class = [len(data_ids) for data_ids in self.valid_data_ids_per_detect_class]
        self.test_data_size_per_detect_class = [len(data_ids) for data_ids in self.test_data_ids_per_detect_class]
        
        self.train_loaded = False  
        self.valid_loaded = False
        self.test_loaded = False
        
        self.batch_size_per_detect_class = int(self.batch_size / (self.learn_class_num))
        self.missing_size = self.batch_size - self.batch_size_per_detect_class
        
        self.prioritize_percent = prioritize_percent
        self.prioritize_size = prioritize_size
                
        self.new_batch_size = int(self.batch_size * (1 - self.prioritize_percent))
        self.priority_batch_size = self.batch_size - self.new_batch_size

        self.prioritize_memory = []
        self.prioritize_weights = []
        
        self.sample_num = sample_num
        self.min_value = min_value
        self.amplitude = amplitude
        
    def load_element_twelve_leads(self, element_id):
        
        return load_element_twelve_leads(self.collection, element_id, self.sample_num, self.min_value, self.amplitude)

    def release_dataset(self):
        
        self.train_data_index = 0
        self.valid_data_index = 0
        self.test_data_index = 0

        self.train_data_per_detect_class = [None for _ in range(self.learn_class_num)]
        self.valid_data_per_detect_class = [None for _ in range(self.learn_class_num)]
        self.test_data_per_detect_class = [None for _ in range(self.learn_class_num)]

        self.train_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]
        self.valid_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]
        self.test_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]

        self.train_loaded = False  
        self.valid_loaded = False 
        self.test_loaded = False        
        
    def extract_element_raw_diagnosis(self, element_id):
        
        element = self.collection.find_one({"_id": element_id})
        raw_diagnosis = element['RestingECG']['Diagnosis']['DiagnosisStatement']
        
        return raw_diagnosis
        
    def extract_element_diagnosis(self, element_id):
        
        element = self.collection.find_one({"_id": element_id})
        raw_diagnosis = element['RestingECG']['Diagnosis']['DiagnosisStatement']
        diagnosis = format_diagnosis(raw_diagnosis, maintain_number=True)
        
        return diagnosis[0]

    def load_element(self, element_id, extract_qrs):

        return 
    
    def load_data(self, data_ids, extract_qrs):
    
        return
    
        
    def load(self, train: bool, valid: bool, test: bool, extract_qrs: bool):
        
        self.train_data_index = 0
        self.valid_data_index = 0
        self.test_data_index = 0
    
        if train:
            self.train_data_per_detect_class = [self.load_data(train_data_ids, extract_qrs) for train_data_ids in self.train_data_ids_per_detect_class]
            self.train_loaded = True
            
        else:
            self.train_data_per_detect_class = [None for _ in range(self.learn_class_num)]
            self.train_loaded = False

        if valid:
            self.valid_data_per_detect_class = [self.load_data(valid_data_ids, extract_qrs) for valid_data_ids in self.valid_data_ids_per_detect_class]
            self.valid_loaded = True
            
        else:
            self.valid_data_per_detect_class = [None for _ in range(self.learn_class_num)]
            self.valid_loaded = False

        if test:
            self.test_data_per_detect_class = [self.load_data(test_data_ids, extract_qrs) for test_data_ids in self.test_data_ids_per_detect_class]
            self.test_loaded = True
            
        else:
            self.test_data_per_detect_class = [None for _ in range(self.learn_class_num)]
            self.test_loaded = False

        self.train_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]
        self.valid_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]
        self.test_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]


    def shuffle(self, subset: str):

        if subset == 'train':

            if self.train_loaded:
                
                for index in range(self.learn_class_num):
                    train_data, train_data_ids = self.train_data_per_detect_class[index], self.train_data_ids_per_detect_class[index]
                    temp = list(zip(train_data, train_data_ids))
                    random.shuffle(temp)
                    train_data, train_data_ids = zip(*temp)
                    self.train_data_per_detect_class[index], self.train_data_ids_per_detect_class[index] = list(train_data), list(train_data_ids)
                
            else:
                
                for index in range(self.learn_class_num):
                    train_data_ids = self.train_data_ids_per_detect_class[index]
                    random.shuffle(train_data_ids)
                    self.train_data_ids_per_detect_class[index] = list(train_data_ids)
                
            self.train_data_index = 0
            self.train_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]

        elif subset == 'valid':

            if self.valid_loaded:
                
                for index in range(self.learn_class_num):
                    valid_data, valid_data_ids = self.valid_data_per_detect_class[index], self.valid_data_ids_per_detect_class[index]
                    temp = list(zip(valid_data, valid_data_ids))
                    random.shuffle(temp)
                    valid_data, valid_data_ids = zip(*temp)
                    self.valid_data_per_detect_class[index], self.valid_data_ids_per_detect_class[index] = list(valid_data), list(valid_data_ids)
            
            else:
                
                for index in range(self.learn_class_num):
                    valid_data_ids = self.valid_data_ids_per_detect_class[index]
                    random.shuffle(valid_data_ids)
                    self.valid_data_ids_per_detect_class[index] = list(valid_data_ids)
                
            self.valid_data_index = 0
            self.valid_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]

        elif subset == 'test':

            if self.test_loaded:
                
                for index in range(self.learn_class_num):
                    test_data, test_data_ids = self.test_data_per_detect_class[index], self.test_data_ids_per_detect_class[index]
                    temp = list(zip(test_data, test_data_ids))
                    random.shuffle(temp)
                    test_data, test_data_ids = zip(*temp)
                    self.test_data_per_detect_class[index], self.test_data_ids_per_detect_class[index] = list(test_data), list(test_data_ids)
                
            else:
                
                for index in range(self.learn_class_num):
                    test_data_ids = self.test_data_ids_per_detect_class[index]
                    random.shuffle(test_data_ids)
                    self.test_data_ids_per_detect_class[index] = list(test_data_ids)
                
            self.test_data_index = 0
            self.test_data_index_per_detect_class = [0 for _ in range(self.learn_class_num)]
        
        else:
            raise ValueError
        
        
    def get_random_data_ids(self, subset: str, size: int = 1):
        
        if subset == 'train':
    
            return random.sample(self.train_data_ids, size)

        elif subset == 'valid':

            return random.sample(self.valid_data_ids, size)

        elif subset == 'test':

            return random.sample(self.test_data_ids, size)
        
        else:
            raise ValueError


    def get_random_element(self, subset: str, extract_qrs = False):

        if subset == 'train':
            
            random_index = np.random.randint(self.train_data_size)

            if self.train_loaded:

                return random.sample(list(set(chain.from_iterable(self.train_data_per_detect_class))))
            
            else:

                element_id = self.train_data_ids[random_index]              

                return self.load_element(element_id, extract_qrs)

        elif subset == 'valid':
            
            random_index = np.random.randint(self.valid_data_size)

            if self.valid_loaded:

                return random.sample(list(set(chain.from_iterable(self.valid_data_per_detect_class))))
            
            else:

                element_id = self.valid_data_ids[random_index]

                return self.load_element(element_id, extract_qrs)
            
        elif subset == 'test':
            
            random_index = np.random.randint(self.test_data_size)

            if self.test_loaded:

                return random.sample(list(set(chain.from_iterable(self.test_data_per_detect_class))))
            
            else:

                element_id = self.test_data_ids[random_index]     

                return self.load_element(element_id, extract_qrs)
        
        else:
            raise ValueError
            

    def get_next_element(self, subset: str, extract_qrs = False):

        if subset == 'train':

            if self.train_data_index < self.train_data_size:

                if self.train_loaded:
                    
                    total_train_data_size = 0
                    
                    for train_data, train_data_size in zip(self.train_data_per_detect_class, self.train_data_size_per_detect_class):
                        
                        if self.train_data_index < total_train_data_size + train_data_size:
                        
                            element = train_data[self.train_data_index - total_train_data_size]
                            break
                            
                        else:
                            
                            total_train_data_size += train_data_size
                            

                else:

                    element_id = self.train_data_ids[self.train_data_index]
                    element = self.load_element(element_id, extract_qrs)

                self.train_data_index += 1

                return element

            else:
                return None

        elif subset == 'valid':

            if self.valid_data_index < self.valid_data_size:

                if self.valid_loaded:
                    
                    total_valid_data_size = 0
                    
                    for valid_data, valid_data_size in zip(self.valid_data_per_detect_class, self.valid_data_size_per_detect_class):
                        
                        if self.valid_data_index < total_valid_data_size + valid_data_size:
                        
                            element = valid_data[self.valid_data_index - total_valid_data_size]
                            break
                            
                        else:
                            
                            total_valid_data_size += valid_data_size

                else:

                    element_id = self.valid_data_ids[self.valid_data_index]
                    element = self.load_element(element_id, extract_qrs)

                self.valid_data_index += 1

                return element

            else:
                return None

        elif subset == 'test':

            if self.test_data_index < self.test_data_size:

                if self.test_loaded:
                    
                    total_test_data_size = 0
                    
                    for test_data, test_data_size in zip(self.test_data_per_detect_class, self.test_data_size_per_detect_class):
                        
                        if self.test_data_index < total_test_data_size + test_data_size:
                        
                            element = test_data[self.test_data_index - total_test_data_size]
                            break
                            
                        else:
                            
                            total_test_data_size += test_data_size

                else:

                    element_id = self.test_data_ids[self.test_data_index]
                    element = self.load_element(element_id, extract_qrs)

                self.test_data_index += 1

                return element

            else:
                return None
        
        else:

            raise ValueError


    def get_random_batch(self, subset: str, extract_qrs = False):

        if subset == 'train':

            if self.train_loaded:

                return random.sample(list(set(chain.from_iterable(self.train_data_per_detect_class))))

            else:
                
                random_indexes = np.random.randint(0, self.train_data_size, self.batch_size)

                batch_ids = [self.train_data_ids[index] for index in random_indexes]

                return self.load_data(batch_ids, extract_qrs)

        elif subset == 'valid':

            if self.valid_loaded:

                return random.sample(list(set(chain.from_iterable(self.valid_data_per_detect_class))))

            else:

                random_indexes = np.random.randint(0, self.valid_data_size, self.batch_size)

                batch_ids = [self.valid_data_ids[index] for index in random_indexes]

                return self.load_data(batch_ids, extract_qrs)

        elif subset == 'test':

            if self.test_loaded:

                return random.sample(list(set(chain.from_iterable(self.test_data_per_detect_class))))

            else:

                random_indexes = np.random.randint(0, self.test_data_size, self.batch_size)

                batch_ids = [self.test_data_ids[index] for index in random_indexes]

                return self.load_data(batch_ids, extract_qrs)

        else:
            raise ValueError

    def update_priority_weights(self, batch_weights):

        if self.prioritize_percent > 0:

            self.prioritize_weights += batch_weights.tolist()[:self.new_batch_size]

    def get_next_batch(self, subset: str, extract_qrs = None):

        if subset == 'train':

            if len(self.prioritize_weights) > self.prioritize_size:
    
                self.prioritize_memory = self.prioritize_memory[-self.prioritize_size:]
                self.prioritize_weights = self.prioritize_weights[-self.prioritize_size:]

            if self.train_data_index + self.batch_size <= self.train_data_size:

                if self.priority_batch_size > 0:
                    
                    if len(self.prioritize_weights) == self.prioritize_size:
                    
                        raise ValueError
                        
                    else:
                    
                        raise ValueError

                else:
                    
                    batch_indexes_per_detect_class = []
                    
                    for index in range(self.learn_class_num):
                    
                        if self.train_data_index_per_detect_class[index] + self.batch_size_per_detect_class <= self.train_data_size_per_detect_class[index]:
                            
                            batch_indexes_per_detect_class.append(list(range(self.train_data_index_per_detect_class[index], self.train_data_index_per_detect_class[index] + self.batch_size_per_detect_class)))
                            self.train_data_index_per_detect_class[index] += self.batch_size_per_detect_class
                        
                        else:
                        
                            batch_indexes_per_detect_class.append(list(range(self.train_data_index_per_detect_class[index], self.train_data_size_per_detect_class[index])))
                            batch_indexes_per_detect_class[-1] += list(range(self.batch_size_per_detect_class + self.train_data_index_per_detect_class[index] - self.train_data_size_per_detect_class[index]))
                            self.train_data_index_per_detect_class[index] = self.batch_size_per_detect_class + self.train_data_index_per_detect_class[index] - self.train_data_size_per_detect_class[index]
                            
                    if self.missing_size < 0:
                        
                        index = np.random.randint(self.learn_class_num)
                        
                        if self.train_data_index_per_detect_class[index] + self.missing_size <= self.train_data_size_per_detect_class[index]:
                            
                            batch_indexes_per_detect_class[index] += list(range(self.train_data_index_per_detect_class[index], self.train_data_index_per_detect_class[index] + self.missing_size))
                            self.train_data_index_per_detect_class[index] += self.missing_size
                        
                        else:
                        
                            batch_indexes_per_detect_class[index] +=  list(range(self.train_data_index_per_detect_class[index], self.train_data_size_per_detect_class[index]))
                            batch_indexes_per_detect_class[index] += list(range(self.missing_size + self.train_data_index_per_detect_class[index] - self.train_data_size_per_detect_class[index]))
                            self.train_data_index_per_detect_class[index] = self.missing_size + self.train_data_index_per_detect_class[index] - self.train_data_size_per_detect_class[index]

                self.train_data_index += self.batch_size

                if self.train_loaded:
                    
                    batch_data = []
                    
                    for batch_indexes, train_data in zip(batch_indexes_per_detect_class, self.train_data_per_detect_class):

                        batch_data += [train_data[index] for index in batch_indexes]
                    
                    return batch_data

                else:

                    batch_data_ids = []
                    
                    for batch_indexes, train_data_ids in zip(batch_indexes_per_detect_class, self.train_data_ids_per_detect_class):
                        
                        batch_data_ids += [train_data_ids[index] for index in batch_indexes]

                    return self.load_data(batch_data_ids, extract_qrs)
            
            else:

                return None
            
        elif subset == 'valid':

            if self.valid_data_index + self.batch_size <= self.valid_data_size:
                    
                batch_indexes_per_detect_class = []
                
                for index in range(self.learn_class_num):
                    
                    if self.valid_data_index_per_detect_class[index] + self.batch_size_per_detect_class <= self.valid_data_size_per_detect_class[index]:
                        
                        batch_indexes_per_detect_class.append(list(range(self.valid_data_index_per_detect_class[index], self.valid_data_index_per_detect_class[index] + self.batch_size_per_detect_class)))
                        self.valid_data_index_per_detect_class[index] += self.batch_size_per_detect_class
                    
                    else:
                    
                        batch_indexes_per_detect_class.append(list(range(self.valid_data_index_per_detect_class[index], self.valid_data_size_per_detect_class[index])))
                        batch_indexes_per_detect_class[-1] += list(range(self.batch_size_per_detect_class + self.valid_data_index_per_detect_class[index] - self.valid_data_size_per_detect_class[index]))
                        self.valid_data_index_per_detect_class[index] = self.batch_size_per_detect_class + self.valid_data_index_per_detect_class[index] - self.valid_data_size_per_detect_class[index]
                        
                if self.missing_size < 0:
                    
                    index = np.random.randint(self.learn_class_num)
                    
                    if self.valid_data_index_per_detect_class[index] + self.missing_size <= self.valid_data_size_per_detect_class[index]:
                        
                        batch_indexes_per_detect_class[index] += list(range(self.valid_data_index_per_detect_class[index], self.valid_data_index_per_detect_class[index] + self.missing_size))
                        self.valid_data_index_per_detect_class[index] += self.missing_size
                    
                    else:
                    
                        batch_indexes_per_detect_class[index] +=  list(range(self.valid_data_index_per_detect_class[index], self.valid_data_size_per_detect_class[index]))
                        batch_indexes_per_detect_class[index] += list(range(self.missing_size + self.valid_data_index_per_detect_class[index] - self.valid_data_size_per_detect_class[index]))
                        self.valid_data_index_per_detect_class[index] = self.missing_size + self.valid_data_index_per_detect_class[index] - self.valid_data_size_per_detect_class[index]

                self.valid_data_index += self.batch_size

                if self.valid_loaded:
                    
                    batch_data = []
                    
                    for batch_indexes, valid_data in zip(batch_indexes_per_detect_class, self.valid_data_per_detect_class):
                        
                        batch_data += [valid_data[index] for index in batch_indexes]
                            
                    return batch_data

                else:

                    batch_data_ids = []
                    
                    for batch_indexes, valid_data_ids in zip(batch_indexes_per_detect_class, self.valid_data_ids_per_detect_class):
                        
                        batch_data_ids += [valid_data_ids[index] for index in batch_indexes]

                    return self.load_data(batch_data_ids, extract_qrs)
            
            else:

                return None

        elif subset == 'test':

            if self.test_data_index + self.batch_size <= self.test_data_size:
                    
                batch_indexes_per_detect_class = []
                
                for index in range(self.learn_class_num):
                
                    if self.test_data_index_per_detect_class[index] + self.batch_size_per_detect_class <= self.test_data_size_per_detect_class[index]:
                        
                        batch_indexes_per_detect_class.append(list(range(self.test_data_index_per_detect_class[index], self.test_data_index_per_detect_class[index] + self.batch_size_per_detect_class)))
                        self.test_data_index_per_detect_class[index] += self.batch_size_per_detect_class
                    
                    else:
                    
                        batch_indexes_per_detect_class.append(list(range(self.test_data_index_per_detect_class[index], self.test_data_size_per_detect_class[index])))
                        batch_indexes_per_detect_class[-1] += list(range(self.batch_size_per_detect_class + self.test_data_index_per_detect_class[index] - self.test_data_size_per_detect_class[index]))
                        self.test_data_index_per_detect_class[index] = self.batch_size_per_detect_class + self.test_data_index_per_detect_class[index] - self.test_data_size_per_detect_class[index]
                        
                if self.missing_size < 0:
                    
                    index = np.random.randint(self.learn_class_num)
                    
                    if self.test_data_index_per_detect_class[index] + self.missing_size <= self.test_data_size_per_detect_class[index]:
                        
                        batch_indexes_per_detect_class[index] += list(range(self.test_data_index_per_detect_class[index], self.test_data_index_per_detect_class[index] + self.missing_size))
                        self.test_data_index_per_detect_class[index] += self.missing_size
                    
                    else:
                    
                        batch_indexes_per_detect_class[index] +=  list(range(self.test_data_index_per_detect_class[index], self.test_data_size_per_detect_class[index]))
                        batch_indexes_per_detect_class[index] += list(range(self.missing_size + self.test_data_index_per_detect_class[index] - self.test_data_size_per_detect_class[index]))
                        self.test_data_index_per_detect_class[index] = self.missing_size + self.test_data_index_per_detect_class[index] - self.test_data_size_per_detect_class[index]

                if self.test_loaded:
                    
                    batch_data = []
                    
                    for batch_indexes, test_data in zip(batch_indexes_per_detect_class, self.test_data_per_detect_class):
                        
                        batch_data += [test_data[index] for index in batch_indexes]
                        
                    return batch_data

                else:

                    batch_data_ids = []
                    
                    for batch_indexes, test_data_ids in zip(batch_indexes_per_detect_class, self.test_data_ids_per_detect_class):
                        
                        batch_data_ids += [test_data_ids[index] for index in batch_indexes]

                    return self.load_data(batch_data_ids, extract_qrs)
            
            else:

                return None
        
        else:

            raise ValueError