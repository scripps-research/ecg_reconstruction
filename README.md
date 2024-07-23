# AI-Enhanced Reconstruction of the 12-Lead Electrocardiogram via 3-Leads with Accurate Clinical Assessment

This repository contains the full Python code for designing and training the Deep Learning models presented in the paper "AI-Enhanced Reconstruction of the 12-Lead Electrocardiogram via 3-Leads with Accurate Clinical Assessment" (https://doi.org/10.1038/s41746-024-01193-7).

## Author

- [@masonfed](https://github.com/masonfed)

## Main functions

The framework includes the following main scripts:

### explore_dataset.py ###

* It makes it possible to carry out an explorative analysis on the overall dataset, associating each element with multiple classes, according to the clinical labels used in the dataset, and computing statistical metrics describing the different classes.

### process_dataset.py ###

* It makes it possible to (i) discard the corrupted elements from the dataset and divide the remaining elements into three groups, which are used as training, validation, and testing sets for the analysis. 

### single_reconstruction.py ###

* It makes it possible to design and train a deep-learning
model taking a subset of ECG leads as input and generating a full 12-lead ECG as output, considering as loss function the mathematical difference between the reconstructed signal and the original 12-lead ECG.

### single_classification.py ###

* It makes it possible to design and train a deep-learning
model taking a subset of ECG leads as input and determining if the ECG is associated with a specific class of the dataset, considering as a loss function the detection accuracy of the architecture. 

### single_recon_classif.py ###

* It makes it possible to design and train a deep-learning
model taking a subset of ECG leads as input and generating a full 12-lead ECG as output, considering as reconstruction loss the probability of correctly associating the ECG signal with any class of the dataset.

### multi_reconstruction.py ###

* It makes it possible to design and train multiple deep-learning
models taking a subset of ECG leads as input and generating a full 12-lead ECG as output, considering as loss function the mathematical difference between the reconstructed signal and the original 12-lead ECG.

### multi_classification.py ###

* It makes it possible to design and train multiple deep-learning
models taking a subset of ECG leads as input and determining if the ECG is associated with a specific class of the dataset, considering as a loss function the detection accuracy of the architecture. 

### multi_recon_classif.py ###

* It makes it possible to design and train multiple deep-learning
models taking a subset of ECG leads as input and generating a full 12-lead ECG as output, considering as reconstruction loss the probability of correctly associating the ECG signal with any class of the dataset.

## Additional functions

The framework includes the following additional scripts:

### generate_dataclass.py ###

* It makes it possible to define a new class from those already in the dataset, where the new class is given by the union or intersection of multiple clinical labels.

### clean_dataclass.py ###

* It makes it possible to discern the elements of a given class between corrupted and cleaned elements while collecting information about the differences between the two groups.

### analyze_dataclass.py ###

* It makes it possible to compute the probability for the elements of a given class to be associated with the different clinical labels used in the dataset.

### process_dataclass.py ###

* It makes it possible to sequentially execute the "clean_dataclass" and "analyze_dataclass" scripts over the same class of elements.
