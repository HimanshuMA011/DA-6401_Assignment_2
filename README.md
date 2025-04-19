# DA-6401_Assignment_2

# Overview
The purpose of this assignment was three fold
1. Building and training a CNN model from scratch for iNaturalist image data classification.
2. Fine tune a pretrained model on the iNaturalist dataset.
3. Use a pretrained Object Detection model for a cool application


The link to the wandb report:
https://wandb.ai/ma23c011-indian-institute-of-technology-madras/Small_cnn_model/reports/DA6401-Assignment-2--VmlldzoxMjMzMjQwNg

## Part A: Developing a Lightweight CNN from Scratch

### Overview

In the first section of the assignment, I constructed a custom convolutional neural network (CNN) from scratch using PyTorch. The goal was to analyze how different design and training choices affect model performance. To streamline experimentation, I employed Weights & Biases (WandB) for managing hyperparameter sweeps and performance logging.

---

### Questions 1–3: Architecture Setup and Hyperparameter Tuning

The notebook `Q1toQ3.ipynb` handles:
- Dataset preparation for training and validation.
- Defining a comprehensive sweep configuration.
- Running multiple training sessions with varying hyperparameters using WandB’s Bayesian optimization approach.

The hyperparameter sweep includes tuning the number of filters, kernel sizes, activations, dropout, and more:

```
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'
    },
    'parameters': {
        'conv_filters': {
            'values': [[32, 32, 32, 32, 32], [512, 256, 128, 64, 32], [32, 64, 128, 256, 512]]
        },
        'kernel_size': {
            'values': [[3, 3, 3, 3, 3], [5, 3, 3, 3, 3]]
        },
        'conv_activation': {
            'values': ['ReLU', 'SiLU', 'GeLU','Mish']
        },
        'dense_neurons': {
            'values': [128, 256]
        },
        'dense_activation': {
            'values': ['ReLU', 'SiLU']
        },
        'dropout': {
            'values': [0.2, 0.3]
        },
        'batch_normalise': {
            'values': [True, False]
        },
        'lr': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'data_augmentation': {
            'values': [True, False]
        }
    }
}
```
Best model from training :
'batch_size': 64, 
'conv_activation': 'GeLU', 
'conv_filters': [32, 64, 128, 256, 512], 
'kernel_size': [3, 3, 3, 3, 3], 
'dense_activation': 'SiLU', 
'dense_neurons': 256, 
'dropout': 0.3, 
'lr': 0.0001, 
'data_augmentation': False
