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

In the first section of the assignment, I constructed a custom convolutional neural network (CNN) from scratch using PyTorch.
---
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

### Testing:

In order to test the best trained model on the test data set, a test script has been written that:
1. Evaluates the test accuracy
2. Plots a confusion matrix
3. Plots sample images, the associated predictions and true labels.
 
The commands to run the testing script is simply:

```python3 test.py```

# Part-B Using Pre-trained Models for Image Classification

The code Dataset.py : [iNaturalist dataset](https://github.com/vamsikrishnamohan/DA6401-Assingment2-/blob/main/data_loading.py) loads the images.

Model is ResNet50

Sweep Config
```python
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"}, 
    "parameters": {
        "epochs": {"values": [5]},
        "batch_size": {"values": [64, 128]},
        "denselayer_size": {"values": [64, 128]},
        "l_rate": {"values": [0.001, 0.0001]},
        "optimizer": {"values": ["Adam"]},
        "dropout": {"values": [0.2, 0.4]},
        "model_version": {"values": ["resnet50"]},
        "activation": {"values": ["relu", "leakyrelu"]}
    }
}
```
