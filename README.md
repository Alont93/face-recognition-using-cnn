# face-recognition-cnn
Deep Convolutional Network for Face Classification.


Using a dataset of 200 identities in total, this project will present possible solution to build a classifier
using CNNs implemented with PyTorch. We tested out three different architectures:
1. The first model presented is the baseline model we were provided. This model is only used as a guideline for what our other models are at least supposed to perform.
2. **TronNet**, is an extension of what we were provided in the baseline. We made it deeper, in hope of it learning more advanced features.
3. we are using **transfer learning** to initialize a **ResNet18** model. Here we are downloading a pre-trained model while switching out the fully connected layers to fit our problem of 200 different identities.


This project includes:
* Image pre-processing such as: normalization and rotations.
* Xavier weight initialization.
* Our new cnn architecture for solving face classification problem - TronNet.
* Weights Visualization of selected layers.
* Loss Visualizations and comperisons.


## Setup
Install the dependencies in the requirements.txt file.

## Overview
**cnn.py**:
Main class, this is also where the training and testing happens

**models.py**:
All of our Pytorch models is located here

**utils.py**:
Helper methods 


## Configuration
Most of our parameters can be configured using command line. 

Our default settings is listed below
```
SETTINGS = {
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'LR': 0.001,
    'DECAY': 0,
    'NUM_CLASSES': 201,
    'RANDOM_SEED': 42,
    'WLOSS': True,
    'K-FOLD': False,
    'K-FOLD-NUMBER': 2, # if k-fold is enabled
    'NNET': None,
    'TRANSFORMER': "default",
    'DATA_PATHS': {
        'TRAIN_CSV': 'train.csv',
        'TEST_CSV': 'test.csv',
        'DATASET_PATH': './datasets/cs154-fa19-public/'
    }
}
```

To run with a small data set, add the parameter --mini True
