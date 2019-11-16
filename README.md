# Deep-Learning-Assignment-3
Deep Convolutional Network for Face Classification


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