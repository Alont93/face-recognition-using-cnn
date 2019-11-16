from torch.utils.data import Dataset
from PIL import Image
import torchvision.models as models
import torch
import torch.nn as nn
import pandas as pd
import os


def num_flat_features(inputs):
    # Get the dimensions of the layers excluding the inputs
    size = inputs.size()[1:]
    # Track the number of features
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Nnet(nn.Module):
    def __init__(self, num_classes=201):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 21, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 20, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 15, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 7, 5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1183, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, num_classes)
        )

        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, num_flat_features(x))
        return self.fc(x)

    def __str__(self):
        return "Nnet"

    def __repr__(self):
        return "Nnet"


class TronNet(nn.Module):
    def __init__(self, num_classes=201):
        super(TronNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 21, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 30, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(30, 35, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(35),
            nn.ReLU(inplace=True),
            nn.Conv2d(35, 25, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
            nn.Conv2d(25, 20, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 15, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.Conv2d(15, 10, 5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1210, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, num_classes)
        )

        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, num_flat_features(x))
        return self.fc(x)

    def __str__(self):
        return "TronNet"

    def __repr__(self):
        return "TronNet"




