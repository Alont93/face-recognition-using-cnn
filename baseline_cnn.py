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


class Loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self):
        """
        Calculate the weight for each class based on number of samples of that class. Also weight = 1
        for the unknown class 0.
        :return: torch.Tensor - size = [#classes + 1,]
        """
        label_counts = self.frame.label.value_counts().sort_index()  # Count images pr label
        class_weights = 1 / torch.Tensor(label_counts.to_list())
        class_weights = torch.cat((torch.Tensor([1.0]), class_weights), 0)
        class_weights = class_weights.float()
        return class_weights


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


class AlexNet(nn.Module):
    def __init__(self, num_classes=201):
        super(AlexNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 192, 3, stride=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 384, 3, stride=1),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(384, 256, 3, stride=1),
            nn.Conv2d(256, 256, 3, stride=1),
            nn.Conv2d(256, 256, 3, stride=1),
            nn.MaxPool2d(3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, input):
        x = self.main(input)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def __str__(self):
        return "AlexNet"

    def __repr__(self):
        return "AlexNet"


class AlexNet2(nn.Module):
    def __init__(self, num_classes=201):
        super(AlexNet2, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, x):
        x = self.main(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return "AlexNet"

    def __repr__(self):
        return "AlexNet"


class TransferNet(nn.Module):
    def __init__(self, num_classes=201):
        super(TransferNet, self).__init__()
        self.main = models.vgg11(pretrained=True)

        # Freeze parameters, so gradient not computed here
        for param in self.main.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.main.classifier[0].in_features
        self.main.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 300, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(300, num_classes, bias=True)
        )
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def forward(self, input):
        x = self.main.features(input)
        x = self.main.avgpool(x)
        x = x.view(-1, num_flat_features(x))
        return self.main.classifier(x)

    def __str__(self):
        return "TransferNet"

    def __repr__(self):
        return "TransferNet"





