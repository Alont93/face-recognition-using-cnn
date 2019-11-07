from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os


class loader(Dataset):
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
    def __init__(self):
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
            nn.Linear(300, 201),
            # nn.Softmax()
        )

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, self.num_flat_features(x))
        return self.fc(x)


transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
dataset = loader('train.csv', './datasets/cs154-fa19-public/', transform=transform)
print(dataset.get_class_weights())

