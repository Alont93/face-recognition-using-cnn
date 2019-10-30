from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn

class loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    
class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 21 , 3, stride=2, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(__, 20, 3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(__),
            nn.ReLU(inplace=True),
            nn.Conv2d(__, 15, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(__),
            nn.ReLU( inplace=True),
            nn.Conv2d(__,7, 5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(__),
            nn.ReLU( inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(__, __),
            nn.ReLU( inplace=True),
            nn.Linear(__, 201),
            nn.Softmax()
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
        x=self.main(input)
        x=x.view(-1, self.num_flat_features(x))
        return self.fc(x)

        
