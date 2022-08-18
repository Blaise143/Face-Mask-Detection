import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

    def forward(self, X):
        return self.convs(X)

class Classifier(nn.Module):
    def __init__(self, in_channels,  num_classes) -> None:
        super(Classifier, self).__init__()

        conv1 = ConvolutionBlock(in_channels= in_channels, out_channels=8)
        conv2 = ConvolutionBlock(in_channels= 8, out_channels= 16)
        conv3 = ConvolutionBlock(in_channels= 16, out_channels=32)

        self.layers = nn.Sequential(conv1, conv2, conv3)

        self.fc = nn.Sequential(
            nn.Linear(in_features= 36864, out_features= 40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=num_classes)
        )

    def forward(self, X):
        X = self.layers(X)
        X = torch.flatten(X)
        X = self.fc(X)
        #X = F.log_softmax(X,dim=-1)

        return X


def get_dataloaders(train_path, test_path, valid_path):
    #train_path = 'Mask_Data/train'
    #test_path = 'Mask_Data/test'
    #valid_path = 'Mask_Data/val'
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    training = torchvision.datasets.ImageFolder(train_path, transform=train_transforms)
    testing = torchvision.datasets.ImageFolder(test_path, transform=test_transforms)
    valid = torchvision.datasets.ImageFolder(valid_path, transform=test_transforms)
    train_loader = DataLoader(training, batch_size=4, shuffle=True)
    test_loader = DataLoader(testing, batch_size=32, shuffle=False)
    val_loader = DataLoader(valid, batch_size=32, shuffle=False)

    return train_loader, test_loader, val_loader


# DEALING WITH DATASETS




def train_model(model = Classifier(in_channels=3,num_classes=4)):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    EPOCHS = 2
    train_loader, test_loader, valid_loader = get_dataloaders(train_path='Mask_Data/train',
                                                              test_path='Mask_Data/train',
                                                              valid_path='Mask_Data/val')
    model.train()
    for epoch in tqdm(range(EPOCHS)):

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # Forward pass
            y_pred = model(inputs)

            # Loss Function
            loss = criterion(y_pred, labels)

            # BackPropagation
            loss.backward()
            optimizer.step()

            # Zero Gradients
            optimizer.zero_grad()

            if batch_idx % 50:
                print(loss.item())
train_model()