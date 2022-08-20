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
            nn.Linear(in_features=20, out_features=num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, X):
        X = self.layers(X)
        X = torch.flatten(X)
        X = self.fc(X)
        #X = F.softmax(X, dim=-1)

        return X


def get_dataloaders(path: str, training: bool):
    global transforms
    if training:
        transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
        ])
    else:
        transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    data = torchvision.datasets.ImageFolder(
        path, transform=transforms)

    data_loader = DataLoader(data, batch_size=32, shuffle=training)

    return data_loader

def train_model(model = Classifier(in_channels=3,num_classes=4)):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    EPOCHS = 2
    train_loader = get_dataloaders(path='Mask_Data/train', training=True)
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        #for batch_idx, (inputs, labels) in enumerate(train_loader):
        for batch in train_loader:
            inputs, labels = batch

            # Forward pass
            y_pred = model.forward(inputs)
            print('y_pred shape: ', y_pred.shape)

            # Calculating loss
            loss = criterion(input= y_pred, target=labels)

            # BackPropagation
            loss.backward()
            optimizer.step()

            # Zero Gradients
            optimizer.zero_grad()

        print(loss)
train_model()