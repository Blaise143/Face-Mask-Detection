import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm.notebook import tqdm


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        '''
        :param in_channels:  Number of Channels entering the convolution layer
        :param out_channels: Number of filter for the convolution layer
        '''

        super(ConvolutionBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
    def forward(self, X):
        return self.convs(X)


class Classifier(nn.Module):
    def __init__(self, in_channels: int,  num_classes: int) -> None:
        '''
        :param in_channels: Number of channels of the input image
        :param num_classes: Number of classes
        '''
        super(Classifier, self).__init__()

        conv1 = ConvolutionBlock(in_channels=in_channels, out_channels=8)
        dropout1 = nn.Dropout2d(0.5)
        conv2 = ConvolutionBlock(in_channels=8, out_channels=16)
        dropout2 = nn.Dropout2d(0.5)
        conv3 = ConvolutionBlock(in_channels=16, out_channels=32)

        self.layers = nn.Sequential(conv1,dropout1, conv2,dropout2, conv3)

        self.fc = nn.Sequential(
            nn.Linear(in_features=1152, out_features=40),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=40, out_features=20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=20, out_features=num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, X):
        '''
        Emplement forward pass
        '''
        X = self.layers(X)
        X = X.view(X.shape[0], -1)
        X = self.fc(X)
        return X

def get_dataloaders(path: str, training: bool):
    '''
    returns a dataloader object given a path to a directory depending on whether or not it is a trainset
    :param path: String
    :param training: Boolean
    :return:
    '''
    transforms = None
    if training:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip()
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor()
        ])

    data = torchvision.datasets.ImageFolder(
        path, transform=transforms)

    data_loader = DataLoader(data, batch_size=32, shuffle=training)

    return data_loader

def calculate_accuracy(y_true, y_pred) -> float:
    '''
    Calculate accuracy of a model
    :param y_true: True labels
    :param y_pred: predictions
    :return:
    '''
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy


train_loss = []
train_acc = []
val_loss = []
val_acc = []

model = Classifier(in_channels=3, num_classes=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
EPOCHS = 30
train_loader = get_dataloaders(path='Mask_Data/train', training=True)
valid_loader = get_dataloaders(path='Mask_Data/val', training=False)
model.train()
for epoch in tqdm(range(EPOCHS)):
    # for batch_idx, (inputs, labels) in enumerate(train_loader):
    for batch in train_loader:
        inputs, labels = batch

        # Forward pass
        y_pred = model.forward(inputs)
        # print('y_pred shape: ', y_pred.shape)

        # Calculating Accuracy
        _, predicted = torch.max(y_pred, dim=1)
        train_accuracy = calculate_accuracy(y_true=labels, y_pred=predicted)

        # Calculating loss
        loss = criterion(input=y_pred, target=labels)

        # BackPropagation
        loss.backward()
        optimizer.step()

        # Zero Gradients
        optimizer.zero_grad()

        # train_loss.append(loss.item())

    with torch.no_grad():
        validation_loss = 0
        model.eval()
        for batch in valid_loader:
            val_input, val_labels = batch
            preds = model.forward(val_input)
            _, eval_preds = torch.max(preds, dim=1)

            valid_accuracy = calculate_accuracy(y_true=val_labels, y_pred=eval_preds)

            valid_loss = criterion(input=preds, target=val_labels)

            validation_loss += valid_loss.item()
            # val_loss.append(validation_loss)

    train_loss.append(loss.item())
    train_acc.append(train_accuracy)
    val_loss.append(validation_loss)
    val_acc.append(valid_accuracy)
    print('Epoch: {}, Loss: {}, Train Accuracy: {}, Val Accuracy: {}'.format(
        epoch, loss.item(), train_accuracy, valid_accuracy))


import matplotlib.pyplot as plt
plt.plot(train_loss, label = 'Training Loss')
plt.plot(val_loss, label = "Validation Loss")
plt.plot(val_acc, label = 'Val Accuracy')
plt.plot(train_acc, label = 'Train Accuracy')
plt.grid()
plt.legend()
plt.show()
plt.close()

test_set = get_dataloaders(path='Mask_Data/test', training=False)
with torch.no_grad():
    test_loss = 0
    model.eval()
    for batch in test_set:
        test_input, test_labels = batch
        preds = model.forward(test_input)
        _, test_preds = torch.max(preds, dim=1)

        test_accuracy: float = calculate_accuracy(y_true=test_labels, y_pred=test_preds)

print('TestAccuracy: ', test_accuracy)