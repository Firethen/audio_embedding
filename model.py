import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
1. 데이터 전처리, 증강(audio segmentation -> mel-spec -> augmentation)
a. device에 맞게 최적화
b. 데이터 증강
'''


class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        #input shape: (128, 128, 1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0)
        #output shape: (122, 122, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        #output shape: (61, 61, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=0)
        #output shape: (28, 28, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #output shape: (14, 14, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        #output shape: (12, 12, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        #output shape: (6, 6, 256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        #output shape: (4, 4, 512)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        #output shape: (2, 2, 512)
        self.bn5 = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()
        #output shape: (2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(2048, 1024)
        #output shape: (1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        #output shape: (256)
        self.dropout3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(256, 64)
        #output shape: (64)
        self.fc4 = nn.Linear(64, 32)
        #feature embedding이므로 여기까지만.

        #output shape: (32)
        #self.fc5 = nn.Linear(32, n_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.bn5(x)
        x = self.flatten(x)
        x = self.bn6(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        #x = self.fc5(x)
        #x = F.softmax(x, dim=1)
        return x



def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

