import torch
from torch.utils.data import DataLoader, Dataset
from conv_net import ConvNet
from dataloader import MusicDataset
from pytorch_metric_learning.losses import ContrastiveLoss
import torch.nn as nn
import torch.optim as optim



def train_loop(dataloader, model, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.squeeze()
        print(f"train_loop output.shape: {output.shape}")
        print(f"train_loop target.shape: {target.shape}")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def valid_loop(dataloader, model, criterion, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.squeeze()
            print(f"valid_loop output.shape: {output.shape}")
            print(f"valid_loop target.shape: {target.shape}")
            loss = criterion(output, target)
            #metric 계산, 출력(prediction)



def run_model(**kwargs):
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    model = ConvNet(n_classes=10).to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = kwargs['num_epochs']

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loop(train_loader, model, criterion, optimizer, device)
        valid_loop(valid_loader, model, criterion, device)



'''
데이터로더를 통해 받아온 데이터를 모델에 넣어줌
(loss, optim등을 정의)
train: 모델에서 나온 결과와 실제 라벨을 비교해서 loss 계산
train: backprop
valid: inference하여 metric 계산
'''