import torch
from torch.utils.data import DataLoader, Dataset
from conv_net import ConvNet
from dataloader import MusicDataset

'''
model = ConvNet(n_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
'''


dataloader = DataLoader(MusicDataset(csv_path, h5_path), batch_size=32, shuffle=True)

def train_loop(dataloader, model, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def valid_loop(dataloader, model, criterion, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            #metric 계산, 출력(prediction)



def run_model(**kwargs):
    model = kwargs['model']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    device = kwargs['device']
    num_epochs = kwargs['num_epochs']

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loop(train_loader, model, criterion, optimizer, device)
        valid_loop(valid_loader, model, criterion, device)