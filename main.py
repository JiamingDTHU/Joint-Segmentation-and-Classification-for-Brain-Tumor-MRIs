import torch
from Net import *
from UNet import *
from dice_score import *
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TumorDataset import *
import matplotlib.pyplot as plt
from utils import *
from multiLoss import *
from dice_score import *

def train(model: UNet, 
          device: torch.device, 
          batch_size: int, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          epoch: int):
    '''
    one epoch training process, containing: forwarding, calculating loss value, back propagation, printing some of the training progress
    '''
    running_loss = 0.
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets, labels = data
        inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
        optimizer.zero_grad() # set gradient of last epoch to zero
        outputs = model(inputs)
        outputs = F.interpolate(F.pad(outputs, (32, 32, 32, 32), value=0), size=(512, 512), mode='bilinear', align_corners=False)
        loss = criterion(outputs[:, 0], targets)
        loss.backward() # backward the gradient
        optimizer.step() # update parameters
        running_loss += loss.item() # sum of total loss
        log_times = 5
        log_interval = len(train_loader) // log_times
        if batch_idx % log_interval == 0:
            print('epoch: {}; num_batches: {}; loss {}'.format(epoch, batch_idx, running_loss / (log_interval if batch_idx else 1)))
            running_loss = 0.
    return

@torch.no_grad()
def test(model: UNet, 
         device: torch.device, 
         test_loader: DataLoader,
         criterion: torch.nn.Module):
    '''
    testing the accuracy of current partly-trained model and print
    '''
    total_loss = 0
    for batch_idx, data in enumerate(test_loader, 0):
        images, targets, labels = data
        images, targets, labels = images.to(device), targets.to(device), labels.to(device)
        outputs = model(images)
        outputs = F.interpolate(F.pad(outputs, (32, 32, 32, 32), value=0), size=(512, 512), mode='bilinear', align_corners=False)
        total_loss += criterion(outputs[:, 0], targets)
    print(f'epoch BCE loss: {total_loss / batch_idx}')
    targets = targets.cpu()
    outputs = outputs.cpu()
    outputs[outputs < 0.5] = 0
    outputs[outputs >= 0.5] = 1
    predict = outputs[:, 0]
    print('epoch dice score: ', 1 - dice_loss(predict, targets).item())
    return

def main():
    batch_size = 16
    num_epoch = 200
    model = UNet(1, 2, False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(192),
                                   transforms.ToTensor(),
                                  ])
    train_dataset = TumorDataset(dataset_dir='./dataset', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = TumorDataset(dataset_dir='./dataset', train=False, transform=transform)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    for epoch in range(num_epoch):
        train(model, device, batch_size, train_loader, optimizer, criterion, epoch)
        test(model, device, valid_loader, criterion)

if __name__ == '__main__':
    main()