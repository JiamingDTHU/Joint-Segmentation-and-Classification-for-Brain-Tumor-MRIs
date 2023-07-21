import torch
from Net import *
from UNet import *
from dice_score import *
from torchvision import transforms
from torch.utils.data import DataLoader
from TumorDataset import *
import matplotlib.pyplot as plt
from utils import *
from multiLoss import *

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
    running_loss=0.
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets, labels=data
        inputs, targets, labels=inputs.to(device), targets.to(device), labels.to(device)
        optimizer.zero_grad() # set gradient of last epoch to zero
        outputs=model(inputs)
        loss=criterion(outputs, targets)
        loss.backward() # backward the gradient
        optimizer.step() # update parameters
        running_loss+=loss.item() # sum of total loss
        if batch_idx % 300 == 299:
            print('第{}轮次已训练{}批样本, 本批次平均loss值: {}'.format(epoch+1, batch_idx+1, running_loss/300))
            running_loss=0.
    return

def test(model: UNet, 
         device: torch.device, 
         test_loader: DataLoader,
         criterion: torch.nn.Module):
    '''
    testing the accuracy of current partly-trained model and print
    '''
    total=0
    total_loss=0
    with torch.no_grad():
        for data in test_loader:
            images, targets, labels=data
            images, targets, labels=images.to(device), targets.to(device), labels.to(device)
            outputs=model(images)
            total+=labels.size(0)
            total_loss+=criterion(outputs, targets)
        # print('accuracy on test set: {}%\naverage dice score: {}'.format(100*correct/total, total_dice/total))
        print(f'current epoch average cross entropy loss: {total_loss/total}')
        target=np.array(targets.cpu())
        output=np.array(outputs.cpu())
        output[output<=0.5]=0
        output[output>0.5]=1
        # brain_MRI=image[0, 0].copy()
        # groun_truth=target[0].copy()
        predict=output[0, 1].copy()
        print('Dice score', my_dice_score(predict, target))
    return total_loss/total

def main():
    batch_size=1
    num_epoch=1
    model=UNet(1, 2, False)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transforms=transforms.Compose([torchvision.transforms.Resize(256),
                                   torchvision.transforms.CenterCrop(192),
                                   transforms.ToTensor(),
                                  ])
    train_dataset=TumorDataset(dataset_dir='./dataset', train=True, transform=transforms)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset=TumorDataset(dataset_dir='./dataset', train=False, transform=transforms)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-8)
    for epoch in range(num_epoch):
        train(model, device, batch_size, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()