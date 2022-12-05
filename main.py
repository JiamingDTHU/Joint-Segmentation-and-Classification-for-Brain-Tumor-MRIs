import torch
from Net import *
from dice_score import *
from torchvision import transforms
from torch.utils.data import DataLoader
from tumorDataset import *
import matplotlib.pyplot as plt
from multiLoss import *

def train(model: cUNet, 
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
        outputs1, outputs2=model(inputs)
        loss=criterion(outputs1, outputs2, labels, targets, 'classification')
        loss.backward() # backward the gradient
        optimizer.step() # update parameters
        
        running_loss+=loss.item() # sum of total loss
        if batch_idx % 50 == 49:
            print('第{}轮次已训练{}批样本, 本批次平均loss值: {}'.format(epoch+1, batch_idx+1, running_loss/50))
            running_loss=0.
    return

def test(model: cUNet, 
         device: torch.device, 
         test_loader: DataLoader):
    '''
    testing the accuracy of current partly-trained model and print
    '''
    correct=0
    total=0
    total_dice=0
    with torch.no_grad():
        for data in test_loader:
            images, targets, labels=data
            images, targets, labels=images.to(device), targets.to(device), labels.to(device)
            outputs1, outputs2=model(images)
            _, predicted=torch.max(outputs1.data, dim=1)
            total+=labels.size(0)
            correct+=(predicted-labels<1e-6).sum().item()
            total_dice+=dice_coeff(outputs2[:, 0], targets)
    print('accuracy on test set: {}%\naverage dice score: {}'.format(100*correct/total, total_dice/total))

    return

def main():
    batch_size=8
    epochs=5
    model=cUNet()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1402, ), (0.8402, ))])
    train_dataset=TumorDataset(dataset_dir='./dataset/training/', train=True, transform=transform)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset=TumorDataset(dataset_dir='./dataset/testing/', train=False, transform=transform)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    criterion=torch.nn.CrossEntropyLoss()
    multiloss=MultiLoss(device, criterion, dice_loss)
    optimizer=torch.optim.SGD(multiloss.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(epochs):
        train(model, device, batch_size, train_loader, optimizer, multiloss, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()