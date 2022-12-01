import torch
from Net import *
from dice_score import *
from torchvision import transforms
from torch.utils.data import DataLoader
from tumorDataset import *

def train(model: cUNet, 
          device: torch.device, 
          batch_size: int, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion1: torch.nn.Module, 
          criterion2: torch.nn.Module, 
          epoch: int):
    '''
    one epoch training process, containing: forwarding, calculating loss value, back propagation, printing some of the training progress
    '''
    running_loss=0.
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets, labels=data
        inputs, targets, labels=inputs.to(device), targets.to(device), labels.to(device)
        optimizer.zero_grad() # 将上一个轮次训练的梯度清零
        outputs1, outputs2=model(inputs)
        l1=criterion1(outputs1, labels) # loss of classification
        l2=criterion2(outputs2[:, 0], targets) # loss of segmentation
        s1=np.random.randn()
        s2=np.random.randn()
        loss=(l1+l2)/2 # calculate Multi-task loss
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
            outputs=model(images)
            _, predicted=torch.max(outputs.data, dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            total_dice+=dice_coeff(images, targets)
    print('accuracy on test set: {}\ndice score{}%'.format(100*correct/total, total_dice/total))
    
    return

def main():
    batch_size=8
    epochs=10
    model=cUNet()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1402, ), (0.8402, ))])
    train_dataset=TumorDataset(dataset_dir='./dataset/', train=True, transform=transform)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset=TumorDataset(dataset_dir='./dataset/', train=False, transform=transform)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)

    for epoch in range(epochs):
        train(model, device, batch_size, train_loader, optimizer, criterion, dice_loss, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()