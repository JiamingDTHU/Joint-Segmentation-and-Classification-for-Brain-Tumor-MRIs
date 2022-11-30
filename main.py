import torch
from Net import *
from dice_score import *
from torchvision import transforms
from torch.utils.data import DataLoader
from tumorDataset import *
from train import *

def train(model: cUNet, 
          device: torch.device, 
          batch_size: int, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          cls_loss: torch.nn.Module, 
          seg_loss: function, 
          epoch: int):
    '''
    one epoch training process, containing: forwarding, calculating loss value, back propagation, printing some of the training progress
    '''
    pass

def test(model: cUNet, 
         device: torch.device, 
         test_loader: DataLoader):
    '''
    testing the accuracy of current partly-trained model and print
    '''
    pass

def main():
    batch_size=8
    epochs=10
    model=cUNet()
    device=torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.3, ), (0.8, ))])
    train_dataset=TumorDataset(dataset_dir='./dataset/', train=True)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size, transform=transform)
    test_dataset=TumorDataset(dataset_dir='./dataset/', train=False)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size, transform=transform)
    criterion=torch.nn.CrossEntropyLoss()
    # optimizer=torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)

    # for epoch in range(epochs):
        # train(model, device, batch_size, train_loader, optimizer, criterion, dice_loss, epoch)
        # test(model, device, test_loader)
    optimizer=torch.optim.Adam(cUNet.parameters(),lr=0.0003)
    cUNet,train_process=train_eval(cUNet,train_loader,0.8,criterion,optimizer,num_epochs=3)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch,train_process.train_loss_all,'ro-',label='Train loss')
    plt.plot(train_process.epoch,train_process.val_loss_all,'bs-',label='Val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(train_process.epoch,train_process.train_acc_all,'ro-',label='Train acc')
    plt.plot(train_process.epoch,train_process.val_acc_all,'bs-',label='Val acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')

if __name__ == '__main__':
    main()