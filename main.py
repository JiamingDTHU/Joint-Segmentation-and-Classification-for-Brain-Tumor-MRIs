import torch
from Net import *
from UNet import *
from dice_score import *
from torchvision import transforms
from torch.utils.data import DataLoader
from tumorDataset import *
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
        # outputs1, outputs2=model(inputs)
        outputs2=model(inputs)
        loss=criterion(outputs2, targets)
        loss.backward() # backward the gradient
        optimizer.step() # update parameters
        
        running_loss+=loss.item() # sum of total loss
        if batch_idx % 2740 == 2739:
            print('第{}轮次已训练{}批样本, 本批次平均loss值: {}'.format(epoch+1, batch_idx+1, running_loss/2740))
            running_loss=0.
    return

def test(model: cUNet, 
         device: torch.device, 
         test_loader: DataLoader,
         criterion: torch.nn.Module):
    '''
    testing the accuracy of current partly-trained model and print
    '''
    correct=0
    total=0
    total_loss=0
    with torch.no_grad():
        for data in test_loader:
            images, targets, labels=data
            images, targets, labels=images.to(device), targets.to(device), labels.to(device)
            # outputs1, outputs2=model(images)
            outputs2=model(images)
            # _, predicted=torch.max(outputs1.data, dim=1)
            total+=labels.size(0)
            # correct+=(predicted-labels<1e-6).sum().item()
            total_loss+=criterion(outputs2, targets)
    # print('accuracy on test set: {}%\naverage dice score: {}'.format(100*correct/total, total_dice/total))
    print(f'current epoch average cross entropy loss: {total_loss/total}')
    image=np.array(images.cpu())
    target=np.array(targets.cpu())
    output2=np.array(outputs2.cpu())

    print('statistic measures', np.max(output2[0][0]), np.min(output2[0][0]), np.mean(output2[0][0]), np.median(output2[0][0]))
    plt.figure(figsize=(15, 30))
    plt.subplot(131)
    plt.imshow(image[0][0], 'gray')
    plt.subplot(132)
    plt.imshow(target[0], 'gray')
    plt.title('target')
    plt.subplot(133)
    plt.imshow(output2[0][0], 'gray')
    plt.title('predict')
    plt.show()
    return

def main():
    batch_size=1
    epochs=1
    model=UNet(1, 2)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform=transforms.Compose([transforms.ToTensor()])
    train_dataset=TumorDataset(dataset_dir='./dataset/training/', train=True, transform=transform)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset=TumorDataset(dataset_dir='./dataset/testing/', train=False, transform=transform)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(epochs):
        train(model, device, batch_size, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()