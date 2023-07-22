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
from torch.optim import Adam
from MyTransforms import Rerange, Resize, RandomCrop, CenterCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

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
        outputs = F.interpolate(F.pad(outputs, (16, 16, 16, 16), value=0), size=(512, 512), mode='bilinear', align_corners=False)
        # outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
        loss = criterion(outputs[:, 1], targets)
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
def eval(model: UNet, 
         device: torch.device, 
         valid_loader: DataLoader,
         criterion: torch.nn.Module):
    '''
    evaluate the accuracy of current partly-trained model and print
    '''
    total_loss = 0
    for batch_idx, data in enumerate(valid_loader, 0):
        images, targets, labels = data
        images, targets, labels = images.to(device), targets.to(device), labels.to(device)
        outputs = model(images)
        outputs = F.interpolate(F.pad(outputs, (16, 16, 16, 16), value=0), size=(512, 512), mode='bilinear', align_corners=False)
        # outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
        total_loss += criterion(outputs[:, 1], targets).item()
    print(f'epoch BCE loss: {total_loss / len(valid_loader)}')
    targets = targets.cpu()
    outputs = outputs.cpu()
    outputs[outputs < 0.5] = 0
    outputs[outputs >= 0.5] = 1
    predict = outputs[:, 1]
    print('epoch dice score: ', 1 - dice_loss(predict, targets).item())
    return total_loss / batch_idx

def main():
    batch_size = 16
    num_epoch = 10
    model = UNet(1, 2, False)
    try:
        model.load_state_dict("optim_params.pth")
        flag = 1
    except:
        flag = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # transform = transforms.Compose([transforms.Resize(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(0.268,
    #                                                      0.420),
    #                                 ])
    transform_train = transforms.Compose([Rerange(),
                                    Resize(256),
                                    RandomCrop(224),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    RandomRotation(),
                                    # transforms.ToTensor(),
                                    ])
    transform_valid = transforms.Compose([Rerange(),
                                    Resize(256),
                                    CenterCrop(224),
                                    # transforms.ToTensor(),
                                    ])
    train_dataset = TumorDataset(dataset_dir='./dataset', train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = TumorDataset(dataset_dir='./dataset', train=False, transform=transform_valid)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.999])
    
    for epoch in range(num_epoch):
        train(model, device, batch_size, train_loader, optimizer, criterion, epoch)
        cur_loss = eval(model, device, valid_loader, criterion)
        if cur_loss < min_loss:
            min_loss = cur_loss
            torch.save(model.state_dict(), "optim_params.pth")
            print(f"epoch {epoch}: update optimal model parameters")

if __name__ == '__main__':
    main()