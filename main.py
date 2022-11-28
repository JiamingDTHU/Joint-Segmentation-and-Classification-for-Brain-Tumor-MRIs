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
          cls_criterion: function, 
          seg_criterion: function, 
          epoch: int):
    '''
    one epoch training process, containing
    '''
    pass

def test(model: cUNet, 
         device: torch.device, 
         test_loader: DataLoader):
    '''
    testing the accuracy of current partly-trained model
    '''
    pass

def main():
    batch_size=8
    epochs=10
    model=cUNet()
    device=torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
    train_dataset=TumorDataset(dataset_dir='./dataset/', train=True)
    train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size, transform=transforms.ToTensor)
    test_dataset=TumorDataset(dataset_dir='./dataset/', train=False)
    test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size, transform=transforms.ToTensor)
    for epoch in range(epochs):
        train()
        test()

if __name__ == '__main__':
    main()