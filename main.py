import torch
from UNet import *
from dice_score import *
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TumorDataset import *
import matplotlib.pyplot as plt
from dice_score import *
from torch.optim import Adam, lr_scheduler
from MyTransforms import Rerange, Resize, RandomCrop, CenterCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

def train(model: UNet, 
          device: torch.device, 
          train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          epoch: int,
          batch_size: int):
    '''
    one epoch training process, containing: forwarding, calculating loss value, back propagation, printing some of the training progress
    '''
    total_loss = 0.
    running_loss = 0.
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets, labels = data
        inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
        optimizer.zero_grad() # set gradient of last epoch to zero
        outputs = model(inputs)
        loss = criterion(outputs[:, 0, :, :], targets[:, 0, :, :])
        loss.backward() # backward the gradient
        optimizer.step() # update parameters
        running_loss += loss.item() # sum of total loss
        total_loss += loss.item()
        log_times = 5
        log_interval = len(train_loader) // log_times
        if batch_idx % log_interval == 0:
            print('epoch: {}; num_batches: [{} | {}]; loss {}'.format(epoch, batch_idx, len(train_loader), running_loss / (log_interval if batch_idx else 1) / batch_size))
            running_loss = 0.
    print(f'epoch: {epoch}; training loss: {total_loss / len(train_loader) / batch_size}')
    return total_loss / len(train_loader) / batch_size

@torch.no_grad()
def eval(model: UNet, 
         device: torch.device, 
         valid_loader: DataLoader,
         criterion: torch.nn.Module,
         epoch: int,
         batch_size):
    '''
    evaluate the accuracy of current partly-trained model and print
    '''
    total_loss = 0.
    total_dice_score = 0.
    for batch_idx, data in enumerate(valid_loader, 0):
        # calculate BCE loss
        images, targets, labels = data
        images, targets, labels = images.to(device), targets.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs[:, 0, :, :], targets[:, 0, :, :]).item()
        # calculate dice score
        images_ = images.cpu()[:, 0, :, :]
        targets_ = targets.cpu()[:, 0, :, :]
        outputs_ = outputs.cpu()
        outputs_[outputs_ < 0.5] = 0.
        outputs_[outputs_ >= 0.5] = 1.
        predicts = outputs_[:, 0, :, :]
        total_dice_score += (1 - dice_loss(predicts, targets_).item())
        if batch_idx == 0:
            toPIL = transforms.ToPILImage()
            for i in range(4):
                sample_image = toPIL(images_[i])
                sample_mask = toPIL(targets_[i])
                sample_prediction = toPIL(predicts[i])
                sample_image.save(f"sample image {i + 1}.jpg")
                sample_mask.save(f"sample mask {i + 1}.jpg")
                sample_prediction.save(f"sample prediction {i + 1}.jpg")
    
    print(f'epoch: {epoch}; validation loss: {total_loss / len(valid_loader) / batch_size}')
    print('dice score: ', total_dice_score / len(valid_loader) / batch_size)
    return total_loss / len(valid_loader) / batch_size

def main():
    batch_size = 32
    num_epoch = 200
    lr = 1e-2
    model = UNet(1, 2, False)
    # if os.path.exists("optim_params.pth"):
    #     model.load_state_dict(torch.load("optim_params.pth"))
    #     print("=> Model loaded from 'optim_params.pth'")
    # else:
    #     print("=> The model will be randomly initialized")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    transform_train = transforms.Compose([Resize(256),
                                    RandomCrop(224),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    RandomRotation(),
                                    ])
    transform_valid = transforms.Compose([Resize(256),
                                    CenterCrop(224),
                                    ])
    train_dataset = TumorDataset(dataset_dir='./dataset', train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = TumorDataset(dataset_dir='./dataset', train=False, transform=transform_valid)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    
    min_loss = float("inf")
    epochs = []
    train_losses = []
    valid_losses = []
    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, batch_size)
        valid_loss = eval(model, device, valid_loader, criterion, epoch, batch_size)
        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # plot current losses and save as .jpg file
        plt.clf()
        plt.plot(epochs, train_losses, label='Training')
        plt.plot(epochs, valid_losses, label='Validation')
        plt.title('Loss curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png")
        if valid_loss < min_loss:
            min_loss = valid_loss
            # torch.save(model.state_dict(), "optim_params.pth")
            print(f"epoch {epoch}: update optimal model parameters")
        scheduler.step()

if __name__ == '__main__':
    main()