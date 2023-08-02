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
            print('epoch: {}; num_batches: [{} | {}]; dice loss {}'.format(epoch, batch_idx, len(train_loader), running_loss / (log_interval if batch_idx else 1) / batch_size))
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
    total_iou = 0.
    for batch_idx, data in enumerate(valid_loader, 0):
        # calculate dice loss
        images, targets, labels = data
        images, targets, labels = images.to(device), targets.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs[:, 0, :, :], targets[:, 0, :, :]).item()
        # calculate mean intersection over union
        images_ = images.cpu()[:, 0, :, :]
        targets_ = targets.cpu()[:, 0, :, :]
        outputs_ = outputs.cpu()[:, 0, :, :]
        predicts = outputs_.clone()
        predicts[outputs_ < 0.5] = 0.
        predicts[outputs_ >= 0.5] = 1.
        total_iou += ((torch.sum(outputs_ * targets_) + 1e-6) / (torch.sum(outputs_) + torch.sum(targets_) - torch.sum(outputs_ * targets_) + 1e-6)).item()
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
    print(f'epoch: {epoch}; mIoU: ', total_iou / len(valid_loader) / batch_size)
    return total_loss / len(valid_loader) / batch_size

def main():
    batch_size = 16
    num_epoch = 200
    lr = 1e-4
    wd = 1e-3
    model = UNet(1, 2, False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, wd=wd)
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint["model_state"])
        best_model_state = checkpoint["best_model_state"]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        min_loss = checkpoint["min_loss"]
        print("=> Model loaded from 'checkpoint.pth'")
    else:
        min_loss = float("inf")
        best_model_state = model.state_dict()
        start_epoch = 0
        print("=> The model will be randomly initialized")

    
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
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = dice_loss

    epochs = []
    train_losses = []
    valid_losses = []
    if start_epoch >= num_epoch:
        print("Already reached maximum epoch")
        return
    for epoch in range(start_epoch, num_epoch):
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
        plt.savefig(f"loss_plot_from_epoch_{start_epoch}.png")
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_model_state = model.state_dict()
            print(f"epoch {epoch}: update best model state")
        scheduler.step()
        torch.save({
            "model_state": model.state_dict(),
            "best_model_state": best_model_state,
            "min_loss": min_loss,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
        }, "checkpoint.pth")

if __name__ == '__main__':
    main()