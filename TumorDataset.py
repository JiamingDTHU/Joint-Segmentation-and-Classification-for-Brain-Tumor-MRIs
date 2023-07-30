import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
from MyTransforms import Rerange, Resize, RandomCrop, CenterCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

class TumorDataset(Dataset):
    def __init__(self, dataset_dir: str, train: bool = True, transform: transforms = None):
        if train:
            self.data_dir = os.path.join(dataset_dir, "train")
        else:
            self.data_dir = os.path.join(dataset_dir, "valid")
        self.transform = transform
        self.name_list = os.listdir(self.data_dir)
        if ".DS_Store" in self.name_list:
            self.name_list.remove(".DS_Store")
        

    def __len__(self):
        return len(self.name_list)
        

    def __getitem__(self, idx):
        '''Get data from dataset and return its image, mask and label fields'''
        path = os.path.join(self.data_dir, str(self.name_list[idx]))
        file_data = h5py.File(path, "r")
        image = np.array(file_data["cjdata"]["image"])
        image = (image - np.min(image)) / (np.max(image)- np.min(image))
        mask = np.array(file_data["cjdata"]['tumorMask'])
        label = np.array(file_data["cjdata"]["label"]).item() - 1
        if self.transform:
            concatenated = np.concatenate((image.reshape((1, *image.shape)), mask.reshape((1, *mask.shape))), axis=0)
            transformed = self.transform(concatenated)
            image, mask = transformed[0:1, :, :], transformed[1:, :, :]
        
        return (
            torch.as_tensor(image).float(), 
            torch.as_tensor(mask).float(), 
            torch.as_tensor(label).float(),
        )
        
    
    
if __name__ == "__main__":
    # transform_train = torchvision.transforms.Compose([
    #     # torchvision.transforms.Grayscale(num_output_channels=3),
    #     # torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(3.0/4.0, 4.0/3.0)),
    #     # torchvision.transforms.RandomHorizontalFlip(), 
    #     # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #     torchvision.transforms.ToTensor(), 
    #     # torchvision.transforms.Normalize([0.485, 0.456, 0.406],
    #     #                                  [0.229, 0.224, 0.225])
    # ])
    # transform_valid = torchvision.transforms.Compose([
    #     # torchvision.transforms.Resize(256), 
    #     # torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     ])
    transform_train = transforms.Compose([Resize(256),
                                    RandomCrop(224),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    RandomRotation(),
                                    ])
    transform_valid = transforms.Compose([Resize(256),
                                    CenterCrop(224),
                                    ])
    train_dataset = TumorDataset(dataset_dir="./dataset", train=True, transform=transform_train)
    valid_dataset = TumorDataset(dataset_dir="./dataset", train=False, transform=transform_valid)
    train_iter = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    valid_iter = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=True)
    
    
        
    
    