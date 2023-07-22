import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
from MyTransforms import Rerange, RandomCrop, Resize, CenterCrop

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
        # image = Image.fromarray(np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image))))
        # image = 
        mask = np.array(file_data["cjdata"]['tumorMask'])
        label = np.array(file_data["cjdata"]["label"]).item() - 1
        if self.transform:
            transformed = self.transform(np.concatenate((image.reshape(1, *image.shape), mask.reshape(1, *mask.shape)), axis=0))
            image, mask = transformed[0], transformed[1]
        # image = torch.unsqueeze(image, dim=0)
        # mask = torch.unsqueeze(mask, dim=0)
        
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
    transform_train = transforms.Compose([Rerange(),
                                    # Resize(256),
                                    # RandomCrop(224),
                                    transforms.ToTensor(),
                                    ])
    transform_valid = transforms.Compose([Rerange(),
                                    # Resize(256),
                                    # CenterCrop(224),
                                    transforms.ToTensor(),
                                    ])
    train_dataset = TumorDataset(dataset_dir="./dataset", train=True, transform=transform_train)
    valid_dataset = TumorDataset(dataset_dir="./dataset", train=False, transform=transform_valid)
    train_iter = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    valid_iter = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=True)
    
    # 计算图像的均值以及方差
    pixel_value_sum = 0.
    pixel_value_sum_sq = 0.
    for images, masks, labels in train_iter:
        images.requires_grad_(False)
        pixel_value_sum += torch.sum(images).item()
    
    for images, masks, labels in valid_iter:
        images.requires_grad_(False)
        pixel_value_sum += torch.sum(images).item()
    
    avg_pixel_value = pixel_value_sum / ((len(train_iter) + len(valid_iter)) * 4 * 512 * 512)
    print(avg_pixel_value)
    
    for images, masks, labels in train_iter:
        images.requires_grad_(False)
        pixel_value_sum_sq += torch.sum((images - avg_pixel_value) ** 2).item()
    
    for images, masks, labels in valid_iter:
        images.requires_grad_(False)
        pixel_value_sum_sq += torch.sum((images - avg_pixel_value) ** 2).item()
        
    std_pixel_value = (pixel_value_sum_sq / ((len(train_iter) + len(valid_iter)) * 4 * 512 * 512 - 1) ) ** 0.5
    print(std_pixel_value)
    
        
    
    