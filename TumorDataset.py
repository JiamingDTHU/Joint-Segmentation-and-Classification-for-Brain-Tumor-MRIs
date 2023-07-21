import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

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
        mask = np.array(file_data["cjdata"]['tumorMask'])
        label = np.array(file_data["cjdata"]["label"]).item() - 1
        if self.transform:
            image = self.transform(image)
        # image = torch.unsqueeze(image, dim=0)
        # mask = torch.unsqueeze(mask, dim=0)
        
        return (
            torch.as_tensor(image).float(), 
            torch.as_tensor(mask).long(), 
            torch.as_tensor(label).long(),
        )
        
    
    
if __name__ == "__main__":
    dataset = TumorDataset(dataset_dir="./dataset")
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(3.0/4.0, 4.0/3.0)),
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256), 
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
        ])
    train_iter = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    valid_iter = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=True)
    
    # for images, masks, labels in valid_iter:
    #     print(images.shape)
    #     print(masks.shape)
    #     print(labels.shape)