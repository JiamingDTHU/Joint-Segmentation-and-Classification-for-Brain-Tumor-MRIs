import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from utils import *
import cv2

class TumorDataset(Dataset):
    def __init__(self, dataset_dir: str = './dataset/', train: bool = True, transform: transforms = None):
        self.dataset_dir=dataset_dir
        self.transform=transform
        self.train=train
        

    def __len__(self):
        '''90% training set and 10% testing set'''
        if self.train:
            return 2757
        else:
            return 307
        

    def __getitem__(self, idx):
        '''Get data from dataset and return its image, mask and label fields'''
        
        assert idx in range(1, 3065), 'index out of range: 1~3064'
        
        if self.train:
            path=self.dataset_dir+'training/'+f'{idx}.mat'
        else:
            path=self.dataset_dir+'testing/'+f'{idx}.mat'
        image=self.load(path, 'image')
        mask=self.load(path, 'tumorMask')
        label=self.load(path, 'label')
        if self.transform:
            image=self.transform(image)
        
        return (
            torch.as_tensor(image).float(), 
            torch.as_tensor(mask).float(), 
            torch.as_tensor(label-1).long()
        )
        
    @staticmethod
    def load(path, field):
        '''Load and preprocess a single .mat data file'''
        
        assert field in ['image', 'label', 'tumorMask'], 'Incorrect data field'
        
        with h5py.File(path, 'r') as f:
            result=np.array(f['cjdata'][field])
        
        if field == 'image':
            # scale to range 0~1
            if result.shape == (256, 256):
                result=cv2.resize(result, (512, 512), interpolation=cv2.INTER_CUBIC)
            temp=result.copy()
            temp[temp>=256]=255
            result = temp/256.
        elif field == 'label':
            result = int(result[0][0])
        
        return result
    
    
