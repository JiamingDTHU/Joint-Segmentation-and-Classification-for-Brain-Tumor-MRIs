import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from utils import *

class TumorDataset(Dataset):
    def __init__(self, dataset_dir: str, train: bool = True, transform: transforms = None):
        self.dataset_dir=dataset_dir
        self.transform=transform
        self.train=train
        

    def __len__(self):
        '''90% training set and 10% testing set'''
        if self.train:
            return 2742
        else:
            return 307
        

    def __getitem__(self, idx):
        '''Get data from dataset and return its image, mask and label fields'''
        for i in os.walk(self.dataset_dir):
            name_list=i[2]
            try:
                name_list.remove('.DS_Store')
            except:
                pass
        path=self.dataset_dir+name_list[idx]
        image=self.load(path, 'image')
        mask=self.load(path, 'tumorMask')
        label=int(self.load(path, 'label'))-1
        image=preprocessing(image) # 高斯模糊，中值滤波， 对比度增强， 归一化
        image, mask=dataAug(image, mask) # 数据增强，包括旋转，镜像，对图像与mask施加同样的操作
        if self.transform:
            image=self.transform(image)
        
        return (
            torch.as_tensor(image).float(), 
            torch.as_tensor(mask).long(), 
            torch.as_tensor(label).long()
        )
        
    @staticmethod
    def load(path, field):
        '''Load and preprocess a single .mat data file'''
        
        assert field in ['image', 'label', 'tumorMask', 'tumorBorder'], 'Incorrect data field'
        
        with h5py.File(path, 'r') as f:
            result=np.array(f['cjdata'][field])
        
        return result
    
    
