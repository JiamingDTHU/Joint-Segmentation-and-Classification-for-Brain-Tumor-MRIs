import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class TumorDataset(Dataset):
    def __init__(self, dataset_dir: str = './dataset/', train: bool = True):
        self.dataset_dir=dataset_dir
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
        mask=self.load(path, 'mask')
        label=self.load(path, 'label')
        
        return {
            'image': torch.as_tensor(image.copy()).float(), 
            'mask': torch.as_tensor(mask.copy()).int(), 
            'label': torch.as_tensor(label.copy()).int()
        }
        
    @staticmethod
    def load(path, field):
        '''Load and preprocess a single .mat data file'''
        
        assert field in ['image', 'label', 'tumorMask'], 'Incorrect data field'
        
        with h5py.File(path, 'r') as f:
            result=np.array(f['cjdata'][field])
        
        if field == 'image':
            # scale to range 0~1
            result.astype('uint8')
            result /= 256
        elif field == 'label':
            result = int(result[0][0])
        
        return result
    
    
