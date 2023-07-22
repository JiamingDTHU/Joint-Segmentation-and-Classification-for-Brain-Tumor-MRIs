import numpy as np
from scipy.ndimage import zoom
import torch
from torchvision import transforms


def rerange(arr):
    
    if not isinstance(arr, (np.ndarray, torch.tensor)):
        raise ValueError("输入类型错误：必须是NumPy数组或PyTorch张量")
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
def resize(arr, dst_shape):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    scale_height = dst_shape[-2] / arr.shape[-2]
    scale_width = dst_shape[-1] / arr.shape[-1]
    
    return zoom(arr, (*arr.shape[:-2], scale_height, scale_width), order=1)

def random_crop(arr, dst_shape):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    if not isinstance(dst_shape, (tuple, list)):
        if isinstance(dst_shape, int):
            dst_shape = (dst_shape, dst_shape)
        else:
            raise TypeError("形状参数类型错误：整数、列表或元组")
    
    if dst_shape[-2] > arr.shape[-2] or dst_shape[-1] > arr.shape[-1]:
        raise ValueError("裁剪后的尺寸必须不大于原尺寸")
    
    start_row = np.random.randint(0, arr.shape[-2] - dst_shape[-2] + 1)
    start_col = np.random.randint(0, arr.shape[-1] - dst_shape[-1] + 1)
    
    return arr[..., start_row:start_row + dst_shape[-2], start_col:start_col + dst_shape[-1]]

def center_crop(arr, dst_shape):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    if not isinstance(dst_shape, (tuple, list)):
        if isinstance(dst_shape, int):
            dst_shape = (dst_shape, dst_shape)
        else:
            raise TypeError("形状参数类型错误：整数、列表或元组")
    
    if dst_shape[-2] > arr.shape[-2] or dst_shape[-1] > arr.shape[-1]:
        raise ValueError("裁剪后的尺寸必须不大于原尺寸")
    
    start_row = (arr.shape[-2] - dst_shape[-2]) // 2
    start_col = (arr.shape[-1] - dst_shape[-1]) // 2
    
    return arr[..., start_row:start_row + dst_shape[-2], start_col:start_col + dst_shape[-1]]

def random_rotation(arr):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    rot_times = np.random.randint(0, 4)
    return np.rot90(arr, rot_times, (-2, -1)).copy()

def random_horizontal_flip(arr):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    flip = np.random.randint(0, 2)
    if flip:
        return arr[..., ::-1].copy()
    else:
        return arr.copy()
    
def random_vertical_flip(arr):
    
    if len(arr.shape) < 2:
        raise ValueError("输入数组必须是大于二维的数组")
    
    flip = np.random.randint(0, 2)
    if flip:
        return arr[..., ::-1, :].copy()
    else:
        return arr.copy()
    
class Rerange:
    def __call__(self, arr):
        
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise ValueError("输入类型错误：必须是NumPy数组或PyTorch张量")
        
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

class Resize:
    def __init__(self, dst_shape):
        
        if not isinstance(dst_shape, (tuple, list)):
            if isinstance(dst_shape, int):
                dst_shape = (dst_shape, dst_shape)
            else:
                raise TypeError("形状参数类型错误：整数、列表或元组")
            
        self.dst_shape = dst_shape

    def __call__(self, arr):
        
        if len(arr.shape) < 2:
            raise ValueError("输入数组必须是大于二维的数组")
        
        scale_height = self.dst_shape[-2] / arr.shape[-2]
        scale_width = self.dst_shape[-1] / arr.shape[-1]

        return zoom(arr, (*([1]*(len(arr.shape) - 2)), scale_height, scale_width), order=1)

class RandomCrop:
    def __init__(self, dst_shape):
        
        if not isinstance(dst_shape, (tuple, list)):
            if isinstance(dst_shape, int):
                dst_shape = (dst_shape, dst_shape)
            else:
                raise TypeError("形状参数类型错误：整数、列表或元组")
            
        self.dst_shape = dst_shape

    def __call__(self, arr):
        
        if len(arr.shape) < 2:
            raise ValueError("输入数组必须是大于二维的数组")

        if self.dst_shape[-2] > arr.shape[-2] or self.dst_shape[-1] > arr.shape[-1]:
            raise ValueError("裁剪后的尺寸必须不大于原尺寸")

        start_row = np.random.randint(0, arr.shape[-2] - self.dst_shape[-2] + 1)
        start_col = np.random.randint(0, arr.shape[-1] - self.dst_shape[-1] + 1)

        return arr[..., start_row:start_row + self.dst_shape[-2], start_col:start_col + self.dst_shape[-1]]

class CenterCrop:
    def __init__(self, dst_shape):
        
        if not isinstance(dst_shape, (tuple, list)):
            if isinstance(dst_shape, int):
                dst_shape = (dst_shape, dst_shape)
            else:
                raise TypeError("形状参数类型错误：整数、列表或元组")
            
        self.dst_shape = dst_shape

    def __call__(self, arr):
        
        if len(arr.shape) < 2:
            raise ValueError("输入数组必须是大于二维的数组")

        if self.dst_shape[-2] > arr.shape[-2] or self.dst_shape[-1] > arr.shape[-1]:
            raise ValueError("裁剪后的尺寸必须不大于原尺寸")

        start_row = (arr.shape[-2] - self.dst_shape[-2]) // 2
        start_col = (arr.shape[-1] - self.dst_shape[-1]) // 2

        return arr[..., start_row:start_row + self.dst_shape[-2], start_col:start_col + self.dst_shape[-1]]
    
class RandomRotation:
    def __call__(self, arr):
        return random_rotation(arr)
    
class RandomHorizontalFlip:
    def __call__(self, arr):
        return random_horizontal_flip(arr)

class RandomVerticalFlip:
    def __call__(self, arr):
        return random_vertical_flip(arr)
    
if __name__ == "__main__":
    my_transforms = transforms.Compose([Rerange(),
                                     Resize(256),
                                     RandomCrop(224),
                                     CenterCrop(192),
                                     RandomHorizontalFlip(),
                                     RandomVerticalFlip(),
                                     RandomRotation(),
                                     ])
    # my_transforms = transforms.Compose([Rerange(),
    #                                  Resize(256),
    #                                  RandomCrop(224),
    #                                  CenterCrop(192),])
    
    input_array = np.random.randn(2, 512, 512)
    output_array = my_transforms(input_array)
    print(output_array.shape)
    print(np.max(output_array), np.min(output_array))