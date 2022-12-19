'''Some useful utilities'''
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def bilinear_interpolation(src: np.ndarray, dst_shape: tuple):
    '''Bilinear interpolation'''
    (src_height, src_width) = src.shape
    (dst_height, dst_width) = dst_shape
    
    dst = np.zeros((dst_height, dst_width))
    for dst_x in range(dst_height):
        for dst_y in range(dst_width):
            src_x = (dst_x+0.5) * (src_height/dst_height) - 0.5
            src_y = (dst_y+0.5) * (src_width/dst_width) - 0.5
            
            i, j = int(src_x), int(src_y)
            u, v = src_x - i, src_y - j
            f = (1-u)*(1-v)*src[i,j] + (1-u)*v*src[i,j+1] + u*(1-v)*src[i+1,j] + u*v*src[i+1,j+1]
            f = np.clip(f, 0, 255)

            dst[dst_x, dst_y] = f
    return dst


def S(x):
    if abs(x) <= 1: 
        y = 1- 2*np.power(x,2) + abs(np.power(x,3))
    elif abs(x)>1 and abs(x)<2:
        y = 4 - 8*abs(x) + 5*np.power(x,2) - abs(np.power(x,3))
    else:
        y = 0
    return y


# 三次卷积插值
def bicubic_interpolation(src: np.ndarray, dst_shape: tuple):
    '''Bicubic interpolation'''
    (src_height, src_width) = src.shape
    (dst_height, dst_width) = dst_shape
    
    dst = np.zeros((dst_height, dst_width))
    for dst_x in range(dst_height):
        for dst_y in range(dst_width):

            src_x = (dst_x+0.5) * (src_height/dst_height) - 0.5
            src_y = (dst_y+0.5) * (src_width/dst_width) - 0.5
            i, j = int(src_x), int(src_y)
            u, v = src_x - i, src_y - j
            
            x1 = min(max(0, i-1), src_height-4)
            x2 = x1 + 4
            y1 = min(max(0, j-1), src_width-4)
            y2 = y1 + 4
            
            A = np.array([S(u+1), S(u), S(u-1), S(u-2)])
            C = np.array([S(v+1), S(v), S(v-1), S(v-2)])
            B = src[x1:x2, y1:y2]
            f0 = A @ B @ C.T
            f = np.clip(f0, 0, 255)  

            dst[dst_x, dst_y] = f
            
    return dst


def nearest_interpolation(src: np.ndarray, dst_shape: tuple):
    '''Nearest interpolation'''
    (src_height, src_width) = src.shape
    (dst_height, dst_width) = dst_shape

    dst = np.zeros(shape = (dst_height, dst_width))
    for dst_x in range(dst_height):
        for dst_y in range(dst_width):

            src_x = dst_x * (src_height/dst_height)
            src_y = dst_y * (src_width/dst_width)
            
            src_x = int(src_x)             
            src_y = int(src_y)

            dst[dst_x, dst_y] = src[src_x, src_y]
            
    return dst

def interpolation(img: np.ndarray, dst_shape: tuple, mode: str = 'bilinear'):
    if mode == 'bilinear':
        result=bilinear_interpolation(img, dst_shape)
    elif mode == 'nearest':
        result=nearest_interpolation(img, dst_shape)
    elif mode == 'bicubic':
        result = bicubic_interpolation(img, dst_shape)
    
    return result

def identical(img):
    return img

def rot90(img):
    result=np.rot90(img, 1, axes=(0, 1))
    return result.copy()

def rot180(img):
    result=np.rot90(img, 2, axes=(0, 1))
    return result.copy()

def rot270(img):
    result=np.rot90(img, 3, axes=(0, 1))
    return result.copy()

def hormir(img):
    result=np.fliplr(img)
    return result.copy()

def vertmir(img):
    result=np.flipud(img)
    return result.copy()

def medianFilter(img, kernelshape=(3, 3), paddle='zero'):
    (M, N)=img.shape
    (m, n)=kernelshape
    result=np.zeros(img.shape, float)
    if paddle == 'zero':
        temp=np.zeros((M+2*int(m/2), N+2*int(n/2)))
    else:
        temp=np.zeros((M+2*int(m/2), N+2*int(n/2)))
    temp[int(m/2):int(m/2)+M, int(n/2):int(n/2)+N]=img.copy()
    for i in range(0, M):
        for j in range(0, N):
            result[i, j]=np.median(temp[i:i+m, j:j+n].copy())
    return result

def preprocessing(img):
    normed=((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
    gaussed=cv2.GaussianBlur(normed, (5, 5), 0.5)
    # plt.imshow(gaussed, 'gray')
    # plt.title('gauss')
    # plt.show()
    normed=((gaussed-np.min(gaussed))/(np.max(gaussed)-np.min(gaussed))*255).astype(np.uint8)
    meded=cv2.medianBlur(normed, 5)
    # plt.imshow(meded, 'gray')
    # plt.title('median')
    # plt.show()
    normed=((meded-np.min(meded))/(np.max(meded)-np.min(meded))*255).astype(np.uint8)
    clahe=cv2.createCLAHE(2., (8, 8))
    enhanced=clahe.apply(normed)
    # plt.imshow(enhanced, 'gray')
    # plt.title('contrast enhance')
    # plt.show()
    result=(enhanced-np.min(enhanced))/(np.max(enhanced)-np.min(enhanced))
    return result.copy()

def dataAug(img, mask):
    trans_func=np.random.choice([identical, rot90, rot180, rot270, hormir, vertmir])
    img_res, mask_res=trans_func(img), trans_func(mask)
    return img_res, mask_res

def myCrossEntropyLoss(output, target):
    output=output.detach().numpy()[0, 1].reshape((512, 512))
    target=target.detach().numpy().reshape((512, 512))
    loss=-np.sum(target*np.log(output+1e-12)+(1-target)*np.log(1-output+1e-12))/(512**2)
    return loss

def my_dice_score(set_A, set_B):
    inter=np.sum(set_A*set_B)
    union=np.sum(set_A+set_B)
    return 2*inter/(union+1e-12)