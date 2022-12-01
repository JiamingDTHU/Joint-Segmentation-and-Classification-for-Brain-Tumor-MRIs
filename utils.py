'''Some useful utilities'''

import numpy as np

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
