import torch
from torch import Tensor

def cross_entropy(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon = 1e-6):
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        return -torch.sum(input * torch.log(target))/len(input.reshape(-1))
    else:
        entropy = 0
        for i in range(input.shape[0]):
            entropy += cross_entropy(input[i, ...], target[i, ...])
        return entropy / input.shape[0]

def multiclass_cross_entropy(input: Tensor, target: Tensor, reduce_batch_first:bool = False, epsilon = 1e-6):#channel不用定义吗
    assert input.size() == target.size()
    entropy = 0
    for channel in range(input.shape[1]):
        entropy += cross_entropy(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    return entropy / input.shape[1]

def cross_entropy_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    assert input.size() == target.size()
    fn = multiclass_cross_entropy if multiclass else cross_entropy
    return fn(input, target, reduce_batch_first=True)