import torch
from torch import Tensor

def MSE(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon = 1e-6):
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        return torch.sum((input.reshape(-1) - target.reshape(-1))**2)
    else:
        entropy = 0
        for i in range(input.shape[0]):
            entropy += MSE(input[i, ...], target[i, ...])
        return entropy / input.shape[0]

def multiclass_MSE(input: Tensor, target: Tensor, reduce_batch_first:bool = False, epsilon = 1e-6):
    assert input.size() == target.size()
    mse = 0
    for channel in range(input.shape[1]):
        mse += MSE(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    return mse / input.shape[1]

def MSE_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    assert input.size() == target.size()
    fn = multiclass_MSE if multiclass else MSE
    return fn(input, target, reduce_batch_first=True)

