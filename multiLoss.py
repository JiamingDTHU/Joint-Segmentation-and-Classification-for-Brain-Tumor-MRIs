import numpy as np
import torch
import torch.nn.functional as F

class MultiLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, device, criterion1, criterion2):
        super().__init__()
        self.s1=torch.nn.Parameter(torch.randn(1, device=device, requires_grad=True)*0.1)
        self.s2=torch.nn.Parameter(torch.randn(1, device=device, requires_grad=True)*0.1)
        self.c1=criterion1
        self.c2=criterion2
    
    def forward(self, outputs1, outputs2, labels, targets, mode='multi-task'):
        l1=self.c1(outputs1, labels)
        l2=self.c2(outputs2[:, 0], targets)
        result=torch.div(l1, torch.square(self.s1).multiply(2))+torch.div(l2, torch.square(self.s2).multiply(2))+torch.log(torch.mul(self.s1, self.s2))
        if mode=='multi-task':
            return result
        elif mode=='classification':
            return l1
        elif mode=='segmentation':
            return l2
        pass
    
def test_loss(inputs, targets):
    inputs, targets=inputs.reshape((-1, 1, 512, 512)), targets.reshape((-1, 1, 512, 512))
    return F.cross_entropy(inputs, targets, reduction='mean')