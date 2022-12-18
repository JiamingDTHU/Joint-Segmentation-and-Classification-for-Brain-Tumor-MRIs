import torch
import torch.nn.functional as F

class ConvBNReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
            torch.nn.BatchNorm2d(out_channels), 
            torch.nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.layers(x)

class imgEncodeBlock1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers=torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            torch.nn.MaxPool2d(2, 2)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class imgEncodeBlock2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers=torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            torch.nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class imgEncodeBlock3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers=torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=1),
            torch.nn.Dropout2d(0.5, True),

        )
    
    def forward(self, x):
        return self.layers(x)

class EdgeEncodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers=torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            torch.nn.AvgPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

class EdgeGuidanceBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv1=torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2=torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        
    def forward(self, X, E):
        gamma=torch.sigmoid(self.conv1(E))
        beta=gamma*X
        output=F.relu(self.conv2(gamma+beta))
        return output