import torch
import torch.nn.functional as F


# 搭建网络
class double_conv2d_bn(torch.nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1):
        super(double_conv2d_bn, self).__init__() 
        self.conv1=torch.nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=kernel_size, stride=stride, 
                                   padding=padding, bias=True)
        self.conv2=torch.nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=kernel_size, stride=stride, 
                                   padding=padding, bias=True)
        self.bn1=torch.nn.BatchNorm2d(out_channels)
        self.bn2=torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        output=F.relu(self.bn1(self.conv1(x)))
        output=F.relu(self.bn2(self.conv2(output)))
        return output

class deconv2d_bn(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(deconv2d_bn, self).__init__()
        self.conv1=torch.nn.ConvTranspose2d(in_channels, out_channels, 
                                   kernel_size=kernel_size, 
                                   stride=stride, bias=True)
        self.bn1=torch.nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        output=F.relu(self.bn1(self.conv1(x)))
        return output
        
class cUNet(torch.nn.Module):
    def __init__(self):
        super(cUNet, self).__init__()
        self.down_conv1=double_conv2d_bn(1, 8) 
        self.down_conv2=double_conv2d_bn(8, 16)  
        self.down_conv3=double_conv2d_bn(16, 32)
        self.down_conv4=double_conv2d_bn(32, 64)
        self.down_conv5=double_conv2d_bn(64, 128)
        self.up_conv1=double_conv2d_bn(128, 64)
        self.up_conv2=double_conv2d_bn(64, 32)
        self.up_conv3=double_conv2d_bn(32, 16)
        self.up_conv4=double_conv2d_bn(16, 8)
        self.up_conv5=torch.nn.Conv2d(8, 1, kernel_size=3, 
                                          stride=1, padding=1, bias=True)
        self.deconv1=deconv2d_bn(128, 64)
        self.deconv2=deconv2d_bn(64, 32)
        self.deconv3=deconv2d_bn(32, 16)
        self.deconv4=deconv2d_bn(16, 8)
    
    def forward(self,x):
        conv1 = self.down_conv1(x)
        pool1 = F.max_pool2d(conv1, 2)
        
        conv2 = self.down_conv2(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        
        conv3 = self.down_conv3(pool2)
        pool3 = F.max_pool2d(conv2, 2)
        
        conv4 = self.down_conv4(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        
        conv5 = self.down_conv5(pool4)
        
        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.up_conv1(concat1)
        
        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.up_conv2(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.up_conv3(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.up_conv4(concat4)
        output = self.up_conv5(conv9)
        
        return output
        
        