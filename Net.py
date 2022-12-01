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
        self.encode_flat1=double_conv2d_bn(1, 16) 
        self.encode_flat2=double_conv2d_bn(16, 32)  
        self.encode_flat3=double_conv2d_bn(32, 64)
        self.encode_flat4=double_conv2d_bn(64, 128)
        self.encode_flat5=double_conv2d_bn(128, 256)
        self.l1=torch.nn.Linear(1024, 3)
        self.decode_flat1=double_conv2d_bn(256, 128)
        self.decode_flat2=double_conv2d_bn(128, 64)
        self.decode_flat3=double_conv2d_bn(64, 32)
        self.decode_flat4=double_conv2d_bn(32, 16)
        self.decode_flat5=torch.nn.Conv2d(16, 1, kernel_size=3, 
                                          stride=1, padding=1, bias=True)
        self.deconv1=deconv2d_bn(256, 128)
        self.deconv2=deconv2d_bn(128, 64)
        self.deconv3=deconv2d_bn(64, 32)
        self.deconv4=deconv2d_bn(32, 16)
        
    def forward(self,x):
        conv1 = self.encode_flat1(x) # size(8, 16, 512, 512)
        pool1 = F.max_pool2d(conv1, 2) # size(8, 16, 256, 256)
        
        conv2 = self.encode_flat2(pool1) # size(8, 32, 256, 256)
        pool2 = F.max_pool2d(conv2, 2) # size(8, 32, 128, 128)
        
        conv3 = self.encode_flat3(pool2) # size(8, 64, 128, 128)
        pool3 = F.max_pool2d(conv3, 2) # size(8, 64, 64, 64)
        
        conv4 = self.encode_flat4(pool3) # size(8, 128, 64, 64)
        pool4 = F.max_pool2d(conv4, 2) # size(8, 128, 32, 32)
        
        conv5 = self.encode_flat5(pool4) # size(8, 256, 32, 32)
        
        convt1 = self.deconv1(conv5) # size(8, 128, 64, 64)
        concat1 = torch.cat([convt1,conv4],dim=1) # size(8, 256, 64, 64)
        conv6 = self.decode_flat1(concat1) # size(8, 128, 64, 64)
        
        convt2 = self.deconv2(conv6) # size(8, 64, 128, 128)
        concat2 = torch.cat([convt2,conv3],dim=1) # size(8, 128, 128, 128)
        conv7 = self.decode_flat2(concat2) # size(8, 64, 128, 128)
        
        convt3 = self.deconv3(conv7) # size(8, 32, 256, 256)
        concat3 = torch.cat([convt3,conv2],dim=1) # size(8, 64, 256, 256)
        conv8 = self.decode_flat3(concat3) # size(8, 32, 256, 256)
        
        convt4 = self.deconv4(conv8) # size(8, 16, 512, 512)
        concat4 = torch.cat([convt4,conv1],dim=1) # size(8, 32, 512, 512)
        conv9 = self.decode_flat4(concat4) # sie(8, 16, 512, 512)
        
        output1 = self.l1(conv5[:, 0, :, :].view(-1, 1024))  # classification branch
        output2 = self.decode_flat5(conv9) # segmentation branch
        
        return output1, output2
        
        