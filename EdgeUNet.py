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

class ImageEncodeBlock1(torch.nn.Module):
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
    
class ImageEncodeBlock2(torch.nn.Module):
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
    
class ImageEncodeBlock3(torch.nn.Module):
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

class DeconvBNReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)

class DecodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=True):
        super().__init__()
        if interpolate:
            self.upsample=torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv=torch.nn.Sequential(
                ConvBNReLU(in_channels, out_channels), 
                ConvBNReLU(out_channels, out_channels)
                )
        else:
            self.upsample=DeconvBNReLU(in_channels, out_channels)
            self.conv=torch.nn.Sequential(
                ConvBNReLU(out_channels, out_channels), 
                ConvBNReLU(out_channels, out_channels)
                )

    def forward(self, X, E):
        concatenated=torch.cat([E, self.upsample(X)], dim=1)
        output=self.conv(concatenated)
        return output

class EdgeUNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=2, interpolate=True):
        super().__init__()
        self.image_encode_block1=ImageEncodeBlock1(in_channels, 64)
        self.image_encode_block2=ImageEncodeBlock1(64, 128)
        self.image_encode_block3=ImageEncodeBlock2(128, 256)
        self.image_encode_block4=ImageEncodeBlock2(256, 512)
        self.image_encode_block5=ImageEncodeBlock3(512, 1024)

        self.edge_encode_block1=EdgeEncodeBlock(in_channels, 64)
        self.edge_encode_block2=EdgeEncodeBlock(64, 128)
        self.edge_encode_block3=EdgeEncodeBlock(128, 256)
        self.edge_encode_block4=EdgeEncodeBlock(256, 512)

        self.edge_guidance_block1=EdgeGuidanceBlock(64, 64)
        self.edge_guidance_block2=EdgeGuidanceBlock(128, 128)
        self.edge_guidance_block3=EdgeGuidanceBlock(256, 256)
        self.edge_guidance_block4=EdgeGuidanceBlock(512, 512)

        self.decode_block1=DecodeBlock(1024, 512, interpolate=interpolate)
        self.decode_block2=DecodeBlock(512, 256, interpolate=interpolate)
        self.decode_block3=DecodeBlock(256, 128, interpolate=interpolate)
        self.decode_block4=DecodeBlock(128, 64, interpolate=interpolate)

        self.out_conv=torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, image, edge):
        image1=self.image_encode_block1(image)
        image2=self.image_encode_block2(image1)
        image3=self.image_encode_block3(image2)
        image4=self.image_encode_block4(image3)
        image5=self.image_encode_block5(image4)

        edge1=self.edge_encode_block1(edge)
        edge2=self.edge_encode_block2(edge1)
        edge3=self.edge_encode_block3(edge2)
        edge4=self.edge_encode_block4(edge3)

        EGB1=self.edge_guidance_block1(image1, edge1)
        EGB2=self.edge_guidance_block2(image2, edge2)
        EGB3=self.edge_guidance_block3(image3, edge3)
        EGB4=self.edge_guidance_block4(image4, edge4)

        decode1=self.decode_block1(image5, EGB4)
        decode2=self.decode_block2(decode1, EGB3)
        decode3=self.decode_block3(decode2, EGB2)
        decode4=self.decode_block4(decode3, EGB1)
        
        output=self.out_conv(decode4)
        return F.sigmoid(output)