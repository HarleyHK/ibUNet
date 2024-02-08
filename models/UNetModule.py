""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

from InceptionEncoder import IncepEncoder
from InceptionEncoder import generation_init_weights

# from unet_parts import *
# ============================================================================================
""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
g_prediction_task ="UnKnown"

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        global g_prediction_task
        if ("DRC" == g_prediction_task):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),#nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels),#nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),#nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels))#nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True)
        elif ("Congestion" == g_prediction_task):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels),#nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(num_parameters=out_channels))#nn.LeakyReLU(0.2, inplace=True),#nn.ReLU(inplace=True)
        else:
            print("ERROR on prediction task!!")
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2), #nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            pass
            #---no advantage on our case ! Harley 2023/11/25
            #---self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# =============================================================================================

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, task="UnKnown"):
        global g_prediction_task
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        _SCALE_ = 2
        g_prediction_task = task

        self.inc = DoubleConv(n_channels, 64//_SCALE_)
        self.down1 = Down(64//_SCALE_, 128//_SCALE_)
        self.down2 = Down(128//_SCALE_, 256//_SCALE_)
        self.down3 = Down(256//_SCALE_, 512//_SCALE_)
        self.down4 = Down(512//_SCALE_, 512//_SCALE_)
        self.up1 = Up(1024//_SCALE_, 256//_SCALE_, bilinear)
        self.up2 = Up(512//_SCALE_, 128//_SCALE_, bilinear)
        self.up3 = Up(256//_SCALE_, 64//_SCALE_, bilinear)
        self.up4 = Up(128//_SCALE_, 64//_SCALE_, bilinear)
        self.outc = OutConv(64//_SCALE_, n_classes)

        self.incepEncoder = IncepEncoder(True, 1, g_prediction_task,512//_SCALE_)
        self.Conv4Inception = DoubleConv(512//_SCALE_, 512//_SCALE_)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5B= self.incepEncoder(x5) #Insert the Inception Module at the bottleneck
        x5C= self.Conv4Inception(x5B)
        x = self.up1(x5C, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # -------------------------------------------------
    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)
    # -------------------------------------------------