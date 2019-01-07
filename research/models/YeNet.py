import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

SRM_npy = np.load('SRM_Kernels.npy')

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1,1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()

class YeNet(nn.Module):
    def __init__(self, with_bn=False, threshold=3):
        super(YeNet, self).__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(1, 0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x
        self.block2 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.block4 = ConvBlock(30, 30, 3, with_bn=self.with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=self.with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=self.with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=self.with_bn)
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=self.with_bn)
        self.ip1 = nn.Linear(16, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x[:,0:1:,:]) + self.preprocessing(x[:,1:2,:,:]) + self.preprocessing(x[:,2:3,:,:])
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = torch.mean(x.view(x.size(0), 16, -1), 2)
        x = self.ip1(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0. ,0.01)
                mod.bias.data.zero_()
