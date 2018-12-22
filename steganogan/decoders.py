# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BasicDecoder(nn.Module):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth, hidden_size):

        super(BasicDecoder, self).__init__()

        self.data_depth = data_depth

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=data_depth, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.

    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """

    def __init__(self, data_depth, hidden_size):
        super(DenseDecoder, self).__init__()
        self.data_depth = data_depth
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size * 2,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size * 3,
                      out_channels=data_depth,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat((x1, x2), dim=1))
        x4 = self.conv4(torch.cat((x1, x2, x3), dim=1))
        return x4
