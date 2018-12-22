# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BasicCritic(nn.Module):
    """
    The BasicCritic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).

    Input: (N, 3, H, W)
    Output: (N, 1)
    """

    def __init__(self, hidden_size):

        super(BasicCritic, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=3)
        )

    def forward(self, x):

        x = self.layers(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)

        return x
