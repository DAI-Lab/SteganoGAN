# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.onnx
from torchvision.ops.deform_conv import DeformConv2d

#input = torch.rand(4, 3, 10, 10)
kh, kw = 3, 3
#weight = torch.rand(5, 3, kh, kw)
# offset and mask should have the same spatial size as the output
# of the convolution. In this case, for an input of 10, stride of 1
# and kernel size of 3, without padding, the output size is 8
#offset = torch.rand(4, 2 * kh * kw, 8, 8)
#mask = torch.rand(4, kh * kw, 8, 8)


class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = False

    def _conv2d(self, in_channels, out_channels):
        return DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            input = torch.rand(4, 3, 10, 10)
            weight = torch.rand(5, 3, kh, kw)
            offset = torch.rand(4, 2 * kh * kw, 8, 8)
            mask = torch.rand(4, kh * kw, 8, 8)
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
            nn.Tanh(),
        )
        return self.features, self.layers

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        # Transform to version 1
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [data], dim=1))
            x_list.append(x)

        if self.add_image:
            x = image + x

        return x


class ResidualEncoder(BasicEncoder):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
        )
        return self.features, self.layers


class DenseEncoder(BasicEncoder):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4
