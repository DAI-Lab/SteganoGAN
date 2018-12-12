import torch
import torch.nn as nn


class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = False

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_layers(self):
        return (
            nn.Sequential(
                self._conv2d(3, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
            ),
            nn.Sequential(
                self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
                self._conv2d(self.hidden_size, 3),
                nn.Tanh(),
            )
        )

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.layers = self._build_layers()

    def forward(self, image, data):
        x = self.layers[0](image)
        x_list = [x]

        for layer in self.layers[1:]:
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

    def _build_layers(self):
        return (
            nn.Sequential(
                self._conv2d(3, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
            ),
            nn.Sequential(
                self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
                self._conv2d(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
                self._conv2d(self.hidden_size, 3),
            )
        )


class DenseEncoder(BasicEncoder):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    add_image = True

    def _build_layers(self):
        return (
            nn.Sequential(
                self._conv2d(3, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
            ),
            nn.Sequential(
                self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
            ),
            nn.Sequential(
                self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(self.hidden_size),
            ),
            nn.Sequential(
                self._conv2d(self.hidden_size * 2 + self.data_depth, 3)
            )
        )
