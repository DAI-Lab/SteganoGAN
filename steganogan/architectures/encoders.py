import torch
import torch.nn as nn


class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth, hidden_size):

        super(BasicEncoder, self).__init__()

        self.data_depth = data_depth

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )

        calc_channels = hidden_size + data_depth

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels=calc_channels,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),

            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image, data):

        x = self.features(image)
        x = self.layers(torch.cat((x, data), dim=1))

        return x


class ResidualEncoder(nn.Module):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth, hidden_size):

        super(ResidualEncoder, self).__init__()

        self.data_depth = data_depth

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )

        calc_channels = hidden_size + data_depth

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels=calc_channels,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),

            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),

            nn.Conv2d(in_channels=hidden_size, out_channels=3, kernel_size=3, padding=1),
        )

    def forward(self, image, data):

        x = self.features(image)
        x = self.layers(torch.cat((x, data), dim=1))

        return image + x


class DenseEncoder(nn.Module):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth, hidden_size):

        super(DenseEncoder, self).__init__()

        self.data_depth = data_depth

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )

        conv2_channels = hidden_size + data_depth

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv2_channels,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),

            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )

        conv3_channels = (hidden_size * 2) + data_depth

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=conv3_channels,
                      out_channels=hidden_size,
                      kernel_size=3,
                      padding=1),

            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )

        conv4_channels = (hidden_size * 3) + data_depth

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=conv4_channels, out_channels=3, kernel_size=3, padding=1),
        )

    def forward(self, image, data):

        x1 = self.conv1(image)
        x2 = self.conv2(torch.cat((x1, data), dim=1))
        x3 = self.conv3(torch.cat((x1, x2, data), dim=1))
        x4 = self.conv4(torch.cat((x1, x2, x3, data), dim=1))

        return image + x4
