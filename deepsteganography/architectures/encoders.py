import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    """
    The BaseEncoder module takes an cover image and a data tensor and combines 
    them into a steganographic image.

    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth, hidden_size):
        super(BaseEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size + data_depth, out_channels=hidden_size, kernel_size=3, padding=1),
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
