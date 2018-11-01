import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, data_depth, hidden_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=data_depth+hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, data, image):
        data = data*2.0-1.0
        image = image*2.0-1.0
        y = self.conv1(image)
        y = self.conv2(torch.cat((y, data), dim=1))
        return (torch.tanh(torch.tan(image) + y) + 1.0) / 2.0


class Decoder(nn.Module):

    def __init__(self, data_depth, hidden_size):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=data_depth, kernel_size=3, padding=1)
        )

    def forward(self, image):
        return self.conv(image*2.0-1.0)

class Critic(nn.Module):

    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        )
        self.linear = nn.Linear(hidden_size, 1)
        for p in self.parameters():
            p.data.clamp_(-0.01, 0.01)

    def forward(self, image):
        y = self.conv(image*2.0-1.0)
        y = torch.mean(y, dim=2)
        y = torch.mean(y, dim=2)
        return self.linear(y)
