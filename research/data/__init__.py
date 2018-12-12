import os

import torch
import torchvision
from torchvision import transforms

__dirname__ = os.path.dirname(__file__)

_norm = [0.5] * 3

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_norm, _norm),
])


def load_dataset(dataset, batch_size=8):
    dir_name = '{}/train'.format(dataset)

    train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(__dirname__, dir_name), transform=transform),

        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    val_dir = '{}/val'.format(dataset)

    val = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(__dirname__, val_dir), transform=transform),

        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return train, val
