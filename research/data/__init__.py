import os

import torch
import torchvision
import numpy as np
from torchvision import transforms

__dirname__ = os.path.dirname(__file__)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])



class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


def load_dataset(dataset, batch_size=2, limit=np.inf):
    train = torch.utils.data.DataLoader(
        ImageFolder(os.path.join(__dirname__, dataset, "train"), transform, limit),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val = torch.utils.data.DataLoader(
        ImageFolder(os.path.join(__dirname__, dataset, "val"), transform, limit),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return train, val
