# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torchvision
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=DEFAULT_TRANSFORM, limit=np.inf,
                 shuffle=True, num_workers=4, batch_size=4, *args, **kwargs):

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            path)

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )
