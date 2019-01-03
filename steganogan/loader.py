# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision
from torchvision import transforms

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):

        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )
