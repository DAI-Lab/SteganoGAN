import torch

import torchvision
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(480, pad_if_needed=True),
    transforms.ToTensor()
])
train = torch.utils.data.ConcatDataset([
    torchvision.datasets.ImageFolder('data/caltech256', transform=transform),
    torchvision.datasets.ImageFolder('data/', transform=transform),
])
train, test = torch.utils.data.random_split(train, [len(train) - 10000, 10000])
train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
test = torch.utils.data.DataLoader(test, shuffle=True, num_workers=8)
