import torch

import torchvision
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(480, pad_if_needed=True),
    transforms.ToTensor()
])


def load_train_test(path):
    train = torchvision.datasets.ImageFolder(path, transform=transform)
    train, test = torch.utils.data.random_split(
        train, [len(train) - 10000, 10000])
    train = torch.utils.data.DataLoader(train, shuffle=True, num_workers=8)
    test = torch.utils.data.DataLoader(test, shuffle=True, num_workers=8)
    return train, test
