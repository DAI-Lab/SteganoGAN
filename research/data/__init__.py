import os
import torch
import torchvision
from torchvision import transforms

__dirname__ = os.path.dirname(__file__)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(128, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def load_dataset(dataset, batch_size=8):
    train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(__dirname__, "%s/train" % dataset), transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(__dirname__, "%s/val" % dataset), transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return train, val
