import os
import torch
import torchvision
from torchvision import transforms

__dirname__ = os.path.dirname(__file__)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(200, pad_if_needed=True),
    transforms.ToTensor()
])

def load_dataset(data_dir, batch_size):
    data_dir = os.path.join(__dirname__, data_dir)
    if not os.path.exists(data_dir):
        raise ValueError("%s not found", data_dir)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
