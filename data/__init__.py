import torch
from sklearn.model_selection import train_test_split

cached = torch.load("data/caltech256.pt")
train, test = train_test_split(cached)
del cached

def yield_images(mode="train"):
    if mode == "train":
        yield from train
    else:
        yield from test
