import os
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])


def load_XY(args):
    I = []
    for path in glob(args.target + "/images/*.png"):
        I.append(os.path.basename(path).replace(".png", ""))
    I_train, I_test = train_test_split(I)

    X_train, Y_train = [], []
    for i in I_train:
        X_train.append(os.path.join(args.cover, "images/%s.jpg" % i))
        X_train.append(os.path.join(args.target, "images/%s.png" % i))
        Y_train.extend([0, 1])    
    for x in X_train:
        assert os.path.exists(x), x

    stegas = args.stega[:args.num_stega] if args.num_stega else args.stega
    print("Using:", stegas)
    X_test, Y_test = [], []
    for i in I_test:
        X_test.append(os.path.join(args.cover, "images/%s.jpg" % i))
        Y_test.append(0)
        for stega in stegas: # only use first N stega
            X_test.append(os.path.join(stega, "images/%s.png" % i))
            Y_test.append(1)
    for x in X_test:
        assert os.path.exists(x), x

    train_loader = ImageFolder("./", transform=DEFAULT_TRANSFORM)
    train_loader.samples = list(zip(X_train, Y_train))
    train_loader = DataLoader(train_loader, batch_size=32, shuffle=True)

    test_loader = ImageFolder("./", transform=DEFAULT_TRANSFORM)
    test_loader.samples = list(zip(X_test, Y_test))
    test_loader = DataLoader(test_loader, batch_size=32, shuffle=False)

    return train_loader, test_loader
