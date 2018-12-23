import argparse
import torch
import json
from time import time
from steganogan import SteganoGAN
from steganogan.loader import DataLoader
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import BasicEncoder, ResidualEncoder, DenseEncoder

torch.manual_seed(42)
timestamp = int(time())

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=32, type=int)
parser.add_argument('--encoder', default="basic", type=str)
parser.add_argument('--data_depth', default=1, type=int)
parser.add_argument('--hidden_size', default=32, type=int)
parser.add_argument('--dataset', default="div2k", type=str)
args = parser.parse_args()

train = DataLoader('data/%s/train/' % args.dataset)
validation = DataLoader('data/%s/val/' % args.dataset)

steganogan = SteganoGAN(
    data_depth=args.data_depth,
    encoder={
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder],
    decoder=DenseDecoder,
    critic=BasicCritic,
    hidden_size=args.hidden_size,
    cuda=True,
    verbose=True,
    log_dir='models/%s' % timestamp
)
with open('models/%s/config.json' % timestamp, "wt") as fout:
    fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))
steganogan.fit(train, validation, epochs=args.epochs)
steganogan.save('models/%s/weights.steg' % timestamp)
