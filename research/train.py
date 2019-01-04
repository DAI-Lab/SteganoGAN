#!/usr/bin/env python3
import argparse
import torch
import json
import os
from time import time
from steganogan import SteganoGAN
from steganogan.loader import DataLoader
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import BasicEncoder, ResidualEncoder, DenseEncoder

def main():
    torch.manual_seed(42)
    timestamp = str(int(time()*1000))

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--encoder', default="basic", type=str)
    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--output', default=False, type=str)
    args = parser.parse_args()

    train = DataLoader(os.path.join("data", args.dataset, "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False)

    encoder = {
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder]
    steganogan = SteganoGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        hidden_size=args.hidden_size,
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', timestamp)
    )
    with open(os.path.join("models", timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    steganogan.fit(train, validation, epochs=args.epochs)
    steganogan.save(os.path.join("models", timestamp, "weights.steg"))
    if args.output:
        steganogan.save(args.output)

if __name__ == '__main__':
    main()
