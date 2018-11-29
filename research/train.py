import os
import gc
import time
import json
import argparse
import imageio
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import ssim
from data import load_dataset
from deepsteganography.architectures import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--epochs', default=32, type=int)
parser.add_argument('--dataset', default="div2k", type=str)
parser.add_argument('--data_depth', default=1, type=int)
parser.add_argument('--hidden_size', default=32, type=int)
parser.add_argument('--architecture', default="basic", type=str)
args = parser.parse_args()
args.device = torch.device('cpu')
if args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')

# Load datasets
train, val = load_dataset(args.dataset)
exemplar = next(iter(val))[0]

# Load the architecture
encoder = {
    "basic": BasicEncoder,
    "residual": ResidualEncoder,
    "dense": DenseEncoder,
}[args.architecture]

# Load models
critic = BasicCritic(args.hidden_size).to(args.device)
decoder = BasicDecoder(args.data_depth, args.hidden_size).to(args.device)
encoder = BasicEncoder(args.data_depth, args.hidden_size).to(args.device)

critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
decoder_optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=1e-4)

# cover image -> (cover, y_true, stega, y_pred)
def inference(cover, quantize=False):
    N, _, H, W = cover.size()
    cover = cover.to(args.device)
    y_true = torch.zeros((N, args.data_depth, H, W), device=args.device).random_(0, 2)

    stega = encoder(cover, y_true)
    if quantize:
        stega = (255.0 * (stega + 1.0) / 2.0).long()
        stega = 2.0 * stega.float() / 255.0 - 1.0
    y_pred = decoder(stega)

    return cover, y_true, stega, y_pred

# (cover, y_true, stega, y_pred) -> metrics
def evaluate(cover, y_true, stega, y_pred):
    encoder_mse = F.mse_loss(stega, cover)
    decoder_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    decoder_acc = (y_pred >= 0.0).eq(y_true >= 0.5).sum().float() / y_true.numel()
    cover_score = torch.mean(critic(cover))
    stega_score = torch.mean(critic(stega))
    return encoder_mse, decoder_loss, decoder_acc, cover_score, stega_score

# Logging
working_dir = "results/%s/" % int(time.time())
os.mkdir(working_dir)
os.mkdir(working_dir + "weights")
os.mkdir(working_dir + "samples")
with open(working_dir + "/config.json", "wt") as fout:
    fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

# Start training
history = []
for epoch in range(1, args.epochs+1):
    metrics = {
        "val.encoder_mse": [],
        "val.decoder_loss": [],
        "val.decoder_acc": [],
        "val.cover_score": [],
        "val.stega_score": [],
        "val.ssim": [],
        "val.psnr": [],
        "train.encoder_mse": [],
        "train.decoder_loss": [],
        "train.decoder_acc": [],
        "train.cover_score": [],
        "train.stega_score": [],
    }

    # Train the critic
    for cover, _ in tqdm(train):
        gc.collect()
        cover, y_true, stega, y_pred = inference(cover)
        _, _, _, cover_score, stega_score = evaluate(cover, y_true, stega, y_pred)

        critic_optimizer.zero_grad()
        (cover_score - stega_score).backward(retain_graph=True)
        critic_optimizer.step()
        for p in critic.parameters():
            p.data.clamp_(-0.1, 0.1)

        metrics["train.cover_score"].append(cover_score.item())
        metrics["train.stega_score"].append(stega_score.item())

    # Train the encoder/decoder
    for cover, _ in tqdm(train):
        gc.collect()
        cover, y_true, stega, y_pred = inference(cover)
        encoder_mse, decoder_loss, decoder_acc, _, stega_score = evaluate(cover, y_true, stega, y_pred)

        decoder_optimizer.zero_grad()
        (100.0 * encoder_mse + decoder_loss + stega_score).backward()
        decoder_optimizer.step()

        metrics["train.encoder_mse"].append(encoder_mse.item())
        metrics["train.decoder_loss"].append(decoder_loss.item())
        metrics["train.decoder_acc"].append(decoder_acc.item())

    # Validation
    for cover_image, _ in tqdm(val):
        gc.collect()
        cover, y_true, stega, y_pred = inference(cover_image, quantize=True)
        encoder_mse, decoder_loss, decoder_acc, cover_score, stega_score = evaluate(cover, y_true, stega, y_pred)

        metrics["val.encoder_mse"].append(encoder_mse.item())
        metrics["val.decoder_loss"].append(decoder_loss.item())
        metrics["val.decoder_acc"].append(decoder_acc.item())
        metrics["val.cover_score"].append(cover_score.item())
        metrics["val.stega_score"].append(stega_score.item())
        metrics["val.ssim"].append(ssim(cover, stega).item())
        metrics["val.psnr"].append(10 * torch.log10(4 / encoder_mse).item())
        

    # Exemplar
    cover, y_true, stega, y_pred = inference(exemplar)
    for i in range(stega.size(0)):
        image = (cover[i].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
        imageio.imwrite(working_dir + "samples/%s.cover.png" % i, (255.0 * image).astype("uint8"))

        image = (stega[i].clamp(-1.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
        imageio.imwrite(working_dir + "samples/%s.stega-%02d.png" % (i, epoch), (255.0 * image).astype("uint8"))
    
    # Logging
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    metrics["epoch"] = epoch
    history.append(metrics)
    with open(working_dir + "/train.log", "wt") as fout:
        fout.write(json.dumps(history, indent=2))
    torch.save((encoder, decoder, critic), working_dir + "weights/%s.acc-%.03f.pt" % (epoch, metrics["val.decoder_acc"]))
