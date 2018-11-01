import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
import imageio
import os, json, time
from tqdm import tqdm
from itertools import chain
from data import load_dataset
from model import Encoder, Decoder, Critic

# config
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=64, type=int)
parser.add_argument('--data_depth', default=1, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--hidden_size', default=32, type=int)
parser.add_argument('--mse_weight', default=1.0, type=float)
parser.add_argument('--critic_weight', default=1.0, type=float)
parser.add_argument('--clip_grad_norm', default=0.25, type=float)
parser.add_argument('--dataset', default="div2k", type=str)
args = parser.parse_args()
args.device = torch.device('cpu')
if torch.cuda.is_available():
    args.device = torch.device('cuda')

# data
test = load_dataset("%s/test" % args.dataset, batch_size=args.batch_size)
train = load_dataset("%s/train" % args.dataset, batch_size=args.batch_size)
exemplar = next(iter(test))[0]

# models
encoder = Encoder(args.data_depth, args.hidden_size).to(args.device)
decoder = Decoder(args.data_depth, args.hidden_size).to(args.device)
critic = Critic(args.hidden_size).to(args.device)

# optimizers
g_optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-4)
c_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

def inference(cover_image, quantize=False):
    N, _, H, W = cover_image.size()
    cover_image = cover_image.to(args.device)
    y_true = torch.zeros((N, args.data_depth, H, W), device=args.device).random_(0, 2)

    stega_image = encoder(y_true, cover_image)
    if quantize:
        stega_image = (stega_image * 255.0).long()
        stega_image = stega_image.float() / 255.0
    y_pred = decoder(stega_image)

    return cover_image, y_true, stega_image, y_pred

def score(cover_image, y_true, stega_image, y_pred):
    acc = (y_pred >= 0.0).eq(y_true >= 0.5).sum().float() / y_true.numel()
    mse_loss = F.mse_loss(stega_image, cover_image)
    decoder_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    cover_score = torch.mean(critic(cover_image))
    stega_score = torch.mean(critic(stega_image))
    return acc, mse_loss, decoder_loss, cover_score, stega_score

# training
def run_epoch():
    metrics = {
        "train.acc": [],
        "train.mse_loss": [],
        "train.decoder_loss": [],
        "train.cover_score": [],
        "train.stega_score": [],
        "test.acc": [],
        "test.mse_loss": [],
        "test.decoder_loss": [],
        "test.cover_score": [],
        "test.stega_score": [],
    }

    # pre-train
    for cover_image, _ in tqdm(train):
        cover_image, y_true, stega_image, y_pred = inference(cover_image)
        acc, mse_loss, decoder_loss, cover_score, stega_score = score(cover_image, y_true, stega_image, y_pred)

        c_optimizer.zero_grad()
        (cover_score - stega_score).backward(retain_graph=True)
        c_optimizer.step()
        for p in critic.parameters():
            p.data.clamp_(-0.01, 0.01)

    # train
    for cover_image, _ in tqdm(train):
        cover_image, y_true, stega_image, y_pred = inference(cover_image)
        acc, mse_loss, decoder_loss, cover_score, stega_score = score(cover_image, y_true, stega_image, y_pred)

        # train the critic
        c_optimizer.zero_grad()
        (cover_score - stega_score).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.clip_grad_norm)
        c_optimizer.step()
        for p in critic.parameters():
            p.data.clamp_(-0.01, 0.01)

        # train the encoder + decoder
        g_optimizer.zero_grad()
        (args.critic_weight*stega_score + args.mse_weight*mse_loss + decoder_loss).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad_norm)
        g_optimizer.step()

        metrics["train.acc"].append(acc.item())
        metrics["train.mse_loss"].append(mse_loss.item())
        metrics["train.decoder_loss"].append(decoder_loss.item())
        metrics["train.cover_score"].append(cover_score.item())
        metrics["train.stega_score"].append(stega_score.item())

    # test
    for cover_image, _ in tqdm(test):
        cover_image, y_true, stega_image, y_pred = inference(cover_image, quantize=True)
        acc, mse_loss, decoder_loss, cover_score, stega_score = score(cover_image, y_true, stega_image, y_pred)

        metrics["test.acc"].append(acc.item())
        metrics["test.mse_loss"].append(mse_loss.item())
        metrics["test.decoder_loss"].append(decoder_loss.item())
        metrics["test.cover_score"].append(cover_score.item())
        metrics["test.stega_score"].append(stega_score.item())

    cover_image, y_true, stega_image, y_pred = inference(exemplar)
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    return metrics, cover_image, stega_image

if __name__ == "__main__":
    results_dir = "results/%s/" % int(time.time())
    os.mkdir(results_dir)
    os.mkdir(results_dir + "samples")

    with open(results_dir + "config.json", "wt") as fout:
        config = {
            "epochs": args.epochs,
            "dataset": args.dataset,
            "data_depth": args.data_depth,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "mse_weight": args.mse_weight,
            "critic_weight": args.critic_weight,
        }
        json.dump(config, fout)
        print(config)

    fout = open(results_dir + "train.log", "wt")
    for epoch in range(args.epochs):
        metrics, cover_image, stega_image = run_epoch()
        metrics["epoch"] = epoch
        fout.write(json.dumps(metrics) + "\n")
        fout.flush()

        for i in range(stega_image.size(0)):
            image = (255 * cover_image[i,:,:,:].cpu().permute(1, 2, 0).detach().numpy()).astype("uint8")
            imageio.imwrite(results_dir + "samples/%s.%s-cover.png" % (epoch, i), image)

            image = (255 * stega_image[i,:,:,:].cpu().permute(1, 2, 0).detach().numpy()).astype("uint8")
            imageio.imwrite(results_dir + "samples/%s.%s-stega.png" % (epoch, i), image)

        torch.cuda.empty_cache()
