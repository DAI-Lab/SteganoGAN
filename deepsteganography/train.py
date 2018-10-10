import argparse
import json
import os
import time

import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm

from architectures import SteganographerV1
from data import load_train_test

parser = argparse.ArgumentParser(description='Train a steganography nodel.')
parser.add_argument('--epochs', type=int, default=16)
parser.add_argument(
    '--dataset',
    type=str,
    default='data/caltech256',
    help='data/caltech256 | data/mscoco')
parser.add_argument(
    '--test_dataset',
    type=str,
    default=None,
    help='data/caltech256 | data/mscoco')
parser.add_argument('--data_depth', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=16)
args = parser.parse_args()

class ModelTrainer(object):

    def __init__(self, dataset, test_dataset, data_depth, hidden_dim, batch_size=4, lr=1e-4):
        self.data_depth = data_depth
        self.batch_size = batch_size
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.epoch = 0
        self.model = SteganographerV1(data_depth, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_set, self.test_set = load_train_test(dataset)
        if test_dataset:
            _, self.test_set = load_train_test(test_dataset)

        self._log_dir = "pretrained/%s.%s/" % (
            int(time.time()), type(self.model).__name__)
        os.mkdir(self._log_dir)
        with open(self._log_dir + "manifest.json", "wt") as fout:
            json.dump({
                "dataset": dataset,
                "test_dataset": test_dataset,
                "data_depth": data_depth,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "lr": lr,
            }, fout)

    def step(self):
        self.epoch += 1
        _, d_train_loss, c_train_loss = self._train(self.train_set)
        d_test_acc, d_test_loss, c_test_loss = self._test(self.test_set)

        torch.save(self.model, self._log_dir + "epoch-%s.acc-%.03f.pt" % (self.epoch, d_test_acc))

    def _train(self, train_set):
        iterator = tqdm(train_set)
        d_acc, d_loss, c_loss, N = 0.0, 0.0, 0.0, 0
        for image, _ in iterator:
            self.optimizer.zero_grad()
            image, data = self._make_pair(image)
            y, decoding_acc, decoding_loss, classifier_loss = self.model(
                image, data)
            (decoding_loss + classifier_loss).backward()
            self.optimizer.step()

            d_acc += decoding_acc
            d_loss += decoding_loss.item()
            c_loss += classifier_loss.item()
            N += 1

            iterator.set_description(
                "Epoch %s, Decoder %s / %s, Classifier %s" %
                (self.epoch, d_acc / N, d_loss / N, c_loss / N))
        return d_acc / N, d_loss / N, c_loss / N

    def _test(self, test_set):
        iterator = tqdm(test_set)
        d_acc, d_loss, c_loss, N = 0.0, 0.0, 0.0, 0
        for image, _ in iterator:
            image, data = self._make_pair(image)
            y, decoding_acc, decoding_loss, classifier_loss = self.model(
                image, data)

            d_acc += decoding_acc
            d_loss += decoding_loss.item()
            c_loss += classifier_loss.item()
            N += 1

            iterator.set_description(
                "Epoch %s, Decoder %s / %s, Classifier %s" %
                (self.epoch, d_acc / N, d_loss / N, c_loss / N))
        return d_acc / N, d_loss / N, c_loss / N

    def _make_pair(self, image):
        _, _, height, width = image.size()
        image = autograd.Variable(
            image.to(
                self.device).expand(
                self.batch_size,
                3,
                height,
                width))
        data = autograd.Variable(
            torch.zeros(
                (self.batch_size, self.data_depth, height, width)).random_(
                0, 2).to(
                self.device))
        return image, data


if __name__ == "__main__":
    trainer = ModelTrainer(args.dataset, args.test_dataset, args.data_depth, args.hidden_dim)
    for _ in range(args.epochs):
        trainer.step()
