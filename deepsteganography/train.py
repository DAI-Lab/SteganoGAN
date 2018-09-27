import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm

from data import test, train
from model import Steganographer


class ModelTrainer(object):

    def __init__(self, data_depth=1, batch_size=4, lr=1e-4):
        self.data_depth = data_depth
        self.batch_size = batch_size
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.epoch = 0
        self.model = Steganographer(data_depth).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def step(self, train_set, test_set):
        self.epoch += 1
        d_train_loss, c_train_loss = self._train(train_set)
        d_test_loss, c_test_loss = self._test(test_set)
        torch.save(
            self.model, "weights/epoch-%s.loss-%.02f.pt" %
            (self.epoch, d_test_loss))

    def _train(self, train_set):
        iterator = tqdm(train_set)
        d_loss, c_loss, N = 0.0, 0.0, 0
        for image, _ in iterator:
            self.optimizer.zero_grad()
            image, data = self._make_pair(image)
            y, decoding_loss, classifier_loss = self.model(image, data)
            (decoding_loss + classifier_loss).backward()
            self.optimizer.step()

            d_loss += decoding_loss.item()
            c_loss += classifier_loss.item()
            N += 1

            iterator.set_description(
                "Epoch %s, Decoder %s, Classifier %s" %
                (self.epoch, d_loss / N, c_loss / N))
        return d_loss / N, c_loss / N

    def _test(self, test_set):
        iterator = tqdm(test_set)
        d_loss, c_loss, N = 0.0, 0.0, 0
        for image, _ in iterator:
            image, data = self._make_pair(image)
            y, decoding_loss, classifier_loss = self.model(image, data)

            d_loss += decoding_loss.item()
            c_loss += classifier_loss.item()
            N += 1

            iterator.set_description(
                "Epoch %s, Decoder %s, Classifier %s" %
                (self.epoch, d_loss / N, c_loss / N))
        return d_loss / N, c_loss / N

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
    trainer = ModelTrainer()
    for _ in range(32):
        trainer.step(train, test)
