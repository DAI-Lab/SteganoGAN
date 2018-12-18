# -*- coding: utf-8 -*-

import gc
import inspect
import json
import os
import pickle
# import time
from collections import Counter

import imageio
import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan.utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')


class SteganoGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def _get_device(self):
        """Returns torch device"""
        if self.cuda and torch.cuda.is_available():
            return torch.device('cuda')

        return torch.device('cpu')

    def __init__(self, data_depth, encoder, decoder, critic, cuda=False, train_path=DEFAULT_PATH,
                 verbose=False, torch_save=False, save_epoch=False, **kwargs):

        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.cuda = cuda
        self.device = self._get_device()

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.train_path = train_path
        self.verbose = verbose
        self.train_metrics = None
        self.torch_save = torch_save
        self.save_epoch = save_epoch

    def encode(self, image, output, text):
        """Encode an image.
        Args:
            image(str): Path to the image to be encoded.
            output(str): Path to where we want to save the encoded image.
            text(str): String of what we want to encode inside the image.
        """
        # Force to use cpu when encode / decode
        if self.cuda:
            self.encoder.to(torch.device('cpu'))

        image = imread(image, pilmode='RGB') / 127.5 - 1.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)

        _, _, height, width = image.size()
        data = self._make_payload(width, height, self.data_depth, text)

        encoded_image = self.encoder(image, data)[0].clamp(-1.0, 1.0)
        #  print(encoded_image.min(), encoded_image.max())
        encoded_image = (encoded_image.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, encoded_image.astype('uint8'))
        print('Encoding completed.')

    def decode(self, image):

        if self.cuda:
            self.decoder.to(torch.device('cpu'))

        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')
        candidate, count = candidates.most_common(1)[0]
        return candidate

    def _make_payload(self, width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = text_to_bits(text) + [0] * 32

        data = message
        while len(data) < width * height * depth:
            data += message
        data = data[:width * height * depth]

        return torch.FloatTensor(data).view(1, depth, height, width)

    def _inference(self, cover, quantize=False):
        """cover image -> (cover, y_true, stega, y_pred)"""
        N, _, H, W = cover.size()
        cover = cover.to(self.device)
        y_true = torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

        stega = self.encoder(cover, y_true)
        if quantize:
            stega = (255.0 * (stega + 1.0) / 2.0).long()
            stega = 2.0 * stega.float() / 255.0 - 1.0
        y_pred = self.decoder(stega)

        return cover, y_true, stega, y_pred

    def _evaluate(self, cover, y_true, stega, y_pred):
        """(cover, y_true, stega, y_pred) -> metrics"""
        encoder_mse = mse_loss(stega, cover)
        decoder_loss = binary_cross_entropy_with_logits(y_pred, y_true)
        decoder_acc = (y_pred >= 0.0).eq(y_true >= 0.5).sum().float() / y_true.numel()
        cover_score = torch.mean(self.critic(cover))
        stega_score = torch.mean(self.critic(stega))

        return encoder_mse, decoder_loss, decoder_acc, cover_score, stega_score

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, decoder_optimizer

    def _create_folders(self):
        # Logging
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(os.path.join(self.train_path, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(self.train_path, 'samples'), exist_ok=True)

    def _create_metrics(self):
        return {
            'val.encoder_mse': list(),
            'val.decoder_loss': list(),
            'val.decoder_acc': list(),
            'val.cover_score': list(),
            'val.stega_score': list(),
            'val.ssim': list(),
            'val.psnr': list(),
            'val.bpp': list(),
            'train.encoder_mse': list(),
            'train.decoder_loss': list(),
            'train.decoder_acc': list(),
            'train.cover_score': list(),
            'train.stega_score': list(),
        }

    def _train_critic(self, train, metrics):
        """Train the critic"""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover, y_true, stega, y_pred = self._inference(cover)
            _, _, _, cover_score, stega_score = self._evaluate(cover, y_true, stega, y_pred)

            self.critic_optimizer.zero_grad()
            (cover_score - stega_score).backward(retain_graph=True)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            if self.verbose:
                metrics['train.cover_score'].append(cover_score.item())
                metrics['train.stega_score'].append(stega_score.item())

    def _train_encoder_decoder(self, train, metrics):
        """Train the encoder, decoder"""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover, y_true, stega, y_pred = self._inference(cover)

            evaluate_result = self._evaluate(cover, y_true, stega, y_pred)
            encoder_mse, decoder_loss, decoder_acc, _, stega_score = evaluate_result

            self.decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + stega_score).backward()
            self.decoder_optimizer.step()

            if self.verbose:
                metrics['train.encoder_mse'].append(encoder_mse.item())
                metrics['train.decoder_loss'].append(decoder_loss.item())
                metrics['train.decoder_acc'].append(decoder_acc.item())

    def _train_validate(self, validate, metrics):
        """Train the validation"""
        for cover_image, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover, y_true, stega, y_pred = self._inference(cover_image, quantize=True)

            evaluate_result = self._evaluate(cover, y_true, stega, y_pred)
            encoder_mse, decoder_loss, decoder_acc, cover_score, stega_score = evaluate_result

            if self.verbose:
                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.stega_score'].append(stega_score.item())
                metrics['val.ssim'].append(ssim(cover, stega).item())
                metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
                metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def _train_exemplar(self, exemplar, epoch):
        cover, y_true, stega, y_pred = self._inference(exemplar)
        for i in range(stega.size(0)):
            imwi_dir = self.train_path + 'samples/{}.cover.png'.format(i)
            image = (cover[i].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(imwi_dir, (255.0 * image).astype('uint8'))
            image_output = self.train_path + 'samples/{}.stega-{:2d}.png'.format(i, epoch)
            _img = (stega[i].clamp(-1.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() + 1.0)
            image = _img / 2.0
            imageio.imwrite(image_output, (255.0 * image).astype('uint8'))

    def fit(self, train, validate, epochs=5):
        """Train a new model with the given ImageLoader class."""

        # In case we changed the device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        exemplar = next(iter(validate))[0]
        self._create_folders()  # Needed for exemplar

        if self.verbose:
            history = list()

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = self._create_metrics()
            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            # Train the critic
            self._train_critic(train, metrics)

            # Train the encoder/decoder
            self._train_encoder_decoder(train, metrics)

            # Validation
            self._train_validate(validate, metrics)

            # Exemplar
            self._train_exemplar(exemplar, epoch)

            # Logging
            if self.verbose:
                self.train_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
                self.train_metrics['epoch'] = epoch
                history.append(self.train_metrics)

                with open(self.train_path + '/train.log', 'wt') as fout:
                    fout.write(json.dumps(history, indent=4))

                save_name = '{}.acc-{:03f}.pt'.format(epoch, self.train_metrics['val.decoder_acc'])

            if self.save_epoch and self.verbose:
                self.save(os.path.join(self.train_path, save_name))

            if self.torch_save and self.verbose:
                if self.save_epoch:
                    self.save(os.path.join(self.train_path, save_name))

                sv_dir = 'weights/{}.acc-{:03f}.pt'.format(epoch,
                                                           self.train_metrics['val.decoder_acc'])

                save_dir = os.path.join(self.train_path, sv_dir)
                torch.save((self.encoder, self.decoder, self.critic), save_dir)

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()

    def save(self, path):
        """Save the fitted model in the given path. Raises an exception if there is no model."""
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        """Loads an instance of SteganoGAN from the given path."""
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
