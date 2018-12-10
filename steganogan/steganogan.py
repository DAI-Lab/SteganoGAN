import os
import gc
import json
import time
import imageio
import torch
import torch.nn.functional as F
import torch.optim as optim
# import sys

from tqdm import tqdm
from data import load_dataset
from collections import Counter
from glob import glob

from imageio import imread, imwrite

from steganogan.critics import BasicCritic
from steganogan.decoder import BasicDecoder
from steganogan.encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from steganogan.utils.coding import bits_to_bytearray, bytearray_to_text, text_to_bits

# Discover pretrained models
pretrained_models = []
sys.modules['architectures'] = architectures
for path in glob(os.path.join(os.path.dirname(__file__), "pretrained/*.pt")):
    pretrained_models.append(path)

class Steganographer(object):

    def __init__(self):
        path_to_model = pretrained_models[0] # TODO: design better way to select models
        self.encoder, self.decoder, _ = torch.load(path_to_model, map_location=lambda storage, loc: storage)

    def encode(self, image, text, stega):
        assert isinstance(image, str), "path to input image"
        assert isinstance(text, str), "text message"
        assert isinstance(stega, str), "path to output image"

        image = imread(image, pilmode="RGB") / 127.5 - 1.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)

        _, _, height, width = image.size()
        data_depth = self.decoder(image).size(1) # TODO: use data_depth prop
        data = self._make_payload(width, height, data_depth, text)

        output = self.encoder(image, data)[0].clamp(-1.0, 1.0)
        print(output.min(), output.max())
        output = (output.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(stega, output.astype("uint8"))

    def decode(self, stega):
        assert isinstance(stega, str)
        if not os.path.exists(stega):
            raise ValueError("Unable to read %s." % stega)

        # extract a bit vector
        stega = imread(stega, pilmode="RGB") / 255.0
        stega = torch.FloatTensor(stega).permute(2, 1, 0).unsqueeze(0)
        stega = self.decoder(stega).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = stega.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b"\x00\x00\x00\x00"):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError("Failed to find message.")
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


    def fit(self):
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
        for epoch in range(1, args.epochs + 1):
            metrics = {
                "val.encoder_mse": [],
                "val.decoder_loss": [],
                "val.decoder_acc": [],
                "val.cover_score": [],
                "val.stega_score": [],
                "val.ssim": [],
                "val.psnr": [],
                "val.bpp": [],
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

                encoder_mse, decoder_loss, decoder_acc, _, stega_score = evaluate(cover,
                                                                                  y_true,
                                                                                  stega,
                                                                                  y_pred)

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
                encoder_mse, decoder_loss, decoder_acc, cover_score, stega_score = evaluate(cover,
                                                                                            y_true,
                                                                                            stega,
                                                                                            y_pred)
                metrics["val.encoder_mse"].append(encoder_mse.item())
                metrics["val.decoder_loss"].append(decoder_loss.item())
                metrics["val.decoder_acc"].append(decoder_acc.item())
                metrics["val.cover_score"].append(cover_score.item())
                metrics["val.stega_score"].append(stega_score.item())
                metrics["val.ssim"].append(ssim(cover, stega).item())
                metrics["val.psnr"].append(10 * torch.log10(4 / encoder_mse).item())
                metrics["val.bpp"].append(args.data_depth * (2 * decoder_acc.item() - 1))
            # Exemplar
            cover, y_true, stega, y_pred = inference(exemplar)
            for i in range(stega.size(0)):
                imwi_dir = working_dir + 'samples/{}.cover.png'.format(i)

                image = (cover[i].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
                imageio.imwrite(imwi_dir, (255.0 * image).astype("uint8"))

                image_output = working_dir + "samples/{}.stega-{:2d}.png".format(i, epoch)

                _img = (stega[i].clamp(-1.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() + 1.0)

                image = _img / 2.0

                imageio.imwrite(image_output, (255.0 * image).astype("uint8"))

            # Logging
            metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            metrics["epoch"] = epoch
            history.append(metrics)
            with open(working_dir + "/train.log", "wt") as fout:
                fout.write(json.dumps(history, indent=2))

            sa_dir = working_dir + "weights/{}.acc-{:03f}.pt".format(epoch,
                                                                     metrics["val.decoder_acc"])

            torch.save((encoder, decoder, critic), sa_dir)
