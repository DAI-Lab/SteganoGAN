import os
import sys
import torch
from glob import glob
from collections import Counter
from imageio import imread, imwrite

from . import architectures
from .utils.coding import *

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
