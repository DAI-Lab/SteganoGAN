import os
# fix relative imports
import sys
from collections import Counter
from glob import glob

import torch
from imageio import imread, imwrite

from . import model
from .utils import bits_to_bytearray, bytearray_to_text, text_to_bits

sys.modules['model'] = model


class Steganographer(object):

    def __init__(self):
        weights_dir = os.path.dirname(__file__)
        for path in glob(os.path.join(weights_dir, "weights", "*.pt")):
            self.model = torch.load(
                path, map_location=lambda storage, loc: storage)

    def encode(self, image, text, stega):
        assert isinstance(image, str)
        assert isinstance(text, str)
        assert isinstance(stega, str)

        image = imread(image, pilmode="RGB") / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        _, _, height, width = image.size()

        data = self._make_payload(width, height, text)

        result, _, _ = self.model(image, data)
        output = result[0].permute(2, 1, 0).data.cpu().numpy() * 255.0
        imwrite(stega, output.astype("uint8"))

    def decode(self, stega):
        assert isinstance(stega, str)
        if not os.path.exists(stega):
            raise ValueError("Unable to read %s." % stega)

        # extract a bit vector
        stega = imread(stega, pilmode="RGB") / 255.0
        stega = torch.FloatTensor(stega).permute(2, 1, 0).unsqueeze(0)
        stega = self.model.decoder(stega).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = stega.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b"\x00\x00\x00\x00"):
            candidate = bytearray_to_text(bytearray(candidate))
            candidates[candidate] += 1

        # choose most common message
        candidate, count = candidates.most_common(1)[0]
        return candidate

    def _make_payload(self, width, height, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = text_to_bits(text) + [0] * 32

        data = message
        while len(data) < width * height:
            data += message
        data = data[:width * height]

        return torch.FloatTensor(data).view(1, 1, height, width)
