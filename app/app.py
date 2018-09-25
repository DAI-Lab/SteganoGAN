#!/usr/bin/env python3
import torch
import argparse
import numpy as np

from scipy import misc
from collections import Counter
from model import Steganographer
from utils import *

parser = argparse.ArgumentParser(description='Encode or decode steganographic images.')
parser.add_argument('mode', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--text', type=str)
parser.add_argument('--dest', type=str)
args = parser.parse_args()

if args.mode == "encode":
    if not args.src:  args.src = "demo/kevin.jpg"
    if not args.dest: args.dest = args.src.replace(".jpg", ".out.png")
    if not args.text: args.text = "Hello World! This is a medium length demo message which requires ~100 bytes to store."
else:
    if not args.src:  args.src = "demo/kevin.out.png"

class Wrapper(object):

    def __init__(self, path_to_weights="model/weights.pt"):
        self._model = torch.load(path_to_weights, map_location=lambda storage, loc: storage)
        torch.save(self._model, path_to_weights)

    def encode(self, message, path_to_input, path_to_output):
        image = misc.imread(path_to_input, mode="RGB")/125.0-1.0
        image = torch.FloatTensor(image).permute(2,1,0).unsqueeze(0)
        _, _, height, width = image.size()

        encoded = bytearray_to_bits(text_to_bytearray(message) + bytearray(b"\x0000"))
        data = encoded
        while len(data) < width*height:
            print("Embedding...")
            data += encoded
        data = data[:width*height]
        data = torch.FloatTensor(data).view(1, 1, height, width)

        stega, decoder, classifier = self._model(image, data)
        output = stega[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
        misc.imsave(path_to_output, output)

        if message in self.decode(path_to_output):
            return True
        return False

    def decode(self, path_to_output):
        image = misc.imread(path_to_output, mode="RGB")/125.0-1.0
        image = torch.FloatTensor(image).permute(2,1,0).unsqueeze(0)
        _, _, height, width = image.size()

        data = (self._model.decoder(image) > 0.0).view(-1).data.cpu().numpy().tolist()
        candidates = list(filter(lambda x: x, [bytearray_to_text(bytearray(x)) for x in bits_to_bytearray(data).split(b"\x0000")]))
        return Counter(candidates).most_common(1)[0][0]

wrapper = Wrapper()
if args.mode == "encode":
    if wrapper.encode(args.text, args.src, args.dest):
        print("Success")
    else:
        print("Embedding failed.")
if args.mode == "decode":
    print(wrapper.decode(args.src))
