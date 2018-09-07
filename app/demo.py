import torch
import numpy as np
from torch.autograd import Variable

from scipy import misc
from model import Steganographer

def to_bits(s):
    # Convert string to a list of bits
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def from_bits(bits):
    # Convert a list of bits to a string
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

class Wrapper(object):

    def __init__(self, path_to_weights="model/weights.pt"):
        self._model = torch.load(path_to_weights, map_location=lambda storage, loc: storage)

    def encode(self, message, path_to_input, path_to_output):
        image = misc.imread(path_to_input)/125.0-1.0
        image = torch.FloatTensor(image).permute(2,1,0).unsqueeze(0)
        _, _, height, width = image.size()

        data = to_bits(message)
        while len(data) < width*height - len(to_bits(message)):
            data += to_bits(message)
        data += [0] * (width*height - len(data))
        data = torch.FloatTensor(data).view(1, 1, height, width)

        stega, decoder, classifier = self._model(Variable(image), Variable(data))
        output = stega[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
        misc.imsave(path_to_output, np.maximum(0.0, np.minimum(255.0, output)))

    def decode(self, path_to_output):
        image = misc.imread(path_to_output)/125.0-1.0
        image = torch.FloatTensor(image).permute(2,1,0).unsqueeze(0)
        _, _, height, width = image.size()

        data = (self._model.decoder(Variable(image)) > 0.0).view(-1).data.cpu().numpy().tolist()
        print(from_bits(data)[:50])

wrapper = Wrapper()
wrapper.encode("Hello World!\n", "demo/kalyan.jpg", "demo/output.jpg")
print(wrapper.decode("demo/output.jpg"))
