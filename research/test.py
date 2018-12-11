import argparse
import os

from steganogan import Steganographer
from steganogan.steganogan import pretrained_models


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="in.png")
    parser.add_argument('-o', '--output', default="out.png")
    parser.add_argument('-m', '--model')
    parser.add_argument('message')
    args = parser.parse_args()

    model_path = args.model
    if model_path:
        if model_path == 'last':
            results = os.listdir('results')[-1]
            weights = os.path.join('results', results, 'weights')
            model = os.listdir(weights)[-1]
            model_path = os.path.join(weights, model)

        print('Using model {}'.format(model_path))
        pretrained_models.insert(0, model_path)

    model = Steganographer()

    print("Encoding message: {}".format(args.message))
    model.encode(args.input, args.message, args.output)

    print("Decoding message")
    message = model.decode(args.output)
    print("Decoded message: {}".format(message))
