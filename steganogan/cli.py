# -*- coding: utf-8 -*-

"""Top-level package for SteganoGAN."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev.dev'

import argparse
import logging
import os

from steganogan.models import SteganoGAN

LOGGER = logging.getLogger(__name__)


def _logging_setup(verbosity=1):
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    LOGGER.setLevel(log_level)
    LOGGER.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)


def _get_steganogan(model_type):
    model_name = '{}.p'.format(model_type)

    pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    pickle_path = os.path.join(pretrained_path, model_name)

    return SteganoGAN.load(pickle_path)


def _encode(args):
    """Given loads a pretrained pickel, encodes the image with it."""
    steganogan = _get_steganogan(args.encoder)
    steganogan.encode(args.input, args.output, args.message)


def _decode(args):
    steganogan = _get_steganogan(args.decoder)
    steganogan.decode(args.input)


def _get_parser():

    parser = argparse.ArgumentParser(description='SteganoGAN Command Line Interface')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-a', '--architecture', default='dense',
                        choices=('basic', 'dense', 'residual'))

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Encoder
    subparser = subparsers.add_parser('encode', help='Encode a given image.')
    subparser.set_defaults(action=_encode)
    subparser.add_argument('-o', '--output', default='output.jpg',
                           help='Path and name to save the output image.')
    subparser.add_argument('cover', help='Path to the image to use as cover.')
    subparser.add_argument('message', nargs='+', help='Message to encode.')

    # Decoder
    subparser = subparsers.add_parser('decode', help='Decode message from a given image.')
    subparser.set_defaults(action=_decode)
    subparser.add_argument('image', help='Path to the image with the encoded message.')

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    _logging_setup(args.verbose)

    if not args.action:
        parser.print_help()
        parser.exit()

    print(args.action)
    # args.action(args)
