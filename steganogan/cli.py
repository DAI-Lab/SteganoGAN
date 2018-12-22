# -*- coding: utf-8 -*-

"""Top-level package for SteganoGAN."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev.dev'

import argparse
import os

from steganogan.models import SteganoGAN


def _get_steganogan(args):
    model_name = '{}.steg'.format(args.architecture)
    pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
    model_path = os.path.join(pretrained_path, model_name)
    return SteganoGAN.load(model_path, cuda=not args.cpu, verbose=args.verbose)


def _encode(args):
    """Given loads a pretrained pickel, encodes the image with it."""
    steganogan = _get_steganogan(args)
    steganogan.encode(args.cover, args.output, args.message)


def _decode(args):
    try:
        steganogan = _get_steganogan(args)
        message = steganogan.decode(args.image)

        if args.verbose:
            print('Message successfully decoded:')

        print(message)

    except Exception as e:
        print('ERROR: {}'.format(e))


def _get_parser():

    # Parent Parser - Shared options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parent.add_argument('-a', '--architecture', default='dense', choices=('basic', 'dense'),
                        help='Model architecture. Use the same one for both encoding and decoding')
    parent.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    parser = argparse.ArgumentParser(description='SteganoGAN Command Line Interface')

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Encode Parser
    encode = subparsers.add_parser('encode', parents=[parent],
                                   help='Hide a message into a steganographic image')
    encode.set_defaults(action=_encode)
    encode.add_argument('-o', '--output', default='output.png',
                        help='Path and name to save the output image')
    encode.add_argument('cover', help='Path to the image to use as cover')
    encode.add_argument('message', help='Message to encode')

    # Decode Parser
    decode = subparsers.add_parser('decode', parents=[parent],
                                   help='Read a message from a steganographic image')
    decode.set_defaults(action=_decode)
    decode.add_argument('image', help='Path to the image with the hidden message')

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)
