# -*- coding: utf-8 -*-

import argparse
import warnings

from torch.serialization import SourceChangeWarning

from steganogan.models import SteganoGAN

warnings.filterwarnings('ignore', category=SourceChangeWarning)


def _get_steganogan(args):

    steganogan_kwargs = {
        'cuda': not args.cpu,
        'verbose': args.verbose
    }

    if args.path:
        steganogan_kwargs['path'] = args.path
    else:
        steganogan_kwargs['architecture'] = args.architecture

    return SteganoGAN.load(**steganogan_kwargs)


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
        import traceback
        traceback.print_exc()


def _get_parser():

    # Parent Parser - Shared options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    group = parent.add_mutually_exclusive_group()
    group.add_argument('-a', '--architecture', default='dense',
                       choices={'basic', 'dense', 'residual'},
                       help='Model architecture. Use the same one for both encoding and decoding')

    group.add_argument('-p', '--path', help='Load a pretrained model from a given path.')
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
