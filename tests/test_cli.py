# -*- coding: utf-8 -*-

import os
from unittest import TestCase
from unittest.mock import Mock, call, patch, MagicMock, PropertyMock

from steganogan import cli
from steganogan.models import SteganoGAN

@patch('steganogan.cli.SteganoGAN.load')
def test__get_steganogan(stega_load_mock):
    """
    VALIDATE:
        * The model path is the right one, with the right architecture
        * SteganoGAN.load is called with the right values
        * The output of SteganoGAN.load is returned

    MOCK:
        * SteganoGAN
        * args
    """

    # setup
    params = MagicMock()
    params.cpu = True
    params.architecture = 'basic'
    params.verbose = True

    model_name = '{}.steg'.format(params.architecture)
    parent_path = os.path.dirname(os.path.dirname(__file__))
    stega_path = os.path.join(parent_path, 'steganogan')
    pretrained_path = os.path.join(stega_path, 'pretrained')
    model_path = os.path.join(pretrained_path, model_name)

    # run
    cli_test = cli._get_steganogan(params)

    # assert
    expected_stega_call = call(model_path, cuda=not params.cpu, verbose=params.verbose)

    assert expected_stega_call == stega_load_mock.call_args_list


@patch('steganogan.cli.SteganoGAN')
def test__encode(mock_steganogan):
    """Test that encode has been called with the proper args"""
    """
    VALIDATE:
        * steganogan.encode has been called with the right values

    MOCK:
        * cli._get_steganogan to return a mock of steganogan
    """
    # setup
    params = MagicMock()
    type(params).cpu = PropertyMock(return_value=True)
    type(params).architecture = PropertyMock(return_value='basic')
    type(params).verbose = PropertyMock(return_value=True)

    type(params).cover = PropertyMock(return_value='image.jpg')
    type(params).message = PropertyMock(return_value='Hello world')
    type(params).output = PropertyMock(return_value='output.png')

    model_name = '{}.steg'.format(params.architecture)
    parent_path = os.path.dirname(os.path.dirname(__file__))
    stega_path = os.path.join(parent_path, 'steganogan')
    pretrained_path = os.path.join(stega_path, 'pretrained')
    model_path = os.path.join(pretrained_path, model_name)

    # run
    with patch('steganogan.cli._get_steganogan') as mock_get_steganogan:
        mock_get_steganogan.return_value = mock_steganogan
        cli._encode(params)

    # assert



def test__decode():
    """
    VALIDATE:
        * steganogan.decode has been called with the right values

    MOCK:
        * mock cli._get_steganogan to return a mock of steganogan
    """
    pass


# @patch('steganogan.cli._get_steganogan')
# def test__encode_image(get_stega_mock):
#     """Given a real image, test that it encodes it with the basic.steg in fixtures."""
#     # setup
#     this_dir = os.path.dirname(__file__)
#     img = os.path.join(this_dir, 'fixtures/images/pythia.jpg')
#     output = os.path.join(this_dir, 'fixtures/images/out.png')
#     steg_path = os.path.join(this_dir, 'fixtures/pretrained/basic.steg')
# 
#     steganogan = SteganoGAN.load(steg_path, cuda=False, verbose=False)
# 
#     get_stega_mock.return_value = steganogan
# 
#     args = MagicMock()
#     type(args).cover = PropertyMock(return_value=img)
#     type(args).output = PropertyMock(return_value=output)
#     type(args).message = PropertyMock(return_value='Hello world')
# 
#     # run
#     cli._encode(args)
# 
#     # assert
# 
#     assert 'Hello world' == steganogan.decode(output)
