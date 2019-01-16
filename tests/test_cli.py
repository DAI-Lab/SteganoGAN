# -*- coding: utf-8 -*-

import os
from unittest.mock import MagicMock, patch

from steganogan import cli


@patch('steganogan.cli.SteganoGAN.load')
def test__get_steganogan(mock_steganogan_load):
    """
    Test that:
        * The model path is the right one, with the right architecture
        * SteganoGAN.load is called with the right values
        * The output of SteganoGAN.load is returned
    """

    # setup
    mock_steganogan_load.return_value = 'Steganogan'

    params = MagicMock(
        cpu=True,
        architecture='basic',
        verbose=True,
    )

    model_name = '{}.steg'.format(params.architecture)
    parent_path = os.path.dirname(os.path.dirname(__file__))
    stega_path = os.path.join(parent_path, 'steganogan')
    pretrained_path = os.path.join(stega_path, 'pretrained')
    model_path = os.path.join(pretrained_path, model_name)

    # run
    cli_test = cli._get_steganogan(params)

    # assert
    mock_steganogan_load.assert_called_once_with(
        model_path,
        cuda=False,
        verbose=True
    )

    assert cli_test == 'Steganogan'


@patch('steganogan.cli._get_steganogan')
def test__encode(mock__get_steganogan):
    """Test that encode has been called with the args that the cli recived."""

    # setup
    steganogan = MagicMock()
    mock__get_steganogan.return_value = steganogan

    params = MagicMock(
        cpu=True,
        architecture='basic',
        verbose=True,
        cover='image.jpg',
        output='output.png',
        message='Hello world'
    )

    # run
    cli._encode(params)

    # assert
    mock__get_steganogan.assert_called_once_with(params)
    steganogan.encode.assert_called_once_with('image.jpg', 'output.png', 'Hello world')


@patch('steganogan.cli._get_steganogan')
def test__decode(mock__get_steganogan):
    """
    VALIDATE:
        * steganogan.decode has been called with the right values

    MOCK:
        * mock cli._get_steganogan to return a mock of steganogan
    """
    # setup
    steganogan = MagicMock()
    steganogan.decode.return_value = 'Hello world'

    mock__get_steganogan.return_value = steganogan

    params = MagicMock(
        cpu=True,
        architecture='basic',
        verbose=True,
        image='test_image.png'
    )

    # run
    cli._decode(params)

    # assert
    mock__get_steganogan.assert_called_once_with(params)
    steganogan.decode.assert_called_once_with('test_image.png')
