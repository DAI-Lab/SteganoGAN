# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

import torch

from steganogan import decoders
from tests.utils import assert_called_with_tensors


class TestBasicDecoder(TestCase):
    """
    METHOD:
        __init__(self, data_depth, hidden_size)

    VALIDATE:
        * args
        * layers

    TODO:
        * change code to be like encoders.py in order to be able to use inheritance.
        * from test.utils import assert_called_with_tensors (needed to validate layers).
        * mock decoders.torch.nn.Conv2d in order to assert called with our params.
    """

    """
    METHOD:
        forward(self, x)

    VALIDATE:
        * return value

    TODO:
        * calculate the expected output with the layers.
        * assert the return value and the expected are the same.

    """
    pass


class TestDenseDecoder(TestCase):

    class TestDecoder(decoders.DenseDecoder):
        def __init__(self):
            pass


    @patch('steganogan.decoders.nn.Sequential')
    @patch('steganogan.decoders.nn.Conv2d')
    @patch('steganogan.decoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""
        # setup
        data_depth = 2
        hidden_size = 5

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=10, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=15, out_channels=2, kernel_size=3, padding=1),
        ]

        expected_batch_calls = [call(5), call(5), call(5)]

        # run

        decoder = decoders.DenseDecoder(data_depth, hidden_size)

        # assert
        assert batchnorm_mock.call_args_list == expected_batch_calls
        assert conv2d_mock.call_args_list == expected_conv_calls


    """
    METHOD:
        forward(self, x)

    VALIDATE:
        * return value

    TODO:
        * calculate the expected output with the layers.
        * assert the return value and the expected are the same.
    """
    pass
