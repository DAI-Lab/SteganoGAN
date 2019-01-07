# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

import torch

from steganogan import critics
from tests.utils import assert_called_with_tensors


class TestBasicCritic(TestCase):

    class TestCritic(critics.BasicCritic):
        def __init__(self):
            pass

    @patch('steganogan.critics.nn.Sequential')
    @patch('steganogan.critics.nn.BatchNorm2d')
    @patch('steganogan.critics.nn.Conv2d')
    def test___init__(self, conv2d_mock, batchnorm2d_mock, sequential_mock):
        """Test that conv2d and batchnorm are called when creating a new critic with hidden_size"""

        # setup
        hidden_size = 2

        expected_conv2d_calls = [
            call(in_channels=3, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=1, kernel_size=3)
        ]

        expected_batch_calls = [call(2), call(2), call(2)]

        # run
        crtici = critics.BasicCritic(hidden_size)

        # assert
        assert conv2d_mock.call_args_list == expected_conv2d_calls
        assert batchnorm2d_mock.call_args_list == expected_batch_calls

    def test_forward(self):
        """Test the return value of method forward"""
        # setup
        test_critic = self.TestCritic()

        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        test_critic.layers = layer1

        image = torch.Tensor([[1, 2], [3, 4]])

        call_1 = call(torch.Tensor([[1, 2], [3, 4]]))

        expected = torch.Tensor([[5, 6], [7, 8]])

        expected = torch.mean(expected.view(expected.size(0), -1), dim=1)

        # run
        result = test_critic.forward(image)

        # assert
        assert (result == expected).all()
        assert_called_with_tensors(layer1, [call_1])
