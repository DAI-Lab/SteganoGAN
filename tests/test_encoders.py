# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

import torch

from steganogan import encoders
from tests.utils import assert_called_with_tensors


class TestBasicEncoder(TestCase):

    class TestEncoder(encoders.BasicEncoder):
        def __init__(self):
            pass

    def setUp(self):
        self.test_encoder = self.TestEncoder()

    @patch('steganogan.encoders.nn.Conv2d', autospec=True)
    def test__covn2d(self, conv2d_mock):
        """Conv2d must be called with given args and kernel_size=3 and padding=1"""

        # run
        result = self.test_encoder._conv2d(2, 4)

        # asserts
        assert result == conv2d_mock.return_value
        conv2d_mock.assert_called_once_with(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            padding=1
        )

    @patch('steganogan.encoders.nn.Sequential')
    @patch('steganogan.encoders.nn.Conv2d')
    @patch('steganogan.encoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""
        # setup
        data_depth = 2
        hidden_size = 5

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=7, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=3, kernel_size=3, padding=1),
        ]

        expected_batch_calls = [call(5), call(5), call(5)]

        # run
        encoder = encoders.BasicEncoder(data_depth, hidden_size)

        # assert
        assert batchnorm_mock.call_args_list == expected_batch_calls
        assert conv2d_mock.call_args_list == expected_conv_calls
        assert encoder.add_image == False

    def test_forward_1_layer(self):
        """If there is only one layer it must be called with image as the only argument."""
        # setup
        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        self.test_encoder._models = [layer1]

        # run
        image = torch.Tensor([[1, 2], [3, 4]])
        result = self.test_encoder.forward(image, 'some_data')

        call_1 = call(torch.Tensor([[1, 2], [3, 4]]))

        # assert
        assert (result == torch.Tensor([[5, 6], [7, 8]])).all()
        assert_called_with_tensors(layer1, [call_1])

    def test_forward_more_than_2_layers(self):
        """If there are more than 2 layers, they must be called adding data to each result"""
        # setup
        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        layer2 = Mock(return_value=torch.Tensor([[9, 10], [11, 12]]))
        layer3 = Mock(return_value=torch.Tensor([[13, 14], [15, 16]]))
        self.test_encoder._models = [layer1, layer2, layer3]

        # run
        image = torch.Tensor([[1, 2], [3, 4]])
        data = torch.Tensor([[-1, -2], [-3, -4]])
        result = self.test_encoder.forward(image, data)

        # asserts
        call_layer_1 = call(torch.Tensor([[1, 2], [3, 4]]))
        call_layer_2 = call(torch.Tensor([[5, 6, -1, -2], [7, 8, -3, -4]]))
        call_layer_3 = call(torch.Tensor([[5, 6, 9, 10, -1, -2], [7, 8, 11, 12, -3, -4]]))

        assert_called_with_tensors(layer1, [call_layer_1])
        assert_called_with_tensors(layer2, [call_layer_2])
        assert_called_with_tensors(layer3, [call_layer_3])

        assert (result == torch.Tensor([[13, 14], [15, 16]])).all()

    def test_forward_add_image(self):
        """If add_image is true, image must be added to the result."""
        # setup
        self.test_encoder.add_image = True
        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        layer2 = Mock(return_value=torch.Tensor([[9, 10], [11, 12]]))
        layer3 = Mock(return_value=torch.Tensor([[13, 14], [15, 16]]))
        self.test_encoder._models = [layer1, layer2, layer3]

        # run
        image = torch.Tensor([[1, 2], [3, 4]])
        data = torch.Tensor([[-1, -2], [-3, -4]])
        result = self.test_encoder.forward(image, data)

        # asserts
        call_layer_1 = call(torch.Tensor([[1, 2], [3, 4]]))
        call_layer_2 = call(torch.Tensor([[5, 6, -1, -2], [7, 8, -3, -4]]))
        call_layer_3 = call(torch.Tensor([[5, 6, 9, 10, -1, -2], [7, 8, 11, 12, -3, -4]]))

        assert_called_with_tensors(layer1, [call_layer_1])
        assert_called_with_tensors(layer2, [call_layer_2])
        assert_called_with_tensors(layer3, [call_layer_3])

        assert (result == torch.Tensor([[14, 16], [18, 20]])).all()


class TestResidualEncoder(TestCase):

    @patch('steganogan.encoders.nn.Sequential')
    @patch('steganogan.encoders.nn.Conv2d')
    @patch('steganogan.encoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""
        # setup
        data_depth = 2
        hidden_size = 5

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=7, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=3, kernel_size=3, padding=1),
        ]

        expected_batch_calls = [call(5), call(5), call(5)]

        # run
        encoder = encoders.ResidualEncoder(data_depth, hidden_size)

        # assert
        assert batchnorm_mock.call_args_list == expected_batch_calls
        assert conv2d_mock.call_args_list == expected_conv_calls
        assert encoder.add_image == True


class TestDenseEncoder(TestCase):

    @patch('steganogan.encoders.nn.Sequential')
    @patch('steganogan.encoders.nn.Conv2d')
    @patch('steganogan.encoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""
        # setup
        data_depth = 2
        hidden_size = 5

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=7, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=12, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=17, out_channels=3, kernel_size=3, padding=1),
        ]

        expected_batch_calls = [call(5), call(5), call(5)]

        # run
        encoder = encoders.DenseEncoder(data_depth, hidden_size)

        # assert
        assert batchnorm_mock.call_args_list == expected_batch_calls
        assert conv2d_mock.call_args_list == expected_conv_calls
        assert encoder.add_image == True
