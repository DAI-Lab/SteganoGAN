# -*- coding: utf-8 -*-

import copy
from unittest import TestCase
from unittest.mock import Mock, call, patch

import torch

from steganogan import decoders
from tests.utils import assert_called_with_tensors


class TestBasicDecoder(TestCase):

    class TestDecoder(decoders.BasicDecoder):
        def __init__(self):
            pass

    def setUp(self):
        self.test_decoder = self.TestDecoder()

    @patch('steganogan.decoders.nn.Conv2d', autospec=True)
    def test__covn2d(self, conv2d_mock):
        """Conv2d must be called with given args and kernel_size=3 and padding=1"""

        # run
        result = self.test_decoder._conv2d(2, 4)

        # asserts
        assert result == conv2d_mock.return_value
        conv2d_mock.assert_called_once_with(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            padding=1
        )

    @patch('steganogan.decoders.nn.Sequential')
    @patch('steganogan.decoders.nn.Conv2d')
    @patch('steganogan.decoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""

        # run
        decoders.BasicDecoder(2, 5)

        # assert
        expected_batch_calls = [call(5), call(5), call(5)]
        assert batchnorm_mock.call_args_list == expected_batch_calls

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=2, kernel_size=3, padding=1),
        ]
        assert conv2d_mock.call_args_list == expected_conv_calls

    def test_upgrade_legacy_without_version(self):
        """Upgrade legacy must create self._models from conv1, conv2, conv3, conv4"""

        # setup
        self.test_decoder.layers = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))

        # run
        self.test_decoder.upgrade_legacy()

        # assert
        assert self.test_decoder._models == [self.test_decoder.layers]
        assert self.test_decoder.version == '1'

    @patch('steganogan.decoders.nn.Sequential', autospec=True)
    def test_upgrade_legacy_with_version_1(self, sequential_mock):
        """The object must be the same and not changed by the method"""

        # setup
        decoder = decoders.BasicDecoder(1, 1)
        expected = copy.deepcopy(decoder)

        # run
        decoder.upgrade_legacy()

        # assert
        assert decoder.__dict__ == expected.__dict__

    def test_forward_1_layer(self):
        """If there is only one layer it must be called with image as the only argument."""

        # setup
        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        self.test_decoder._models = [layer1]

        # run
        image = torch.Tensor([[1, 2], [3, 4]])
        result = self.test_decoder.forward(image)

        # assert
        assert (result == torch.Tensor([[5, 6], [7, 8]])).all()

        call_1 = call(torch.Tensor([[1, 2], [3, 4]]))
        assert_called_with_tensors(layer1, [call_1])

    def test_forward_more_than_2_layers(self):
        """If there are more than 2 layers, they must be called adding data to each result"""

        # setup
        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        layer2 = Mock(return_value=torch.Tensor([[9, 10], [11, 12]]))
        layer3 = Mock(return_value=torch.Tensor([[13, 14], [15, 16]]))
        self.test_decoder._models = [layer1, layer2, layer3]

        # run
        image = torch.Tensor([[1, 2], [3, 4]])
        result = self.test_decoder.forward(image)

        # asserts
        call_layer_1 = call(torch.Tensor([[1, 2], [3, 4]]))
        call_layer_2 = call(torch.Tensor([[5, 6], [7, 8]]))
        call_layer_3 = call(torch.Tensor([[5, 6, 9, 10], [7, 8, 11, 12]]))

        assert_called_with_tensors(layer1, [call_layer_1])
        assert_called_with_tensors(layer2, [call_layer_2])
        assert_called_with_tensors(layer3, [call_layer_3])

        assert (result == torch.Tensor([[13, 14], [15, 16]])).all()


class TestDenseDecoder(TestCase):

    class TestDecoder(decoders.DenseDecoder):
        def __init__(self):
            pass

    def test_upgrade_legacy_without_version(self):
        """Upgrade legacy must create self._models from conv1, conv2, conv3, conv4"""

        # setup
        test_decoder = self.TestDecoder()  # instance an empty decoder
        test_decoder.conv1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        test_decoder.conv2 = Mock(return_value=torch.Tensor([[9, 10], [11, 12]]))
        test_decoder.conv3 = Mock(return_value=torch.Tensor([[13, 14], [15, 16]]))
        test_decoder.conv4 = Mock(return_value=torch.Tensor([[17, 18], [19, 20]]))

        # run
        test_decoder.upgrade_legacy()

        # assert
        expected_models = [
            test_decoder.conv1,
            test_decoder.conv2,
            test_decoder.conv3,
            test_decoder.conv4,
        ]
        assert test_decoder._models == expected_models
        assert test_decoder.version == '1'

    @patch('steganogan.decoders.nn.Sequential', autospec=True)
    def test_upgrade_legacy_with_version_1(self, sequential_mock):
        """The object must be the same and not changed by the method"""

        # setup
        decoder = decoders.DenseDecoder(1, 1)
        expected = copy.deepcopy(decoder)

        # run
        decoder.upgrade_legacy()

        # assert
        assert decoder.__dict__ == expected.__dict__

    @patch('steganogan.decoders.nn.Sequential')
    @patch('steganogan.decoders.nn.Conv2d')
    @patch('steganogan.decoders.nn.BatchNorm2d')
    def test___init__(self, batchnorm_mock, conv2d_mock, sequential_mock):
        """Test the init params and that the layers are created correctly"""

        # run
        decoders.DenseDecoder(2, 5)

        # assert
        expected_batch_calls = [call(5), call(5), call(5)]
        assert batchnorm_mock.call_args_list == expected_batch_calls

        expected_conv_calls = [
            call(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=5, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=10, out_channels=5, kernel_size=3, padding=1),
            call(in_channels=15, out_channels=2, kernel_size=3, padding=1),
        ]
        assert conv2d_mock.call_args_list == expected_conv_calls
