# -*- coding: utf-8 -*-

import os
from unittest import TestCase
from unittest.mock import patch

import torch

from steganogan import utils


class TextEncrypting(TestCase):

    @classmethod
    def setUpClass(cls):
        """Read the fixtures used in this test case"""

        test_path = os.path.dirname(__file__)
        test_path_bits = os.path.join(test_path, 'fixtures/hi_to_bits.txt')
        test_path_bytearray = os.path.join(test_path, 'fixtures/bytearray_hi.txt')

        f = open(test_path_bits, 'r')  # this holds the word hi
        cls.bits_hi = f.readlines()
        f.close()
        cls.bits_hi = [int(x) for x in cls.bits_hi[0]]

        f = open(test_path_bytearray, 'rb')
        cls.bytearray_hi = bytearray(f.read())
        f.close()

    def test_text_to_bits(self):
        """Test that our utility converts text into bits."""

        # run
        result = utils.text_to_bits('hi')

        # assert
        assert self.bits_hi == result

    def test_bits_to_text(self):
        """Test that our utility converts bits to text"""

        # run
        result = utils.bits_to_text(self.bits_hi)

        # assert
        assert result == 'hi'

    def test_text_to_bytearray(self):
        """Test text_to_bytearray return's value"""

        # run
        result = utils.text_to_bytearray('hi')

        # assert
        assert (result == self.bytearray_hi)

    def test_bytearray_to_text(self):
        """Test bytearray_to_text method's return value."""

        # run
        result = utils.bytearray_to_text(self.bytearray_hi)

        # assert
        assert (result == 'hi')

    def test_bits_to_bytearray(self):
        """Test bits_to_bytearray if the utility can turn bytearray into bits"""

        # run
        result = utils.bits_to_bytearray(self.bits_hi)

        # assert
        assert result == self.bytearray_hi


def test_gaussian():
    """Test Gaussian method"""

    # run
    result1 = utils.gaussian(2, 0.2)
    result2 = utils.gaussian(3, 0.3)
    result3 = utils.gaussian(4, 0.4)

    # assert
    expected1 = torch.Tensor([3.7266393064783188e-06, 0.9999963045120239])
    expected2 = torch.Tensor([0.0038362585473805666, 0.9923273921012878, 0.0038362585473805666])
    expected3 = torch.Tensor([3.425617933316971e-06, 0.040387753397226334,
                              0.919221043586731, 0.040387753397226334])

    assert (result1 == expected1).all()
    assert (result2 == expected2).all()
    assert (result3 == expected3).all()


def test_first_element():
    """Test for first_element method, it has to return the first element that we pass"""

    # run
    result_1 = utils.first_element(4, 10)
    result_2 = utils.first_element(10, 4)
    result_3 = utils.first_element('hi', 'bye')

    # assert
    assert (result_1 == 4)
    assert (result_2 == 10)
    assert (result_3 == 'hi')


def test_create_window():
    """Tests the return value of create_window utility method"""

    # run
    result_1 = utils.create_window(1, 1)
    result_2 = utils.create_window(2, 1)

    # assert
    expected_1 = torch.Tensor([[[[1.]]]])
    expected_2 = torch.Tensor([[[[0.19773312, 0.24693881], [0.24693881, 0.30838928]]]])

    assert (result_1 == expected_1).all()
    assert (result_2 == expected_2).all()


def test__ssim_size_average_true():
    """
    METHOD:
        _ssim(img1, img2, window, windows_size, channel, size_average=True)

    TESTS:
        test__ssim_size_average_true
        test_ssim_size_average_false

    VALIDATE:
        return value

    """

    # setup
    window = utils.create_window(11, 1)
    img1 = torch.Tensor([[[[0.19773312, 0.24693881], [0.24693881, 0.30838928]]]])

    # run
    result = utils._ssim(img1, img1, window, 11, 1, size_average=True)

    # assert
    assert (result == torch.Tensor([1.])).all()


def test__ssim_size_average_false():
    """
    METHOD:
        _ssim(img1, img2, window, windows_size, channel, size_average=True)

    TESTS:
        test__ssim_size_average_true
        test_ssim_size_average_false

    VALIDATE:
        return value

    """

    # setup
    window = utils.create_window(3, 1)
    img1 = torch.Tensor([[[[4.19773312, 1.24693881], [1.24693881, 2.30838928]]]])

    # run
    result = utils._ssim(img1, img1, window, 3, 1, size_average=False)

    # assert
    assert (result == torch.Tensor([1.])).all()


@patch('steganogan.utils._ssim')
def test_ssim_size_average_true(mock__ssim):
    """
    METHOD:
        ssim(img1, img2, window_size=11, size_average=True)

    TESTS:
        test_ssim_size_average_true
        test_ssim_size_average_false

    VALIDATE:
        return value

    MOCK:
        create_window
        _ssim
    """

    # run
    utils.ssim(
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        window_size=1,
        size_average=True
    )

    # assert
    mock__ssim.assert_called_once_with(
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        1,
        1,
        True
    )


@patch('steganogan.utils._ssim')
def test_ssim_size_average_false(mock__ssim):

    # run
    utils.ssim(
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        window_size=1,
        size_average=False
    )

    # assert
    mock__ssim.assert_called_once_with(
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        torch.Tensor([[[[1.]]]]),
        1,
        1,
        False
    )
