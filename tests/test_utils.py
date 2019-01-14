# -*- coding: utf-8 -*-

import zlib

import torch
from reedsolo import RSCodec

from steganogan import utils

def test_text_to_bits():
    """
    METHOD:
        text_to_bits(text)

    TESTS:
        test_text_to_bits

    VALIDATE:
        return value

    MOCK:
        bytearray_to_bits
        text_to_bytearray
    """
    pass


def test_bits_to_text():
    """
    METHOD:
        bits_to_text(bits)

    TESTS:
        test_bits_to_text

    VALIDATE:
        return value

    MOCK:
        bytearray_to_text
        bits_to_bytearray
    """
    pass


def test_gaussian():
    """Test Gaussian method"""
    # setup
    expected1 = torch.Tensor([3.7266393064783188e-06, 0.9999963045120239])
    expected2 = torch.Tensor([0.0038362585473805666, 0.9923273921012878, 0.0038362585473805666])
    expected3 = torch.Tensor([3.425617933316971e-06, 0.040387753397226334,
                              0.919221043586731, 0.040387753397226334])

    # run
    result1 = utils.gaussian(2, 0.2)
    result2 = utils.gaussian(3, 0.3)
    result3 = utils.gaussian(4, 0.4)

    # assert
    assert (result1 == expected1).all()
    assert (result2 == expected2).all()
    assert (result3 == expected3).all()


def test_text_to_bytearray():
    """Test text_To_bytearray.
    We want to test if th encoder it's returning the same value for RSCodec(250)

    METHOD:
        text_to_bytearray(text)

    TEST:
        test_text_to_bytearray

    VALIDATE:
        return value

    TODO:
        remove assert from that method

    FIXTURE:
        expected_result calculated by zlib.compress and rs.encode

    """
    # setup
    rs = RSCodec(250)
    expected = rs.encode(zlib.compress('Hello world'.encode('utf-8')))

    # run
    result = utils.text_to_bytearray('Hello world')

    # assert
    assert (result == expected)


def test_bytearray_to_text():
    """Test bytearray_to_text method.
    We want to test if we can decode with RSCodec(250) and decompress with zlib a text.

    METHOD:
        bytearray_to_text(x)

    TEST:
        test_bytearray_to_text

    VALIDATE:
        return value

    FIXTURE:
        x

    """
    # setup
    rs = RSCodec(250)
    text = rs.encode(zlib.compress('Hello world'.encode('utf-8')))

    expected = zlib.decompress(rs.decode(text)).decode('utf-8')

    # run
    result = utils.bytearray_to_text(text)

    # assert
    assert (result == expected)


def test_bits_to_bytearray():
    """Test bits_to_bytearray.
    We want to test if we can turn bytearray into bits

    METHOD:
        bits_to_bytearray(bits)

    TESTS:
        test_bits_to_bytearray

    FIXTURES:
        bits_value
    """
    pass


def test_first_element():
    """Test for first_element method, it has to return the first element that we pass"""
    # setup
    my_first_element = 1
    my_second_element = 2

    # run
    result = utils.first_element(my_first_element, my_second_element)

    # assert
    expected = my_first_element
    assert (result == expected)


def test_create_window():
    """Tests to create a 2d window with 1.5 sigma for the gaussian method
    Args:
        windows_size = 4
        channel = 2

    METHOD:
        create_window(window_size, channel)

    TEST:
        test_create_window

    VALIDATE:
        return value

    MOCK:
        gaussain??
    """
    # setup
    expected_window = utils.gaussian(4, 1.5).unsqueeze(1)
    expected_window = expected_window.mm(expected_window.t()).float().unsqueeze(0).unsqueeze(0)
    expected_window = expected_window.expand(2, 1, 4, 4).contiguous()

    # run
    result = utils.create_window(4, 2)

    # assert
    assert (result == expected_window).all()


def _ssim():
    """
    METHOD:
        _ssim(img1, img2, window, windows_size, channel, size_average=True)

    TESTS:
        test__ssim_size_average_true
        test_ssim_size_average_false

    VALIDATE:
        return value

    MOCK:
        conv2d
    """
    pass


def ssim():
    """
    METHOD:
        ssim(img1, img2, window_size=11, size_average=True)

    TESTS:
        test_ssim_window_size_eq_11
        test_ssim_window_size_gt_11
        test_ssim_window_size_lt_11
        test_ssim_size_average_true
        test_ssim_size_average_false

    VALIDATE:
        return value

    MOCK:
        create_window
        _ssim
    """
    pass
