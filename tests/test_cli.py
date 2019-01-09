# -*- coding: utf-8 -*-

from unittest import TestCase


def test__get_steganogan():
    """
    VALIDATE:
        * The model path is the right one, with the right architecture
        * SteganoGAN.load is called with the right values
        * The output of SteganoGAN.load is returned

    MOCK:
        * SteganoGAN
        * args
    """
    pass


def test__encode():
    """
    VALIDATE:
        * steganogan.encode has been called with the right values

    MOCK:
        * cli._get_steganogan to return a mock of steganogan
    """
    pass


def test__decode():
    """
    VALIDATE:
        * steganogan.decode has been called with the right values

    MOCK:
        * mock cli._get_steganogan to return a mock of steganogan
    """
    pass
