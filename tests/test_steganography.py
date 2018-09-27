#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepsteganography.utils` package."""

import os
import unittest

from deepsteganography import Steganographer

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")


class TestUtils(unittest.TestCase):
    """Tests for `Steganographer` class."""

    def tearDown(self):
        os.remove('tests/resources/output.png')

    def test_end_to_end(self):
        """
        Test end-to-end encoding and decoding for a single image/message pair.
        """
        model = Steganographer()

        message = "Hello! This is a relatively short message."
        image = os.path.join(os.path.join(RESOURCE_DIR, "flag.jpg"))
        output = os.path.join(os.path.join(RESOURCE_DIR, "output.png"))

        model.encode(image, message, output)
        self.assertEqual(model.decode(output), message)
