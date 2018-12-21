#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for steganogan package."""
import os
import unittest

from steganogan.loader import DataLoader

from steganogan.architectures.encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from steganogan.critics import BasicCritic
from steganogan.decoders import BasicDecoder

from steganogan.models import SteganoGAN

IN_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures/flag.jpg')
OUT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures/out.jpg')
ENCODED_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures/tested.jpg')


class TestSteganoGAN(unittest.TestCase):

    def setUpClass(self):
        train = DataLoader('features/div2k/train')
        val = DataLoader('features/div2k/val')
        self.model = SteganoGAN(1, BasicEncoder, BasicDecoder, BasicCritic, hidden_size=1)

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def tearDownClass(self):
        pass

    def test_000_something(self):
        """Test something."""
        self.assertTrue(True)

    def test_steganogan_encode(self):
        """Test encoding of an image."""
        self.model.encode(IN_IMAGE, OUT_IMAGE, 'Hello world')
        self.assertTrue(os.path.isfile(OUT_IMAGE))  # Check if this file exists.
        # Remove the file after it was created in case we run more tests.
        if os.path.isfile(OUT_IMAGE):
            os.remove(OUT_IMAGE)

    def test_steganogan_decode(self):
        """Test decoding of an image."""
        # TODO: place an image, named tested.jpg, that has 'Hello World' encoded
        # message = self.model.decode(ENCODED_IMAGE)
        # self.assertEquals(message, 'Hello World')
        pass

    def test_steganogan_fit(self):
        """Test training a model."""
        dummy_dict = {
            'val.encoder_mse': 0.018459019213914872,
            'val.decoder_loss': 0.4292408537864685,
            'val.decoder_acc': 0.7891681122779847,
            'val.cover_score': 0.02765399806201458,
            'val.stega_score': 0.07727440774440765,
            'val.ssim': 0.6747004044055939,
            'val.psnr': 24.24250383377075,
            'val.bpp': 1.7350086736679078,
            'train.encoder_mse': 0.020988403398077934,
            'train.decoder_loss': 0.43256636276841165,
            'train.decoder_acc': 0.7871638134121894,
            'train.cover_score': 0.0282248446252197,
            'train.stega_score': 0.08084110155701638,
            'epoch': 24
        }

        self.model.fit(train, val, epochs=1)

        for key, val in self.model.train_metrics:
            self.assertTrue(key in dummy_dict)
            self.assertEquals(type(val), float)

