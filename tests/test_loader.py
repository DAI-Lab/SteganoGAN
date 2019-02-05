# -*- coding: utf-8 -*-
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from steganogan import loader


class TestDataLoader(TestCase):

    @patch('steganogan.loader.torch.utils.data.DataLoader.__init__')
    @patch('steganogan.loader.ImageFolder')
    def test___init__with_transform(self, image_folder_mock, dataloader_init_mock):
        """Test that DataLoader it's created with a transform that we specify"""

        # setup
        image_folder_mock.return_value = None
        dataloader_init_mock.return_value = None

        # run
        loader.DataLoader('test_path', transform='test_transform', limit=1)

        # assert
        image_folder_mock.assert_called_once_with('test_path', 'test_transform', 1)
        dataloader_init_mock.assert_called_once_with(
            None,
            shuffle=True,
            num_workers=8,
            batch_size=4
        )

    @patch('steganogan.loader.torch.utils.data.DataLoader.__init__')
    @patch('steganogan.loader.ImageFolder')
    def test___init__without_transform(self, image_folder_mock, dataloader_init_mock):
        """Test that DataLoader it's created with a default transform"""

        # setup
        image_folder_mock.return_value = None
        dataloader_init_mock.return_value = None

        # run
        loader.DataLoader('test_path', limit=1)

        # assert
        image_folder_mock.assert_called_once_with('test_path', loader.DEFAULT_TRANSFORM, 1)
        dataloader_init_mock.assert_called_once_with(
            None,
            shuffle=True,
            num_workers=8,
            batch_size=4
        )


class TestImageFolder(TestCase):

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    def test___init__with_limit(self, img_folder_mock):
        """Test that ImageFolder it's created with the limit that we specify"""

        # run
        test_image_folder = loader.ImageFolder('test_path', loader.DEFAULT_TRANSFORM, limit=1)

        # assert
        img_folder_mock.assert_called_once_with('test_path', transform=loader.DEFAULT_TRANSFORM)
        assert test_image_folder.limit == 1

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    def test___init__without_limit(self, img_folder_mock):
        """Test that the ImageFolder it's created with np.inf as limit"""

        # run
        test_image_folder = loader.ImageFolder('test_path', loader.DEFAULT_TRANSFORM)

        # assert
        img_folder_mock.assert_called_once_with('test_path', transform=loader.DEFAULT_TRANSFORM)
        assert test_image_folder.limit == np.inf

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__len__')
    def test___len__length_gt_limit(self, img_fold_len_mock, img_folder_init_mock):
        """Test that limit it's being returned when length it's greater than limit"""

        # setup
        img_fold_len_mock.return_value = 10

        # run
        test_image_folder = loader.ImageFolder('test_path', loader.DEFAULT_TRANSFORM, limit=5)

        # assert
        assert test_image_folder.__len__() == 5

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__len__')
    def test___len__length_lt_limit(self, img_fold_len_mock, img_folder_init_mock):
        """Test that lenght its being returned when it's lower than limit"""

        # setup
        img_fold_len_mock.return_value = 10

        # run
        test_image_folder = loader.ImageFolder('test_path', loader.DEFAULT_TRANSFORM, limit=15)

        # assert
        assert test_image_folder.__len__() < 15
