# -*- coding: utf-8 -*-
import numpy as np

from unittest import TestCase
from unittest.mock import Mock, call, patch

from steganogan import loader


class TestDataLoader(TestCase):

    @patch('steganogan.loader.torch.utils.data.DataLoader.__init__')
    @patch('steganogan.loader.ImageFolder')
    def test___init__with_transform(self, image_folder_mock, dataloader_init_mock):
        """Test that DataLoader it's created with a transform that we specify"""
        # setup
        image_folder_mock.return_value = None
        dataloader_init_mock.return_value = None
        path = 'test_path'
        limit = 1
        num_workers = 8
        batch_size = 4
        transform = 'test_transformation'

        # run
        data = loader.DataLoader(path, transform=transform, limit=limit)

        # assert
        expected_dataloader_call = [call(None, shuffle=True, num_workers=8, batch_size=4)]
        expected_image_folder_call = [call(path, transform, limit)]

        assert expected_image_folder_call == image_folder_mock.call_args_list
        assert expected_dataloader_call == dataloader_init_mock.call_args_list

    @patch('steganogan.loader.torch.utils.data.DataLoader.__init__')
    @patch('steganogan.loader.ImageFolder')
    def test___init__without_transform(self, image_folder_mock, dataloader_init_mock):
        """Test that DataLoader it's created with a default transform"""
        # setup
        image_folder_mock.return_value = None
        dataloader_init_mock.return_value = None
        path = 'test_path'
        limit = 1
        num_workers = 8
        batch_size = 4
        transform = loader.DEFAULT_TRANSFORM

        # run
        data = loader.DataLoader(path, limit=limit)

        # assert
        expected_dataloader_call = [call(None, shuffle=True, num_workers=8, batch_size=4)]
        expected_image_folder_call = [call(path, transform, limit)]

        assert expected_image_folder_call == image_folder_mock.call_args_list
        assert expected_dataloader_call == dataloader_init_mock.call_args_list


class TestImageFolder(TestCase):
    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    def test___init__with_limit(self, img_folder_mock):
        """Test that ImageFolder it's created with the limit that we specify"""
        # setup
        limit = 1
        path = 'test_path'
        transform = loader.DEFAULT_TRANSFORM

        # run
        test_image_folder = loader.ImageFolder(path, transform, limit=limit)

        # assert
        expected_call = [call(path, transform=transform)]
        assert expected_call == img_folder_mock.call_args_list
        assert limit == test_image_folder.limit

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    def test___init__without_limit(self, img_folder_mock):
        """Test that the ImageFolder it's created with np.inf as limit"""
        # setup
        path = 'test_path'
        limit = np.inf
        transform = loader.DEFAULT_TRANSFORM

        # run
        test_image_folder = loader.ImageFolder(path, transform)

        # assert
        expected_call = [call(path, transform=transform)]
        assert expected_call == img_folder_mock.call_args_list
        assert limit == test_image_folder.limit

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__len__')
    def test___len__length_gt_limit(self, img_fold_len_mock, img_folder_init_mock):
        """Test that limit it's being returned when length it's greater than limit"""
        # setup
        img_fold_len_mock.return_value = 10
        limit = 5
        path = 'test_path'
        transform = loader.DEFAULT_TRANSFORM

        # run
        test_image_folder = loader.ImageFolder(path, transform, limit=limit)

        # assert
        assert test_image_folder.__len__() == limit

    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__init__')
    @patch('steganogan.loader.torchvision.datasets.ImageFolder.__len__')
    def test___len__length_lt_limit(self, img_fold_len_mock, img_folder_init_mock):
        """Test that lenght its being returned when it's lower than limit"""
        # setup
        img_fold_len_mock.return_value = 10
        limit = 15
        path = 'test_path'
        transform = loader.DEFAULT_TRANSFORM

        # run
        test_image_folder = loader.ImageFolder(path, transform, limit=limit)


        # assert
        assert limit > test_image_folder.__len__()
