# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

import os

import torch

from steganogan import encoders, critics, decoders, models


class TestSteganoGAN(TestCase):

    class VoidSteganoGAN(models.SteganoGAN):
        """Can be used to create a steganogan class with empty attributes"""
        def __init__(self):
            pass
    """
    METHOD:
        __init__(self, data_depth, encoder, decoder, critic,
                 cuda=True, verbose, log_dir, **kwargs)

    VALIDATE:
        * attributes

    TESTS:
        test__init__whithout_logdir
        test__init__with_logdir

    MOCK:
        os.makedirs
        steganogan.models.set_device
    """
    def test___init__without_logdir(self):

        # setup
        cuda_patch.return_value = True
        data_depth = 1
        encoder = encoders.BasicEncoder
        decoder = decoders.BasicDecoder
        critic = critics.BasicCritic
        cuda = True
        verbose = True
        log_dir = None
        hidden_size = 2

        # run
        steganogan = models.SteganoGAN(data_depth, encoder, decoder, critic, cuda=cuda,
                                       verbose=verbose, log_dir=log_dir, hidden_size=hidden_size)

        # assert
        expected_data_depth = data_depth
        expected_encoder = encoder
        expected_decoder = decoder
        expected_critic = critic

        expected_device = torch.device('cuda')
        expected_critic_optimizer = None
        expected_decoder_optimizer = None
        expected_fit_metrics = None

        expected_history = list()
        expected_log_dir = log_dir

        expected_to_call = [call(expected_device), call(expected_device), call(expected_device)]


        assert expected_data_depth == steganogan.data_depth
        assert expected_encoder == type(steganogan.encoder)
        assert expected_decoder == type(steganogan.decoder)
        assert expected_critic == type(steganogan.critic)
        assert expected_device == steganogan.device
        assert expected_critic_optimizer == steganogan.critic_optimizer
        assert expected_decoder_optimizer == steganogan.decoder_optimizer
        assert expected_fit_metrics == steganogan.fit_metrics
        assert expected_history == steganogan.history
        assert expected_log_dir == steganogan.log_dir
        assert expected_to_call == steganogan.critic.to.call_args_list
        assert expected_to_call == steganogan.decoder.to.call_args_list
        assert expected_to_call == steganogan.encoder.to.call_args_list

    def test___init__log_dir(self):
        pass


    """
    METHOD:
        _get_instance(self, class_or_instance, **kwargs)

    TESTS:
        test__get_instance_is_class:
            if the given object is not a class it is returned unmodified

        test__get_instance_is_not_class:
            if the given object is a class an instance of the class is returned
    """

    """
    METHOD:
        set_device(self, cuda=True)

    TESTS:
        test_set_device_cuda:
            if cuda = True

        test_set_device_cpu:
            if cuda = False

    MOCK:
        torch.cuda.is_available
        self.encoder
        self.decoder
        self.critic
    """

    """
    METHOD:
        _random_data(self, cover)

    TESTS:
        test__random_data:
            torch.zeros it's called with N, H, W for the cover and the device

    MOCK:
        cover.size() to return N, _, H, W
        torch.zeros

    """

    """
    METHOD:
        _encode_decode(self, cover, quantize=False)

    TESTS:
        test__encode_decode_quantize:
            test the return value when quantize it's True

        test__encode_decode_default:
            test the return value when quantize it's False

    MOCK:
        _random_data, self.encoder

    """

    """
    METHOD:
        _critic(self, image)

    TESTS:
        test__critic:
            validate return value

    MOCK:
        image
        torch.mean
    """

    """
    METHOD:
        _get_optimizers(self)

    TESTS:
        test__get_optimizers:
            validate return values calculated with the list of the decoder / encoder and Adam

    MOCK:
        self.decoder.parameters
        self.encoder.parameters
        Adam
    """

    """
    METHOD:
        _fit_critic(self, train, metrics)

    TESTS:
        test__fit_critic:
            validate that it's calling the rest of the mocks with a given data.

    MOCK:
        train
        metrics
        gc.collect
        _random_data
        self.encoder
        self._critic
    """

    """
    METHOD:
        _fit_coders(self, train, metrics)

    TESTS:
        test__fit_coders:
            validate that it's calling the rest of the mocks with a given data.

    MOCK:
        train
        metrics
        gc.collect
        _encode_decode()
        _critic
    """

    """
    METHOD:
        _coding_scores(self, cover, generated, payload, decoded)

    TESTS:
        test__coding_scores:
            validate return values calculated in this method

    MOCK:
        mse_loss
        binary_cross_entropy_with_logits
        payload.numel()
    """

    """
    METHOD:
        _validate(self, validate, metrics)

    TESTS:
        test__validate:
            validate that metrics it's being updated

    MOCK:
        _critic
        _encode_decode
        _coding_scores
        ssim
    """

    """
    METHOD:
        _generate_samples(self, samples_path, cover, epoch)

    TESTS:
        test__generate_samples:
            validate that the method generates samples of images in the given path

    MOCK:
        imageio.imwrite
    """

    """
    METHOD:
        fit(self, train, validate, epochs=5)

    TESTS:
        test_fit_with_logdir:
            test that this method invokes the other methods for fitting.

        test_fit_without_logdir:
            test that we are generating samples

        test_fit_cuda_true:
            validate that we call the torch.cuda.empty_cache

    MOCK:
        train
        validate
        _fit_critic
        _fit_coders
        _validate
        save
        _generate_samples
        torch.cuda.empty_cache

    TODO:
        if self.log_dir can be moved to a method that has that functionality
    """

    """
    METHOD:
        _make_payload(self, width, height, depth, text)

    TESTS:
        test__make_payload:
            validate return value of this method.

    MOCK:
        text_to_bits
        torch.FloatTensor
    """

    """
    METHOD:
        encode(self, cover, output, text)

    TESTS:
        test_encode:
            validate that the image has been encoded

    MOCK:
        encoder
        imread
        _make_payload
        cover
        payload
        imwrite
    """

    """
    METHOD:
        decode(self, cover)

    TESTS:
        test_decode:
            validate that the image has been decoded

    MOCK:
        os.path.exists
        imread
        image
        decoder
        bits_to_bytearray_to_text
    """

    """
    METHOD:
        save(self, path)

    TESTS:
        test_save:
            validate that we are saving the fitted model

    MOCK:
        torch.save
    """

    """
    METHOD:
        load(cls, path, cuda=True, verbose=False)

    TESTS:
        test_load_cuda_true
        test_load_cuda_false
        test_load_verbose_false
        test_load_verbose_true

    MOCK:
        torch.load
    """


    pass
