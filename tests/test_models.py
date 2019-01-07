# -*- coding: utf-8 -*-

from unittest import TestCase


class TestSteganoGAN(TestCase):
    """
    METHOD:
        __init__(self, data_depth, encoder, decoder, critic,
                 cuda=True, verbose, log_dir, **kwargs)

    VALIDATE:
        * attributes

    TODO:
    """

    """
    METHOD:
        _get_instance(self, class_or_instance, **kwargs)

    TESTS:
        test__get_instance_is_class:
            if the given object is not a class it is returned unmodified

        test__get_instance_is_not_class:
            if the given object is a class an instance of the class is returned

    TODO:
    """

    """
    METHOD:
        set_device(self, cuda=True)

    TESTS:
        test_set_device_cuda:
            if cuda = True

        test_set_device_cpu:
            if cuda = False
    TODO:
    """

    """
    METHOD:
        _random_data(self, cover)

    TESTS:
        test__random_data:
            torch.zeros it's called with N, H, W for the cover and the device

    TODO:
    """

    """
    METHOD:
        _encode_decode(self, cover, quantize=False)

    TESTS:
        test__encode_decode_quantize:
            test the return value when quantize it's True

        test__encode_decode_default:
            test the return value when quantize it's False
    """

    """
    METHOD:
        _critic(self, image)

    TESTS:
        test__critic:
            validate return value
    """

    """
    METHOD:
        _get_optimizers(self)

    TESTS:
        test__get_optimizers:
            validate return values calculated with the list of the decoder / encoder and Adam
    """

    """
    METHOD:
        _fit_critic(self, train, metrics)

    TESTS:
        test__fit_critic:
    """

    """
    METHOD:
        _fit_coders(self, train, metrics)

    TESTS:
        test__fit_coders:

    """

    """
    METHOD:
        _coding_scores(self, cover, generated, payload, decoded)

    TESTS:
        test__coding_scores:
            validate return values calculated in this method

    """

    """
    METHOD:
        _validate(self, validate, metrics)

    TESTS:
        test__validate:
    """

    """
    METHOD:
        _generate_samples(self, samples_path, cover, epoch)

    TESTS:
        test__generate_samples:
            validate that the method generates samples of images in the given path
    """

    """
    METHOD:
        fit(self, train, validate, epochs=5)

    TESTS:
        test_fit:
            test that this method invokes the other methods for fitting.

    TODO:
        if self.log_dir can be moved to a method that has that functionality
    """

    """
    METHOD:
        _make_payload(self, width, height, depth, text)

    TESTS:
        test__make_payload:
            validate return value of this method.
    """

    """
    METHOD:
        encode(self, cover, output, text)

    TESTS:
        test_encode:
            validate that the image has been encoded or generated?
    """

    """
    METHOD:
        decode(self, cover)

    TESTS:
        test_decode:
            validate that the image can be processed?
    """

    """
    METHOD:
        load(cls, path, cuda=True, verbose=False)

    TESTS:
        test_load:
            validate that a model can be loaded from a torch pickle
    """

    """
    METHOD:
        save(self, path)

    TESTS:
        test_save:
            validate that we are saving the fitted model
    """

    pass
