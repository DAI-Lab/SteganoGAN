# -*- coding: utf-8 -*-
import numpy as np

import os
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import torch

from steganogan import critics, decoders, encoders, models


class TestSteganoGAN(TestCase):

    @classmethod
    def setUpClass(cls):
        test_path = os.path.dirname(__file__)
        cls.fixtures_path = os.path.join(test_path, 'fixtures')

    class VoidSteganoGAN(models.SteganoGAN):
        """Can be used to create a steganogan class with empty attributes"""
        def __init__(self):
            pass

    def test__get_instance_is_class(self):
        """Test when given an class it returns a instance of it"""

        # setup
        steganogan = self.VoidSteganoGAN()

        # run
        encoder = steganogan._get_instance(
            encoders.BasicEncoder, {'hidden_size': 2, 'data_depth': 1})

        critic = steganogan._get_instance(
            critics.BasicCritic, {'hidden_size': 2, 'data_depth': 1})

        decoder = steganogan._get_instance(
            decoders.BasicDecoder, {'hidden_size': 2, 'data_depth': 1})

        # assert
        assert isinstance(encoder, encoders.BasicEncoder)
        assert isinstance(critic, critics.BasicCritic)
        assert isinstance(decoder, decoders.BasicDecoder)

    def test__get_instance_is_instance(self):
        """Test when given a instance of a class it returns the same instace of it"""

        # setup
        steganogan = self.VoidSteganoGAN()
        encoder = encoders.BasicEncoder(1, 2)
        decoder = decoders.BasicDecoder(1, 2)
        critic = critics.BasicCritic(1)

        # run
        res_encoder = steganogan._get_instance(encoder, {})
        res_decoder = steganogan._get_instance(decoder, {})
        res_critic = steganogan._get_instance(critic, {})

        # assert
        assert res_encoder == encoder
        assert res_decoder == decoder
        assert res_critic == critic

    @patch('steganogan.models.torch.cuda.is_available')
    def test_set_device_cuda(self, mock_cuda_is_available):
        """Test that we create a device with torch.cuda.device('cuda') for our architectures"""

        # setup
        mock_cuda_is_available.return_value = True

        steganogan = self.VoidSteganoGAN()

        steganogan.verbose = True  # needed inside the method
        steganogan.encoder = MagicMock()
        steganogan.decoder = MagicMock()
        steganogan.critic = MagicMock()

        # run
        steganogan.set_device()

        # assert
        assert steganogan.device == torch.device('cuda')
        steganogan.encoder.to.assert_called_once_with(torch.device('cuda'))
        steganogan.decoder.to.assert_called_once_with(torch.device('cuda'))
        steganogan.critic.to.assert_called_once_with(torch.device('cuda'))

    @patch('steganogan.models.torch.cuda.is_available')
    def test_set_device_cpu(self, mock_cuda_is_available):
        """Test that we create a device with torch.cuda.device('cuda') for our architectures"""

        # setup
        mock_cuda_is_available.return_value = True

        steganogan = self.VoidSteganoGAN()

        steganogan.verbose = True  # needed inside the method
        steganogan.encoder = MagicMock()
        steganogan.decoder = MagicMock()
        steganogan.critic = MagicMock()

        # run
        steganogan.set_device(cuda=False)

        # assert
        assert steganogan.device == torch.device('cpu')
        steganogan.encoder.to.assert_called_once_with(torch.device('cpu'))
        steganogan.decoder.to.assert_called_once_with(torch.device('cpu'))
        steganogan.critic.to.assert_called_once_with(torch.device('cpu'))

    @patch('os.makedirs')
    @patch('steganogan.models.SteganoGAN.set_device')
    @patch('steganogan.models.SteganoGAN._get_instance')
    def test___init__without_logdir(self, mock__get_instance, mock_set_device, mock_os_makedirs):
        """Test creating an instance of SteganoGAN without a log dir"""

        # run
        steganogan = models.SteganoGAN(1, encoders.BasicEncoder, decoders.BasicDecoder,
                                       critics.BasicCritic, hidden_size=5)

        # assert
        expected_get_instance_calls = [
            call(encoders.BasicEncoder, {'hidden_size': 5, 'data_depth': 1}),
            call(decoders.BasicDecoder, {'hidden_size': 5, 'data_depth': 1}),
            call(critics.BasicCritic, {'hidden_size': 5, 'data_depth': 1}),
        ]

        mock_set_device.assert_called_once_with(False)
        mock_os_makedirs.assert_not_called()
        assert mock__get_instance.call_args_list == expected_get_instance_calls

        assert steganogan.data_depth == 1
        assert steganogan.critic_optimizer is None
        assert steganogan.decoder_optimizer is None
        assert steganogan.fit_metrics is None
        assert steganogan.history == list()

    @patch('os.makedirs')
    @patch('steganogan.models.SteganoGAN.set_device')
    @patch('steganogan.models.SteganoGAN._get_instance')
    def test___init__log_dir(self, mock__get_instance, mock_set_device, mock_os_makedirs):
        """Test creating an instance of SteganoGAN with a log dir"""

        # run
        steganogan = models.SteganoGAN(1, encoders.BasicEncoder, decoders.BasicDecoder,
                                       critics.BasicCritic, log_dir='test_dir', hidden_size=5)

        # assert
        expected_get_instance_calls = [
            call(encoders.BasicEncoder, {'hidden_size': 5, 'data_depth': 1}),
            call(decoders.BasicDecoder, {'hidden_size': 5, 'data_depth': 1}),
            call(critics.BasicCritic, {'hidden_size': 5, 'data_depth': 1}),
        ]

        expected_os_makedirs_calls = [
            call('test_dir', exist_ok=True),
            call(os.path.join('test_dir', 'samples'), exist_ok=True),
        ]

        mock_set_device.assert_called_once_with(False)
        mock_os_makedirs.call_args_list == expected_os_makedirs_calls
        assert mock__get_instance.call_args_list == expected_get_instance_calls

        assert steganogan.data_depth == 1
        assert steganogan.critic_optimizer is None
        assert steganogan.decoder_optimizer is None
        assert steganogan.fit_metrics is None
        assert steganogan.history == list()

    @patch('steganogan.models.torch.zeros')
    def test__random_data(self, mock_torch_zeros):
        """Test that we generate random data by calling torch.zeros"""

        # setup
        cover = MagicMock()
        cover.size.return_value = (1, 2, 3, 4)

        steganogan = self.VoidSteganoGAN()
        steganogan.device = 'cpu'
        steganogan.data_depth = 1

        # run
        steganogan._random_data(cover)

        # assert
        mock_torch_zeros.called_once_with((1, 1, 3, 4), device='cpu')
        mock_torch_zeros.random_.called_once_with(0, 2)

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
    @patch('steganogan.models.SteganoGAN._random_data')
    def test__encode_decode_quantize_false(self, mock__random_data):
        """Test that we encode *random data* and then decode it. """

        # setup
        random_data_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/random.ts'))
        mock__random_data.return_value = random_data_fixture

        cover_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))

        steganogan = self.VoidSteganoGAN()
        steganogan.encoder = MagicMock()
        steganogan.decoder = MagicMock()
        steganogan.encoder.return_value = cover_fixture
        steganogan.decoder.return_value = random_data_fixture

        # run
        generated, payload, decoded = steganogan._encode_decode(cover_fixture)

        # assert
        steganogan.encoder.assert_called_once_with(cover_fixture, random_data_fixture)
        steganogan.decoder.assert_called_once_with(cover_fixture)

        assert (generated == cover_fixture).all()
        assert (payload == random_data_fixture).all()
        assert (decoded == random_data_fixture).all()

    def test__encode_decode_quantize_true(self):
        """Test that we aply quantize to our generated image"""
        pass

    def test__critic(self):
        """Test that _critic it's calling torch.mean method with the critic's tensor"""

        # setup
        image_fixture  = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))
        steganogan = self.VoidSteganoGAN()
        steganogan.critic = MagicMock(return_value=image_fixture)

        # run
        result = steganogan._critic(image_fixture)

        # assert
        steganogan.critic.assert_called_once_with(image_fixture)

        assert (result == torch.Tensor(np.array(-0.3195187))).all()

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
    @patch('steganogan.models.Adam')
    def test__get_optimizers(self, mock_Adam):
        """Test the return values of this method"""

        # setup
        mock_Adam.return_value = 1
        steganogan = self.VoidSteganoGAN()

        steganogan.decoder = MagicMock()
        steganogan.encoder = MagicMock()
        steganogan.critic = MagicMock()

        steganogan.decoder.parameters.return_value = [1]
        steganogan.encoder.parameters.return_value = [1]
        steganogan.critic.parameters.return_value = [1]

        # run
        result_1, result_2 = steganogan._get_optimizers()

        # assert
        expected_Adam_calls = [call([1], lr=1e-4), call([1, 1], lr=1e-4)]

        steganogan.decoder.parameters.assert_called_once()
        steganogan.encoder.parameters.assert_called_once()
        steganogan.critic.parameters.assert_called_once()

        assert mock_Adam.call_args_list == expected_Adam_calls
        assert result_1 == 1
        assert result_2 == 1

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
    @patch('steganogan.models.SteganoGAN._random_data')
    @patch('steganogan.models.SteganoGAN._critic')
    def test__fit_critic(self, mock__random_data, mock__critic):
        """Test that fit critic it's being called properly"""

        # setup
        cover = MagicMock()
        train = [(cover, 1)]

        mock__critic.return_value = cover
        mock__random_data.return_value = cover

        steganogan = self.VoidSteganoGAN()
        steganogan.verbose = True
        steganogan.encoder = MagicMock()
        steganogan.critic = MagicMock()
        steganogan.device = MagicMock()
        steganogan.critic_optimizer = MagicMock()

        metrics = {'train.cover_score': list(), 'train.generated_score': list()}

        # run
        steganogan._fit_critic(train, metrics)

        # assert
        steganogan.encoder.assert_called_once_with(cover.to(), cover)
        mock__critic.assert_called_once_with(cover.to())

        steganogan.critic_optimizer.zero_grad.assert_called_once()
        steganogan.critic_optimizer.step.assert_called_once()
        steganogan.critic.parameters.assert_called_once()

        assert metrics['train.cover_score'] is not list()
        assert metrics['train.generated_score'] is not list()

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
