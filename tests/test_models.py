# -*- coding: utf-8 -*-
import os
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from steganogan import critics, decoders, encoders, models
from tests.utils import assert_called_with_tensors


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

        class Test:
            def __init__(self):
                pass

        # setup
        steganogan = self.VoidSteganoGAN()

        # run
        encoder = steganogan._get_instance(
            encoders.BasicEncoder, {'hidden_size': 2, 'data_depth': 1})

        critic = steganogan._get_instance(
            critics.BasicCritic, {'hidden_size': 2, 'data_depth': 1})

        decoder = steganogan._get_instance(
            decoders.BasicDecoder, {'hidden_size': 2, 'data_depth': 1})

        test = steganogan._get_instance(Test, {})

        # assert
        assert isinstance(encoder, encoders.BasicEncoder)
        assert isinstance(critic, critics.BasicCritic)
        assert isinstance(decoder, decoders.BasicDecoder)
        assert isinstance(test, Test)

    def test__get_instance_is_instance(self):
        """Test when given a instance of a class it returns the same instace of it"""

        class Test:
            def __init__(self):
                pass

        # setup
        steganogan = self.VoidSteganoGAN()
        encoder = encoders.BasicEncoder(1, 2)
        decoder = decoders.BasicDecoder(1, 2)
        critic = critics.BasicCritic(1)
        test = Test()

        # run
        res_encoder = steganogan._get_instance(encoder, {})
        res_decoder = steganogan._get_instance(decoder, {})
        res_critic = steganogan._get_instance(critic, {})
        res_test = steganogan._get_instance(test, {})

        # assert
        assert res_encoder == encoder
        assert res_decoder == decoder
        assert res_critic == critic
        assert res_test == test

    @patch('steganogan.models.torch.cuda.is_available')
    def test_set_device_cuda(self, mock_cuda_is_available):
        """Test that we create a device with torch.cuda.device('cuda') for our architectures"""

        # setup
        mock_cuda_is_available.return_value = True
        steganogan = self.VoidSteganoGAN()

        steganogan.verbose = False
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

        steganogan.verbose = False  # needed inside the method
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

    @patch('steganogan.models.SteganoGAN._random_data')
    def test__encode_decode_quantize_false(self, mock__random_data):
        """Test that we encode *random data* and then decode it. """

        # setup
        random_data_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/random.ts'))
        mock__random_data.return_value = random_data_fixture

        cover_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))

        steganogan = self.VoidSteganoGAN()
        steganogan.encoder = MagicMock(return_value=cover_fixture)
        steganogan.decoder = MagicMock(return_value=MagicMock())

        # run
        generated, payload, decoded = steganogan._encode_decode(cover_fixture)

        # assert
        mock__random_data.called_once_with(cover_fixture)

        steganogan.encoder.assert_called_once_with(cover_fixture, random_data_fixture)
        steganogan.decoder.assert_called_once_with(cover_fixture)

        assert (generated == cover_fixture).all()
        assert (payload == random_data_fixture).all()
        assert decoded == steganogan.decoder.return_value

    @patch('steganogan.models.SteganoGAN._random_data')
    def test__encode_decode_quantize_true(self, mock__random_data):
        """Test that we apply quantize and then decode it. """

        # setup
        random_data_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/random.ts'))
        mock__random_data.return_value = random_data_fixture

        cover_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))

        steganogan = self.VoidSteganoGAN()
        steganogan.encoder = MagicMock(return_value=cover_fixture)
        steganogan.decoder = MagicMock(return_value=MagicMock())

        # run
        generated, payload, decoded = steganogan._encode_decode(cover_fixture, quantize=True)

        # assert
        expected_quantize_cover = (255 * (cover_fixture + 1.0) / 2.0).long()
        expected_quantize_cover = 2.0 * expected_quantize_cover.float() / 255.0 - 1.0

        mock__random_data.called_once_with(cover_fixture)

        steganogan.encoder.assert_called_once_with(cover_fixture, random_data_fixture)
        assert_called_with_tensors(steganogan.decoder, [call(expected_quantize_cover)])

        assert (generated == expected_quantize_cover).all()
        assert (payload == random_data_fixture).all()
        assert decoded == steganogan.decoder.return_value

    def test__critic(self):
        """Test that _critic it's calling torch.mean method with the critic's tensor"""

        # setup
        image_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))
        steganogan = self.VoidSteganoGAN()
        steganogan.critic = MagicMock(return_value=image_fixture)

        # run
        result = steganogan._critic(image_fixture)

        # assert
        steganogan.critic.assert_called_once_with(image_fixture)

        assert (result == torch.Tensor(np.array(-0.3195187))).all()

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

        steganogan.decoder.parameters.assert_called_once_with()
        steganogan.encoder.parameters.assert_called_once_with()
        steganogan.critic.parameters.assert_called_once_with()

        assert mock_Adam.call_args_list == expected_Adam_calls
        assert result_1 == 1
        assert result_2 == 1

    @patch('steganogan.models.gc')
    @patch('steganogan.models.SteganoGAN._critic')
    @patch('steganogan.models.SteganoGAN._random_data')
    def test__fit_critic(self, mock__random_data, mock__critic, mock_gc):
        """Test that fit critic it's being called properly"""

        # setup
        cover = MagicMock()
        cover.to.return_value = cover  # This way we use cover instead of cover.to

        generated = MagicMock()
        payload = MagicMock()
        cover_score = MagicMock()
        generated_score = MagicMock()

        mock__critic.side_effect = [cover_score, generated_score]
        mock__random_data.return_value = payload

        cover_backward = cover_score - generated_score

        steganogan = self.VoidSteganoGAN()
        steganogan.verbose = False
        steganogan.encoder = MagicMock(return_value=generated)
        steganogan.critic = MagicMock()
        steganogan.device = MagicMock()
        steganogan.critic_optimizer = MagicMock()

        metrics = {'train.cover_score': list(), 'train.generated_score': list()}

        # run
        steganogan._fit_critic([(cover, 1)], metrics)

        # assert
        mock_gc.collect.assert_called_once_with()
        cover.to.assert_called_once_with(steganogan.device)
        mock__random_data.asser_called_once_with(cover)

        steganogan.critic_optimizer.zero_grad.assert_called_once_with()
        steganogan.encoder.assert_called_once_with(cover, payload)
        steganogan.critic_optimizer.step.assert_called_once_with()
        steganogan.critic.parameters.assert_called_once_with()

        cover_backward.backward.assert_called_once_with(retain_graph=False)

        assert mock__critic.call_args_list == [call(cover), call(generated)]
        assert metrics['train.cover_score'][0] == cover_score.item()
        assert metrics['train.generated_score'][0] == generated_score.item()

    @patch('steganogan.models.SteganoGAN._coding_scores')
    @patch('steganogan.models.gc')
    @patch('steganogan.models.SteganoGAN._critic')
    @patch('steganogan.models.SteganoGAN._encode_decode')
    def test__fit_coders(self, mock__encode_decode, mock__critic, mock_gc, mock__coding_scores):
        """Test that fit coders calls and proceeds the data."""

        # setup
        cover = MagicMock()
        cover.to.return_value = cover
        generated = MagicMock()
        payload = MagicMock()
        decoded = MagicMock()

        encoder_mse = MagicMock()
        decoder_loss = MagicMock()
        decoder_acc = MagicMock()

        generated_score = MagicMock()

        mock__encode_decode.return_value = (generated, payload, decoded)

        mock__critic.return_value = generated_score
        mock__coding_scores.return_value = (encoder_mse, decoder_loss, decoder_acc)

        mock_backward = (100.0 * encoder_mse + decoder_loss + generated_score)

        steganogan = self.VoidSteganoGAN()
        steganogan.verbose = False
        steganogan.critic = MagicMock()
        steganogan.device = MagicMock()
        steganogan.decoder_optimizer = MagicMock()

        metrics = {
            'train.encoder_mse': list(),
            'train.decoder_loss': list(),
            'train.decoder_acc': list(),
        }

        # run
        steganogan._fit_coders([(cover, 1)], metrics)

        # assert
        mock_gc.collect.assert_called_once_with()
        cover.to.assert_called_once_with(steganogan.device)

        mock__encode_decode.assert_called_once_with(cover)
        mock__coding_scores.assert_called_once_with(cover, generated, payload, decoded)
        mock__critic.assert_called_once_with(generated)

        steganogan.decoder_optimizer.zero_grad.assert_called_once_with()
        mock_backward.backward.assert_called_once_with()
        steganogan.decoder_optimizer.step.assert_called_once_with()

        assert metrics['train.encoder_mse'][0] == encoder_mse.item()
        assert metrics['train.decoder_loss'][0] == decoder_loss.item()
        assert metrics['train.decoder_acc'][0] == decoder_acc.item()

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
    @patch('steganogan.models.mse_loss')
    @patch('steganogan.models.binary_cross_entropy_with_logits')
    def test__coding_scores(self, mock_binary, mock_mse_loss):
        """Test _coding_scores method that returns expected value"""

        # setup
        payload = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))
        decoded = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))
        mock_binary.return_value = MagicMock()
        mock_mse_loss.return_value = MagicMock()

        cover = MagicMock()
        generated = MagicMock()

        steganogan = self.VoidSteganoGAN()

        # run
        res_encoder_mse, res_decoder_loss, res_decoder_acc = steganogan._coding_scores(
            cover, generated, payload, decoded)

        # assert
        expected_dec_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        mock_mse_loss.assert_called_once_with(generated, cover)
        mock_binary.assert_called_once_with(decoded, payload)

        assert res_encoder_mse == mock_mse_loss.return_value
        assert res_decoder_loss == mock_binary.return_value
        assert res_decoder_acc == expected_dec_acc

    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.SteganoGAN._encode_decode')
    @patch('steganogan.models.SteganoGAN._coding_scores')
    @patch('steganogan.models.SteganoGAN._critic')
    @patch('steganogan.models.ssim')
    @patch('steganogan.models.torch.log10')
    def test__validate(self, mock_torch_log, mock_ssim, mock__critic,
                       mock__coding_scores, mock__encode_decode, mock_gc):
        """Test _validate method in SteganoGAN"""

        # setup
        cover = MagicMock()
        cover.to.return_value = cover

        generated = MagicMock()
        payload = MagicMock()
        decoded = MagicMock()

        encoder_mse = MagicMock()
        decoder_loss = MagicMock()
        decoder_acc = MagicMock()
        decoder_acc.item.return_value = 1

        generated_score = MagicMock()
        cover_score = MagicMock()

        ssim_return_mock = MagicMock()
        log_mock = MagicMock()
        log_mock.item.return_value = 2.0
        # log_res_mock = 10 * log_mock  # TODO: assert that this .item has been called

        mock__encode_decode.return_value = (generated, payload, decoded)
        mock_ssim.return_value = ssim_return_mock
        mock_torch_log.return_value = log_mock

        mock__critic.side_effect = [generated_score, cover_score]
        mock__coding_scores.return_value = (encoder_mse, decoder_loss, decoder_acc)

        steganogan = self.VoidSteganoGAN()
        steganogan.verbose = False
        steganogan.critic = MagicMock()
        steganogan.device = MagicMock()
        steganogan.decoder_optimizer = MagicMock()
        steganogan.data_depth = 1

        metrics = {
            'val.encoder_mse': list(),
            'val.decoder_loss': list(),
            'val.decoder_acc': list(),
            'val.cover_score': list(),
            'val.generated_score': list(),
            'val.ssim': list(),
            'val.psnr': list(),
            'val.bpp': list(),
        }

        # run
        steganogan._validate([(cover, 1)], metrics)

        # assert
        expected_metrics = {
            'val.encoder_mse': [encoder_mse.item()],
            'val.decoder_loss': [decoder_loss.item()],
            'val.decoder_acc': [decoder_acc.item()],
            'val.cover_score': [cover_score.item()],
            'val.generated_score': [generated_score.item()],
            'val.ssim': [ssim_return_mock.item()],
            'val.psnr': [20.0],
            'val.bpp': [1],
        }

        mock_gc.assert_called_once_with()
        mock__encode_decode.assert_called_once_with(cover, quantize=True)
        mock__coding_scores.assert_called_once_with(cover, generated, payload, decoded)
        mock__critic.call_args_list == [call(generated), call(cover)]

        assert decoder_acc.item.call_args_list == [call(), call(), call()]
        assert encoder_mse.item.call_args_list == [call(), call()]
        assert decoder_loss.item.call_args_list == [call(), call()]
        assert cover_score.item.call_args_list == [call(), call()]

        mock_ssim.called_once_with(cover, generated)
        ssim_return_mock.item.called_once_with()
        mock_torch_log.assert_called_once_with(4 / encoder_mse)
        log_mock.item.called_once_with()

        assert expected_metrics == metrics

    """
    METHOD:
        _generate_samples(self, samples_path, cover, epoch)

    TESTS:
        test__generate_samples:
            validate that the method generates samples of images in the given path

    MOCK:
        imageio.imwrite
    """
    @patch('steganogan.models.SteganoGAN._encode_decode')
    @patch('steganogan.models.imageio.imwrite')
    def test__generate_samples(self, mock_imwrite, mock__encode_decode):
        """Validate that we are calling to imwrite in order to save to disk"""

        # setup
        steganogan = self.VoidSteganoGAN()
        steganogan.device = MagicMock()

        cover = MagicMock()
        cover.to.return_value = cover
        cover_permute_mock = MagicMock()
        cover_detach_mock = MagicMock()
        cover_cpu_mock = MagicMock()

        cover[0].permute.return_value = cover_permute_mock
        cover_permute_mock.detach.return_value = cover_detach_mock
        cover_detach_mock.return_value = cover_cpu_mock
        cover_cpu_mock.numpy.return_value = 1.0

        generated = MagicMock()
        generated.to.return_value = generated
        generated_permute_mock = MagicMock()
        generated_detach_mock = MagicMock()
        generated_cpu_mock = MagicMock()

        generated[0].permute.return_value = generated_permute_mock
        generated_permute_mock.detach.return_value = generated_detach_mock
        generated_detach_mock.return_value = generated_cpu_mock
        generated_cpu_mock.numpy.return_value = 1.0

        payload = MagicMock()
        decoded = MagicMock()

        generated.size.return_value = 1

        mock__encode_decode.return_value = (generated, payload, decoded)

        # run
        # steganogan._generate_samples(samples_path='test_path', cover, 10)

        # assert
        # generated.size.assert_called_once_with(0)
        # cover.to.assert_called_once_with(steganogan.device)

        # cover_permute_mock.assert_called_once_with(1, 2, 0)
        # cover_detach.assert_called_once_with()
        # cover_cpu.assert_called_once_with()
        # cover_cpu.numpy.assert_called_once_with()

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
    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    @patch('steganogan.models.SteganoGAN._get_optimizers')
    def test_fit_optimizer_is_none(self, mock__get_optimizers, mock_cuda_empty, mock_gc_collect,
                                   mock__validate, mock__fit_coders, mock__fit_critic):
        """
        Ensure that _get_optimizers it's being called and epochs it's being reset.
        Ensure that the rest of code runs as expected.
        """

        # setup
        def validate_side_effect(x, metrics):
            self.validate_item_call = deepcopy(x)
            self.validate_dict_call = deepcopy(metrics)
            for k, v in metrics.items():
                v.append(1)

        def critic_side_effect(x, metrics):
            self.critic_item_call = deepcopy(x)
            self.critic_dict_call = deepcopy(metrics)

        def coders_side_effect(x, metrics):
            self.coders_item_call = deepcopy(x)
            self.coders_dict_call = deepcopy(metrics)

        steganogan = self.VoidSteganoGAN()
        steganogan.log_dir = None
        steganogan.verbose = False
        steganogan.critic_optimizer = None
        steganogan.cuda = False

        mock__get_optimizers.return_value = (1, 2)

        mock__validate.side_effect = validate_side_effect
        mock__fit_critic.side_effect = critic_side_effect
        mock__fit_coders.side_effect = coders_side_effect

        # run
        steganogan.fit('some_train', 'some_validate', epochs=1)

        # assert
        expected_metrics = {field: list() for field in models.METRIC_FIELDS}

        mock__get_optimizers.assert_called_once_with()
        assert steganogan.critic_optimizer == 1
        assert steganogan.decoder_optimizer == 2

        assert self.critic_dict_call == expected_metrics
        assert self.critic_item_call == 'some_train'

        assert self.coders_dict_call == expected_metrics
        assert self.coders_item_call == 'some_train'

        assert self.validate_dict_call == expected_metrics
        assert self.validate_item_call == 'some_validate'

        assert steganogan.epochs == 1

        mock_gc_collect.assert_called_once_with()

    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    @patch('steganogan.models.SteganoGAN._get_optimizers')
    def test_fit_with_optimizers(self, mock__get_optimizers, mock_cuda_empty, mock_gc_collect,
                                 mock__validate, mock__fit_coders, mock__fit_critic):
        """
        Ensure that _get_optimizers it's being called and epochs it's being reset.
        Ensure that the rest of code runs as expected.
        """

        # setup
        def validate_side_effect(x, metrics):
            self.validate_item_call = deepcopy(x)
            self.validate_dict_call = deepcopy(metrics)
            for k, v in metrics.items():
                v.append(1)

        def critic_side_effect(x, metrics):
            self.critic_item_call = deepcopy(x)
            self.critic_dict_call = deepcopy(metrics)

        def coders_side_effect(x, metrics):
            self.coders_item_call = deepcopy(x)
            self.coders_dict_call = deepcopy(metrics)

        steganogan = self.VoidSteganoGAN()
        steganogan.log_dir = None
        steganogan.verbose = False
        steganogan.critic_optimizer = 'some_optimizer'
        steganogan.cuda = False
        steganogan.epochs = 1

        mock__validate.side_effect = validate_side_effect
        mock__fit_critic.side_effect = critic_side_effect
        mock__fit_coders.side_effect = coders_side_effect

        # run
        steganogan.fit('some_train', 'some_validate', epochs=1)

        # assert
        expected_metrics = {field: list() for field in models.METRIC_FIELDS}

        mock__get_optimizers.assert_not_called()

        assert self.critic_dict_call == expected_metrics
        assert self.critic_item_call == 'some_train'

        assert self.coders_dict_call == expected_metrics
        assert self.coders_item_call == 'some_train'

        assert self.validate_dict_call == expected_metrics
        assert self.validate_item_call == 'some_validate'

        assert steganogan.epochs == 2

        mock_gc_collect.assert_called_once_with()

    def test_fit_with_log_dir(self):
        """Test that the block of code after if log_dir it's executed"""
        # TODO: fix models first?
        pass

    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    def test_fit_with_cuda(self, mock_cuda_empty, mock_gc_collect, mock__validate,
                           mock__fit_coders, mock__fit_critic):
        """Test that we call the torch.cuda.empty when cuda it's true"""

        # setup
        def validate_side_effect(x, metrics):
            self.validate_item_call = deepcopy(x)
            self.validate_dict_call = deepcopy(metrics)
            for k, v in metrics.items():
                v.append(1)

        def critic_side_effect(x, metrics):
            self.critic_item_call = deepcopy(x)
            self.critic_dict_call = deepcopy(metrics)

        def coders_side_effect(x, metrics):
            self.coders_item_call = deepcopy(x)
            self.coders_dict_call = deepcopy(metrics)

        steganogan = self.VoidSteganoGAN()
        steganogan.log_dir = None
        steganogan.verbose = False
        steganogan.critic_optimizer = 'some_optimizer'
        steganogan.cuda = True
        steganogan.epochs = 1

        mock__validate.side_effect = validate_side_effect
        mock__fit_critic.side_effect = critic_side_effect
        mock__fit_coders.side_effect = coders_side_effect

        # run
        steganogan.fit('some_train', 'some_validate', epochs=1)

        # assert
        expected_metrics = {field: list() for field in models.METRIC_FIELDS}

        assert self.critic_dict_call == expected_metrics
        assert self.critic_item_call == 'some_train'

        assert self.coders_dict_call == expected_metrics
        assert self.coders_item_call == 'some_train'

        assert self.validate_dict_call == expected_metrics
        assert self.validate_item_call == 'some_validate'

        assert steganogan.epochs == 2

        mock_cuda_empty.assert_called_once_with()

        mock_gc_collect.assert_called_once_with()

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
