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
        """Test generate random data by calling torch.zeros"""

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
        """Test encode *random data* and decode data"""

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
        """Test apply quantize and then call decode"""

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
        """Test _critic is calling torch.mean method with the critic's tensor"""

        # setup
        image_fixture = torch.load(os.path.join(self.fixtures_path, 'tensors/img.ts'))
        steganogan = self.VoidSteganoGAN()
        steganogan.critic = MagicMock(return_value=image_fixture)

        # run
        result = steganogan._critic(image_fixture)

        # assert
        steganogan.critic.assert_called_once_with(image_fixture)

        assert ((result - torch.Tensor(np.array(-0.3195187))).abs() < 1e-5).all()

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
        """Test fit critic it's processing as expected"""

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
        """Test fit coders calls and proceeds the data."""

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
        """Test _validate method in SteganoGAN it's updating the metrics and others."""

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

    @patch('steganogan.models.SteganoGAN._encode_decode')
    @patch('steganogan.models.imageio.imwrite')
    def test__generate_samples(self, mock_imwrite, mock__encode_decode):
        """Test _generate_samples is saving to disk with imwrite"""

        # setup
        steganogan = self.VoidSteganoGAN()
        steganogan.device = 'some_device'

        cover = MagicMock()
        cover.to.return_value = cover
        cover_detach = cover.__getitem__.return_value.permute.return_value.detach
        cover_numpy = cover_detach.return_value.cpu.return_value.numpy.return_value
        cover_image = cover_numpy.__add__.return_value.__truediv__.return_value
        cover_image_iw = cover_image.__rmul__.return_value

        generated = MagicMock()
        generated.to.return_value = generated
        gen_permute = generated.__getitem__.return_value.clamp.return_value.permute
        gen_detach = gen_permute.return_value.detach
        gen_numpy = gen_detach.return_value.cpu.return_value.numpy.return_value
        gen_image = gen_numpy.__add__.return_value.__truediv__.return_value
        gen_image_iw = gen_image.__rmul__.return_value

        generated.size.return_value = 1

        payload = MagicMock()
        decoded = MagicMock()

        mock__encode_decode.return_value = (generated, payload, decoded)

        # run
        steganogan._generate_samples('test_path', cover, 10)

        # assert
        expected_iwrite_call = [
            call('test_path/0.cover.png', cover_image_iw.astype.return_value),
            call('test_path/0.generated-10.png', gen_image_iw.astype.return_value)
        ]

        cover.to.assert_called_once_with('some_device')

        cover.__getitem__.assert_called_once_with(0)
        cover.__getitem__.return_value.permute.assert_called_once_with(1, 2, 0)

        cover_detach.assert_called_once_with()
        cover_detach.return_value.cpu.assert_called_once_with()
        cover_detach.return_value.cpu.return_value.numpy.assert_called_once_with()

        cover_numpy.__add__.assert_called_once_with(1)
        cover_numpy.__add__.return_value.__truediv__.assert_called_once_with(2.0)

        generated.__getitem__.assert_called_once_with(0)
        generated.__getitem__.return_value.clamp.assert_called_once_with(-1.0, 1.0)

        gen_detach.assert_called_once_with()
        gen_detach.return_value.cpu.assert_called_once_with()
        gen_detach.return_value.cpu.return_value.numpy.assert_called_once_with()

        gen_numpy.__add__.assert_called_once_with(1)
        gen_numpy.__add__.return_value.__truediv__.assert_called_once_with(2.0)

        cover_image_iw.astype.assert_called_once_with('uint8')
        gen_image_iw.astype.assert_called_once_with('uint8')

        assert mock_imwrite.call_args_list == expected_iwrite_call

    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    @patch('steganogan.models.SteganoGAN._get_optimizers')
    def test_fit_optimizer_is_none(self, mock__get_optimizers, mock_cuda_empty, mock_gc_collect,
                                   mock__validate, mock__fit_coders, mock__fit_critic):
        """Test fit method without optimizers"""

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
        """Test with existing optimizers"""

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

    @patch('steganogan.models.SteganoGAN._generate_samples')
    @patch('steganogan.models.SteganoGAN.save')
    @patch('steganogan.models.json.dump')
    @patch('steganogan.models.open')
    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    @patch('steganogan.models.SteganoGAN._get_optimizers')
    def test_fit_with_log_dir(self, mock__get_optimizers, mock_cuda_empty, mock_gc_collect,
                              mock__validate, mock__fit_coders, mock__fit_critic, mock_open,
                              mock_dump, mock_save, mock__generate_samples):
        """Test fit with log_dir"""

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
        steganogan.epochs = 0
        steganogan.log_dir = 'any_dir'
        steganogan.samples_path = 'samples_path'
        steganogan.history = list()

        validate = MagicMock()
        validate.__iter__.return_value = ['1', '2']

        mock__validate.side_effect = validate_side_effect
        mock__fit_critic.side_effect = critic_side_effect
        mock__fit_coders.side_effect = coders_side_effect

        # run
        steganogan.fit('some_train', validate, epochs=1)

        # assert
        expected_metrics = {field: list() for field in models.METRIC_FIELDS}
        expected_history = [{field: 1.0 for field in models.METRIC_FIELDS}]
        expected_history[0]['epoch'] = 1

        mock__get_optimizers.assert_not_called()

        assert self.critic_dict_call == expected_metrics
        assert self.critic_item_call == 'some_train'

        assert self.coders_dict_call == expected_metrics
        assert self.coders_item_call == 'some_train'

        assert self.validate_dict_call == expected_metrics
        assert self.validate_item_call == validate

        assert steganogan.epochs == 1
        mock_gc_collect.assert_called_once_with()

        assert steganogan.history == expected_history
        mock_open.assert_called_once_with('any_dir/metrics.log', 'w')

        mock_dump.assert_called_once_with(
            expected_history,
            mock_open.return_value.__enter__.return_value,
            indent=4
        )

        mock_save.assert_called_once_with('any_dir/1.bpp-1.000000.p')
        mock__generate_samples.assert_called_once_with('samples_path', '1', 1)

    @patch('steganogan.models.SteganoGAN._fit_critic')
    @patch('steganogan.models.SteganoGAN._fit_coders')
    @patch('steganogan.models.SteganoGAN._validate')
    @patch('steganogan.models.gc.collect')
    @patch('steganogan.models.torch.cuda.empty_cache')
    def test_fit_with_cuda(self, mock_cuda_empty, mock_gc_collect, mock__validate,
                           mock__fit_coders, mock__fit_critic):
        """Test calling torch.cuda.empty when cuda it's true"""

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

    @patch('steganogan.models.torch.FloatTensor')
    @patch('steganogan.models.text_to_bits')
    def test__make_payload(self, mock_text_to_bits, mock_float_tensor):
        """Test that the return value it's as expected"""

        # setup
        steganogan = self.VoidSteganoGAN()

        payload = MagicMock()
        payload.__add__.return_value.__len__.return_value = 10000

        mock_text_to_bits.return_value = payload

        # run
        result = steganogan._make_payload(2, 3, 4, 'Hello')

        # assert
        expected_payload = payload.__add__.return_value[:1]
        expected_view_call = [call(1, 4, 3, 2), call(1, 4, 3, 2)]
        expected_result = mock_float_tensor.return_value.view(1, 4, 3, 2)

        mock_text_to_bits.assert_called_once_with('Hello')
        payload.__add__.assert_called_once_with([0] * 32)

        payload.__add__.return_value.__getitem__.call_list_args == [call(slice(None, 24, None)),
                                                                    call(slice(None, 24, None))]

        mock_float_tensor.assert_called_once_with(expected_payload)

        assert mock_float_tensor.return_value.view.call_args_list == expected_view_call
        assert result == expected_result

    @patch('steganogan.models.imread')
    @patch('steganogan.models.imwrite')
    @patch('steganogan.models.torch.FloatTensor')
    @patch('steganogan.models.SteganoGAN._make_payload')
    def test_encode(self, mock__make_payload, mock_float_tensor, mock_imwrite, mock_imread):
        """Test that we encode the imagewith our encoder"""

        # setup
        cover = MagicMock()
        cover = MagicMock()
        torched_cover = MagicMock()
        payload = MagicMock()
        generated = MagicMock()
        _gen = generated.__getitem__.return_value.clamp.return_value
        _generated = _gen.permute.return_value.detach.return_value.cpu.return_value
        _gen_astype = _generated.numpy.return_value.__add__.return_value.__mul__.return_value

        mock_imread.return_value = cover

        _cover_for_torch = cover.__truediv__.return_value.__sub__.return_value
        mock_float_tensor.return_value.permute.return_value.unsqueeze.return_value = torched_cover

        cover.to.return_value = cover
        payload.to.return_value = payload

        steganogan = self.VoidSteganoGAN()
        steganogan.device = 'some_device'
        steganogan.data_depth = 2
        steganogan.encoder = MagicMock(return_value=generated)
        steganogan.verbose = False

        mock__make_payload.return_value = payload

        # run
        steganogan.encode(cover, 'some_output', 'Hello')

        # assert
        mock_imread.assert_called_once_with(cover, pilmode='RGB')
        mock_float_tensor.assert_called_once_with(_cover_for_torch)
        mock_float_tensor.return_value.permute.assert_called_once_with(2, 1, 0)
        mock_float_tensor.return_value.permute.return_value.unsqueeze.assert_called_once_with(0)

        mock_float_tensor.return_value.permute.return_value.unsqueeze.assert_called_once_with(0)
        torched_cover.size.assert_called_once_with()
        assert torched_cover.size.return_value.__getitem__.call_args_list == [call(3), call(2)]

        mock__make_payload.assert_called_once_with(
            torched_cover.size.return_value.__getitem__.return_value,
            torched_cover.size.return_value.__getitem__.return_value,
            2,
            'Hello'
        )

        torched_cover.to.assert_called_once_with('some_device')
        payload.to.assert_called_once_with('some_device')

        steganogan.encoder.assert_called_once_with(torched_cover.to(), payload)
        generated.__getitem__.return_value.clamp.assert_called_once_with(-1.0, 1.0)

        _gen.permute.assert_called_once_with(2, 1, 0)
        _gen.permute.return_value.detach.assert_called_once_with()
        _gen.permute.return_value.detach.return_value.cpu.assert_called_once_with()
        _generated.numpy.assert_called_once_with()
        _generated.numpy.return_value.__add__.assert_called_once_with(1)
        _generated.numpy.return_value.__add__.return_value.__mul__.assert_called_once_with(127.5)
        _gen_astype.astype.assert_called_once_with('uint8')

        mock_imwrite.assert_called_once_with('some_output', _gen_astype.astype('unit8'))

    def test_decode_image_is_not_file(self):
        """Raise an exception if image is not a file"""

        # setup
        steganogan = self.VoidSteganoGAN()

        # run
        with self.assertRaises(ValueError):
            steganogan.decode('not_existing_path')

    @patch('steganogan.models.os.path.exists')
    @patch('steganogan.models.bytearray_to_text')
    @patch('steganogan.models.torch.FloatTensor')
    @patch('steganogan.models.imread')
    @patch('steganogan.models.bits_to_bytearray')
    def test_decode_image_zero_candidates(self, mock_bits, mock_imread, mock_float_tensor,
                                          mock_bytearray_to_text, mock_os):
        """Raise an exception when no candidates are found"""

        # setup
        steganogan = self.VoidSteganoGAN()
        steganogan.decoder = MagicMock()
        steganogan.device = 'some_device'

        image = MagicMock()
        tensor_image = MagicMock()
        tensor_image.to.return_value = tensor_image
        _image = MagicMock()

        mock_imread.return_value = image
        mock_float_tensor.return_value.permute.return_value.unsqueeze.return_value = tensor_image

        steganogan.decoder.return_value.view.return_value.__gt__.return_value = _image

        bits_image = _image.data.cpu.return_value.numpy.return_value.tolist.return_value

        mock_bytearray_to_text.return_value = None
        mock_os.return_value = True

        # run/assert
        with self.assertRaises(ValueError):
            steganogan.decode('existing_path')

        # asserts
        mock_imread.assert_called_once_with('existing_path', pilmode='RGB')
        tensor_image.to.assert_called_once_with('some_device')

        image.__truediv__.assert_called_once_with(255.0)

        mock_float_tensor.assert_called_once_with(image.__truediv__.return_value)
        mock_float_tensor.return_value.permute.assert_called_once_with(2, 1, 0)
        mock_float_tensor.return_value.permute.return_value.unsqueeze.assert_called_once_with(0)

        mock_bits.assert_called_once_with(bits_image)

    @patch('steganogan.models.os.path.exists')
    @patch('steganogan.models.bytearray_to_text')
    @patch('steganogan.models.torch.FloatTensor')
    @patch('steganogan.models.imread')
    @patch('steganogan.models.bits_to_bytearray')
    def test_decode_image(self, mock_bits, mock_imread, mock_float_tensor,
                          mock_bytearray_to_text, mock_os):
        """Test decode a image and finding a candidate"""

        # setup
        steganogan = self.VoidSteganoGAN()
        steganogan.decoder = MagicMock()
        steganogan.device = 'some_device'

        image = MagicMock()
        tensor_image = MagicMock()
        tensor_image.to.return_value = tensor_image

        _image = MagicMock()
        candidate = MagicMock()
        _candidate = MagicMock()

        mock_imread.return_value = image
        mock_float_tensor.return_value.permute.return_value.unsqueeze.return_value = tensor_image

        steganogan.decoder.return_value.view.return_value.__gt__.return_value = _image

        bits_image = _image.data.cpu.return_value.numpy.return_value.tolist.return_value
        mock_bits.return_value.split.return_value = [candidate]

        mock_bytearray_to_text.return_value = _candidate

        mock_os.return_value = True

        # run
        result = steganogan.decode('existing_path')

        # asserts
        mock_imread.assert_called_once_with('existing_path', pilmode='RGB')
        tensor_image.to.assert_called_once_with('some_device')

        image.__truediv__.assert_called_once_with(255.0)

        mock_float_tensor.assert_called_once_with(image.__truediv__.return_value)
        mock_float_tensor.return_value.permute.assert_called_once_with(2, 1, 0)
        mock_float_tensor.return_value.permute.return_value.unsqueeze.assert_called_once_with(0)

        mock_bits.assert_called_once_with(bits_image)
        mock_bytearray_to_text.assert_called_once_with(bytearray(candidate))

        assert result == _candidate

    @patch('steganogan.models.torch.save')
    def test_save(self, mock_save):
        """Test saving the instance with torch"""

        # setup
        steganogan = self.VoidSteganoGAN()

        # run
        steganogan.save('some_path')

        # assert
        mock_save.assert_called_once_with(steganogan, 'some_path')

    @patch('steganogan.models.os.path.join')
    @patch('steganogan.models.torch.load')
    def test_load(self, mock_load, os_path_mock):
        """Test loading a default architecture"""

        # setup
        steganogan = MagicMock()
        mock_load.return_value = steganogan

        # run
        result = models.SteganoGAN.load('some_architecture', cuda=False, verbose=False)

        # assert
        mock_load.assert_called_once_with(os_path_mock.return_value, map_location='cpu')

        assert not steganogan.verbose

        steganogan.encoder.upgrade_legacy.assert_called_once_with()
        steganogan.decoder.upgrade_legacy.assert_called_once_with()
        steganogan.critic.upgrade_legacy.assert_called_once_with()

        steganogan.set_device.assert_called_once_with(False)

        assert result == steganogan

    @patch('steganogan.models.os.path.join')
    @patch('steganogan.models.torch.load')
    def test_load_cuda_verbose_true(self, mock_load, os_path_mock):
        """Test loading a path, with cuda and verbose True"""

        # setup
        steganogan = MagicMock()
        mock_load.return_value = steganogan

        # run
        result = models.SteganoGAN.load(architecture='some_architecture', cuda=True, verbose=True)

        # assert
        mock_load.assert_called_once_with(os_path_mock.return_value, map_location='cpu')

        assert steganogan.verbose

        steganogan.encoder.upgrade_legacy.assert_called_once_with()
        steganogan.decoder.upgrade_legacy.assert_called_once_with()
        steganogan.critic.upgrade_legacy.assert_called_once_with()

        steganogan.set_device.assert_called_once_with(True)

        assert result == steganogan

    @patch('steganogan.models.torch.load')
    def test_load_path_no_architecture(self, mock_load):
        """Test loading a model when passing a path and architecture is None"""

        # setup
        steganogan = MagicMock()
        mock_load.return_value = steganogan

        # run
        result = models.SteganoGAN.load(
            architecture=None, path='some_path', cuda=True, verbose=True)

        # assert
        mock_load.assert_called_once_with('some_path', map_location='cpu')

        assert result == steganogan

    def test_load_path_and_architecture(self):

        # run / assert
        with self.assertRaises(ValueError):
            models.SteganoGAN.load(architecture='some_arch', path='some_path')
