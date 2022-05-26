"""Tests that supported models can call forward() and sample()."""

import tempfile
import unittest

import torch
from torch import distributions

from pytorch_generative import models
from pytorch_generative.models import autoregressive
from pytorch_generative.models import vae


class DummyLoader:
    """Dummy data loader used for integration testing."""

    def __init__(self, channels, size):
        self._xs = torch.rand((1, channels, size, size))
        self._ys = torch.tensor([0])

    def __iter__(self):
        self._exhausted = False
        return self

    def __next__(self):
        if not self._exhausted:
            self._exhausted = True
            return self._xs, self._ys
        raise StopIteration()


class IntegrationTests(unittest.TestCase):
    """Main (integration) tests for implemented models."""

    def _test_integration(self, module, in_channels=1, in_size=28):
        dummy_loader = DummyLoader(in_channels, in_size)
        with tempfile.TemporaryDirectory() as log_dir:
            module.reproduce(
                n_epochs=1, log_dir=log_dir, n_gpus=0, debug_loader=dummy_loader
            )

    def test_NADE(self):
        self._test_integration(autoregressive.nade)

    def test_MADE(self):
        self._test_integration(autoregressive.made)

    def test_PixelCNN(self):
        self._test_integration(autoregressive.pixel_cnn)

    def test_GatedPixelCNN(self):
        self._test_integration(autoregressive.gated_pixel_cnn)

    def test_PixelSnail(self):
        self._test_integration(autoregressive.pixel_snail)

    def test_ImageGPT(self):
        self._test_integration(autoregressive.image_gpt)

    def test_FullyVisibleBeliefNetwork(self):
        self._test_integration(autoregressive.fvbn)

    def test_VAE(self):
        self._test_integration(vae.vae, in_size=32)

    def test_BetaVAE(self):
        self._test_integration(vae.beta_vae, in_size=32)

    def test_VeryDeepVAE(self):
        self._test_integration(vae.vd_vae, in_size=32)

    def test_VectorQuantizedVAE(self):
        self._test_integration(vae.vq_vae, in_channels=3)

    def test_VectorQuantizedVAE2(self):
        self._test_integration(vae.vq_vae_2, in_channels=3)


class MultipleChannelsTests(unittest.TestCase):
    """Tests models correctness when using multiple input and output channels."""

    def _test_multiple_channels(self, model, conditional_sample=False):
        # Test forward().
        batch = torch.rand(2, 3, 8, 8)
        model(batch)

        # Test unconditional sample().
        model.sample(n_samples=2)

        # Test that conditional sample() only modifies pixels < 0.
        if conditional_sample:
            batch[:, :, 1:, :] = -1
            sample = model.sample(conditioned_on=batch)
            self.assertTrue((sample[:, :, 0, :] == batch[:, :, 0, :]).all())

    def test_PixelCNN(self):
        model = models.PixelCNN(
            in_channels=3,
            out_channels=3,
            n_residual=1,
            residual_channels=1,
            head_channels=1,
        )
        self._test_multiple_channels(model, conditional_sample=True)

    def test_GatedPixelCNN(self):
        model = models.GatedPixelCNN(
            in_channels=3, out_channels=3, n_gated=1, gated_channels=1, head_channels=1
        )
        self._test_multiple_channels(model, conditional_sample=True)

    def test_PixelSNAIL(self):
        model = models.PixelSNAIL(
            in_channels=3,
            out_channels=3,
            n_channels=2,
            n_pixel_snail_blocks=1,
            n_residual_blocks=1,
            attention_key_channels=1,
            attention_value_channels=1,
        )
        self._test_multiple_channels(model, conditional_sample=True)

    def test_ImageGPT(self):
        model = models.ImageGPT(
            in_channels=3,
            out_channels=3,
            in_size=8,
            n_transformer_blocks=1,
            n_attention_heads=2,
            n_embedding_channels=4,
        )
        self._test_multiple_channels(model, conditional_sample=True)

    def test_VAE(self):
        model = models.VAE(
            in_channels=3,
            out_channels=3,
            latent_channels=1,
            strides=[2, 2],
            hidden_channels=1,
            residual_channels=1,
        )
        self._test_multiple_channels(model)

    def test_VeryDeepVAE(self):
        from pytorch_generative.models.vae.vd_vae import StackConfig

        model = models.VeryDeepVAE(
            in_channels=3,
            out_channels=3,
            input_resolution=8,
            stack_configs=[
                StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
                StackConfig(n_encoder_blocks=1, n_decoder_blocks=1),
            ],
            latent_channels=1,
            bottleneck_channels=1,
        )
        self._test_multiple_channels(model)

    def test_MixtureModel(self):
        n_features = 3 * 8 * 8
        # Test GaussianMixtureModel.
        model = models.GaussianMixtureModel(n_components=3, n_features=n_features)
        self._test_multiple_channels(model)

        # Test BernoulliMixtureModel.
        model = models.BernoulliMixtureModel(n_components=3, n_features=n_features)
        self._test_multiple_channels(model)


class TestKernelDensityEstimators(unittest.TestCase):
    def test_smoke_tests(self):
        train_Xs = batch = torch.rand(4, 3)

        # Test ParzenWindowKernel.
        model = models.KernelDensityEstimator(
            kernel=models.ParzenWindowKernel(bandwidth=0.1), train_Xs=train_Xs
        )
        model(batch)
        model.sample(2)

        # Test GaussianKernel.
        model = models.KernelDensityEstimator(
            kernel=models.GaussianKernel(bandwidth=0.1), train_Xs=train_Xs
        )
        model(batch)
        model.sample(2)

    def test_multidimensional_support_GaussianKernel(self):
        # Gaussian 2D data.
        train_Xs = torch.normal(torch.zeros((100, 2)), torch.ones((100, 2)))
        kernel = models.GaussianKernel(bandwidth=1.0)
        model = models.KernelDensityEstimator(train_Xs, kernel)

        # Build meshgrid.
        dx = 0.1
        X = torch.arange(-8, 8, dx)
        Y = torch.arange(-8, 8, dx)
        xx, yy = torch.meshgrid(X, Y)
        meshgrid = torch.stack((xx, yy), axis=2).view(-1, 2)

        log_probs = model(meshgrid)
        integral = torch.sum(torch.exp(log_probs) * dx * dx)
        torch.testing.assert_allclose(integral, 1.0)

    def test_multidimensional_support_ParzenWindowKernel(self):
        # Gaussian 2D data.
        train_Xs = torch.normal(torch.zeros((100, 2)), torch.ones((100, 2)))
        kernel = models.ParzenWindowKernel(bandwidth=1.0)
        model = models.KernelDensityEstimator(train_Xs, kernel)

        # Build meshgrid.
        dx = 0.1
        X = torch.arange(-8, 8, dx)
        Y = torch.arange(-8, 8, dx)
        xx, yy = torch.meshgrid(X, Y)
        meshgrid = torch.stack((xx, yy), axis=2).view(-1, 2)

        log_probs = model(meshgrid)
        integral = torch.sum(torch.exp(log_probs) * dx * dx)
        torch.testing.assert_allclose(integral, 1.0)
